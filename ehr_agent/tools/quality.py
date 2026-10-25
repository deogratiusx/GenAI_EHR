"""Data-quality tools: contradictions between documents and corrupted exports.

Checks (deterministic):
  * anthropometry — height/weight from numeric fields and free text, conflicting values across documents;
  * lab identities — MCHC ≈ Hb/Hct×100, MCH ≈ Hb/RBC, WBC differential ≈ 100 %, bilirubin total = direct + indirect;
  * text in numeric lab fields — a phrase from another document in a lab cell (row shift in the export);
  * implausible values — outside physiological hard limits (`analytes.PLAUSIBLE`).
A panel with any of these is "suspicious": every value from it carries a НЕДОСТОВЕРНО mark in the lab
tools, because models do not reliably call this tool by themselves.
"""

from __future__ import annotations

import re

from ..knowledge.analytes import implausible
from ..sources import Record
from .base import spec

# Urinalysis descriptors that are text by nature ("слегка мутная", "соломенно-жёлтый"), not a shifted phrase.
QUALITATIVE = re.compile(
    r"^(?:слегка|умеренно|незначительно|сильно)?\s*(?:мутн|прозрачн|соломенн|светло|тёмн|темн|желт|жёлт|янтарн|кисл|щелочн|нейтральн)\w*",
    re.I,
)

SPECS = [
    spec(
        "check_consistency",
        "Проверка качества данных: противоречия роста/веса между документами, нарушения внутренних тождеств "
        "анализов (MCHC, MCH, сумма лейкоформулы, билирубин), текст в числовых полях и физиологически невозможные значения "
        "(признаки сдвига строк при выгрузке).",
    ),
]


class QualityTools:
    def _init_quality(self) -> None:
        self.suspicious_groups: dict[str, str] = {}
        for issue in self._lab_identity_issues():
            for rid in issue["ids"]:
                self.suspicious_groups[self.by_id[rid].group] = f"нарушено тождество «{issue['check']}»"
        for issue in self._type_issues():
            self.suspicious_groups[self.by_id[issue["id"]].group] = "в числовом поле стоит текст из другого документа"
        for issue in self._implausible_issues():
            self.suspicious_groups.setdefault(self.by_id[issue["id"]].group, f"физиологически невозможное значение ({issue['analyte']})")

    def quality_mark(self, rec: Record) -> str | None:
        why = self.suspicious_groups.get(rec.group)
        return f"НЕДОСТОВЕРНО: в панели «{rec.group[:40]}» {why} — вероятен сдвиг строк при выгрузке" if why else None

    # ------------------------------------------------------------------ tool

    def check_consistency(self) -> dict:
        issues = self._anthropometry_issues() + self._lab_identity_issues() + self._type_issues() + self._implausible_issues()
        return {
            "issues": issues,
            "count": len(issues),
            "note": "Проверены: рост/вес/ИМТ между документами, внутренние тождества анализов, "
            "текст в числовых полях (признак сдвига строк при выгрузке), физиологически невозможные значения.",
        }

    # ------------------------------------------------------------------ checks

    def anthropometry(self) -> dict[str, list[tuple[Record, float]]]:
        """Height/weight from numeric fields and free text ("162 см 79 кг", "Масса тела: 100 кг. Рост: 170 см")."""
        found: dict[str, list[tuple[Record, float]]] = {"Вес": [], "Рост": []}
        for r in self.p.records:
            pl = r.param.strip().lower()
            if r.num and re.match(r"^(вес|масса тела)", pl):
                found["Вес"].append((r, r.num))
            elif r.num and pl.startswith("рост"):
                found["Рост"].append((r, r.num))
            elif r.num is None and r.value:
                t = r.value.lower()
                for m in re.finditer(r"(?:рост\s*)?(\d{3})\s*см[,;\s]*(?:вес|масса тела|масс тела)?\s*(\d{2,3}(?:[.,]\d)?)\s*кг", t):
                    found["Рост"].append((r, float(m.group(1))))
                    found["Вес"].append((r, float(m.group(2).replace(",", "."))))
                for m in re.finditer(r"(?:масса тела|вес)\s*:?\s*(\d{2,3}(?:[.,]\d)?)\s*(?:кг|kg)", t):
                    if not any(x is r for x, _ in found["Вес"]):
                        found["Вес"].append((r, float(m.group(1).replace(",", "."))))
                for m in re.finditer(r"рост\s*:?\s*(\d{3})\s*(?:см|cm)", t):
                    if not any(x is r for x, _ in found["Рост"]):
                        found["Рост"].append((r, float(m.group(1))))
        return found

    def _anthropometry_issues(self) -> list[dict]:
        out = []
        for name, pairs in self.anthropometry().items():
            nums = {r.id: v for r, v in pairs}
            if len(set(nums.values())) > 1:
                out.append(
                    {
                        "type": "conflict",
                        "field": name,
                        "values": [{"id": r.id, "date": r.date, "source": r.source, "value": nums[r.id]} for r, _ in pairs],
                    }
                )
        return out

    def _lab_identity_issues(self) -> list[dict]:
        first: dict[str, tuple[Record, float]] = {}
        for r, a in self.labs:
            first.setdefault(a.code, (r, r.num))
        out = []

        def check(name, codes, expected, actual, tol):
            if abs(expected - actual) > tol * max(1.0, abs(actual)):
                out.append(
                    {
                        "type": "lab_identity",
                        "check": name,
                        "expected": round(expected, 2),
                        "reported": actual,
                        "ids": [first[c][0].id for c in codes if c in first],
                    }
                )

        def g(code):
            return first[code][1] if code in first else None

        if None not in (g("HGB"), g("HCT"), g("MCHC")) and g("HCT"):
            # MCHC comes in g/L (check-ups) or g/dL (PDF lab blanks); compare in g/L.
            # (a PDF blank may even print "г/л" next to 34, so the value decides: MCHC < 60 is g/dL).
            scale = 10 if g("MCHC") < 60 else 1
            check("MCHC ≈ HGB/HCT×100", ("HGB", "HCT", "MCHC"), g("HGB") / g("HCT") * 100 / scale, g("MCHC"), 0.1)
        if None not in (g("HGB"), g("RBC"), g("MCH")) and g("RBC"):
            check("MCH ≈ HGB/RBC", ("HGB", "RBC", "MCH"), g("HGB") / g("RBC"), g("MCH"), 0.1)
        diff = ("NEUP", "LYMP", "MONOP", "EOSP", "BASOP")
        if None not in [g(c) for c in diff]:
            check("сумма лейкоформулы ≈ 100%", diff, 100.0, sum(g(c) for c in diff), 0.05)
        if None not in (g("TBIL"), g("DBIL"), g("IBIL")):
            check("билирубин общий = прямой + непрямой", ("TBIL", "DBIL", "IBIL"), g("DBIL") + g("IBIL"), g("TBIL"), 0.08)
        return out

    def _type_issues(self) -> list[dict]:
        out = []
        for r in self.p.records:
            if r.specialty != "лаборатория" or not r.group.startswith(("Клинический анализ крови", "Общий анализ мочи")):
                continue
            if QUALITATIVE.match(r.value.strip()):
                continue
            if r.num is None and re.search(r"[а-я]{4,}\s+[а-я]{3,}", r.value.lower()) and len(r.value) > 12:
                out.append(
                    {
                        "type": "text_in_lab_field",
                        "id": r.id,
                        "param": r.param,
                        "value": r.value[:80],
                        "meaning": "в поле анализа стоит фраза из другого документа — вероятен сдвиг значений в этом блоке",
                    }
                )
        return out

    def _implausible_issues(self) -> list[dict]:
        return [
            {
                "type": "implausible",
                "id": r.id,
                "analyte": a.name,
                "value": r.num,
                "meaning": "значение вне физиологически возможных пределов — ошибка единиц или выгрузки",
            }
            for r, a in self.labs
            if implausible(a, r.num, r.ref)
        ]
