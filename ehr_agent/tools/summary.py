"""Summary tools: lab dynamics, the evidence dossier of the whole record, and the report schema.

A summary fails mostly by omission: a model that chooses its own tool calls skips whole sections.
`build_dossier` therefore enumerates the record deterministically — every abnormal or changing lab
value with its trend, every study conclusion, diagnosis, medication by phase, recommendation and
data-quality issue — ranks each item by clinical significance (`knowledge/critical.py`) and lists
the items a report must not omit (`must_mention`). The model's job is the verdict and the wording;
the harness then checks the report against the dossier (`agent/harness.check_report`).

The report is returned through `final_answer` with one field per section (`FINAL_REPORT`), and
`render_report` turns it into one Markdown document, the same for every model.
"""

from __future__ import annotations

import re

from ..knowledge.analytes import parse_ref, resolve_analyte
from ..knowledge.critical import TEXT_RULES, lab_level, text_findings
from ..knowledge.dynamics import describe
from ..sources import CONCLUSION_FIELDS
from .base import spec

SPECS = [
    spec(
        "get_trend",
        "Динамика лабораторного показателя: все измерения по датам, первое, последнее и пиковое значения, изменение в %, "
        "значимо ли оно (порог — reference change value: аналитическая + биологическая вариация) и нормализовался ли показатель.",
        {"analyte": {"type": "string", "description": "Название показателя (тропонин, АСТ, креатинин, калий…)"}},
        ["analyte"],
    ),
    spec(
        "build_dossier",
        "Полное досье карты для сводки: отклонения анализов с динамикой, исследования, диагнозы, препараты по фазам, "
        "рекомендации, проблемы качества данных. Каждый пункт с уровнем значимости (критично / значимо) и id строк; "
        "must_mention — пункты, которые нельзя пропустить в отчёте.",
    ),
]

REPORT_SECTIONS = {
    "key_findings": "Ключевые выводы",
    "labs": "Анализы",
    "dynamics": "Динамика показателей",
    "studies": "Инструментальные исследования",
    "diagnoses": "Диагнозы",
    "therapy": "Терапия",
    "data_quality": "Противоречия и качество данных",
    "recommendations": "Рекомендации",
    "dismissed": "Сознательно не включено",
}
_LINE = "Каждый пункт с новой строки: факт — значение — вердикт (норма / отклонение / критично) — динамика — [id]."
FINAL_REPORT = spec(
    "final_answer",
    "Итоговый отчёт по карте, по разделам. " + _LINE + " Критичное — первым. Пустой раздел оставьте пустым.",
    {
        "key_findings": {"type": "string", "description": "3–6 главных выводов для врача, критичное первым"},
        "labs": {"type": "string", "description": "Отклонения анализов с вердиктом"},
        "dynamics": {"type": "string", "description": "Значимые изменения показателей во времени"},
        "studies": {"type": "string", "description": "Находки инструментальных исследований"},
        "diagnoses": {"type": "string", "description": "Диагнозы (МКБ-10 и текстовые)"},
        "therapy": {"type": "string", "description": "Препараты: до госпитализации, в стационаре, при выписке"},
        "data_quality": {"type": "string", "description": "Противоречия между документами и недостоверные данные"},
        "recommendations": {"type": "string", "description": "Рекомендации специалистов и план"},
        "dismissed": {"type": "string", "description": "Пункты must_mention, сознательно не включённые, с причиной"},
        "citations": {"type": "array", "items": {"type": "string"}, "description": "id строк-источников"},
        "status": {"type": "string", "enum": ["answered", "not_found", "uncertain"]},
    },
    ["key_findings", "citations", "status"],
)


_PLACEHOLDER = re.compile(r"^(?:-|—|нет|не\s+применимо|не\s+найдено|нет\s+данных|отсутствует)\.?$", re.I)
_ECHO = re.compile(r"^\s*(?:error:|отч[её]т\s+неполон|проверка\s+\w+\s+не\s+пройдена)", re.I)


def meaningful(text) -> bool:
    """A section with content: not empty, not a placeholder, not JSON debris, not an echo of a harness message."""
    t = str(text or "").strip()
    return len(re.findall(r"[а-яёa-z0-9]", t, re.I)) >= 3 and not _PLACEHOLDER.match(t) and not _ECHO.match(t)


def render_report(args: dict) -> str:
    """Report fields → one Markdown document (sections without content skipped)."""
    return "\n\n".join(f"## {title}\n{str(args[key]).strip()}" for key, title in REPORT_SECTIONS.items() if meaningful(args.get(key)))


def merge_reports(prev: dict | None, new: dict) -> dict:
    """A repaired report adds to the previous one: a section the model left empty keeps its earlier text."""
    if not prev:
        return new
    out = {**new}
    for key in REPORT_SECTIONS:
        if not meaningful(new.get(key)) and meaningful(prev.get(key)):
            out[key] = prev[key]
    out["citations"] = list(dict.fromkeys([*(prev.get("citations") or []), *(new.get("citations") or [])]))
    out["answer"] = render_report(out)
    return out


_RECS = re.compile(r"рекомендац|план\b|назначен|лечение", re.I)
_DISCLAIMER = re.compile(r"не является диагнозом", re.I)
LEVEL_ORDER = {"критично": 0, "значимо": 1, "фон": 2}


class SummaryTools:
    # ------------------------------------------------------------------ helpers

    def _upper(self, r, a) -> float | None:
        if r.ref:
            return parse_ref(r.ref)[1]
        return a.interval(self.p.sex)[1]

    def _series(self) -> dict[str, tuple]:
        """code → (analyte, points sorted by time) over trustworthy numeric lab values."""
        out: dict[str, tuple] = {}
        for r, a in self.labs:
            if r.num is None or self.quality_mark(r):
                continue
            iv = self._interval(r, a)
            level, why = lab_level(a.code, r.num, iv["flag"], self._upper(r, a))
            point = {
                "date": r.date,
                "value": r.num,
                "flag": iv["flag"],
                "id": r.id,
                "row": r.row,
                "unit": iv["unit"],
                "reference": iv["reference"],
                "level": level,
                "reason": why,
            }
            out.setdefault(a.code, (a, []))[1].append(point)
        for _, pts in out.values():
            pts.sort(key=lambda p: (p["date"], p["row"]))
        return out

    # ------------------------------------------------------------------ tools

    def get_trend(self, analyte: str) -> dict:
        cands = resolve_analyte(analyte)
        if not cands:
            return {"error": f"Анализ «{analyte}» не распознан.", "hint": "Например: тропонин, АСТ, креатинин, калий, гемоглобин."}
        series = self._series()
        items = []
        for a in cands:
            if a.code not in series:
                continue
            pts = series[a.code][1]
            item = {
                "analyte": a.name,
                "unit": pts[-1]["unit"],
                "reference": pts[-1]["reference"],
                "points": [{k: p[k] for k in ("date", "value", "flag", "id")} for p in pts[:20]],
            }
            if len(pts) > 1:
                item.update(describe(a.code, pts))
            else:
                item["note"] = "Одно измерение — динамику оценить нельзя."
            items.append(item)
        if not items:
            return {"found": False, "analyte": ", ".join(a.name for a in cands), "note": "Этот анализ у пациента не выполнялся."}
        return {
            "found": True,
            "items": items,
            "note": "direction учитывает только изменения больше reference change value (rcv_pct); course — сравнение с нормой.",
        }

    def build_dossier(self) -> dict:
        must: list[dict] = []  # label, ids, section, regex that recognises the item in a report

        # Labs: every analyte abnormal at some point or with a significant change.
        labs = []
        for code, (a, pts) in self._series().items():
            level = min((p["level"] for p in pts), key=LEVEL_ORDER.get)
            last = pts[-1]
            trend = describe(code, pts) if len(pts) > 1 else None
            changing = trend and trend["direction"] != "без значимых изменений"
            if level == "фон" and not changing:
                continue
            at_level = [p for p in pts if p["level"] == level]
            worst = min(at_level, key=lambda p: p["value"]) if at_level[0]["flag"] == "L" else max(at_level, key=lambda p: p["value"])
            item = {
                "analyte": a.name,
                "level": level,
                "value": last["value"],
                "unit": last["unit"],
                "reference": last["reference"],
                "flag": last["flag"],
                "id": last["id"],
            }
            if worst["reason"]:
                item["reason"] = worst["reason"]
            if trend:
                item["trend"] = (
                    f"{trend['first']['value']:g} → {trend['last']['value']:g} ({trend['change_pct']:+.0f} %, "
                    f"{trend['direction']}; пик {trend['peak']['value']:g} {trend['peak']['date']}); {trend['course']}"
                )
                item["ids"] = list(dict.fromkeys([trend["first"]["id"], trend["peak"]["id"], last["id"]]))
            labs.append(item)
            if level == "критично":
                stem = re.escape(a.name.lower().split(" (")[0].split()[0][:6])
                must.append(
                    {
                        "label": f"{a.name}: {worst['value']:g} {last['unit']} ({worst['reason']})",
                        "ids": item.get("ids", [last["id"]]),
                        "section": "labs",
                        "match": stem,
                    }
                )
        labs.sort(key=lambda x: (LEVEL_ORDER[x["level"]], x["analyte"]))

        # Studies, diagnoses: action-requiring text findings are critical.
        studies = []
        for s in self.entities.studies:
            found = text_findings(s.conclusion)
            ef = (s.values or {}).get("ФВ, %")
            if ef and float(ef) < 40:
                found.append("сниженная фракция выброса (< 40 %)")
            studies.append(
                {
                    "kind": s.kind,
                    "date": s.date,
                    "id": s.id,
                    "conclusion": s.conclusion[:260],
                    **({"values": s.values} if s.values else {}),
                    "level": "критично" if found else "значимо",
                }
            )
            for label in found:
                must.append({"label": f"{s.kind}: {label}", "ids": [s.id], "section": "studies", "match": _rule_rx(label)})
        dx = self.get_diagnoses()
        diagnoses = [{"icd10": d["icd10"], "name": d["name"], "ids": d["ids"][:2]} for d in dx["icd10"]]
        diagnoses += [{"text": d["text"], "ids": [d["id"]]} for d in dx["text_diagnoses"][:10]]
        for d in diagnoses:
            text = f"{{{d['icd10']}}} {d['name']}" if "icd10" in d else d["text"]
            for label in text_findings(text):
                if not any(m["label"].endswith(label) for m in must):
                    must.append({"label": f"диагноз: {label}", "ids": d["ids"], "section": "diagnoses", "match": _rule_rx(label)})

        # Medications by phase, recommendations.
        therapy: dict[str, list[str]] = {}
        for m in self.entities.meds:
            line = f"{m.drug} {m.dose}".strip() + f" [{m.id}]"
            bucket = therapy.setdefault(m.phase, [])
            if len(bucket) < 12 and not any(x.startswith(f"{m.drug} {m.dose}".strip()) for x in bucket):
                bucket.append(line)
        recs, seen = [], set()
        for r in self.p.records:
            if r.specialty == "лаборатория" or not _RECS.search(r.param) or len(r.value) < 4 or _DISCLAIMER.search(r.value):
                continue
            if not CONCLUSION_FIELDS.match(r.param.strip()) and not _RECS.match(r.param.strip()):
                continue
            key = r.value[:80]
            if key not in seen and len(recs) < 12:
                seen.add(key)
                recs.append({"id": r.id, "specialty": r.specialty, "text": r.value[:200]})

        # Data quality: conflicts and corrupted panels must be reported.
        quality = []
        for issue in self.check_consistency()["issues"]:
            if issue["type"] == "conflict":
                vals = sorted({v["value"] for v in issue["values"]})
                ids = [next(v["id"] for v in issue["values"] if v["value"] == x) for x in vals][:4]
                text, rx = f"{issue['field']}: разные значения в документах ({', '.join(f'{v:g}' for v in vals[:6])})", "вес|рост|масс"
            else:
                ids = issue.get("ids") or [issue.get("id")]
                text, rx = (
                    f"{issue.get('check') or issue.get('analyte') or issue['type']}: недостоверные данные",
                    "недостовер|сдвиг|ошибк|некоррект|искаж",
                )
            quality.append({"issue": text, "ids": [i for i in ids if i]})
            if not any(m["section"] == "data_quality" and m["match"] == rx for m in must):
                must.append({"label": text, "ids": quality[-1]["ids"], "section": "data_quality", "match": rx})

        uniq: dict[str, dict] = {}  # one item per label; ids merged (the same finding in several ECGs)
        for m in must:
            if m["label"] in uniq:
                uniq[m["label"]]["ids"] = list(dict.fromkeys(uniq[m["label"]]["ids"] + m["ids"]))[:4]
            else:
                uniq[m["label"]] = m
        must = list(uniq.values())
        self._must = must
        return {
            "patient": {
                "sex": self.p.sex,
                "age": self.p.age,
                "age_note": self.p.age_note,
                "period": [min(r.date for r in self.p.records), max(r.date for r in self.p.records)],
            },
            "must_mention": [{k: m[k] for k in ("label", "ids", "section")} for m in must],
            "labs": labs[:40],
            "studies": studies[:15],
            "diagnoses": diagnoses[:20],
            "therapy": therapy,
            "recommendations": recs,
            "data_quality": quality[:8],
            "note": "level: критично / значимо. trend: первое → последнее (изменение, направление с учётом RCV; пик); course — относительно нормы.",
        }

    def dossier(self) -> dict:
        """The dossier, built once per patient — for the harness and the evaluation."""
        if not hasattr(self, "_dossier"):
            self._dossier = self.build_dossier()
        return self._dossier

    def must_mention(self) -> list[dict]:
        self.dossier()
        return self._must


def _rule_rx(label: str) -> str:
    """The detection regex of a text rule, reused to recognise the finding in a report."""
    if "фракция выброса" in label:
        return r"фв|фракци\w* выброса"
    for rx, name in TEXT_RULES:
        if label == name:
            return rx
    return re.escape(label.split()[0][:6])
