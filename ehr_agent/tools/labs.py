"""Lab tools: values with reference intervals, out-of-range panels, deterministic calculators."""

from __future__ import annotations

import re

from ..config import ref_policy
from ..knowledge.analytes import CATALOG, resolve_analyte
from ..knowledge.lexicon import PANEL_CODES, PANEL_NAMES
from ..knowledge.reference import resolve_reference
from ..sources import Record
from .base import OFFSET, page, spec

SPECS = [
    spec(
        "get_lab",
        "Значения лабораторного показателя с референсным интервалом и флагом H/L/N. Принимает русское/латинское "
        "название или сокращение (ЛПНП, АЛТ, гемоглобин, ТТГ, витамин D), а также рост, вес, ИМТ. "
        "Возвращает found=false, если анализ не выполнялся.",
        {"analyte": {"type": "string", "description": "Название показателя"}},
        ["analyte"],
    ),
    spec(
        "list_abnormal_labs",
        "Все лабораторные показатели вне референсного интервала. Необязательный фильтр panel: " + ", ".join(PANEL_NAMES) + ".",
        {"panel": {"type": "string", "description": "Фильтр по панели или названию"}, **OFFSET},
    ),
    spec(
        "calculate",
        "Детерминированный расчёт: bmi(weight_kg,height_cm), non_hdl(chol,hdl), homa_ir(glucose,insulin), "
        "de_ritis(ast,alt), egfr(creatinine мкмоль/л) — СКФ CKD-EPI 2021, fib4(ast, alt, platelets) — индекс фиброза; "
        "возраст и пол подставляются из карты. Передавайте только нужные для формулы числа.",
        {
            "formula": {"type": "string", "enum": ["bmi", "non_hdl", "homa_ir", "de_ritis", "egfr", "fib4"]},
            **{
                k: {"type": "number", "description": d}
                for k, d in [
                    ("weight_kg", "вес, кг"),
                    ("height_cm", "рост, см"),
                    ("chol", "общий холестерин, ммоль/л"),
                    ("hdl", "ЛПВП, ммоль/л"),
                    ("glucose", "глюкоза, ммоль/л"),
                    ("insulin", "инсулин, мкЕд/мл"),
                    ("ast", "АСТ, Ед/л"),
                    ("alt", "АЛТ, Ед/л"),
                    ("creatinine", "креатинин, мкмоль/л"),
                    ("platelets", "тромбоциты, ×10⁹/л"),
                ]
            },
        },
        ["formula"],
    ),
]


class LabTools:
    def _interval(self, r: Record, a) -> dict:
        """Reference for one value under the active policy; the source layer is reported when it matters."""
        res = resolve_reference(r, a, self.p.sex, self.p.age)
        out = {"unit": r.unit or a.unit, "reference": res["reference"], "flag": res["flag"]}
        if ref_policy() == "layered" or r.ref:
            out["reference_source"] = res["source"]
        return out

    # ------------------------------------------------------------------ tools

    def get_lab(self, analyte: str) -> dict:
        if re.fullmatch(r"\s*(вес|масса( тела)?|рост|имт|индекс массы тела)\s*", analyte.lower()):
            key = "Рост" if "рост" in analyte.lower() else "Вес"
            vals = [{"id": r.id, "date": r.date, "source": r.source, "value": v} for r, v in self.anthropometry()[key]]
            out = {"found": bool(vals), "items": vals}
            if len({v["value"] for v in vals}) > 1:
                out["warning"] = "Значения различаются между документами — укажите расхождение."
            return out
        cands = resolve_analyte(analyte)
        if not cands:
            # Not in the catalog: raw lab rows whose name contains the query, without judgement.
            raw = [
                r
                for r in self.p.records
                if r.specialty == "лаборатория"
                and analyte.lower() in f"{r.group} {r.param}".lower()
                and r.param.lower() != "общий комментарий"
            ]
            if raw:
                return {
                    "analyte": analyte,
                    "catalog": False,
                    "note": "Показателя нет в справочнике норм; показаны сырые строки без оценки.",
                    "items": [r.short(200) for r in raw],
                }
            return {
                "error": f"Анализ «{analyte}» не распознан.",
                "hint": "Используйте русское или латинское название, например 'ЛПНП', 'АЛТ', 'гемоглобин', 'ТТГ'.",
                "known": [a.name for a in CATALOG][:40],
            }
        items = [
            {"id": r.id, "date": r.date, "analyte": a.name, "value": r.num, **self._interval(r, a)}
            for a in cands
            for r, ra in self.labs
            if ra.code == a.code
        ]
        if not items:
            return {
                "analyte": ", ".join(a.name for a in cands),
                "found": False,
                "note": "Этот анализ у пациента не выполнялся (в карте нет такой строки).",
            }
        if ref_policy() == "layered" or any("reference_source" in it for it in items):
            note = (
                "Источник интервала — в reference_source: бланк лаборатории, порог клинических рекомендаций, "
                f"популяционный интервал NHANES по полу и возрасту ({self.p.age_note or 'возраст из карты'}) "
                "или ориентировочный справочник. flag: H выше, L ниже, N в интервале, ? интервал не задан."
            )
        else:
            note = "Интервалы ориентировочные (взрослые, по полу); бланк лаборатории в выгрузке отсутствует. flag: H выше, L ниже, N в интервале."
        res = {"found": True, "items": items, "note": note}
        marks = {it["id"]: self.quality_mark(self.by_id[it["id"]]) for it in items}
        for it in items:
            if marks[it["id"]]:
                it["quality"] = marks[it["id"]]
        if any(marks.values()):
            res["warning"] = "Значения помечены как недостоверные; не интерпретируйте их клинически, ответьте status=uncertain."
        return res

    def list_abnormal_labs(self, panel: str | None = None, offset: int = 0) -> dict:
        codes = None
        if panel:
            pl = panel.lower().replace("ё", "е")
            hits = [c for key, c in PANEL_CODES.items() if re.search(key, pl)]
            codes = set().union(*hits) if hits else {a.code for r, a in self.labs if pl in r.group.lower() or pl in a.name.lower()}
            if not codes:
                return {
                    "error": f"Панель «{panel}» не распознана.",
                    "available_panels": PANEL_NAMES,
                    "hint": "Вызовите без panel, чтобы получить все отклонения.",
                }
        items = []
        for r, a in self.labs:
            iv = self._interval(r, a)
            if iv["flag"] in ("N", "?") or (codes is not None and a.code not in codes):
                continue
            item = {
                "id": r.id,
                "date": r.date,
                "analyte": a.name,
                "value": r.num,
                **iv,
                "interval_reliability": "бланк лаборатории" if r.ref else ("высокая" if a.stable else "зависит от лаборатории"),
            }
            mark = self.quality_mark(r)
            if mark:
                item["quality"] = mark
            items.append(item)
        chunk, meta = page(items, offset, 60)
        out = {
            "items": chunk,
            "count": len(items),
            **meta,
            "checked": sorted({a.name for r, a in self.labs if codes is None or a.code in codes}),
        }
        if not items:
            out["note"] = "Все проверенные показатели в пределах ориентировочных интервалов."
        if self._lab_identity_issues():
            out["warning"] = (
                "Найдены внутренние несоответствия лабораторных данных — значения могут быть сдвинуты при выгрузке. См. check_consistency."
            )
        return out

    def calculate(self, formula: str, **v) -> dict:
        f = formula.lower().strip()
        age, sex = v.get("age") or self.p.age, self.p.sex
        try:
            if f == "bmi":
                h = float(v["height_cm"]) / 100
                return {"result": round(float(v["weight_kg"]) / h / h, 2), "unit": "кг/м²"}
            if f == "non_hdl":
                return {"result": round(float(v["chol"]) - float(v["hdl"]), 2), "unit": "ммоль/л"}
            if f == "homa_ir":
                return {"result": round(float(v["glucose"]) * float(v["insulin"]) / 22.5, 2)}
            if f == "de_ritis":
                return {"result": round(float(v["ast"]) / float(v["alt"]), 2)}
            if f == "egfr":  # CKD-EPI 2021, race-free
                scr = float(v["creatinine"]) / 88.4
                k, alpha = (0.7, -0.241) if sex == "F" else (0.9, -0.302)
                egfr = 142 * min(scr / k, 1) ** alpha * max(scr / k, 1) ** -1.200 * 0.9938 ** float(age) * (1.012 if sex == "F" else 1)
                return {
                    "result": round(egfr, 1),
                    "unit": "мл/мин/1,73 м²",
                    "formula": "CKD-EPI 2021",
                    "age_used": age,
                    "age_note": self.p.age_note,
                }
            if f == "fib4":
                fib = float(age) * float(v["ast"]) / (float(v["platelets"]) * float(v["alt"]) ** 0.5)
                zone = "низкий риск фиброза (<1.30)" if fib < 1.3 else "высокий риск (>2.67)" if fib > 2.67 else "серая зона"
                return {"result": round(fib, 2), "interpretation": zone, "age_used": age, "age_note": self.p.age_note}
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            return {"error": f"Неверные аргументы для {formula}: {exc}"}
        return {"error": "Неизвестная формула. Доступно: bmi, non_hdl, homa_ir, de_ritis, egfr, fib4."}
