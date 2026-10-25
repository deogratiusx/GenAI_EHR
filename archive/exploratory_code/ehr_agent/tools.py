"""Tool sets the agent can call over one patient's EHR.

Three conditions share the same data and the same ``final_answer`` contract:

* ``domain``  — typed medical tools (lab lookup with reference intervals, abnormal
  labs, note search, specialist conclusions, cross-document consistency checks).
* ``generic`` — a single ``run_python`` tool over a pandas DataFrame, the way
  general-purpose assistants analyse data (code interpreter).
* ``fullctx`` — no tools; the whole record is serialized into the prompt.

Every tool returns JSON-serializable dicts and is deterministic: arithmetic and
reference intervals never come from the model.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import re
from collections import Counter
from typing import Any, Callable

from .data import Patient, Record
from .refs import CATALOG, flag, interval_text, lab_values, ref_policy, resolve_analyte, resolve_reference

MAX_TOOL_CHARS = 3500

CONCLUSION_FIELDS = re.compile(
    r"^(заключение|диагноз|диагноз по мкб|диагноз клинический|рекомендации|"
    r"рекомендации \(немедикаментозное лечение\)|лекарственные назначения|план обследования|"
    r"заключение узи сердца)\b",
    re.I,
)
BOILERPLATE = re.compile(r"^(защита врача|печать назначений|трудоспособность|повторный при[её]м|название протокола)", re.I)


def _truncate(obj: Any, limit: int = MAX_TOOL_CHARS) -> Any:
    s = json.dumps(obj, ensure_ascii=False)
    if len(s) <= limit:
        return obj
    if isinstance(obj, dict):
        for key in ("items", "results", "documents", "issues"):
            if isinstance(obj.get(key), list):
                items = list(obj[key])
                while items and len(json.dumps({**obj, key: items}, ensure_ascii=False)) > limit - 120:
                    items.pop()
                return {**obj, key: items, "truncated": f"показано {len(items)} из {len(obj[key])}; уточните запрос"}
    return {"text": s[: limit - 40] + "…", "truncated": True}


def _source(r: Record) -> str:
    return r.visit_title or r.protocol


# ---------------------------------------------------------------- text search

_TOKEN = re.compile(r"[a-zа-я0-9]+", re.I)


def _stem(tok: str) -> str:
    tok = tok.lower().replace("ё", "е")
    return tok[:5] if len(tok) > 5 else tok


def _tokens(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text) if len(t) > 1]


# Abbreviations and lay terms used in Russian protocols; a query term pulls in its group.
SYNONYMS: list[set[str]] = [
    {"фракция выброса", "фв"},
    {"курение", "курит", "табак", "сигареты", "вредные привычки"},
    {"алкоголь", "вредные привычки"},
    {"толщина интима медиа", "тим", "кзим"},
    {"гастроскопия", "эгдс", "фгдс", "эзофагогастродуоденоскопия"},
    {"колоноскопия", "фкс"},
    {"щитовидная", "щитовидной", "тиреоид"},
    {"узел", "узлы", "образование", "tirads"},
    {"давление", "ад", "гипертензия", "гипертония"},
    {"аллергия", "аллергические", "непереносимость"},
    {"лекарства", "препараты", "назначения", "прием лекарственных препаратов"},
    {"печень", "печени", "гепатоз", "стеатоз", "эластометрия"},
    {"почки", "почечн", "чашечно"},
    {"сердце", "эхокардиография", "эхокг"},
    {"сонные артерии", "брахиоцефальные", "бца", "маг"},
]


# v2 only: study names as doctors abbreviate them vs. protocol titles in the export.
SYNONYMS_V2: list[set[str]] = SYNONYMS + [
    {"мрт", "магнитно-резонансная томография", "мр"},
    {"кт", "мскт", "компьютерная томография"},
    {"узи", "ультразвуковое исследование", "эхографические"},
    {"экг", "электрокардиограмма"},
    {"холтер", "холтеровское мониторирование", "суточное мониторирование"},
    {"вены", "триплексное сканирование вен", "флебо"},
    {"колено", "коленного", "коленных", "мениск"},
    {"головного мозга", "головной мозг", "мозга", "мр картина"},
]


def _expand(query: str, synonyms: list[set[str]] = SYNONYMS) -> str:
    q = query.lower()
    extra = [" ".join(g) for g in synonyms if any(term in q for term in g)]
    return " ".join([query, *extra])


class NoteIndex:
    """BM25 over textual fields (everything that is not a bare lab number)."""

    def __init__(self, p: Patient, with_source: bool = False, synonyms: list[set[str]] = SYNONYMS):
        self.synonyms = synonyms
        # Ultrasound protocols use short all-caps rows ("ЖЕЛЧНЫЙ ПУЗЫРЬ") as section headers;
        # they are not indexed themselves but prefix the rows below them in the same document.
        self.docs: list[Record] = []
        texts: list[str] = []
        header, header_visit = "", ""
        for r in p.records:
            if not r.value or BOILERPLATE.match(r.param):
                continue
            if r.value.isupper() and len(r.value) < 40:
                header, header_visit = r.value, r.visit
                continue
            if r.visit != header_visit:
                header = ""
            if r.specialty == "лаборатория" and r.num is not None:
                continue
            if r.num is None and len(r.value) <= 2:
                continue
            self.docs.append(r)
            src = f"{_source(r)} {r.specialty} " if with_source else ""
            texts.append(f"{src}{header} {r.group} {r.param} {r.value}")
        self.texts = texts
        self.tf = [Counter(_tokens(t)) for t in texts]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = sum(self.len) / max(1, len(self.len))
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, k: int, specialty: str | None = None) -> list[tuple[float, Record]]:
        q = list(dict.fromkeys(_tokens(_expand(query, self.synonyms))))
        scored = []
        for r, c, L in zip(self.docs, self.tf, self.len):
            if specialty and specialty.lower() not in r.specialty.lower() and specialty.lower() not in _source(r).lower():
                continue
            s = 0.0
            for t in q:
                f = c.get(t, 0)
                if f:
                    s += self.idf.get(t, 0) * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * L / self.avg))
            if s > 0:
                scored.append((s, r))
        scored.sort(key=lambda x: -x[0])
        return scored[:k]


# ---------------------------------------------------------------- domain tools

PANEL_CODES: dict[str, set[str]] = {
    r"печен|печён|гепат|aлт|алт|аст|билируб|трансаминаз": {"ALT", "AST", "GGT", "ALP", "TBIL", "DBIL", "IBIL", "ALB", "TP"},
    r"липид|холест|атероген|лпнп|лпвп": {"CHOL", "LDL", "HDL", "VLDL", "TG", "AI", "LPA"},
    r"оак|общ\w* анализ крови|клиническ\w* анализ крови|гемограм|кров[ьи] общ|лейкоформ": {
        "WBC", "RBC", "HGB", "HCT", "MCV", "MCH", "MCHC", "RDW", "PLT", "MPV", "ESR", "NEUP", "LYMP", "MONOP",
        "EOSP", "BASOP", "NEU", "LYM", "MONO", "EOS", "BASO"},
    r"щитовид|тиреоид|ттг|т4": {"TSH", "FT4"},
    r"углевод|глюк|диабет|сахар|инсулин": {"GLU", "HBA1C", "INS"},
    r"почк|почеч|креатин|мочевин|мочев\w* кисл": {"CREA", "UREA", "URIC"},
    r"желез|ферритин|анеми": {"FE", "FERR", "TSAT", "HGB"},
    r"коагул|сверт|гемостаз": {"INR", "APTT", "PT", "PTI", "TT", "FIB"},
    r"онкомарк|опухол": {"PSA", "AFP", "CEA", "CA199", "CA153", "CA125"},
    r"электролит|калий|натрий|хлор|кальц": {"K", "NA", "CL", "CA"},
    r"воспал|срб|c-реакт|с-реакт": {"CRP", "ESR", "WBC"},
}
PANEL_NAMES = ["печёночные пробы", "липидограмма", "общий анализ крови", "щитовидная железа", "углеводный обмен",
               "почечные показатели", "обмен железа", "коагулограмма", "онкомаркеры", "электролиты", "воспаление"]


class DomainTools:
    def __init__(self, p: Patient):
        self.p = p
        self.index = NoteIndex(p)
        self.labs = lab_values(p)

    # 1
    def get_overview(self) -> dict:
        visits: dict[str, dict] = {}
        for r in self.p.records:
            v = visits.setdefault(r.visit + "|" + r.protocol, {
                "document_id": r.visit, "date": r.date, "source": _source(r),
                "specialty": r.specialty, "fields": 0})
            v["fields"] += 1
        docs = sorted(visits.values(), key=lambda d: (d["date"], d["specialty"]))
        return {
            "patient": self.p.pid,
            "sex": {"M": "мужской", "F": "женский"}.get(self.p.sex or "", "не указан"),
            "dates": sorted({r.date for r in self.p.records}),
            "specialties": sorted({d["specialty"] for d in docs}),
            "lab_results": len(self.labs),
            "documents": docs,
        }

    # 2
    def get_lab(self, analyte: str) -> dict:
        cands = resolve_analyte(analyte)
        if not cands:
            # Fall back to raw lab rows whose name contains the query (non-catalog tests).
            raw = [r for r in self.p.records if r.specialty == "лаборатория"
                   and analyte.lower() in f"{r.group} {r.param}".lower()
                   and r.param.lower() != "общий комментарий"]
            if raw:
                return {"analyte": analyte, "catalog": False,
                        "note": "Показателя нет в справочнике норм; показаны сырые строки без оценки.",
                        "items": [r.short(200) for r in raw]}
            return {"error": f"Анализ «{analyte}» не распознан.",
                    "hint": "Используйте русское или латинское название, например 'ЛПНП', 'АЛТ', 'гемоглобин', 'ТТГ'.",
                    "known": [a.name for a in CATALOG][:40]}
        items = []
        for a in cands:
            for r, ra in self.labs:
                if ra.code == a.code:
                    items.append({"id": r.id, "date": r.date, "analyte": a.name, "value": r.num, **self._interval(r, a)})
        if not items:
            return {"analyte": ", ".join(a.name for a in cands), "found": False,
                    "note": "Этот анализ у пациента не выполнялся (в карте нет такой строки)."}
        if ref_policy() == "layered" or any("reference_source" in it for it in items):
            note = ("Источник интервала — в reference_source: бланк лаборатории, порог клинических рекомендаций, "
                    f"популяционный интервал NHANES по полу и возрасту ({self.p.age_note or 'возраст из карты'}) "
                    "или ориентировочный справочник. flag: H выше, L ниже, N в интервале, ? интервал не задан.")
        else:
            note = "Интервалы ориентировочные (взрослые, по полу); бланк лаборатории в выгрузке отсутствует. flag: H выше, L ниже, N в интервале."
        return {"found": True, "items": items, "note": note}

    def _interval(self, r: Record, a) -> dict:
        """Reference for one value under the active policy (EHR_REF_POLICY); the source is always reported."""
        res = resolve_reference(r, a, self.p.sex, self.p.age)
        out = {"unit": r.unit or a.unit, "reference": res["reference"], "flag": res["flag"]}
        if ref_policy() == "layered" or r.ref:
            out["reference_source"] = res["source"]
        return out

    # 3
    def list_abnormal_labs(self, panel: str | None = None) -> dict:
        codes = None
        if panel:
            pl = panel.lower().replace("ё", "е")
            codes = set().union(*[c for key, c in PANEL_CODES.items() if re.search(key, pl)]) if any(
                re.search(key, pl) for key in PANEL_CODES) else None
            if codes is None:
                hits = {a.code for r, a in self.labs if pl in r.group.lower() or pl in a.name.lower()}
                if not hits:
                    return {"error": f"Панель «{panel}» не распознана.",
                            "available_panels": PANEL_NAMES, "hint": "Вызовите без panel, чтобы получить все отклонения."}
                codes = hits
        items = []
        for r, a in self.labs:
            iv = self._interval(r, a)
            if iv["flag"] in ("N", "?"):
                continue
            if codes is not None and a.code not in codes:
                continue
            items.append({"id": r.id, "date": r.date, "analyte": a.name, "value": r.num, **iv,
                          "interval_reliability": "бланк лаборатории" if r.ref else ("высокая" if a.stable else "зависит от лаборатории")})
        issues = self._lab_identity_issues()
        checked = sorted({a.name for r, a in self.labs if codes is None or a.code in codes})
        out = {"items": items, "count": len(items), "checked": checked}
        if not items:
            out["note"] = "Все проверенные показатели в пределах ориентировочных интервалов."
        if issues:
            out["warning"] = "Найдены внутренние несоответствия лабораторных данных — значения могут быть сдвинуты при выгрузке. См. check_consistency."
        return out

    # 4
    def search_notes(self, query: str, specialty: str | None = None, top_k: int = 6) -> dict:
        top_k = max(1, min(int(top_k), 12))
        hits = self.index.search(query, top_k, specialty)
        if not hits:
            alt = self.index.search(query, top_k) if specialty else []
            return {"results": [], "note": "Ничего не найдено."
                    + (" Без фильтра по специальности есть совпадения — повторите без specialty." if alt else
                       " Попробуйте синонимы или более общий термин.")}
        return {"results": [{**r.short(400), "specialty": r.specialty, "score": round(s, 2)} for s, r in hits]}

    # 5
    def get_conclusions(self, specialty: str | None = None) -> dict:
        items = []
        for r in self.p.records:
            if not CONCLUSION_FIELDS.match(r.param.strip()) or not r.value or r.value in {".", "-"}:
                continue
            if specialty and specialty.lower() not in r.specialty.lower() and specialty.lower() not in _source(r).lower():
                continue
            items.append({"id": r.id, "date": r.date, "specialty": r.specialty, "source": _source(r),
                          "field": r.param, "text": r.value[:500]})
        if not items:
            specs = sorted({r.specialty for r in self.p.records})
            return {"items": [], "note": f"Заключений не найдено. Доступные специальности: {specs}"}
        return {"items": items}

    # 5b
    def get_diagnoses(self) -> dict:
        coded: dict[str, dict] = {}
        coded_visits = set()
        for r in self.p.records:
            for c, n in re.findall(r"\{([A-Z]\d{2}(?:\.\d{1,2})?)\}\s*([^{}]*)", r.value):
                e = coded.setdefault(c, {"icd10": c, "name": n.strip()[:80], "specialists": [], "ids": []})
                if r.specialty not in e["specialists"]:
                    e["specialists"].append(r.specialty)
                e["ids"].append(r.id)
                coded_visits.add(r.visit)
        text = [{"id": r.id, "specialty": r.specialty, "text": r.value[:160]}
                for r in self.p.records
                if re.match(r"^диагноз", r.param.strip(), re.I) and r.value not in {".", "-", ""}
                and r.visit not in coded_visits and not re.search(r"\{[A-Z]\d{2}", r.value)]
        return {"icd10": list(coded.values()), "text_diagnoses": text,
                "note": "icd10 — уникальные коды МКБ-10; text_diagnoses — диагнозы из документов без кода."}

    # 6
    def get_document(self, document_id: str) -> dict:
        recs = [r for r in self.p.records if r.visit == str(document_id).strip()]
        if not recs:
            return {"error": f"Документ {document_id} не найден. Возьмите document_id из get_overview."}
        return {"document_id": document_id, "date": recs[0].date, "source": _source(recs[0]),
                "items": [{"id": r.id, "group": r.group, "param": r.param, "value": r.value[:350]}
                          for r in recs if not BOILERPLATE.match(r.param)]}

    # 7
    def check_consistency(self) -> dict:
        issues = self._anthropometry_issues() + self._lab_identity_issues() + self._type_issues()
        return {"issues": issues, "count": len(issues),
                "note": "Проверены: рост/вес/ИМТ между документами, внутренние тождества анализов, "
                        "текст в числовых полях (признак сдвига строк при выгрузке)."}

    def _anthropometry(self) -> dict[str, list[tuple[Record, float]]]:
        """Height/weight from numeric fields and from free text ("162 см 79 кг", "рост 177 см, вес 70 кг")."""
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
                # "Масса тела: 100 кг. Рост: 170 см", "Вес 100 kg; Рост 170 cm" (inpatient records)
                for m in re.finditer(r"(?:масса тела|вес)\s*:?\s*(\d{2,3}(?:[.,]\d)?)\s*(?:кг|kg)", t):
                    if not any(x is r for x, _ in found["Вес"]):
                        found["Вес"].append((r, float(m.group(1).replace(",", "."))))
                for m in re.finditer(r"рост\s*:?\s*(\d{3})\s*(?:см|cm)", t):
                    if not any(x is r for x, _ in found["Рост"]):
                        found["Рост"].append((r, float(m.group(1))))
        return found

    def _anthropometry_issues(self) -> list[dict]:
        out = []
        for name, pairs in self._anthropometry().items():
            vals = [r for r, _ in pairs]
            nums = {r.id: v for r, v in pairs}
            distinct = set(nums.values())
            if len(distinct) > 1:
                out.append({"type": "conflict", "field": name,
                            "values": [{"id": r.id, "date": r.date, "source": _source(r), "value": nums[r.id]} for r in vals]})
        return out

    def _lab_identity_issues(self) -> list[dict]:
        v: dict[str, tuple[Record, float]] = {}
        for r, a in self.labs:
            v.setdefault(a.code, (r, r.num))
        out = []

        def chk(name, ids, expected, actual, tol):
            if abs(expected - actual) > tol * max(1.0, abs(actual)):
                out.append({"type": "lab_identity", "check": name, "expected": round(expected, 2),
                            "reported": actual, "ids": ids})

        g = lambda c: v[c][1] if c in v else None  # noqa: E731
        ids = lambda *cs: [v[c][0].id for c in cs if c in v]  # noqa: E731
        if None not in (g("HGB"), g("HCT"), g("MCHC")) and g("HCT"):
            chk("MCHC ≈ HGB/HCT×100", ids("HGB", "HCT", "MCHC"), g("HGB") / g("HCT") * 100, g("MCHC"), 0.1)
        if None not in (g("HGB"), g("RBC"), g("MCH")) and g("RBC"):
            chk("MCH ≈ HGB/RBC", ids("HGB", "RBC", "MCH"), g("HGB") / g("RBC"), g("MCH"), 0.1)
        parts = [g(c) for c in ("NEUP", "LYMP", "MONOP", "EOSP", "BASOP")]
        if None not in parts:
            chk("сумма лейкоформулы ≈ 100%", ids("NEUP", "LYMP", "MONOP", "EOSP", "BASOP"), 100.0, sum(parts), 0.05)
        if None not in (g("TBIL"), g("DBIL"), g("IBIL")):
            chk("билирубин общий = прямой + непрямой", ids("TBIL", "DBIL", "IBIL"), g("DBIL") + g("IBIL"), g("TBIL"), 0.08)
        return out

    def _type_issues(self) -> list[dict]:
        out = []
        numeric_groups = ("Клинический анализ крови", "Общий анализ мочи")
        for r in self.p.records:
            if r.specialty != "лаборатория" or not r.group.startswith(numeric_groups):
                continue
            if r.num is None and re.search(r"[а-я]{4,}\s+[а-я]{3,}", r.value.lower()) and len(r.value) > 12:
                out.append({"type": "text_in_lab_field", "id": r.id, "param": r.param, "value": r.value[:80],
                            "meaning": "в поле анализа стоит фраза из другого документа — вероятен сдвиг значений в этом блоке"})
        return out

    # 8 (pure computation)
    def calculate(self, formula: str, **values) -> dict:
        f = formula.lower().strip()
        try:
            if f in {"bmi", "имт"}:
                w, h = float(values["weight_kg"]), float(values["height_cm"]) / 100
                return {"result": round(w / h / h, 2), "unit": "кг/м²"}
            if f in {"non_hdl", "не-лпвп"}:
                return {"result": round(float(values["chol"]) - float(values["hdl"]), 2), "unit": "ммоль/л"}
            if f in {"homa_ir", "homa"}:
                return {"result": round(float(values["glucose"]) * float(values["insulin"]) / 22.5, 2)}
            if f in {"de_ritis", "аст/алт"}:
                return {"result": round(float(values["ast"]) / float(values["alt"]), 2)}
        except (KeyError, ValueError, ZeroDivisionError) as exc:
            return {"error": f"Неверные аргументы для {formula}: {exc}"}
        return {"error": "Неизвестная формула. Доступно: bmi(weight_kg,height_cm), non_hdl(chol,hdl), "
                         "homa_ir(glucose,insulin), de_ritis(ast,alt)."}


S = lambda **kw: kw  # noqa: E731

DOMAIN_SPECS: list[dict] = [
    S(name="get_overview", description="Обзор карты: пол, даты, список документов (document_id, специальность, источник). Вызывайте первым, если не знаете, где искать.",
      parameters={"type": "object", "properties": {}, "required": []}),
    S(name="get_lab", description="Значения лабораторного показателя с референсным интервалом и флагом H/L/N. Принимает русское/латинское название или сокращение (ЛПНП, АЛТ, гемоглобин, ТТГ, витамин D). Возвращает found=false, если анализ не выполнялся.",
      parameters={"type": "object", "properties": {"analyte": {"type": "string", "description": "Название показателя"}}, "required": ["analyte"]}),
    S(name="list_abnormal_labs", description="Все лабораторные показатели вне референсного интервала. Необязательный фильтр panel: печёночные пробы, липидограмма, общий анализ крови, щитовидная железа, углеводный обмен, почечные показатели, обмен железа, коагулограмма, онкомаркеры, электролиты, воспаление.",
      parameters={"type": "object", "properties": {"panel": {"type": "string", "description": "Фильтр по панели или названию"}}, "required": []}),
    S(name="search_notes", description="Полнотекстовый поиск по текстовым полям протоколов (жалобы, анамнез, описания УЗИ/КТ/МРТ, заключения). Возвращает фрагменты с id строк.",
      parameters={"type": "object", "properties": {
          "query": {"type": "string", "description": "Ключевые слова"},
          "specialty": {"type": "string", "description": "Необязательно: специальность или исследование (УЗИ, кардиолог, офтальмолог…)"},
          "top_k": {"type": "integer", "description": "Сколько результатов (1–12), по умолчанию 6"}}, "required": ["query"]}),
    S(name="get_conclusions", description="Заключения, диагнозы (включая МКБ), рекомендации и назначения всех специалистов или одной специальности. Основной инструмент для сводки и сопоставления мнений врачей.",
      parameters={"type": "object", "properties": {"specialty": {"type": "string", "description": "Необязательно: специальность"}}, "required": []}),
    S(name="get_diagnoses", description="Компактный список всех диагнозов: коды МКБ-10 с названиями и текстовые диагнозы, с указанием специалиста и id строки.",
      parameters={"type": "object", "properties": {}, "required": []}),
    S(name="get_document", description="Все поля одного документа по document_id из get_overview.",
      parameters={"type": "object", "properties": {"document_id": {"type": "string"}}, "required": ["document_id"]}),
    S(name="check_consistency", description="Проверка качества данных: противоречия роста/веса между документами, нарушения внутренних тождеств анализов (MCHC, MCH, сумма лейкоформулы, билирубин), текст в числовых полях (сдвиг строк выгрузки).",
      parameters={"type": "object", "properties": {}, "required": []}),
    S(name="calculate", description="Детерминированный расчёт: bmi(weight_kg,height_cm), non_hdl(chol,hdl), homa_ir(glucose,insulin), de_ritis(ast,alt). Передавайте только нужные для формулы числа.",
      parameters={"type": "object", "properties": {
          "formula": {"type": "string", "enum": ["bmi", "non_hdl", "homa_ir", "de_ritis"]},
          **{k: {"type": "number", "description": d} for k, d in [
              ("weight_kg", "вес, кг"), ("height_cm", "рост, см"), ("chol", "общий холестерин, ммоль/л"),
              ("hdl", "ЛПВП, ммоль/л"), ("glucose", "глюкоза, ммоль/л"), ("insulin", "инсулин, мкЕд/мл"),
              ("ast", "АСТ, Ед/л"), ("alt", "АЛТ, Ед/л")]}}, "required": ["formula"]}),
]

FINAL_SPEC = S(
    name="final_answer",
    description="Завершить работу и дать ответ врачу. Каждое утверждение должно опираться на id строк карты (формат P00X:rN), полученные из инструментов.",
    parameters={"type": "object", "properties": {
        "answer": {"type": "string", "description": "Ответ на русском, кратко и по существу, с числами и единицами"},
        "citations": {"type": "array", "items": {"type": "string"}, "description": "id строк-источников, например ['P002:r245']"},
        "status": {"type": "string", "enum": ["answered", "not_found", "uncertain"],
                   "description": "not_found — данных в карте нет; uncertain — данные противоречивы или недостоверны"}},
        "required": ["answer", "citations", "status"]},
)

GENERIC_SPECS: list[dict] = [
    S(name="run_python", description=(
        "Выполнить Python-код. Доступен pandas DataFrame `df` со строками карты пациента: "
        "колонки id, date, specialty, source, group, param, value (строка), num (число или NaN). "
        "Результат — то, что напечатано через print (до 3500 символов). Названия групп и параметров — длинные русские строки "
        "(например 'Липидограмма (Холестерол, Триглицериды, ЛПВП, ЛПНП…)'), поэтому сначала изучите структуру "
        "(df.group.unique(), df[df.param.str.contains('…', case=False)]) и фильтруйте через str.contains."),
      parameters={"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}),
]


class GenericTools:
    def __init__(self, p: Patient):
        import pandas as pd
        self.df = pd.DataFrame([{
            "id": r.id, "date": r.date, "specialty": r.specialty, "source": _source(r),
            "group": r.group, "param": r.param, "value": r.value, "num": r.num} for r in p.records])

    def run_python(self, code: str) -> dict:
        import numpy as np
        import pandas as pd
        unescaped = False
        if "\n" not in code and "\\n" in code:
            # Some backends double-escape newlines inside JSON arguments.
            code, unescaped = code.replace("\\n", "\n").replace('\\"', '"'), True
        buf = io.StringIO()
        env = {"df": self.df.copy(), "pd": pd, "np": np, "re": re}
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(code, "<agent>", "exec"), env)  # noqa: S102 - local research sandbox over own data
        except Exception as exc:  # the model sees its own traceback, as in a code interpreter
            return {"error": f"{type(exc).__name__}: {exc}", "stdout": buf.getvalue()[-1500:]}
        out = buf.getvalue()
        if not out.strip():
            return {"stdout": "", "note": "Код ничего не напечатал. Используйте print()."}
        res = {"stdout": out[:MAX_TOOL_CHARS]}
        if unescaped:
            res["harness_fix"] = "unescaped_newlines"
        return res


def serialize_full(p: Patient) -> str:
    lines = []
    for r in p.records:
        if BOILERPLATE.match(r.param) or not r.value:
            continue
        lines.append(f"{r.id} | {r.date} | {_source(r)} | {r.group} | {r.param} = {r.value}")
    return "\n".join(lines)


class DomainToolsV2(DomainTools):
    """Tool fixes from the v1 error analysis (same names and schemas, different behaviour):

    1. the note index includes the study/protocol name and modality synonyms (МРТ, КТ, УЗИ…);
    2. an empty or weak search says that absence of a hit is not absence of the study;
    3. unfiltered overview/conclusions are compact instead of truncated;
    4. data-quality warnings are attached to every lab value from a suspicious panel;
    5. get_lab also serves height/weight/BMI with cross-document conflicts.
    """

    def __init__(self, p: Patient):
        super().__init__(p)
        self.index = NoteIndex(p, with_source=True, synonyms=SYNONYMS_V2)
        self.suspicious_groups: dict[str, str] = {}
        for issue in self._lab_identity_issues():
            for rid in issue["ids"]:
                rec = self.p.by_id()[rid]
                self.suspicious_groups[rec.group] = f"нарушено тождество «{issue['check']}»"
        for issue in self._type_issues():
            rec = self.p.by_id()[issue["id"]]
            self.suspicious_groups[rec.group] = "в числовом поле стоит текст из другого документа"

    def _quality(self, rec: Record) -> str | None:
        why = self.suspicious_groups.get(rec.group)
        return f"НЕДОСТОВЕРНО: в панели «{rec.group[:40]}» {why} — вероятен сдвиг строк при выгрузке" if why else None

    def _docs_matching(self, query: str) -> list[dict]:
        terms = [t for t in _tokens(_expand(query, SYNONYMS_V2)) if len(t) > 2]
        seen, out = set(), []
        for r in self.p.records:
            src = _tokens(f"{_source(r)} {r.specialty}")
            if r.visit + _source(r) not in seen and any(t in src for t in terms):
                seen.add(r.visit + _source(r))
                out.append({"document_id": r.visit, "date": r.date, "source": _source(r)})
        return out[:8]

    def get_overview(self) -> dict:
        by_spec: dict[str, list[str]] = {}
        seen = set()
        for r in self.p.records:
            key = (r.visit, _source(r))
            if key in seen:
                continue
            seen.add(key)
            by_spec.setdefault(r.specialty, []).append(f"{r.visit} · {r.date} · {_source(r)[:60]}")
        return {"patient": self.p.pid, "sex": {"M": "мужской", "F": "женский"}.get(self.p.sex or "", "не указан"),
                "dates": sorted({r.date for r in self.p.records}), "lab_results": len(self.labs),
                "documents_by_specialty": by_spec,
                "data_quality_warnings": sorted(set(self.suspicious_groups.values())) or None}

    def search_notes(self, query: str, specialty: str | None = None, top_k: int = 6) -> dict:
        res = super().search_notes(query, specialty, top_k)
        docs = self._docs_matching(query)
        if docs:
            res["matching_documents"] = docs
        if not res.get("results"):
            res["note"] = ("Совпадений в тексте нет. Отсутствие совпадений НЕ означает, что исследование не проводилось: "
                           "проверьте matching_documents, get_overview или get_conclusions(specialty).")
        return res

    def get_conclusions(self, specialty: str | None = None) -> dict:
        if specialty:
            res = super().get_conclusions(specialty)
            if not res["items"]:
                docs = self._docs_matching(specialty)
                if docs:
                    terms = {d["document_id"] for d in docs}
                    res = {"items": [{"id": r.id, "date": r.date, "source": _source(r), "field": r.param, "text": r.value[:500]}
                                     for r in self.p.records if r.visit in terms and CONCLUSION_FIELDS.match(r.param.strip())
                                     and r.value not in {".", "-", ""}]}
            return res
        by_doc: dict[tuple, dict] = {}
        for r in self.p.records:
            if not CONCLUSION_FIELDS.match(r.param.strip()) or r.value in {".", "-", ""} or re.match(r"рекомендации$", r.param.strip(), re.I) and "не является диагнозом" in r.value:
                continue
            d = by_doc.setdefault((r.visit, _source(r)), {"source": _source(r)[:50], "specialty": r.specialty, "ids": [], "text": ""})
            d["ids"].append(r.id)
            if len(d["text"]) < 160:
                d["text"] = (d["text"] + " | " if d["text"] else "") + r.value[:160 - len(d["text"])]
        items = list(by_doc.values())
        return {"items": items, "mode": "compact",
                "note": "Краткий вид по документам. Для полного текста вызовите get_conclusions(specialty=…)."}

    def get_lab(self, analyte: str) -> dict:
        if re.fullmatch(r"\s*(вес|масса( тела)?|рост|имт|индекс массы тела)\s*", analyte.lower()):
            pairs = self._anthropometry()
            key = "Рост" if "рост" in analyte.lower() else "Вес"
            vals = [{"id": r.id, "date": r.date, "source": _source(r), "value": v} for r, v in pairs[key]]
            out = {"found": bool(vals), "items": vals}
            if len({v["value"] for v in vals}) > 1:
                out["warning"] = "Значения различаются между документами — укажите расхождение."
            return out
        res = super().get_lab(analyte)
        if res.get("items"):
            warn = set()
            for it in res["items"]:
                q = self._quality(self.p.by_id()[it["id"]])
                if q:
                    it["quality"] = q
                    warn.add(q)
            if warn:
                res["warning"] = "Значения помечены как недостоверные; не интерпретируйте их клинически, ответьте status=uncertain."
        return res

    def list_abnormal_labs(self, panel: str | None = None) -> dict:
        res = super().list_abnormal_labs(panel)
        for it in res.get("items", []):
            q = self._quality(self.p.by_id()[it["id"]])
            if q:
                it["quality"] = q
        return res


class DomainToolsRI(DomainTools):
    """v1 tools + population intervals by sex and (assumed) age from NHANES, guideline decision limits
    for lipids/glucose/HbA1c/vitamin D, and age-aware calculators (eGFR CKD-EPI 2021, FIB-4)."""

    def _judge(self, a, value: float) -> dict:
        from .refs import DECISION_LIMITS, nhanes_interval
        out = {"reference_indicative": interval_text(a, self.p.sex), "flag_indicative": flag(a, value, self.p.sex)}
        ri = nhanes_interval(a.code, self.p.sex, self.p.age)
        if ri:
            f = "L" if value < ri["lower"] else "H" if value > ri["upper"] else "N"
            out.update(reference_population=f"{ri['lower']:g}–{ri['upper']:g} {a.unit} (NHANES, {ri['partition']}, n={ri['n']})",
                       flag_population=f)
        if a.code in DECISION_LIMITS:
            out.update(decision_limit=DECISION_LIMITS[a.code], flag=out["flag_indicative"], basis="порог клинических рекомендаций")
        else:
            out.update(flag=out.get("flag_population", out["flag_indicative"]),
                       basis="популяционный интервал по полу и возрасту" if ri else "ориентировочный интервал")
        return out

    def get_lab(self, analyte: str) -> dict:
        res = super().get_lab(analyte)
        if res.get("items"):
            for it in res["items"]:
                a = next(a for r, a in self.labs if r.id == it["id"])
                it.pop("reference", None)
                it.pop("flag", None)
                it.update(self._judge(a, it["value"]))
            res["note"] = (f"Пол {self.p.sex or '?'}, возраст {self.p.age or '?'} ({self.p.age_note or 'из карты'}). "
                           "flag — итоговая оценка: для липидов, глюкозы, HbA1c и витамина D по порогам рекомендаций, "
                           "для остальных по популяционному интервалу NHANES (США, другие анализаторы) с учётом пола и возраста.")
        return res

    def list_abnormal_labs(self, panel: str | None = None) -> dict:
        res = super().list_abnormal_labs(panel)
        if "error" in res:
            return res
        codes = None
        if panel:
            pl = panel.lower().replace("ё", "е")
            hit = [c for key, c in PANEL_CODES.items() if re.search(key, pl)]
            codes = set().union(*hit) if hit else {a.code for _, a in self.labs if pl in a.name.lower()}
        items = []
        for r, a in self.labs:
            if codes is not None and a.code not in codes:
                continue
            j = self._judge(a, r.num)
            if j["flag"] != "N":
                items.append({"id": r.id, "analyte": a.name, "value": r.num, "unit": a.unit, **j})
        return {"items": items, "count": len(items),
                "note": f"Возраст {self.p.age} — {self.p.age_note}." if self.p.age else "Возраст неизвестен."}

    def calculate(self, formula: str, **values) -> dict:
        f = formula.lower().strip()
        age = values.get("age") or self.p.age
        sex = self.p.sex
        try:
            if f == "egfr":
                scr = float(values["creatinine"]) / 88.4
                k, alpha = (0.7, -0.241) if sex == "F" else (0.9, -0.302)
                egfr = 142 * min(scr / k, 1) ** alpha * max(scr / k, 1) ** -1.200 * 0.9938 ** float(age) * (1.012 if sex == "F" else 1)
                return {"result": round(egfr, 1), "unit": "мл/мин/1,73 м²", "formula": "CKD-EPI 2021 (без расы)",
                        "age_used": age, "age_note": self.p.age_note}
            if f == "fib4":
                fib = float(age) * float(values["ast"]) / (float(values["platelets"]) * float(values["alt"]) ** 0.5)
                zone = "низкий риск фиброза (<1.30)" if fib < 1.3 else "высокий риск (>2.67)" if fib > 2.67 else "серая зона"
                return {"result": round(fib, 2), "interpretation": zone, "age_used": age, "age_note": self.p.age_note}
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            return {"error": f"Неверные аргументы для {formula}: {exc}"}
        return super().calculate(formula, **values)


RI_SPECS = [dict(s) for s in DOMAIN_SPECS]
for _s in RI_SPECS:
    if _s["name"] == "calculate":
        _s["description"] = (_s["description"] + " Также egfr(creatinine мкмоль/л) — СКФ CKD-EPI 2021 и fib4(ast, alt, platelets) — "
                             "индекс фиброза; возраст и пол подставляются из карты.")
        _s["parameters"] = {**_s["parameters"], "properties": {
            **_s["parameters"]["properties"],
            "formula": {"type": "string", "enum": ["bmi", "non_hdl", "homa_ir", "de_ritis", "egfr", "fib4"]},
            "creatinine": {"type": "number", "description": "креатинин, мкмоль/л"},
            "platelets": {"type": "number", "description": "тромбоциты, ×10⁹/л"}}}


class DomainToolsV3(DomainToolsV2):
    """v2 + T1 (document-aware search results) + T2 (hybrid BM25+dense search with ontology expansion)."""

    def __init__(self, p: Patient, dense: bool = True, expand: bool = True, doc_context: bool = True):
        super().__init__(p)
        from .retrieval import HybridIndex
        self.hybrid = HybridIndex(p, dense=dense, expand=expand)
        self.doc_context = doc_context

    def search_notes(self, query: str, specialty: str | None = None, top_k: int = 6) -> dict:
        from .retrieval import document_context
        top_k = max(1, min(int(top_k), 12))
        hits = [r for _, r in self.hybrid.search(query, top_k, specialty)]
        if not hits and specialty:
            hits = [r for _, r in self.hybrid.search(query, top_k)]
        if not hits:
            return {"documents": [], "note": "Ничего не найдено. Отсутствие совпадений НЕ означает, что исследования не было: "
                                             "проверьте get_overview или get_conclusions(specialty)."}
        if not self.doc_context:
            return {"results": [{**r.short(400), "specialty": r.specialty} for r in hits]}
        return {"documents": document_context(self.p, hits),
                "note": "Найденные строки (hits) показаны вместе с заключением документа и соседними строками."}


class DomainToolsV4(DomainToolsV3):
    """v3 + T3 typed entities: medications with phase and studies with key values, for every source."""

    def __init__(self, p: Patient):
        super().__init__(p)
        from .entities import Entities
        self.entities = Entities(p)

    def get_medications(self, drug: str | None = None, drug_class: str | None = None, phase: str | None = None) -> dict:
        return self.entities.medications(drug, drug_class, phase)

    def get_studies(self, kind: str | None = None) -> dict:
        return self.entities.study(kind)


V3_SPECS = [dict(s) for s in DOMAIN_SPECS]
for _s in V3_SPECS:
    if _s["name"] == "search_notes":
        _s["description"] = ("Поиск по тексту карты (жалобы, анамнез, протоколы исследований, заключения, назначения, выписка). "
                             "Понимает синонимы, бытовые формулировки и классы препаратов (например 'антитромботическая терапия', "
                             "'жиры в крови'). Возвращает документы: найденные строки вместе с заключением документа.")
_FINAL_V3 = {**FINAL_SPEC, "description": "Завершить работу и дать ответ врачу. Каждое утверждение подкрепите id строк, "
                                          "скопированными точно в том виде, в каком они приведены в поле id вывода инструментов."}
_FINAL_V3["parameters"] = {**FINAL_SPEC["parameters"], "properties": {
    **FINAL_SPEC["parameters"]["properties"],
    "citations": {"type": "array", "items": {"type": "string"}, "description": "id строк-источников, скопированные из вывода инструментов"}}}




def _v4_specs() -> list[dict]:
    return V3_SPECS + [
        S(name="get_medications", description=(
            "Назначения препаратов из всех документов (листы назначений, выписка, анамнез, догоспитальный этап, назначения "
            "специалистов): препарат (МНН, торговые названия приводятся к МНН), класс, доза, кратность, путь, фаза, дата, id. "
            "Фильтры: drug (название), drug_class (антиагрегант, антикоагулянт, гиполипидемический, бета-блокатор, блокатор РАС, "
            "диуретик, глифлозин, ИПП…), phase (при выписке, стационар, до госпитализации, догоспитальный). "
            "Если препарат не упоминается — found=false."),
          parameters={"type": "object", "properties": {
              "drug": {"type": "string", "description": "Название препарата"},
              "drug_class": {"type": "string", "description": "Класс препаратов"},
              "phase": {"type": "string", "description": "Фаза: при выписке / стационар / до госпитализации / догоспитальный"}},
              "required": []}),
        S(name="get_studies", description=(
            "Инструментальные исследования: ЭКГ, ЭхоКГ, холтер, УЗИ, КТ, МРТ, эндоскопия, коронарография, спирография, "
            "эластометрия — дата, заключение и ключевые значения (ФВ, ЧСС, ритм)."),
          parameters={"type": "object", "properties": {"kind": {"type": "string", "description": "Тип исследования"}}, "required": []}),
    ]


def build_toolset(condition: str, p: Patient) -> tuple[list[dict], dict[str, Callable[..., dict]]]:
    if condition in ("domain", "domain_v2"):
        t = DomainTools(p) if condition == "domain" else DomainToolsV2(p)
        fns = {s["name"]: getattr(t, s["name"]) for s in DOMAIN_SPECS}
        return DOMAIN_SPECS + [FINAL_SPEC], fns
    if condition == "generic":
        g = GenericTools(p)
        return GENERIC_SPECS + [FINAL_SPEC], {"run_python": g.run_python}
    if condition == "fullctx":
        return [FINAL_SPEC], {}
    if condition in ("domain_v4", "domain_v41"):
        t = DomainToolsV4(p)
        specs = _v4_specs()
        return specs + [_FINAL_V3], {s["name"]: getattr(t, s["name"]) for s in specs}
    if condition in ("domain_v3", "domain_v3p", "domain_v3r", "domain_v4a", "domain_v41a"):
        t = DomainToolsV3(p)
        return V3_SPECS + [_FINAL_V3], {s["name"]: getattr(t, s["name"]) for s in V3_SPECS}
    if condition == "domain_ri":
        t = DomainToolsRI(p)
        return RI_SPECS + [FINAL_SPEC], {s["name"]: getattr(t, s["name"]) for s in RI_SPECS}
    if condition == "hybrid":  # same record in context as fullctx, plus the v1 domain tools
        t = DomainTools(p)
        return DOMAIN_SPECS + [FINAL_SPEC], {s["name"]: getattr(t, s["name"]) for s in DOMAIN_SPECS}
    raise ValueError(condition)


def validate_args(spec: dict, args: Any) -> tuple[dict | None, str | None]:
    """Light JSON-schema check: object, required keys, primitive types, enums. Drops unknown keys."""
    if args is None:
        args = {}
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            return None, "Аргументы не являются корректным JSON-объектом."
    if not isinstance(args, dict):
        return None, "Аргументы должны быть объектом."
    props = spec["parameters"].get("properties", {})
    for req in spec["parameters"].get("required", []):
        if req not in args or args[req] in (None, ""):
            return None, f"Не передан обязательный аргумент '{req}'."
    clean = {}
    for k, v in args.items():
        if k not in props:
            continue
        typ = props[k].get("type")
        if typ == "number":
            try:
                v = float(str(v).replace(",", "."))
            except (TypeError, ValueError):
                return None, f"'{k}' должен быть числом."
        elif typ == "integer":
            try:
                v = int(v)
            except (TypeError, ValueError):
                return None, f"'{k}' должен быть целым числом."
        elif typ == "string" and not isinstance(v, str):
            v = str(v)
        elif typ == "array" and isinstance(v, str):
            v = [x.strip() for x in re.split(r"[,\s]+", v) if x.strip()]
        elif typ == "object" and isinstance(v, str):
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                return None, f"'{k}' должен быть JSON-объектом."
        if "enum" in props[k] and v not in props[k]["enum"]:
            return None, f"'{k}' должен быть одним из {props[k]['enum']}."
        clean[k] = v
    return clean, None


# v2 compact conclusions are dense by design; they get a larger window instead of being cut.
TOOL_LIMITS = {"get_conclusions": 7000, "search_notes": 6000, "get_medications": 6000, "get_studies": 6000}


def run_tool(fns: dict[str, Callable[..., dict]], name: str, args: dict) -> dict:
    try:
        owner = getattr(fns[name], "__self__", None)
        limit = TOOL_LIMITS.get(name, MAX_TOOL_CHARS) if isinstance(owner, DomainToolsV2) else MAX_TOOL_CHARS
        return _truncate(fns[name](**args), limit)
    except TypeError as exc:
        return {"error": f"Неверные аргументы: {exc}"}
    except Exception as exc:  # tool bug must not kill the episode; it is logged as an error
        return {"error": f"Ошибка инструмента: {type(exc).__name__}: {exc}"}

