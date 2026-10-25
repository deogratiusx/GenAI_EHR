"""T3: source-independent clinical entities extracted from any source's records (FHIR-like, simplified).

MedicationMention ~ FHIR MedicationRequest/Statement: drug (INN), ATC-like class, dose, unit, frequency,
  route, phase (prehospital / home / inpatient / discharge / specialist recommendation), date, row id.
StudyReport ~ FHIR DiagnosticReport: study kind, date, conclusion text, key values (EF, HR, rhythm).

Rule-based extraction over masked record text; the lexicon is curated for the PoC (Russian INN and trade
names of drugs met in cardiology and check-up records), not a licensed drug database.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..sources import Patient, Record

# INN → (class, aliases incl. trade names and inflection stems)
DRUGS: dict[str, tuple[str, tuple[str, ...]]] = {
    "ацетилсалициловая кислота": ("антиагрегант", ("ацетилсалицил", "аспирин", "кардиомагнил", "тромбо асс", "аск ")),
    "клопидогрел": ("антиагрегант", ("клопидогрел", "плавикс", "зилт")),
    "тикагрелор": ("антиагрегант", ("тикагрелор", "брилинт")),
    "прасугрел": ("антиагрегант", ("прасугрел", "эффиент")),
    "гепарин": ("антикоагулянт", ("гепарин", "нфг")),
    "эноксапарин": ("антикоагулянт", ("эноксапарин", "клексан")),
    "фондапаринукс": ("антикоагулянт", ("фондапаринукс", "арикстр")),
    "ривароксабан": ("антикоагулянт", ("ривароксабан", "ксарелто")),
    "апиксабан": ("антикоагулянт", ("апиксабан", "эликвис")),
    "дабигатран": ("антикоагулянт", ("дабигатран", "прадакс")),
    "варфарин": ("антикоагулянт", ("варфарин",)),
    "аторвастатин": ("гиполипидемический", ("аторвастатин", "липримар", "аторис")),
    "розувастатин": ("гиполипидемический", ("розувастатин", "крестор", "розукард")),
    "симвастатин": ("гиполипидемический", ("симвастатин",)),
    "эзетимиб": ("гиполипидемический", ("эзетимиб", "эзетрол")),
    "метопролол": ("бета-блокатор", ("метопролол", "беталок", "эгилок")),
    "бисопролол": ("бета-блокатор", ("бисопролол", "конкор")),
    "карведилол": ("бета-блокатор", ("карведилол",)),
    "периндоприл": ("блокатор РАС", ("периндоприл", "престариум", "нолипрел")),
    "эналаприл": ("блокатор РАС", ("эналаприл",)),
    "лизиноприл": ("блокатор РАС", ("лизиноприл",)),
    "рамиприл": ("блокатор РАС", ("рамиприл",)),
    "лозартан": ("блокатор РАС", ("лозартан", "лориста")),
    "валсартан": ("блокатор РАС", ("валсартан",)),
    "сакубитрил/валсартан": ("блокатор РАС", ("сакубитрил", "юперио")),
    "амлодипин": ("блокатор кальциевых каналов", ("амлодипин",)),
    "лерканидипин": ("блокатор кальциевых каналов", ("лерканидипин",)),
    "индапамид": ("диуретик", ("индапамид", "нолипрел", "арифон")),
    "торасемид": ("диуретик", ("торасемид", "диувер")),
    "фуросемид": ("диуретик", ("фуросемид",)),
    "спиронолактон": ("диуретик", ("спиронолактон", "верошпирон")),
    "эплеренон": ("диуретик", ("эплеренон",)),
    "дапаглифлозин": ("глифлозин", ("дапаглифлозин", "форсиг")),
    "эмпаглифлозин": ("глифлозин", ("эмпаглифлозин", "джардинс")),
    "метформин": ("сахароснижающий", ("метформин", "глюкофаж")),
    "алоглиптин": ("сахароснижающий", ("алоглиптин",)),
    "омепразол": ("ИПП", ("омепразол",)),
    "пантопразол": ("ИПП", ("пантопразол",)),
    "эзомепразол": ("ИПП", ("эзомепразол", "нексиум", "эманера")),
    "рабепразол": ("ИПП", ("рабепразол",)),
    "тегопразан": ("ИПП", ("тегопразан",)),
    "аллопуринол": ("противоподагрический", ("аллопуринол",)),
    "морфин": ("анальгетик", ("морфин",)),
    "нитроглицерин": ("нитрат", ("нитроглицерин", "нитроспрей")),
    "амоксициллин": ("антибиотик", ("амоксициллин", "флемоксин")),
    "кларитромицин": ("антибиотик", ("кларитромицин",)),
    "колекальциферол": ("витамин D", ("колекальциферол", "вигантол", "детримакс", "аквадетрим")),
    "лорноксикам": ("НПВП", ("лорноксикам", "ксефокам", "ксефкам", "ксефркам")),
    "напроксен": ("НПВП", ("напроксен", "тералив")),
    "золмитриптан": ("противомигренозный", ("золмитриптан", "эксенза")),
    "амитриптилин": ("антидепрессант", ("амитриптилин", "амитриптиллин")),
    "левотироксин": ("гормон щитовидной железы", ("левотироксин", "эутирокс", "l-тироксин")),
    "пробиотик": ("пробиотик", ("энтеролактис", "бифиформ", "линекс")),
}
_ALIAS = sorted(((a, inn) for inn, (_, al) in DRUGS.items() for a in al), key=lambda x: -len(x[0]))
_DRUG_RX = re.compile("|".join(re.escape(a) for a, _ in _ALIAS), re.I)
_DOSE_RX = re.compile(r"(\d+(?:[.,]\d+)?)\s*(мг/кг|ед/кг|мг|мкг|г\b|мл|ед\b|ЕД\b|МЕ|ме\b|капс|таб)", re.I)
_FREQ_RX = re.compile(r"(\d)\s*раз[а]?\s*в\s*(день|сутки)|\d\s*р/[дс]|утром и вечером|утром|вечером|однократно|болюсно|длительно", re.I)
_ROUTE_RX = re.compile(r"перорально|внутрь|п/к|подкожно|в/в|внутривенно|в\\в|ингаляц", re.I)

PHASES = [
    ("догоспитальный этап", r"догоспитальн|смп|скорой"),
    ("при выписке", r"при выписке|после выписки|рекомендац\w*\s+при\s+выписк|амбулаторн|по месту жительства|длительно, под контролем"),
    (
        "до госпитализации (постоянно принимал)",
        r"постоянн\w* принима|принимаем\w* медикаментозн|лекарственный анамнез|ранее принимал|анамнез",
    ),
    ("стационар", r"лист назначен|дней приема|перорально, дней|назначения|план лечения|в\s+стационар|получал|инфузион|интенсивн"),
]


# Phase from the field label of the block (the strongest cue in form 003/у), checked after the
# prehospital context and before the free-text context rules.
PARAM_PHASES = [
    ("во время вмешательства", r"описание операции|протокол операции|ход операции|вид анестезии|течени\w* анестез"),
    ("при выписке", r"амбулаторн|при выписке|после выписки"),
    ("до госпитализации (постоянно принимал)", r"история настоящего заболевания|анамнез"),
    ("стационар", r"стационарн|таблетированн|инъекционн|инфузионн|план лечения|^лечение|терапия$|лист назначен"),
]


def drug_mentioned(text: str) -> str | None:
    """The first drug name (INN or trade name) found in a text, or None."""
    m = _DRUG_RX.search(text)
    return m.group(0).lower() if m else None


def _phase(r: Record, ctx: str) -> str:
    if r.patient.startswith("P"):
        return "назначение специалиста (чекап)"
    if re.search(PHASES[0][1], ctx.lower()):
        return PHASES[0][0]
    pl = r.param.lower()
    for name, rx in PARAM_PHASES:
        if re.search(rx, pl):
            return name
    if r.specialty == "лист назначений":
        return "стационар"
    t = f"{r.protocol} {r.param} {ctx}".lower()
    # home medication quoted inside any document (anamnesis) wins over the document type
    if re.search(r"постоянн\w* принима|принимаем\w* медикаментозн|лекарственный анамнез|ранее принимал", t):
        return "до госпитализации (постоянно принимал)"
    if r.specialty == "выписной эпикриз" and re.search(r"рекомендац|назначен|терапи", f"{r.param} {ctx}", re.I):
        return "при выписке"
    for name, rx in PHASES:
        if re.search(rx, t):
            return name
    return "не указана"


@dataclass
class MedicationMention:
    drug: str
    drug_class: str
    id: str
    date: str
    phase: str
    dose: str = ""
    frequency: str = ""
    route: str = ""
    text: str = ""


@dataclass
class StudyReport:
    kind: str
    id: str
    date: str
    source: str
    conclusion: str
    values: dict = field(default_factory=dict)


STUDY_KINDS = {
    "ЭКГ": r"\bэкг\b|электрокардиограф",
    "ЭхоКГ": r"эхокардиограф|эхокг|узи сердца",
    "холтер": r"холтер",
    "УЗИ": r"\bузи\b|ультразвуков|эхографическ|доплер|триплекс",
    "КТ": r"компьютерн\w* томограф|\bкт\b|мскт",
    "МРТ": r"магнитно-резонанс|\bмрт\b|мр-признак|мр картина",
    "эндоскопия": r"гастроскоп|колоноскоп|эгдс|фгдс|ректосигмо",
    "коронарография": r"коронарограф|стентирован|ангиопласт|тбка|чкв",
    "спирография": r"спирограф",
    "эластометрия": r"эластометр|фиброскан",
}
_EF_RX = re.compile(r"(?:фв|фракци\w* выброса)[^0-9]{0,30}(\d{2})\s*%?", re.I)
_HR_RX = re.compile(r"чсс[^0-9]{0,15}(\d{2,3})", re.I)
_RHYTHM_RX = re.compile(r"синусов\w* ритм|фибрилляц\w* предсердий|трепетани\w*", re.I)
_CONCL_PARAM = re.compile(r"^(заключение|диагноз|выводы?)\b", re.I)


class Entities:
    def __init__(self, p: Patient):
        self.p = p
        self.meds: list[MedicationMention] = []
        self.studies: list[StudyReport] = []
        self._extract()

    def _extract(self) -> None:
        for r in self.p.records:
            if r.specialty == "лаборатория" and r.num is not None:
                continue
            text = r.value
            for m in _DRUG_RX.finditer(text):
                inn = next(inn for a, inn in _ALIAS if a.lower() == m.group(0).lower())
                window = text[m.start() : m.start() + 140]
                ctx = text[max(0, m.start() - 160) : m.start()]
                dose = _DOSE_RX.search(window)
                freq = _FREQ_RX.search(window)
                route = _ROUTE_RX.search(window)
                self.meds.append(
                    MedicationMention(
                        drug=inn,
                        drug_class=DRUGS[inn][0],
                        id=r.id,
                        date=r.date,
                        phase=_phase(r, ctx),
                        dose=(dose.group(1) + " " + dose.group(2)) if dose else "",
                        frequency=freq.group(0) if freq else "",
                        route=route.group(0) if route else "",
                        text=text[max(0, m.start() - 40) : m.start() + 160].strip(),
                    )
                )
                if inn in ("периндоприл",) and "нолипрел" in m.group(0).lower():  # fixed combination
                    self.meds.append(
                        MedicationMention(
                            drug="индапамид", drug_class="диуретик", id=r.id, date=r.date, phase=_phase(r, ctx), text=self.meds[-1].text
                        )
                    )
        # Studies: one report per document; the conclusion comes from that document's conclusion rows.
        docs: dict[tuple, list[Record]] = {}
        for r in self.p.records:
            docs.setdefault((r.visit, r.protocol), []).append(r)
        for (_visit, protocol), rows in docs.items():
            head = f"{protocol} {rows[0].visit_title} " + " ".join(r.value[:300] for r in rows[:4])
            if re.search(r"лист\s+назнач|назначенных\s+услуг|услуг\w*\s+статус", head, re.I):
                continue  # lists of ordered services mention studies without being one
            kind = next((k for k, rx in STUDY_KINDS.items() if re.search(rx, head[:900], re.I)), None)
            if kind is None or rows[0].specialty == "лаборатория":
                continue
            concl = [r for r in rows if _CONCL_PARAM.match(r.param.strip()) and len(r.value) > 5]
            body = " ".join(r.value for r in rows)
            values = {}
            ef = _EF_RX.search(body)
            if ef:
                values["ФВ, %"] = ef.group(1)
            hr = _HR_RX.search(body)
            if hr and kind in ("ЭКГ", "холтер", "ЭхоКГ"):
                values["ЧСС"] = hr.group(1)
            rh = _RHYTHM_RX.search(body)
            if rh and kind in ("ЭКГ", "холтер"):
                values["ритм"] = rh.group(0)
            if not concl and not values:
                continue
            pick = concl[0] if concl else rows[0]
            self.studies.append(
                StudyReport(
                    kind=kind,
                    id=pick.id,
                    date=pick.date,
                    source=(pick.visit_title or protocol)[:70],
                    conclusion=pick.value[:450],
                    values=values,
                )
            )

    # ------------------------------------------------------------------ queries

    def medications(self, drug: str | None = None, drug_class: str | None = None, phase: str | None = None) -> dict:
        ms = self.meds
        if drug:
            d = drug.lower().replace("ё", "е")
            inns = {inn for a, inn in _ALIAS if a.lower().strip() in d or d in a.lower() or d in inn}
            ms = [m for m in ms if m.drug in inns or d in m.drug]
            if not ms:
                return {
                    "found": False,
                    "drug": drug,
                    "checked_records": len(self.p.records),
                    "note": f"Препарат «{drug}» в карте не упоминается (проверены все текстовые записи, включая листы назначений и выписку).",
                }
        if drug_class:
            c = drug_class.lower()
            ms = [m for m in ms if c[:5] in m.drug_class.lower()]
        if phase:
            ph = phase.lower()
            ms = [m for m in ms if ph[:5] in m.phase.lower()]
        grouped: dict[str, dict] = {}
        for m in ms:
            g = grouped.setdefault(m.drug, {"drug": m.drug, "class": m.drug_class, "mentions": []})
            key = (m.dose, m.phase, m.frequency)
            if key in {(x["dose"], x["phase"], x["frequency"]) for x in g["mentions"]}:
                continue
            g["mentions"].append(
                {
                    "id": m.id,
                    "date": m.date,
                    "phase": m.phase,
                    "dose": m.dose,
                    "frequency": m.frequency,
                    "route": m.route,
                    "text": m.text[:180],
                }
            )
        items = list(grouped.values())
        for g in items:
            g["mentions"] = g["mentions"][:6]
        return {
            "found": bool(items),
            "medications": items,
            "note": "Фаза определена по контексту записи (лист назначений, выписка, анамнез, догоспитальный этап); дозы — как в тексте.",
        }

    def study(self, kind: str | None = None) -> dict:
        ss = self.studies
        if kind:
            k = kind.lower()
            key = next((name for name, rx in STUDY_KINDS.items() if re.search(rx, k, re.I) or name.lower() in k), None)
            ss = [s for s in ss if s.kind == key] if key else [s for s in ss if k in s.source.lower() or k in s.conclusion.lower()]
            if not ss:
                return {
                    "found": False,
                    "kind": kind,
                    "available": sorted({s.kind for s in self.studies}),
                    "note": "Исследований такого типа в карте нет; см. available.",
                }
        return {
            "found": bool(ss),
            "studies": [
                {"kind": s.kind, "id": s.id, "date": s.date, "source": s.source, "values": s.values, "conclusion": s.conclusion}
                for s in ss[:10]
            ],
        }
