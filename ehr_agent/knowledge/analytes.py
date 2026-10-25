"""Analyte catalog: canonical code, Russian name, synonyms, unit, indicative adult interval.

The indicative intervals are typical adult values of Russian commercial labs: the last layer of the
reference policy (see `reference.py`), used when the source has no lab blank. `stable=True` marks
analytes whose interval barely differs between labs; the benchmark gold relies only on these when
the policy is 'indicative'.

A lab record is mapped to an analyte by (1) a pattern over "group || field" (the check-up export's
naming) or (2) an exact normalized name from `LIS_ALIASES` (the PDF lab system's naming, typos
included). `PLAUSIBLE` holds physiological hard limits used by data-quality checks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..sources import Patient, Record


@dataclass(frozen=True)
class Analyte:
    code: str
    name: str
    pattern: str  # regex over "group || param", lowercase
    unit: str
    low: float | None = None
    high: float | None = None
    low_f: float | None = None  # female overrides
    high_f: float | None = None
    stable: bool = True
    aliases: tuple[str, ...] = ()

    def interval(self, sex: str | None) -> tuple[float | None, float | None]:
        if sex == "F":
            return (self.low_f if self.low_f is not None else self.low, self.high_f if self.high_f is not None else self.high)
        return self.low, self.high


A = Analyte
CATALOG: list[Analyte] = [
    # Biochemistry
    A("ALT", "АЛТ (аланинаминотрансфераза)", r"аланинаминотрансфераз", "Ед/л", None, 41, None, 33, aliases=("алт", "alt", "алат")),
    A("AST", "АСТ (аспартатаминотрансфераза)", r"аспартатаминотрансфераз", "Ед/л", None, 40, None, 32, aliases=("аст", "ast", "асат")),
    A(
        "GGT",
        "ГГТ (гамма-глутамилтранспептидаза)",
        r"гамма-?глю?т",
        "Ед/л",
        None,
        60,
        None,
        40,
        stable=False,
        aliases=("ггт", "ggt", "гамма-гт"),
    ),
    A("ALP", "Щелочная фосфатаза", r"щелочной фосфатаз", "Ед/л", 40, 130, 35, 105, stable=False, aliases=("щф", "alp")),
    A("TBIL", "Билирубин общий", r"билирубин общий", "мкмоль/л", 3.4, 20.5, aliases=("билирубин",)),
    A("DBIL", "Билирубин прямой", r"билирубин прямой", "мкмоль/л", None, 5.0, stable=False),
    A("IBIL", "Билирубин непрямой", r"билирубин непрямой", "мкмоль/л", None, 16.5, stable=False),
    A("TP", "Общий белок", r"общий белок", "г/л", 64, 83, aliases=("белок",)),
    A("ALB", "Альбумин", r"\|\| альбумин$", "г/л", 35, 52),
    A("UREA", "Мочевина", r"мочевина сыворотки", "ммоль/л", 2.8, 7.2, stable=False),
    A("URIC", "Мочевая кислота", r"мочевая кислота", "мкмоль/л", 202, 416, 142, 339, aliases=("урат",)),
    A("CREA", "Креатинин", r"креатинин сыворотки", "мкмоль/л", 62, 106, 44, 80, stable=False),
    A("GLU", "Глюкоза (плазма)", r"глюкоза в плазме", "ммоль/л", 3.9, 6.1, aliases=("сахар",)),
    A(
        "HBA1C",
        "Гликированный гемоглобин HbA1c",
        r"hba1c.*\|\| доля",
        "%",
        None,
        6.0,
        aliases=("гликированный", "hba1c", "гликозилированный"),
    ),
    A("INS", "Инсулин", r"^инсулин", "мкЕд/мл", 2.6, 24.9, stable=False),
    A("FE", "Железо сывороточное", r"железо в сыворотке", "мкмоль/л", 11.6, 31.3, 9.0, 30.4, stable=False),
    A("FERR", "Ферритин", r"^ферритин", "мкг/л", 20, 250, 10, 120, stable=False),
    A("TSAT", "Насыщение трансферрина железом", r"насыщени.*трансферрин", "%", 15, 50, stable=False),
    A("CRP", "С-реактивный белок", r"реактивный белок", "мг/л", None, 5.0, aliases=("срб", "crp")),
    A("AMYP", "Амилаза панкреатическая", r"панкреатическ", "Ед/л", 8, 53, stable=False),
    A("CER", "Церулоплазмин", r"церулоплазмин", "мг/дл", 20, 60, stable=False),
    A("K", "Калий", r"калий в сыворотке", "ммоль/л", 3.5, 5.1),
    A("NA", "Натрий", r"натрий в сыворотке", "ммоль/л", 136, 145),
    A("CL", "Хлор", r"хлор в сыворотке", "ммоль/л", 98, 107),
    A("CA", "Кальций общий", r"кальций", "ммоль/л", 2.15, 2.55),
    # Lipids
    A("CHOL", "Холестерин общий", r"холестерин сыворотки", "ммоль/л", None, 5.2, aliases=("холестерин", "хс", "холестерол")),
    A("LDL", "ЛПНП", r"\(лпнп\)", "ммоль/л", None, 3.0, aliases=("лпнп", "ldl", "липопротеиды низкой плотности")),
    A("HDL", "ЛПВП", r"\(лпвп\)", "ммоль/л", 1.0, None, 1.2, None, aliases=("лпвп", "hdl")),
    A("VLDL", "ЛПОНП", r"\(лпонп\)", "ммоль/л", None, 1.04, stable=False, aliases=("лпонп", "vldl")),
    A("TG", "Триглицериды", r"триглицериды сыворотки", "ммоль/л", None, 1.7, aliases=("тг", "триглицериды", "tg")),
    A("AI", "Коэффициент атерогенности", r"коэффициент атерогенности", "", None, 3.0, stable=False, aliases=("ка", "атерогенности")),
    A("LPA", "Липопротеин (a)", r"липопротеин \(a\)", "мг/дл", None, 30, stable=False, aliases=("lp(a)", "лп(а)")),
    # Hormones / vitamins / markers
    A("TSH", "ТТГ", r"\(ттг\)", "мЕд/л", 0.4, 4.0, aliases=("ттг", "tsh", "тиреотропный")),
    A("FT4", "Т4 свободный", r"\(т4\) свободный", "пмоль/л", 9.0, 19.0, stable=False, aliases=("т4", "ft4", "тироксин")),
    A("VITD", "Витамин D (25-OH)", r"витамин d", "нг/мл", 30, 100, aliases=("витамин д", "25-oh", "кальциферол")),
    A("TESTO", "Тестостерон", r"^тестостерон", "нмоль/л", 8.64, 29.0, 0.29, 1.67, stable=False),
    A("PSA", "ПСА общий", r"\(пса\)", "нг/мл", None, 4.0, aliases=("пса", "psa", "простатспецифический")),
    A("AFP", "Альфа-фетопротеин", r"альфа-фп", "МЕ/мл", None, 7.0, stable=False, aliases=("афп",)),
    A("CEA", "РЭА", r"\(рэа\)", "нг/мл", None, 5.0, stable=False, aliases=("рэа", "cea")),
    A("CA199", "CA 19-9", r"ca 19-9", "Ед/мл", None, 37, stable=False),
    A("CA153", "CA 15-3", r"ca 15-3", "Ед/мл", None, 25, stable=False),
    A("CA125", "CA 125", r"ca 125", "Ед/мл", None, 35, stable=False),
    A("IGE", "IgE общий", r"иммуноглобулины класса e", "МЕ/мл", None, 100, stable=False),
    # Cardiac markers (inpatient source; the blank's own interval is used when present)
    A(
        "TROP",
        "Тропонин I высокочувствительный",
        r"тропонин",
        "нг/л",
        None,
        None,
        stable=False,
        aliases=("тропонин", "тропонин i", "troponin", "hs-tni"),
    ),
    A(
        "CK",
        "Креатинфосфокиназа (КФК)",
        r"^креатинфосфокиназа \|\||\|\| креатинфосфокиназа$|\bкфк\b(?!.?mb)",
        "Ед/л",
        None,
        None,
        stable=False,
        aliases=("кфк", "креатинфосфокиназа", "креатинкиназа", "ck"),
    ),
    A(
        "CKMB",
        "КФК-МВ",
        r"креатинфосфокиназа\s*mb|кфк.?mb|кфк.?мв",
        "Ед/л",
        None,
        None,
        stable=False,
        aliases=("кфк-мв", "кфк mb", "креатинфосфокиназа mb", "ck-mb"),
    ),
    # Coagulation
    A("INR", "МНО", r"\|\| мно", "", 0.8, 1.2),
    A("APTT", "АЧТВ", r"\|\| ачтв", "с", 25, 37, stable=False),
    A("PT", "Протромбиновое время", r"протромбиновое время", "с", 11, 14, stable=False),
    A("PTI", "Протромбиновый индекс", r"протромбиновый индекс", "%", 70, 130, stable=False),
    A("TT", "Тромбиновое время", r"\|\| тромбиновое время", "с", 14, 21, stable=False),
    A("FIB", "Фибриноген", r"\|\| фибриноген", "г/л", 2.0, 4.0),
    # CBC
    A("WBC", "Лейкоциты", r"\(общее количество лейкоцитов\)", "×10⁹/л", 4.0, 9.0, aliases=("лейкоциты", "wbc")),
    A("RBC", "Эритроциты", r"\(общее количество эритроцитов\)", "×10¹²/л", 4.3, 5.7, 3.8, 5.1, aliases=("эритроциты", "rbc")),
    A("HGB", "Гемоглобин", r"\(концентрация гемоглобина\)", "г/л", 132, 173, 117, 155, aliases=("гемоглобин", "hgb", "hb")),
    A("HCT", "Гематокрит", r"\(гематокрит\)", "%", 39, 49, 35, 45),
    A("MCV", "MCV", r"\(средний объем эритроцитов\)", "фл", 80, 100),
    A("MCH", "MCH", r"\(среднее содержание гемоглобина", "пг", 27, 34),
    A("MCHC", "MCHC", r"\(средняя концентрация гемоглобина", "г/л", 320, 360),
    A("RDW", "RDW-CV", r"rdwcv", "%", 11.5, 14.5),
    A("PLT", "Тромбоциты", r"\(общее количество тромбоцитов\)", "×10⁹/л", 150, 400, aliases=("тромбоциты", "plt")),
    A("MPV", "MPV", r"\(средний объем тромбоцита\)", "фл", 7.4, 10.4, stable=False),
    A("ESR", "СОЭ", r"\|\| соэ$", "мм/ч", 2, 15, 2, 20, aliases=("соэ", "esr")),
    A("NEUP", "Нейтрофилы, %", r"neu%", "%", 47, 72),
    A("LYMP", "Лимфоциты, %", r"lym%", "%", 19, 37),
    A("MONOP", "Моноциты, %", r"mono%", "%", 3, 11),
    A("EOSP", "Эозинофилы, %", r"eos%", "%", 0.5, 5),
    A("BASOP", "Базофилы, %", r"baso%", "%", 0, 1),
    A("NEU", "Нейтрофилы, абс.", r"neu \(абсолютное", "×10⁹/л", 1.8, 7.7),
    A("LYM", "Лимфоциты, абс.", r"lym \(абсолютное", "×10⁹/л", 1.0, 4.8),
    A("MONO", "Моноциты, абс.", r"mono \(абсолютное", "×10⁹/л", 0.05, 0.82),
    A("EOS", "Эозинофилы, абс.", r"eos \(абсолютное", "×10⁹/л", 0.02, 0.5),
    A("BASO", "Базофилы, абс.", r"baso \(абсолютное", "×10⁹/л", 0.0, 0.08),
]
BY_CODE = {a.code: a for a in CATALOG}
_COMPILED = [(a, re.compile(a.pattern)) for a in CATALOG]

SKIP_PARAMS = {"общий комментарий", "результат", "интерпретация"}


def analyte_of(rec: Record) -> Analyte | None:
    if rec.specialty != "лаборатория" or rec.num is None:
        return None
    if rec.param.lower() in SKIP_PARAMS:
        return None
    key = f"{rec.group.lower()} || {rec.param.lower()}"
    for a, rx in _COMPILED:
        if rx.search(key):
            return a
    code = _alias_code(rec.param)
    return BY_CODE.get(code) if code else None


# Common LIS spellings (second clinic); typos are part of real data ("Блирубин").
LIS_ALIASES = {
    "креатинин": "CREA",
    "глюкоза": "GLU",
    "мочевина": "UREA",
    "натрий": "NA",
    "калий": "K",
    "хлор": "CL",
    "кальций": "CA",
    "лейкоциты": "WBC",
    "эритроциты": "RBC",
    "гематокрит": "HCT",
    "гемоглобин": "HGB",
    "тромбоциты": "PLT",
    "средний объём эритроцита": "MCV",
    "средний объем эритроцита": "MCV",
    "среднее объемное содержание гемоглобина в эритроците": "MCH",
    "среднее содержание гемоглобина в эритроците": "MCH",
    "средняя концентрация гемоглобина в эритроците": "MCHC",
    "распределение эритроцитов по величине (коэффициент вариации)": "RDW",
    "средний объём тромбоцитов": "MPV",
    "нейтрофилы %": "NEUP",
    "лимфоциты %": "LYMP",
    "моноциты %": "MONOP",
    "эозинофилы %": "EOSP",
    "базофилы %": "BASOP",
    "нейтрофилы": "NEU",
    "лимфоциты": "LYM",
    "моноциты": "MONO",
    "эозинофилы": "EOS",
    "базофилы": "BASO",
    "аст": "AST",
    "алт": "ALT",
    "ггт": "GGT",
    "щелочная фосфатаза": "ALP",
    "билирубин общий": "TBIL",
    "блирубин общий": "TBIL",
    "билирубин прямой": "DBIL",
    "блирубин прямой": "DBIL",
    "общий белок": "TP",
    "альбумин": "ALB",
    "мочевая кислота": "URIC",
    "холестерин": "CHOL",
    "холестерин общий": "CHOL",
    "лпнп": "LDL",
    "лпвп": "HDL",
    "холестерин лпнп": "LDL",
    "холестерин лпвп": "HDL",
    "холестерин липопротеинов низкой плотности": "LDL",
    "холестерин липопротеинов высокой плотности": "HDL",
    "триглицериды": "TG",
    "концентрация фибриногена": "FIB",
    "фибриноген": "FIB",
    "мно": "INR",
    "ачтв": "APTT",
    "протромбиновое время": "PT",
    "с-реактивный белок": "CRP",
    "срб": "CRP",
    "гликированный гемоглобин": "HBA1C",
    "соэ": "ESR",
    "ттг": "TSH",
    "ферритин": "FERR",
    "тропонин i сердечный высокочувствительный": "TROP",
    "креатинфосфокиназа": "CK",
    "креатинфосфокиназа mb": "CKMB",
}


_LIS = {k.replace("ё", "е"): v for k, v in LIS_ALIASES.items()}


def _alias_code(param: str) -> str | None:
    return _LIS.get(re.sub(r"\s+", " ", param.lower().replace("ё", "е").strip()))


def parse_ref(ref: str) -> tuple[float | None, float | None]:
    """'9.2:13.9', '0-29', '<5', '>60' → (low, high)."""
    t = (ref or "").replace(",", ".").replace(" ", "")
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)[:\-–](-?\d+(?:\.\d+)?)", t)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.fullmatch(r"[<≤](\d+(?:\.\d+)?)", t)
    if m:
        return None, float(m.group(1))
    m = re.fullmatch(r"[>≥](\d+(?:\.\d+)?)", t)
    if m:
        return float(m.group(1)), None
    return None, None


def flag(a: Analyte, value: float, sex: str | None) -> str:
    lo, hi = a.interval(sex)
    if lo is not None and value < lo:
        return "L"
    if hi is not None and value > hi:
        return "H"
    return "N"


def interval_text(a: Analyte, sex: str | None) -> str:
    lo, hi = a.interval(sex)
    if lo is not None and hi is not None:
        return f"{lo}–{hi} {a.unit}".strip()
    if hi is not None:
        return f"≤{hi} {a.unit}".strip()
    if lo is not None:
        return f"≥{lo} {a.unit}".strip()
    return "нет"


def lab_values(p: Patient) -> list[tuple[Record, Analyte]]:
    out = []
    for r in p.records:
        a = analyte_of(r)
        if a:
            out.append((r, a))
    return out


def _norm(s: str) -> str:
    return re.sub(r"[^a-zа-я0-9()%]+", " ", s.lower().replace("ё", "е")).strip()


def resolve_analyte(query: str) -> list[Analyte]:
    """Map a free-form analyte name (ru/en/abbreviation) to catalog entries."""
    q = _norm(query)
    if not q:
        return []
    if q.upper() in BY_CODE:
        return [BY_CODE[q.upper()]]
    exact, partial = [], []
    for a in CATALOG:
        names = [_norm(a.name), a.code.lower(), *(_norm(x) for x in a.aliases)]
        if q in names:
            exact.append(a)
        elif any(q in n or (len(n) > 3 and n in q) for n in names):
            partial.append(a)
    return exact or partial


# Physiological hard limits (adult): a value outside is not "abnormal" but implausible — a unit or
# export error (e.g. a row shift). Used by quality checks for every source.
PLAUSIBLE: dict[str, tuple[float, float]] = {
    "HGB": (40, 250),
    "RBC": (1.0, 9.0),
    "HCT": (10, 70),
    "MCV": (50, 130),
    "MCH": (15, 45),
    "MCHC": (250, 400),
    "WBC": (0.3, 100),
    "PLT": (5, 1500),
    "NEUP": (0, 100),
    "LYMP": (0, 100),
    "MONOP": (0, 100),
    "EOSP": (0, 100),
    "BASOP": (0, 20),
    "BASO": (0, 2),
    "GLU": (1, 40),
    "CREA": (15, 2500),
    "UREA": (0.5, 80),
    "K": (1.5, 9.5),
    "NA": (100, 180),
    "CL": (70, 140),
    "TP": (25, 130),
    "ALB": (8, 70),
    "CHOL": (0.5, 25),
    "TG": (0.1, 60),
    "LDL": (0.1, 20),
    "HDL": (0.1, 6),
    "TBIL": (0.5, 900),
}


def implausible(a: Analyte, value: float, lab_ref: str = "") -> bool:
    """A value that cannot be a real measurement in the catalog's units (physiological hard limits).

    Values that carry the issuing lab's own interval are not judged here: their units may differ from
    the catalog (e.g. MCHC in g/dl), and a gross deviation from the lab's interval is pathology
    (troponin in myocardial infarction), not an export error.
    """
    if lab_ref:
        return False
    lo, hi = PLAUSIBLE.get(a.code, (None, None))
    return lo is not None and not (lo <= value <= hi)
