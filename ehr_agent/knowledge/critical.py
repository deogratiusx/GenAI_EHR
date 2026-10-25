"""Clinical significance of findings: what a summary must not omit.

Three levels:
  * «критично» — a value beyond a critical limit (critical-value lists of clinical labs), a large
    multiple of the upper limit, or a text finding that requires action (neoplasm, infarction,
    thrombosis, reduced ejection fraction…);
  * «значимо» — any other value outside its reference interval;
  * «фон» — everything else.
The rules are deterministic and deliberately conservative; they decide what the report checklist
contains, not the clinical verdict, which stays with the model and the doctor.
"""

from __future__ import annotations

import re

# Absolute critical limits (low, high) in catalog units — typical critical-value lists.
LIMITS = {
    "K": (3.0, 6.0, "калий вне критических пределов"),
    "NA": (125, 155, "натрий вне критических пределов"),
    "GLU": (2.8, 20.0, "глюкоза вне критических пределов"),
    "CA": (1.8, 3.2, "кальций вне критических пределов"),
    "HGB": (80, None, "гемоглобин ниже 80 г/л"),
    "PLT": (50, 1000, "тромбоциты вне критических пределов"),
    "WBC": (2.0, 30.0, "лейкоциты вне критических пределов"),
    "INR": (None, 4.0, "МНО выше 4"),
    "CRP": (None, 100, "СРБ выше 100 мг/л"),
    "LDL": (None, 4.9, "ЛПНП ≥ 4.9: порог тяжёлой гиперхолестеринемии"),
    "HBA1C": (None, 6.49, "HbA1c ≥ 6.5 %: диагностический порог сахарного диабета"),
}
# Any value above the upper limit of the interval is critical for these analytes …
ANY_HIGH = {"TROP": "повышение тропонина — повреждение миокарда", "CKMB": "повышение КФК-МВ"}
# … and for these from a multiple of the upper limit.
FOLD_HIGH = {"ALT": 3, "AST": 3, "CREA": 2, "CK": 5, "TBIL": 3, "UREA": 3}

TEXT_RULES = [
    (r"новообразован|опухол|злокачеств|\bрак\b|карцином|\{C\d\d", "новообразование"),
    (r"bi-?rads\D{0,6}[45]|ti-?rads\D{0,6}[45]|lung-?rads\D{0,6}[34]", "подозрительное образование (RADS 4–5)"),
    (r"инфаркт\w* миокард|\{I2[12]", "инфаркт миокарда"),
    (r"инсульт|онмк|\{I6[34]", "инсульт"),
    (r"тромбоэмбол|\bтэла\b|тромбоз", "тромбоз или эмболия"),
    (r"аневризм", "аневризма"),
    (r"фибрилляц\w* предсерд|трепетани\w* предсерд|\{I48", "фибрилляция или трепетание предсердий"),
    (r"стеноз\w*[^.]{0,40}(?:[7-9]\d|100)\s*%", "гемодинамически значимый стеноз"),
]
_TEXT = [(re.compile(rx, re.I), label) for rx, label in TEXT_RULES]
_NEG_BEFORE = re.compile(r"(?:не\s+выявлен\w*|не\s+обнаружен\w*|\bбез\b|\bнет\b|отсутств\w*|исключ\w*)[^.;]{0,40}$", re.I)
_NEG_AFTER = re.compile(r"^[^.;]{0,40}?(?:\bнет\b|не\s+выявлен|не\s+обнаружен|не\s+определ|отсутству)", re.I)
_BENIGN = re.compile(r"доброкачеств\w*[^.;]{0,15}$", re.I)


def lab_level(code: str, value: float, flag: str, upper: float | None) -> tuple[str, str]:
    """(level, reason) of one lab value."""
    lo, hi, why = LIMITS.get(code, (None, None, ""))
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        return "критично", why
    if code in ANY_HIGH and flag == "H":
        return "критично", ANY_HIGH[code]
    if code in FOLD_HIGH and upper and value >= FOLD_HIGH[code] * upper:
        return "критично", f"≥ {FOLD_HIGH[code]}× верхней границы нормы"
    if flag in ("H", "L"):
        return "значимо", "вне референсного интервала"
    return "фон", ""


def text_findings(text: str) -> list[str]:
    """Labels of action-requiring findings in a text, skipping negated ("без признаков тромбоза",
    "признаков тромбоза нет") and benign ("доброкачественное новообразование") mentions."""
    out = []
    for rx, label in _TEXT:
        for m in rx.finditer(text):
            before, after = text[max(0, m.start() - 60) : m.start()], text[m.end() : m.end() + 60]
            if not (_NEG_BEFORE.search(before) or _NEG_AFTER.search(after) or _BENIGN.search(before)):
                out.append(label)
                break
    return out
