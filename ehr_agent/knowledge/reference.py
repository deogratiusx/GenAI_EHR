"""Reference policy: which interval judges a lab value, and where it came from.

Layers, first available wins (policy 'layered'):
  1. the issuing lab's blank (PDF lab tables carry interval and flag);
  2. a guideline decision limit (lipids, glucose, HbA1c, vitamin D) — a healthy population is itself
     dyslipidaemic or vitamin-D deficient, so percentiles would hide the risk;
  3. a population interval by sex and age derived from NHANES (`nhanes.py`, data/nhanes_ri.json);
  4. the indicative catalog interval (`analytes.py`).
Policy 'indicative' uses layers 1 and 4 only. The chosen layer is always returned as `source`.
"""

from __future__ import annotations

import json
from functools import lru_cache

from ..config import DATA_DIR, ref_policy
from ..sources import Record
from .analytes import Analyte, flag, interval_text, parse_ref

NHANES_RI = DATA_DIR / "nhanes_ri.json"
# Analytes judged against guideline decision limits rather than population percentiles: a "normal"
# population is itself dyslipidaemic / vitamin-D deficient, so percentiles would hide the risk.
DECISION_LIMITS = {
    "CHOL": "≤5.2 ммоль/л (желательный уровень)",
    "LDL": "≤3.0 ммоль/л (без факторов риска; ниже при высоком риске)",
    "HDL": "≥1.0 (М) / ≥1.2 (Ж) ммоль/л",
    "TG": "≤1.7 ммоль/л",
    "GLU": "3.9–6.1 ммоль/л натощак",
    "HBA1C": "<6.0 %",
    "VITD": "≥30 нг/мл (20–30 — недостаточность, <20 — дефицит)",
}


@lru_cache(maxsize=1)
def _nhanes() -> dict:
    return json.loads(NHANES_RI.read_text(encoding="utf-8")) if NHANES_RI.exists() else {}


def nhanes_interval(code: str, sex: str | None, age: int | None) -> dict | None:
    parts = _nhanes().get("intervals", {}).get(code) or []
    if not parts or sex not in ("M", "F"):
        return None
    cands = [p for p in parts if p["sex"] == sex]
    if age is not None:
        cands = [p for p in cands if p["age"][0] <= age <= p["age"][1]] or cands
    if not cands:
        return None
    p = cands[0]
    return {"lower": p["lower"], "upper": p["upper"], "n": p["n"], "partition": f"{sex}, {p['age'][0]}–{p['age'][1]} лет"}


# Assay-dependent analytes where the NHANES method differs from Russian lab practice by more than the
# interval width (free T4: ~8–14 vs ~9–19 pmol/L); for these the population layer is skipped.
NHANES_SKIP = {"FT4"}
DECISION_RANGES = {  # numeric form of DECISION_LIMITS: (low, high) by sex
    "CHOL": {None: (None, 5.2)},
    "LDL": {None: (None, 3.0)},
    "TG": {None: (None, 1.7)},
    "HDL": {"M": (1.0, None), "F": (1.2, None)},
    "GLU": {None: (3.9, 6.1)},
    "HBA1C": {None: (None, 6.0)},
    "VITD": {None: (30, 100)},
}


def _flag(v: float, lo: float | None, hi: float | None) -> str:
    return "L" if lo is not None and v < lo else "H" if hi is not None and v > hi else "N"


def resolve_reference(rec: Record, a: Analyte, sex: str | None, age: int | None, policy: str | None = None) -> dict:
    """Pick the reference for one value: lab blank → guideline decision limit → NHANES (sex, age) → indicative.

    Returns {"reference", "flag", "source"}; the source is always reported so a reviewer can see
    which layer judged the value.
    """
    policy = policy or ref_policy()
    if rec.ref:
        lo, hi = parse_ref(rec.ref)
        if lo is not None or hi is not None:
            return {"reference": rec.ref, "flag": _flag(rec.num, lo, hi), "source": "бланк лаборатории"}
        lab = {"Повыш": "H", "Пониж": "L", "Норм": "N"}.get(rec.lab_flag[:5])
        if lab:
            return {"reference": rec.ref or "—", "flag": lab, "source": "бланк лаборатории (флаг)"}
    if policy == "layered":
        if a.code in DECISION_RANGES:
            lo, hi = DECISION_RANGES[a.code].get(sex, DECISION_RANGES[a.code].get(None, (None, None)))
            return {"reference": DECISION_LIMITS[a.code], "flag": _flag(rec.num, lo, hi), "source": "порог клинических рекомендаций"}
        ri = None if a.code in NHANES_SKIP else nhanes_interval(a.code, sex, age)
        if ri:
            return {
                "reference": f"{ri['lower']:g}–{ri['upper']:g} {a.unit}".strip(),
                "flag": _flag(rec.num, ri["lower"], ri["upper"]),
                "source": f"NHANES 2009–2018, {ri['partition']}",
            }
    if a.low is None and a.high is None:
        return {"reference": "нет", "flag": "?", "source": "интервал не задан"}
    return {"reference": interval_text(a, sex), "flag": flag(a, rec.num, sex), "source": "ориентировочный справочник"}
