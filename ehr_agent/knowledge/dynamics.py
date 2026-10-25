"""Dynamics of a lab value: is a change real or within measurement noise?

A difference between two results of one patient is clinically meaningful only when it exceeds the
reference change value (RCV), the combined analytical and within-subject biological variation:

    RCV = 1.96 · √2 · √(CVa² + CVi²)      (two-sided, 95 %)

CVi below are rounded within-subject biological variation estimates (EFLM Biological Variation
Database / Ricós et al.); CVa is a typical analytical imprecision. Values are indicative — enough to
separate a real trend from noise in a proof of concept, not a laboratory's validated RCV.
"""

from __future__ import annotations

import math

CVA_DEFAULT = 3.0  # % analytical imprecision of a routine analyser
CVI_DEFAULT = 10.0  # % when the analyte has no estimate below
CVI = {  # % within-subject biological variation
    "ALT": 9.3, "AST": 9.5, "ALP": 5.3, "TBIL": 21.8, "DBIL": 22.0, "IBIL": 22.0, "TP": 2.6, "ALB": 2.5,
    "UREA": 13.9, "URIC": 8.4, "CREA": 4.4, "GLU": 5.0, "K": 4.1, "NA": 0.6, "CL": 1.2, "CA": 1.8,
    "CHOL": 5.3, "LDL": 7.8, "HDL": 5.8, "TG": 19.9, "CRP": 42.2, "FE": 26.5, "FERR": 14.2,
    "HGB": 2.7, "HCT": 2.8, "RBC": 3.2, "WBC": 11.4, "PLT": 7.3, "NEU": 14.1, "LYM": 10.8, "ESR": 30.0,
    "TROP": 9.7, "CK": 14.5, "CKMB": 19.7, "INR": 4.5, "APTT": 2.7, "FIB": 8.6, "TSH": 17.7, "HBA1C": 1.6,
}  # fmt: skip


def rcv(code: str) -> float:
    """Reference change value, % of the first result."""
    return round(1.96 * math.sqrt(2) * math.hypot(CVA_DEFAULT, CVI.get(code, CVI_DEFAULT)), 1)


def describe(code: str, points: list[dict]) -> dict:
    """Summary of a series of {date, value, flag, id} sorted by time: first, last, peak, change, verdicts."""
    first, last = points[0], points[-1]
    peak = max(points, key=lambda p: p["value"])
    trough = min(points, key=lambda p: p["value"])
    change = (last["value"] - first["value"]) / abs(first["value"]) * 100 if first["value"] else 0.0
    threshold = rcv(code)
    significant = abs(change) > threshold
    out = {
        "n": len(points),
        "first": {k: first[k] for k in ("date", "value", "flag", "id")},
        "last": {k: last[k] for k in ("date", "value", "flag", "id")},
        "peak": {k: peak[k] for k in ("date", "value", "flag", "id")},
        "change_pct": round(change, 1),
        "rcv_pct": threshold,
        "direction": ("рост" if change > 0 else "снижение") if significant else "без значимых изменений",
    }

    def moved(a: float, b: float) -> bool:
        return abs(b - a) / abs(a) * 100 > threshold if a else b != a

    # A rise to an interior peak and a fall after it (troponin in infarction) is neither "рост" nor "снижение".
    if peak is not first and peak is not last and moved(first["value"], peak["value"]) and moved(peak["value"], last["value"]):
        out["direction"] = "рост, затем снижение"
    elif trough is not first and trough is not last and moved(first["value"], trough["value"]) and moved(trough["value"], last["value"]):
        out["direction"] = "снижение, затем рост"
    if trough["value"] < first["value"] and trough is not first and trough is not last:
        out["trough"] = {k: trough[k] for k in ("date", "value", "flag", "id")}
    abnormal_before = any(p["flag"] in ("H", "L") for p in points[:-1])
    if abnormal_before and last["flag"] == "N":
        out["course"] = "нормализовался к последнему измерению"
    elif first["flag"] == "N" and last["flag"] in ("H", "L"):
        out["course"] = "вышел за пределы нормы"
    elif last["flag"] in ("H", "L"):
        out["course"] = "вне нормы на последнем измерении"
    else:
        out["course"] = "в пределах нормы"
    return out
