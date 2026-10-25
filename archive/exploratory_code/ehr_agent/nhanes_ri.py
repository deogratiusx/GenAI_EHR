"""Derive age/sex-partitioned adult reference intervals from NHANES 2017–2018 (direct method).

Proof of concept for a machine-readable, population-based interval source (US population, US
analysers — not the clinic's lab). Reference individuals: age 18–79, not pregnant, BMI 18.5–30,
no diabetes (questionnaire or HbA1c >= 6.5 %), hs-CRP <= 10 mg/L, no self-reported liver condition.
Limits are survey-weighted 2.5th/97.5th percentiles; a partition needs >= 120 individuals
(CLSI EP28-A3c nonparametric minimum), otherwise the next coarser partition is used.

  python -m ehr_agent.nhanes_ri   →  data/nhanes_ri.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DIR = Path(__file__).resolve().parent.parent / "data" / "nhanes"
OUT = DIR.parent / "nhanes_ri.json"
AGE_BANDS = [(18, 39), (40, 59), (60, 79)]
MIN_N = 120

# catalog code → (file, column, factor to our unit, weight column)
MEC, FAST = "WTMEC2YR", "WTSAF2YR"
ANALYTES = {
    "ALT": ("BIOPRO_J", "LBXSATSI", 1, MEC), "AST": ("BIOPRO_J", "LBXSASSI", 1, MEC),
    "GGT": ("BIOPRO_J", "LBXSGTSI", 1, MEC), "ALP": ("BIOPRO_J", "LBXSAPSI", 1, MEC),
    "TBIL": ("BIOPRO_J", "LBDSTBSI", 1, MEC), "TP": ("BIOPRO_J", "LBDSTPSI", 1, MEC),
    "ALB": ("BIOPRO_J", "LBDSALSI", 1, MEC), "UREA": ("BIOPRO_J", "LBDSBUSI", 1, MEC),
    "URIC": ("BIOPRO_J", "LBDSUASI", 1, MEC), "CREA": ("BIOPRO_J", "LBDSCRSI", 1, MEC),
    "K": ("BIOPRO_J", "LBXSKSI", 1, MEC), "NA": ("BIOPRO_J", "LBXSNASI", 1, MEC),
    "CL": ("BIOPRO_J", "LBXSCLSI", 1, MEC), "CA": ("BIOPRO_J", "LBDSCASI", 1, MEC),
    "FE": ("FETIB_J", "LBDIRNSI", 1, MEC), "TSAT": ("FETIB_J", "LBDPCT", 1, MEC),
    "FERR": ("FERTIN_J", "LBDFERSI", 1, MEC), "CRP": ("HSCRP_J", "LBXHSCRP", 1, MEC),
    "CHOL": ("TCHOL_J", "LBDTCSI", 1, MEC), "HDL": ("HDL_J", "LBDHDDSI", 1, MEC),
    "TG": ("TRIGLY_J", "LBDTRSI", 1, FAST), "LDL": ("TRIGLY_J", "LBDLDLSI", 1, FAST),
    "GLU": ("GLU_J", "LBDGLUSI", 1, FAST), "INS": ("INS_J", "LBDINSI", 1 / 6.0, FAST),
    "HBA1C": ("GHB_J", "LBXGH", 1, MEC), "VITD": ("VID_J", "LBXVIDMS", 1 / 2.496, MEC),
    "WBC": ("CBC_J", "LBXWBCSI", 1, MEC), "RBC": ("CBC_J", "LBXRBCSI", 1, MEC),
    "HGB": ("CBC_J", "LBXHGB", 10, MEC), "HCT": ("CBC_J", "LBXHCT", 1, MEC),
    "MCV": ("CBC_J", "LBXMCVSI", 1, MEC), "MCH": ("CBC_J", "LBXMCHSI", 1, MEC),
    "MCHC": ("CBC_J", "LBXMC", 10, MEC), "RDW": ("CBC_J", "LBXRDW", 1, MEC),
    "PLT": ("CBC_J", "LBXPLTSI", 1, MEC), "MPV": ("CBC_J", "LBXMPSI", 1, MEC),
    "NEUP": ("CBC_J", "LBXNEPCT", 1, MEC), "LYMP": ("CBC_J", "LBXLYPCT", 1, MEC),
    "MONOP": ("CBC_J", "LBXMOPCT", 1, MEC), "EOSP": ("CBC_J", "LBXEOPCT", 1, MEC),
    "BASOP": ("CBC_J", "LBXBAPCT", 1, MEC), "NEU": ("CBC_J", "LBDNENO", 1, MEC),
    "LYM": ("CBC_J", "LBDLYMNO", 1, MEC), "MONO": ("CBC_J", "LBDMONO", 1, MEC),
    "EOS": ("CBC_J", "LBDEONO", 1, MEC), "BASO": ("CBC_J", "LBDBANO", 1, MEC),
}


def _read(name: str) -> pd.DataFrame:
    return pd.read_sas(DIR / f"{name}.xpt", format="xport").set_index("SEQN")


def reference_population() -> pd.DataFrame:
    d = _read("DEMO_J")[["RIAGENDR", "RIDAGEYR", "RIDEXPRG", "WTMEC2YR"]]
    d = d.join(_read("BMX_J")[["BMXBMI"]]).join(_read("DIQ_J")[["DIQ010"]]).join(_read("GHB_J")[["LBXGH"]])
    d = d.join(_read("HSCRP_J")[["LBXHSCRP"]]).join(_read("MCQ_J")[["MCQ160L"]])
    keep = (
        d.RIDAGEYR.between(18, 79) & (d.RIDEXPRG != 1) & d.BMXBMI.between(18.5, 30)
        & (d.DIQ010 != 1) & ~(d.LBXGH >= 6.5) & ~(d.LBXHSCRP > 10) & (d.MCQ160L != 1)
    )
    ref = d[keep].copy()
    ref["sex"] = ref.RIAGENDR.map({1: "M", 2: "F"})
    return ref


def wquantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    order = np.argsort(x)
    x, w = x[order], w[order]
    cw = np.cumsum(w) - 0.5 * w
    return float(np.interp(q * w.sum(), cw, x))


# Analytes used by the latent-abnormal-values exclusion (LAVE, after Ichihara et al. / IFCC C-RIDL):
# an individual is dropped from the reference set for analyte X when >= 2 *other* LAVE analytes fall
# outside their provisional sex-specific limits. Fasting-only analytes are left out (subsample).
LAVE_SET = ["ALT", "AST", "GGT", "ALP", "TBIL", "ALB", "TP", "UREA", "CREA", "URIC", "CHOL", "HDL", "HGB", "MCV", "CRP"]


def _values(ref: pd.DataFrame, cache: dict, code: str) -> pd.DataFrame:
    f, col, factor, wcol = ANALYTES[code]
    if f not in cache:
        cache[f] = _read(f)
    src = cache[f]
    extra = [c for c in (col, wcol) if c in src and c not in ref]
    df = ref.join(src[extra], how="inner") if extra else ref.copy()
    weight = df[wcol] if wcol in df else df["WTMEC2YR"]
    df = df.assign(v=df[col] * factor, w=weight).dropna(subset=["v", "w"])
    return df[df.w > 0]


def lave_flags(ref: pd.DataFrame, cache: dict, iterations: int = 2) -> pd.DataFrame:
    """Per individual and LAVE analyte: True if outside the provisional limits of the current iteration."""
    keep = pd.Series(True, index=ref.index)
    flags = pd.DataFrame(False, index=ref.index, columns=LAVE_SET)
    for _ in range(iterations):
        flags[:] = False
        for code in LAVE_SET:
            df = _values(ref, cache, code)
            for sex in ("M", "F"):
                g = df[(df.sex == sex)]
                base = g[keep.reindex(g.index, fill_value=False)]
                lo, hi = wquantile(base.v.values, base.w.values, 0.025), wquantile(base.v.values, base.w.values, 0.975)
                out = g.index[(g.v < lo) | (g.v > hi)]
                flags.loc[out, code] = True
        keep = flags.sum(axis=1) < 2
    return flags


def derive(lave: bool = True) -> dict:
    ref = reference_population()
    cache: dict[str, pd.DataFrame] = {}
    flags = lave_flags(ref, cache) if lave else None
    out: dict[str, list[dict]] = {}
    for code in ANALYTES:
        df = _values(ref, cache, code)
        if flags is not None:
            others = flags.drop(columns=[code], errors="ignore").sum(axis=1)
            df = df[others.reindex(df.index, fill_value=0) < 2]
        parts = []
        for sex in ("M", "F"):
            for lo, hi in AGE_BANDS:
                g = df[(df.sex == sex) & df.RIDAGEYR.between(lo, hi)]
                if len(g) < MIN_N:
                    g = df[df.sex == sex]
                    if len(g) < MIN_N:
                        continue
                parts.append({"sex": sex, "age": [lo, hi], "n": int(len(g)),
                              "lower": round(wquantile(g.v.values, g.w.values, 0.025), 3),
                              "upper": round(wquantile(g.v.values, g.w.values, 0.975), 3)})
        out[code] = parts
    return {"source": "NHANES 2017–2018 (CDC), direct method, survey-weighted 2.5–97.5 percentiles"
                      + (", latent abnormal values exclusion (>= 2 other analytes out of provisional limits)" if lave else ""),
            "reference_population": "age 18–79, not pregnant, BMI 18.5–30, no diabetes, HbA1c < 6.5 %, hs-CRP <= 10 mg/L, no liver condition",
            "n_reference": int(len(ref)), "intervals": out}


# Analytes measured only in other NHANES cycles. LAVE is not applied (those cycles lack the full
# panel here); analyte-specific exclusions follow the C-RIDL protocol where it names one.
EXTRA = {
    "TSH": ("G", "THYROD_G", "LBXTSH1", 1, "WTSA2YR"),
    "FT4": ("G", "THYROD_G", "LBDT4FSI", 1, "WTSA2YR"),
    "PSA": ("F", "PSA_F", "LBXP1", 1, "WTMEC2YR"),
    "TESTO": ("I", "TST_I", "LBXTST", 0.0347, "WTMEC2YR"),
}


def _population(cycle: str) -> pd.DataFrame:
    d = _read(f"DEMO_{cycle}")[["RIAGENDR", "RIDAGEYR", "RIDEXPRG", "WTMEC2YR"]]
    d = d.join(_read(f"BMX_{cycle}")[["BMXBMI"]]).join(_read(f"DIQ_{cycle}")[["DIQ010"]])
    d = d.join(_read(f"GHB_{cycle}")[["LBXGH"]]).join(_read(f"MCQ_{cycle}")[["MCQ160L", "MCQ160M"]])
    keep = (d.RIDAGEYR.between(18, 79) & (d.RIDEXPRG != 1) & d.BMXBMI.between(18.5, 30)
            & (d.DIQ010 != 1) & ~(d.LBXGH >= 6.5) & (d.MCQ160L != 1))
    for crp_file, col, limit in (("CRP_F", "LBXCRP", 1.0), ("HSCRP_I", "LBXHSCRP", 10)):
        if crp_file.endswith(cycle) and (DIR / f"{crp_file}.xpt").exists():
            crp = _read(crp_file)[[col]]
            d = d.join(crp)
            keep &= ~(d[col] > limit)
    ref = d[keep].copy()
    ref["sex"] = ref.RIAGENDR.map({1: "M", 2: "F"})
    return ref


def derive_extra() -> dict:
    out = {}
    for code, (cycle, f, col, factor, wcol) in EXTRA.items():
        ref = _population(cycle)
        src = _read(f)
        df = ref.join(src[[c for c in {col, wcol, "LBXTPO", "LBXATG"} if c in src and c not in ref]], how="inner")
        if code in ("TSH", "FT4"):  # autoimmune thyroiditis and known thyroid disease excluded (C-RIDL)
            df = df[~(df.LBXTPO > 9) & ~(df.LBXATG > 4) & (df.MCQ160M != 1)]
        w = df[wcol] if wcol in df else df.WTMEC2YR
        df = df.assign(v=df[col] * factor, w=w).dropna(subset=["v", "w"])
        df = df[df.w > 0]
        parts = []
        for sex in ("M", "F"):
            for lo, hi in AGE_BANDS:
                g = df[(df.sex == sex) & df.RIDAGEYR.between(lo, hi)]
                if len(g) < MIN_N:
                    g = df[df.sex == sex]
                    if len(g) < MIN_N:
                        continue
                parts.append({"sex": sex, "age": [lo, hi], "n": int(len(g)), "cycle": cycle, "lave": False,
                              "lower": round(wquantile(g.v.values, g.w.values, 0.025), 3),
                              "upper": round(wquantile(g.v.values, g.w.values, 0.975), 3)})
        out[code] = parts
    return out


if __name__ == "__main__":
    raw = derive(lave=False)
    res = derive(lave=True)
    res["intervals"].update(derive_extra())
    res["without_lave"] = raw["intervals"]
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("reference individuals:", res["n_reference"])
    for code in ("ALT", "AST", "TSH", "FT4", "PSA", "TESTO"):
        print(code, [(p["sex"], p["age"], p["n"], p["lower"], p["upper"]) for p in res["intervals"].get(code, [])])
