"""How many lab flags change when the interval source changes (indicative vs NHANES raw vs NHANES+LAVE)?

python -m ehr_agent.eval.ri_compare  →  runs/ri_comparison.md
"""

from __future__ import annotations

import json
from collections import Counter

from ..config import RUNS_DIR
from ..knowledge.analytes import flag, lab_values
from ..knowledge.reference import DECISION_LIMITS, NHANES_RI
from ..sources import get_patient

OUT = RUNS_DIR / "ri_comparison.md"


def _pop_flag(parts: list[dict], sex: str, age: int, v: float) -> str | None:
    c = [p for p in parts if p["sex"] == sex and p["age"][0] <= age <= p["age"][1]]
    if not c:
        return None
    return "L" if v < c[0]["lower"] else "H" if v > c[0]["upper"] else "N"


def main() -> None:
    ri = json.loads(NHANES_RI.read_text(encoding="utf-8"))
    rows, changes = [], Counter()
    for pid in ("P001", "P002", "P003", "P004", "P006"):  # P005 excluded: shifted export
        p = get_patient(pid)
        for r, a in lab_values(p):
            if a.code in DECISION_LIMITS:
                continue
            ind = flag(a, r.num, p.sex)
            raw = _pop_flag(ri["without_lave"].get(a.code, []), p.sex, p.age, r.num)
            lave = _pop_flag(ri["intervals"].get(a.code, []), p.sex, p.age, r.num)
            if raw is None:
                continue
            changes[("ind", ind, "lave", lave)] += 1
            if len({ind, raw, lave}) > 1:
                rows.append(f"| {pid} | {a.name} | {r.num:g} | {ind} | {raw} | {lave} |")
    total = sum(changes.values())
    agree = sum(n for (_, i, _, l), n in changes.items() if i == l)
    md = [
        "# Lab flags: indicative vs NHANES intervals\n",
        f"{total} lab values (P001–P004, P006; lipids/glucose/HbA1c/vitamin D excluded — judged by decision limits).",
        f"Indicative vs NHANES+LAVE agree on {agree}/{total} ({agree / total:.0%}).\n",
        "| Patient | Analyte | Value | Indicative | NHANES raw | NHANES+LAVE |",
        "|---|---|---|---|---|---|",
        *rows,
    ]
    OUT.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
