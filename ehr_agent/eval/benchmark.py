"""Benchmark builder: automatic questions from the data + hand-written gold (`gold.py`) → data/questions*.jsonl.

Automatic categories (gold computed from the record): lookup, abnormal (under the active reference
policy), not_found, icd, anthropo. Hand-written: freetext, integrate, quality, summaries, PDF facts,
probe sets. Each question also gets `support_rows`: every row that independently carries the gold fact,
so failure analysis does not over-count "retrieval misses" when a fact appears in several documents.

  python -m ehr_agent.eval.benchmark                     # indicative policy → data/questions.jsonl
  EHR_REF_POLICY=layered python -m ehr_agent.eval.benchmark   # → data/questions_layered.jsonl
"""

from __future__ import annotations

import json
import re

from ..config import DATA_DIR, ref_policy
from ..knowledge.analytes import BY_CODE, lab_values
from ..knowledge.dynamics import describe
from ..knowledge.reference import resolve_reference
from ..sources import get_patient
from ..tools import EHRTools
from .gold import (
    ANALYTE_LEVELS,
    FAMILY,
    FAMILY_LAYERED,
    FAMILY_OF,
    LOOKUP_ORDER,
    MANUAL,
    MANUAL_LEVELS,
    NOT_FOUND_ORDER,
    PANEL_LEVELS,
    PANELS,
    PDF_Q,
    PDF_SUMMARY,
    PROBE,
    PROBE_ENTITIES,
    PROBE_RETRIEVAL,
    SUMMARY,
    SUMMARY_Q,
)

OUT = DATA_DIR / ("questions.jsonl" if ref_policy() == "indicative" else f"questions_{ref_policy()}.jsonl")
SPLIT = {"P001": "dev", "P002": "dev", "P003": "test", "P004": "test", "P005": "test", "P006": "test"}


def _auto(pid: str) -> list[dict]:
    p = get_patient(pid)
    labs = lab_values(p)
    present = {a.code: (r, a) for r, a in labs}
    qs: list[dict] = []
    i = int(pid[-1])
    # lookup: rotate through the order so patients get different analytes
    order = LOOKUP_ORDER[i % 3 :] + LOOKUP_ORDER[: i % 3]
    for code in [c for c in order if c in present and BY_CODE[c].stable][:3]:
        r, a = present[code]
        qs.append(
            {
                "category": "lookup",
                "question": f"Какой уровень показателя «{a.name}» у пациента?",
                "gold": {"type": "numeric", "value": r.num, "family": FAMILY.get(code)},
                "gold_rows": [r.id],
            }
        )
    # abnormal panels
    for panel, codes in PANELS.items():
        fam_gold, fam_neutral, fam_normal, rows = set(), set(), set(), []
        for r, a in labs:
            if a.code not in codes:
                continue
            fam = FAMILY_OF.get(a.code, a.code)
            ref = resolve_reference(r, a, p.sex, p.age)
            f = ref["flag"]
            reliable = a.stable or ref["source"] != "ориентировочный справочник"
            if fam not in FAMILY or (fam in FAMILY_LAYERED and ref_policy() != "layered"):
                if f != "N":
                    fam_neutral.add(fam)
                continue
            if f != "N" and reliable:
                fam_gold.add(fam)
                rows.append(r.id)
            elif f != "N":
                fam_neutral.add(fam)
            else:
                fam_normal.add(fam)
        if not (fam_gold | fam_normal):
            continue
        qs.append(
            {
                "category": "abnormal",
                "question": f"Какие показатели {panel} у пациента выходят за пределы референсных значений? Перечислите только отклонившиеся показатели.",
                "gold": {
                    "type": "set",
                    "positive": sorted(fam_gold),
                    "negative": sorted(fam_normal - fam_gold),
                    "neutral": sorted(fam_neutral - fam_gold),
                    "ref_policy": ref_policy(),
                },
                "gold_rows": rows,
            }
        )
    # not found
    raw_text = " ".join(f"{r.group} {r.param}".lower() for r in p.records)
    nf = 0
    for code in NOT_FOUND_ORDER:
        a = BY_CODE[code]
        if code in present or (code == "PSA" and p.sex == "M" and "пса" in raw_text):
            continue
        if any(k in raw_text for k in (a.name.lower().split(" (")[0], *a.aliases) if len(k) > 3):
            continue
        qs.append(
            {
                "category": "not_found",
                "question": f"Какой уровень показателя «{a.name}» у пациента?",
                "gold": {"type": "not_found"},
                "gold_rows": [],
            }
        )
        nf += 1
        if nf == 2:
            break
    # ICD codes
    codes, rows = set(), []
    for r in p.records:
        found = re.findall(r"\{([A-Z]\d{2}(?:\.\d{1,2})?)\}", r.value)
        if found:
            codes.update(found)
            rows.append(r.id)
    if codes:
        qs.append(
            {
                "category": "icd",
                "question": "Перечислите все коды МКБ-10, выставленные пациенту специалистами, с указанием специалиста.",
                "gold": {"type": "codes", "codes": sorted(codes)},
                "gold_rows": rows,
            }
        )
    # anthropometry
    issues = EHRTools(p)._anthropometry_issues()
    rows = [v["id"] for i in issues for v in i["values"]]
    qs.append(
        {
            "category": "anthropo",
            "question": "Согласуются ли рост и вес пациента в разных документах карты? Если нет — приведите расхождения.",
            "gold": {"type": "conflict", "conflict": bool(issues), "fields": sorted(i["field"] for i in issues)},
            "gold_rows": rows,
        }
    )
    return qs


def _levels(q: dict) -> tuple[str, str] | None:
    if q["id"] in MANUAL_LEVELS:
        return MANUAL_LEVELS[q["id"]]
    m = re.search(r"«(.+?)»", q["question"])
    if q["category"] in ("lookup", "not_found") and m and m.group(1) in ANALYTE_LEVELS:
        return ANALYTE_LEVELS[m.group(1)]
    for key, v in PANEL_LEVELS.items():
        if q["category"] == "abnormal" and key in q["question"]:
            return v
    return None


def support_rows(q: dict) -> list[str]:
    """Gold rows plus every row whose own text carries all gold facts (or the gold number)."""
    rows = list(q["gold_rows"])
    g = q["gold"]
    p = get_patient(q["patient"])
    if g["type"] == "facts" and g["facts"]:
        for r in p.records:
            t = f"{r.param} {r.value}"
            if r.id not in rows and all(re.search(rx, t, re.I) for rx in g["facts"]):
                rows.append(r.id)
    elif g["type"] == "numeric":
        target = float(g["value"])
        fam = g.get("family")
        for r in p.records:
            if r.id in rows:
                continue
            if r.num is not None and abs(r.num - target) < 1e-9 and (not fam or re.search(fam, f"{r.group} {r.param}".lower())):
                rows.append(r.id)
    return rows


TREND_PRIORITY = ["TROP", "AST", "CK", "CREA", "K", "HGB", "ALT", "GLU", "CRP", "WBC"]
REPORT_Q = (
    "Составьте итоговый отчёт по госпитализации: ключевые выводы, отклонения анализов и их динамика, исследования, "
    "диагнозы, терапия, противоречия в документации, рекомендации."
)


def _num_rx(v: float) -> str:
    """A number as the model may print it: 6.31 / 6,31 / 6.310; 4200 / 4 200 / 4200.0."""
    s = f"{v:g}"
    if "." in s:
        return r"(?<![\d.,])" + s.replace(".", r"[.,]") + r"0*(?!\d)"
    if len(s) > 3:
        s = s[:-3] + r"\s?" + s[-3:]
    return r"(?<![\d.,])" + s + r"(?:[.,]0+)?(?![\d]|[.,]\d)"  # 9061, 9 061, 9061.0


def _trend_questions(pid: str, n: int = 2) -> list[dict]:
    """Dynamics questions from the record: analytes measured ≥ 3 times with a significant course.
    Gold: the peak and last values, the direction of the course and its relation to the norm."""
    series = EHRTools(get_patient(pid))._series()
    out = []
    for code in TREND_PRIORITY:
        if code not in series or len(series[code][1]) < 3:
            continue
        a, pts = series[code]
        d = describe(code, pts)
        if d["direction"] == "без значимых изменений" or all(p["flag"] == "N" for p in pts):
            continue  # a question about dynamics needs a real course and a value outside the norm
        direction = {
            "рост": r"\bрост|повыс|увелич|вырос|нараст",
            "снижение": r"сниж|снизил|уменьш",
            "рост, затем снижение": r"сниж|снизил|уменьш|пик|максим",
            "снижение, затем рост": r"\bрост|повыс|увелич|минимум",
        }
        course = (
            r"нормализ|в\s+пределах\s+норм|вернул\w*\s+к\s+норм|норм\w*\s+значени"
            if d["course"].startswith("нормализ")
            else r"выше\s+норм|ниже\s+норм|повыш|пониж|вне\s+норм|не\s+нормализ|сохраня|остава|остаёт|превыш"
        )
        out.append(
            {
                "category": "dynamics",
                "question": f"Как менялся показатель «{a.name.split(' (')[0]}» за время госпитализации: пик, последнее значение, "
                f"нормализовался ли он?",
                "gold": {
                    "type": "facts",
                    "facts": [_num_rx(d["peak"]["value"]), _num_rx(d["last"]["value"]), direction[d["direction"]], course],
                    "trend": {k: d[k] for k in ("direction", "course", "change_pct", "rcv_pct")},
                },
                "gold_rows": list(dict.fromkeys([d["first"]["id"], d["peak"]["id"], d["last"]["id"]])),
            }
        )
        if len(out) == n:
            break
    return out


def build() -> list[dict]:
    out = []
    for pid in sorted(SPLIT):
        qs = (_auto(pid) if pid != "P005" else []) + MANUAL.get(pid, [])
        for k, q in enumerate(qs, start=1):
            out.append({"id": f"{pid}-{k:02d}", "patient": pid, "split": SPLIT[pid], **q})
        for k, q in enumerate(PROBE.get(pid, []), start=1):
            out.append({"id": f"{pid}-P{k}", "patient": pid, "split": "probe", **q})
    base = [q for q in out if q["split"] == "test"]
    for q in base:
        lv = _levels(q)
        if lv:
            for level, text in zip((2, 3), lv):
                out.append({**q, "id": f"{q['id']}-L{level}", "split": "para", "level": level, "base_id": q["id"], "question": text})
    for pid, gold in SUMMARY.items():
        out.append(
            {
                "id": f"{pid}-SUM",
                "patient": pid,
                "split": "summary",
                "category": "summary",
                "question": SUMMARY_Q,
                "gold": {"type": "checklist", "sections": {k: v for k, v in gold.items() if k != "trap"}, "trap": gold.get("trap")},
                "gold_rows": [],
            }
        )
    for pid, qs in PDF_Q.items():
        for k, q in enumerate(qs, start=1):
            out.append({"id": f"{pid}-{k:02d}", "patient": pid, "split": "pdf", **q})
    for pid, qs in PROBE_RETRIEVAL.items():
        for k, q in enumerate(qs, start=1):
            out.append({"id": f"{pid}-V3{k}", "patient": pid, "split": "probe3", **q})
    for pid, qs in PROBE_ENTITIES.items():
        for k, q in enumerate(qs, start=1):
            out.append({"id": f"{pid}-V4{k}", "patient": pid, "split": "probe4", **q})
    for pid, gold in PDF_SUMMARY.items():
        out.append(
            {
                "id": f"{pid}-SUM",
                "patient": pid,
                "split": "pdf_summary",
                "category": "summary",
                "question": "Составьте краткую сводку госпитализации для врача: диагнозы, ключевые исследования и вмешательства, "
                "отклонения анализов, противоречия в документации, назначения при выписке.",
                "gold": {"type": "checklist", "sections": gold, "trap": None},
                "gold_rows": [],
            }
        )
    for pid in sorted(PDF_Q):
        for k, q in enumerate(_trend_questions(pid), start=1):
            out.append({"id": f"{pid}-T{k}", "patient": pid, "split": "trend", **q})
        if pid not in PDF_SUMMARY:
            # Draft gold: the dossier's must-mention items (critical recall only), to be reviewed by a clinician.
            out.append(
                {
                    "id": f"{pid}-REP",
                    "patient": pid,
                    "split": "report",
                    "category": "summary",
                    "question": REPORT_Q,
                    "gold": {"type": "must", "draft": True},
                    "gold_rows": [],
                }
            )
    return out


if __name__ == "__main__":
    qs = build()
    for q in qs:
        q["support_rows"] = support_rows(q)
    OUT.write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in qs) + "\n", encoding="utf-8")
    from collections import Counter

    print(len(qs), "questions", Counter((q["split"], q["category"]) for q in qs))
