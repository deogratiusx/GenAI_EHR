"""Automatic scoring of an episode against its gold answer.

Correctness is category-specific (see questions.py). Grounding is shared:
  cite_valid   — share of cited ids that exist in the patient record
  cite_recall  — share of gold rows that were cited
  fabricated   — numbers in the answer that occur nowhere in the record or tool outputs
Summaries also get critical_recall: the share of the dossier's must-mention items the answer covers
(the same rule as the report harness check, computed for every system so they stay comparable).
"""

from __future__ import annotations

import json
import re
from functools import cache

from ..sources import get_patient
from ..util import norm_number as _norm_num
from ..util import numbers_in
from .gold import FAMILY

NOT_FOUND_RX = re.compile(
    r"нет\s+данн|не\s+выполн|не\s+найден|отсутству|не\s+проводил|не\s+обнаружен|не\s+определял|не\s+исследовал|нет\s+(результат|информац|сведени)|не\s+сдавал|не\s+указан",
    re.I,
)
FLAG_RX = re.compile(
    r"недостовер|сдвиг|смещ|ошибк\w*\s+(в\s+)?(выгруз|данн|запис)|некорректн|несоответств|не\s+соответству|противореч|сомнительн|сомнени|невозможн|нефизиолог|артефакт|перепута|неправдоподоб|не\s+доверя|нельзя\s+доверя|не\s+может\s+быть|некачествен|нарушен\w*\s+структур",
    re.I,
)
CONFLICT_RX = re.compile(r"расхожд|противореч|не\s+согласу|различ|отлича|разн\w*\s+(значени|данн|документ)|не\s+совпада|несоответств", re.I)
NO_CONFLICT_RX = re.compile(
    r"(нет|не\s+выявлен\w*|не\s+обнаружен\w*|не\s+найден\w*|отсутству\w*)\s+(каких-либо\s+|существенных\s+)?(расхожд|противореч|несоответств)|(?<!не\s)согласуются|(?<!не\s)совпадают|одинаков|идентичн|не\s+различа",
    re.I,
)


NORMAL_CLAUSE = re.compile(r"в\s+норм|норма\b|в\s+пределах|не\s+отклон|без\s+отклон|\(n\)|нормальн|не\s+выход", re.I)
ABNORMAL_CLAUSE = re.compile(r"выше|ниже|повыш|пониж|снижен|вне\s+(норм|пределов|референс)|\b[hl]\b|отклон\w*\s*:|за\s+пределы", re.I)


def _mentions(text: str) -> set[str]:
    """Analyte families the answer claims to be abnormal.

    A family mentioned only in a clause marked as normal ("ЛПВП 1.37 — в норме") is not a claim.
    Clauses are split on sentence/list punctuation; decimal points stay inside a clause.
    """
    t = text.lower()
    found: set[str] = set()
    for clause in re.split(r"[;\n,]|\.(?!\d)", t):
        normal = NORMAL_CLAUSE.search(clause) and not ABNORMAL_CLAUSE.search(clause)
        if normal:
            continue
        found |= {fam for fam, rx in FAMILY.items() if re.search(rx, clause)}
    return found


def correctness(q: dict, final: dict) -> tuple[float, dict]:
    g = q["gold"]
    ans = str(final.get("answer", ""))
    status = final.get("status", "answered")
    t = ans.lower()
    info: dict = {}
    typ = g["type"]
    if typ == "numeric":
        ok = _norm_num(str(g["value"])) in numbers_in(ans)
        return float(ok), info
    if typ == "set":
        m = _mentions(ans)
        pos, neg = set(g["positive"]), set(g["negative"])
        tp, fp = len(pos & m), len(neg & m)
        info.update(tp=tp, fp=fp, fn=len(pos - m))
        if not pos:
            return float(fp == 0), info
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / len(pos)
        return (2 * prec * rec / (prec + rec) if prec + rec else 0.0), info
    if typ == "not_found":
        flagged = status == "not_found" or bool(NOT_FOUND_RX.search(t))
        halluc = not flagged and bool(numbers_in(ans))
        info["hallucinated_value"] = halluc
        return float(flagged), info
    if typ == "codes":
        got = set(re.findall(r"\b([A-Z]\d{2}(?:\.\d{1,2})?)\b", ans))
        gold = set(g["codes"])
        tp = len(got & gold)
        prec = tp / len(got) if got else 0.0
        rec = tp / len(gold)
        info.update(tp=tp, extra=sorted(got - gold), missed=sorted(gold - got))
        return (2 * prec * rec / (prec + rec) if prec + rec else 0.0), info
    if typ == "conflict":
        says_no = bool(NO_CONFLICT_RX.search(t))
        says_yes = bool(CONFLICT_RX.search(t)) and not says_no
        if g["conflict"]:
            fields = [f for f in g["fields"] if f.lower()[:3] in t]
            return (0.5 + 0.5 * len(fields) / len(g["fields"])) if says_yes else 0.0, info
        return float(says_no and not says_yes), info
    if typ == "facts":
        hits = [bool(re.search(rx, t, re.I)) for rx in g["facts"]]
        info["facts"] = hits
        return sum(hits) / len(hits), info
    if typ == "checklist":
        per = {}
        hits_all = []
        for sec, rxs in g["sections"].items():
            hits = [bool(re.search(rx, t, re.I)) for rx in rxs]
            if hits:
                per[sec] = round(sum(hits) / len(hits), 3)
            hits_all += hits
        info["sections"] = per
        if g.get("trap"):
            info["trapped"] = bool(re.search(g["trap"], t, re.I)) and not FLAG_RX.search(t)
        return (sum(hits_all) / len(hits_all) if hits_all else 0.0), info
    if typ == "uncertain":
        flagged = status == "uncertain" or bool(FLAG_RX.search(t))
        trapped = bool(re.search(g["trap"], t, re.I)) and not flagged
        info["trapped"] = trapped
        return float(flagged), info
    raise ValueError(typ)


def grounding(q: dict, ep: dict) -> dict:
    p = get_patient(q["patient"])
    ids = {r.id for r in p.records}
    cites = [str(c).strip() for c in (ep["final"].get("citations") or [])]
    valid = [c for c in cites if c in ids]
    out = {"n_cites": len(cites), "cite_valid": (len(valid) / len(cites)) if cites else None}
    if q["gold_rows"]:
        out["cite_recall"] = len(set(q["gold_rows"]) & set(valid)) / len(q["gold_rows"])
    source = " ".join(r.value for r in p.records)
    for s in ep["trace"]:
        if "result" in s:
            source += " " + json.dumps(s["result"], ensure_ascii=False)
    src_nums = numbers_in(source)
    ans_nums = numbers_in(str(ep["final"].get("answer", "")))
    fab = sorted(n for n in ans_nums if n not in src_nums)
    out.update(n_numbers=len(ans_nums), fabricated=fab)
    return out


@cache
def _must(pid: str) -> list[dict]:
    from ..tools import EHRTools

    return EHRTools(get_patient(pid)).must_mention()


def critical_recall(pid: str, final: dict) -> tuple[float | None, list[str]]:
    from ..agent.harness import coverage

    must = _must(pid)
    if not must:
        return None, []
    covered, missing = coverage(final, must)
    return round(len(covered) / len(must), 3), [m["label"] for m in missing]


def score_episode(q: dict, ep: dict) -> dict:
    if q["gold"]["type"] == "must":  # draft report gold: critical recall is the score
        rec, missing = critical_recall(q["patient"], ep["final"])
        corr, info = (rec or 0.0), {"missing": missing}
    else:
        corr, info = correctness(q, ep["final"])
    if q["category"] == "summary":
        rec, missing = critical_recall(q["patient"], ep["final"])
        info.update(critical_recall=rec, missing=missing)
    return {"correct": round(corr, 3), "detail": info, **grounding(q, ep)}
