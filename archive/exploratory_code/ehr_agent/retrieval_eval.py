"""Tool-level retrieval benchmark, no LLM: does search_notes surface the gold evidence for the question?

Query = the question text (what a model would naturally pass). Variants isolate the v3 components:
  v1           BM25, field text only
  v2           BM25 + source names + modality synonyms
  v2+onto      v2 + ontology/lay expansion (T2a)
  dense        multilingual-e5-small only (T2b)
  hybrid       v2+onto fused with dense by RRF (T2)
Metrics at k (k = rows returned):
  Row@k  a gold row is among the hits
  Ctx@k  a gold row is among the rows the v3 tool returns (hits + document conclusion + neighbours, T1)
  Doc@k  a hit lies in a gold row's document
Questions whose gold rows are only numeric lab values are excluded (get_lab serves them).

  python -m ehr_agent.retrieval_eval  →  runs/retrieval_eval.md
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .data import get_patient
from .questions import OUT as QUESTIONS
from .retrieval import HybridIndex, document_context, document_of
from .tools import SYNONYMS, SYNONYMS_V2, NoteIndex

OUT = Path(__file__).resolve().parent.parent / "runs" / "retrieval_eval.md"
KS = (3, 6, 12)
VARIANTS = ["v1", "v2", "v2+onto", "dense", "hybrid"]
SPLITS = ("test", "probe", "pdf", "probe3")


def _index(variant: str, p):
    if variant == "v1":
        return NoteIndex(p, with_source=False, synonyms=SYNONYMS)
    if variant == "v2":
        return NoteIndex(p, with_source=True, synonyms=SYNONYMS_V2)
    if variant == "v2+onto":
        return HybridIndex(p, dense=False, expand=True)
    if variant == "dense":
        idx = HybridIndex(p, dense=True, expand=False)
        idx.bm25_only_off = True
        return idx
    return HybridIndex(p, dense=True, expand=True)


def _search(variant: str, idx, q: str, k: int):
    if variant == "dense":
        return [r for _, r in idx.dense.search(q, k)]
    return [r for _, r in idx.search(q, k)]


def evaluate() -> dict:
    qs = [json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]
    qs = [q for q in qs if q["gold_rows"] and q["category"] in ("freetext", "integrate", "lookup") and q["split"] in SPLITS]
    res: dict = defaultdict(lambda: {"row": [], "ctx": [], "doc": []})
    cache: dict = {}
    for q in qs:
        src = "PDF" if q["patient"].startswith("K") else "таблица"
        p = get_patient(q["patient"])
        by_id = p.by_id()
        gold = [g for g in q["gold_rows"] if g in by_id]
        if all(by_id[g].specialty == "лаборатория" and by_id[g].num is not None for g in gold):
            continue
        gold_docs = {document_of(by_id[g]) for g in gold}
        for v in VARIANTS:
            if (q["patient"], v) not in cache:
                cache[(q["patient"], v)] = _index(v, p)
            hits = _search(v, cache[(q["patient"], v)], q["question"], max(KS))
            for k in KS:
                top = hits[:k]
                ctx_ids = {row["id"] for d in document_context(p, top, max_docs=k) for row in d["rows"]} | {r.id for r in top}
                d = res[(src, v, k)]
                d["row"].append(bool(set(gold) & {r.id for r in top}))
                d["ctx"].append(bool(set(gold) & ctx_ids))
                d["doc"].append(bool(gold_docs & {document_of(r) for r in top}))
    return res


def main() -> None:
    res = evaluate()
    lines = ["# Retrieval benchmark (search_notes; query = question text)", "",
             "Row = gold row among hits; Ctx = gold row among hits + document conclusion + neighbours (what the v3 tool returns); "
             "Doc = a hit in a gold row's document. Splits: test, probe, probe3 (written before v3), pdf. "
             "Questions answerable only from numeric lab rows are excluded.", "",
             "| Source | Variant | n | " + " | ".join(f"Row@{k}" for k in KS) + " | " + " | ".join(f"Ctx@{k}" for k in KS)
             + " | " + " | ".join(f"Doc@{k}" for k in KS) + " |",
             "|---|---|---|" + "---|" * (3 * len(KS))]
    for src in ("таблица", "PDF"):
        for v in VARIANTS:
            n = len(res[(src, v, KS[0])]["row"])
            if not n:
                continue
            cell = lambda m: " | ".join(f"{sum(res[(src, v, k)][m]) / n:.2f}" for k in KS)  # noqa: E731
            lines.append(f"| {src} | {v} | {n} | {cell('row')} | {cell('ctx')} | {cell('doc')} |")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
