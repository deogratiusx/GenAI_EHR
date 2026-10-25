"""Hybrid search and document-aware results.

`HybridIndex.search` fuses BM25 (query expanded with the ontology) and dense rankings by reciprocal
rank fusion (score = Σ 1/(60 + rank)). `document_context` groups hits by document and attaches the
document's conclusion rows and the hit's neighbours, so a hit in a description brings its conclusion
along (the "right document, wrong row" failure of plain search).
"""

from __future__ import annotations

from ..config import dense_enabled
from ..knowledge.lexicon import expand_query
from ..sources import CONCLUSION_FIELDS, Patient, Record
from .bm25 import BM25Index
from .dense import DenseIndex


class HybridIndex:
    """BM25 (ontology-expanded query) fused with dense ranks by reciprocal rank fusion."""

    def __init__(self, p: Patient, dense: bool = True, expand: bool = True):
        self.p, self.expand, self.use_dense = p, expand, dense
        self.bm25 = BM25Index(p)
        # Without local encoder weights the index degrades to BM25 + ontology (see config.dense_enabled).
        self.dense = DenseIndex(p.pid, self.bm25.docs, self.bm25.texts) if dense and dense_enabled() else None

    def search(self, query: str, k: int, specialty: str | None = None) -> list[tuple[float, Record]]:
        depth = max(50, k * 5)
        q = expand_query(query) if self.expand else query
        lex = self.bm25.search(q, depth, specialty)
        ranks: dict[str, float] = {}
        by_id: dict[str, Record] = {}
        for rank, (_, r) in enumerate(lex):
            ranks[r.id] = ranks.get(r.id, 0) + 1 / (60 + rank)
            by_id[r.id] = r
        if self.dense is not None:
            dense = self.dense.search(query, depth)
            if specialty:
                dense = [(s, r) for s, r in dense if specialty.lower() in r.specialty.lower() or specialty.lower() in r.source.lower()]
            for rank, (_, r) in enumerate(dense):
                ranks[r.id] = ranks.get(r.id, 0) + 1 / (60 + rank)
                by_id[r.id] = r
        best = sorted(ranks.items(), key=lambda kv: -kv[1])[:k]
        return [(score, by_id[rid]) for rid, score in best]


def document_of(r: Record) -> tuple[str, str]:
    return (r.visit, r.protocol)


def document_context(p: Patient, hits: list[Record], max_docs: int = 5, per_doc_chars: int = 900) -> list[dict]:
    """Group hits by document; attach the document's conclusion rows and the hit's neighbours."""
    docs: dict[tuple, list[Record]] = {}
    for r in p.records:
        docs.setdefault(document_of(r), []).append(r)
    out, seen = [], []
    for h in hits:
        key = document_of(h)
        if key in seen:
            next(d for d in out if d["_key"] == key)["hits"].append(h.id)
            continue
        if len(seen) >= max_docs:
            continue
        seen.append(key)
        rows = docs[key]
        idx = next(i for i, r in enumerate(rows) if r.id == h.id)
        pick = {rows[j].id for j in range(max(0, idx - 1), min(len(rows), idx + 2))}
        pick |= {r.id for r in rows if CONCLUSION_FIELDS.match(r.param.strip()) and r.value not in {".", "-", ""}}
        ctx, used = [], 0
        for r in rows:
            if r.id not in pick or not r.value:
                continue
            v = r.value[:400]
            if used + len(v) > per_doc_chars and ctx:
                break
            used += len(v)
            ctx.append({"id": r.id, "param": r.param[:60], "value": v})
        out.append(
            {
                "_key": key,
                "document_id": h.visit,
                "date": h.date,
                "source": h.source[:70],
                "specialty": h.specialty,
                "hits": [h.id],
                "rows": ctx,
            }
        )
    for d in out:
        d.pop("_key")
    return out
