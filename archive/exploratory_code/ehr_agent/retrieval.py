"""Retrieval for tools v3: ontology expansion, hybrid lexical + dense search, document-aware results.

T2a  ontology expansion — drug classes (curated from the WHO ATC hierarchy, Russian INN and trade names),
     a lay → clinical lexicon, ICD-10 block names. Curated by hand for the PoC, not a licensed ontology.
T2b  dense retrieval — multilingual-e5-small, local; embeddings cached per patient on disk.
T2   fusion — reciprocal rank fusion (RRF, k=60) of BM25 and dense rankings.
T1   document-aware results — hits grouped by document, each with the document's conclusion rows and the
     neighbouring rows of the hit, so a hit in a description brings its conclusion along.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from .data import Patient, Record
from .tools import CONCLUSION_FIELDS, SYNONYMS_V2, NoteIndex, _source, _tokens

EMB_DIR = Path(__file__).resolve().parent.parent / "data" / "emb_cache"
EMB_MODEL = str(Path(__file__).resolve().parent.parent / "models" / "multilingual-e5-small")

# concept → terms. A query term from any group pulls in the whole group.
ONTOLOGY: dict[str, set[str]] = {
    # ATC B01AC platelet aggregation inhibitors
    "антиагреганты": {"антиагрегант", "антитромбоцитар", "дезагрегант", "антитромботическ", "ацетилсалицил", "аспирин",
                      "кардиомагнил", "тромбо асс", "клопидогрел", "плавикс", "тикагрелор", "брилинта", "прасугрел", "датт"},
    # ATC B01AA/AB/AE/AF anticoagulants
    "антикоагулянты": {"антикоагулянт", "антитромботическ", "гепарин", "нфг", "эноксапарин", "клексан", "фондапаринукс",
                       "ривароксабан", "ксарелто", "апиксабан", "эликвис", "дабигатран", "прадакса", "варфарин", "ноак"},
    # ATC C10 lipid modifying agents
    "гиполипидемические": {"гиполипидем", "статин", "аторвастатин", "розувастатин", "симвастатин", "питавастатин",
                           "эзетимиб", "алирокумаб", "эволокумаб", "инклисиран", "фенофибрат"},
    "бета-блокаторы": {"бета-блокатор", "бета блокатор", "метопролол", "бисопролол", "карведилол", "небиволол", "конкор"},
    "блокаторы рас": {"иапф", "ингибитор апф", "сартан", "эналаприл", "периндоприл", "нолипрел", "лизиноприл", "рамиприл",
                      "лозартан", "валсартан", "сакубитрил", "юперио"},
    "ипп": {"ипп", "ингибитор протонной помпы", "омепразол", "пантопразол", "эзомепразол", "рабепразол", "нексиум"},
    "диуретики": {"диуретик", "мочегонн", "торасемид", "фуросемид", "индапамид", "спиронолактон", "эплеренон", "верошпирон"},
    "глифлозины": {"глифлозин", "дапаглифлозин", "форсига", "эмпаглифлозин", "джардинс"},
    # lay → clinical
    "липиды": {"жир", "жиры", "холестерин", "липид", "липидограмма", "лпнп", "лпвп", "триглицерид", "дислипидем", "гиперлипидем"},
    "анемия": {"малокров", "анеми", "гемоглобин", "эритроцит", "железо", "ферритин"},
    "углеводы": {"сахар", "глюкоз", "гликированн", "hba1c", "диабет", "гликем"},
    "сердечная функция": {"насосн", "сократим", "фракция выброса", "фв", "эхокардиограф", "эхокг", "simpson", "систолическ"},
    "щитовидная": {"щитовидк", "щитовидн", "ттг", "тиреоид", "т4"},
    "давление": {"давлени", "ад ", "гипертензи", "гипертони", "артериальн"},
    "кишечник": {"кишечник", "кишк", "колоноскоп", "фкс", "толстой кишки", "стул"},
    "желудок": {"желуд", "гастроскоп", "эгдс", "фгдс", "гастрит", "пищевод", "кардии"},
    "колено": {"нога", "ноги", "колен", "мениск", "сустав", "травматолог", "ортопед"},
    "лёгкие": {"легк", "лёгк", "грудной клетки", "огк", "плевр", "бронх", "кт огк"},
    "печень": {"печен", "гепат", "стеатоз", "нажбп", "эластометр", "фиброскан", "алт", "аст"},
    "простата": {"простат", "предстательн", "пса", "труз"},
    "питание": {"питани", "диет", "еда", "есть", "стол №", "сладк", "мучн"},
    "лечение": {"лечени", "лечиться", "назначени", "назначил", "препарат", "терапи", "рекомендац"},
    "выписка": {"выписк", "выписн", "при выписке", "эпикриз", "рекомендации при выписке"},
    "вредные привычки": {"вредные привычки", "алкогол", "курен", "курит", "табак", "сигарет", "пачк"},
    "коронарные вмешательства": {"стент", "чкв", "ткба", "тбка", "ангиопласт", "коронарограф", "сосуд сердца", "сосудах сердца", "реваскуляриз"},
    "сердечные маркеры": {"тропонин", "кфк", "креатинфосфокиназ", "миокардиальн"},
    "инфаркт": {"инфаркт", "оим", "ими", "окс", "i21", "i22", "стенокард", "ибс"},
}


def expand_query(query: str) -> str:
    q = query.lower().replace("ё", "е")
    extra = [" ".join(g) for g in SYNONYMS_V2 if any(t in q for t in g)]
    for terms in ONTOLOGY.values():
        if any(t.replace("ё", "е") in q for t in terms):
            extra.append(" ".join(terms))
    return " ".join([query, *extra])


class DenseIndex:
    _model = None

    def __init__(self, pid: str, docs: list[Record], texts: list[str]):
        self.docs = docs
        key = hashlib.sha1(("\n".join(texts) + EMB_MODEL).encode()).hexdigest()[:16]
        EMB_DIR.mkdir(parents=True, exist_ok=True)
        path = EMB_DIR / f"{pid}_{key}.npy"
        if path.exists():
            self.emb = np.load(path)
        else:
            self.emb = self.model().encode([f"passage: {t[:1200]}" for t in texts], batch_size=64,
                                           normalize_embeddings=True, show_progress_bar=False)
            np.save(path, self.emb)

    @classmethod
    def model(cls):
        if cls._model is None:
            import os
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from sentence_transformers import SentenceTransformer
            cls._model = SentenceTransformer(EMB_MODEL, device="cpu")
        return cls._model

    def search(self, query: str, k: int) -> list[tuple[float, Record]]:
        qv = self.model().encode([f"query: {query}"], normalize_embeddings=True, show_progress_bar=False)[0]
        sims = self.emb @ qv
        order = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.docs[i]) for i in order]


class HybridIndex:
    """BM25 over the v2 text (source names + synonyms) with ontology expansion, fused with dense ranks by RRF."""

    def __init__(self, p: Patient, dense: bool = True, expand: bool = True):
        self.p, self.expand, self.use_dense = p, expand, dense
        self.bm25 = NoteIndex(p, with_source=True, synonyms=SYNONYMS_V2)
        self.dense = DenseIndex(p.pid, self.bm25.docs, self.bm25.texts) if dense else None

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
                dense = [(s, r) for s, r in dense
                         if specialty.lower() in r.specialty.lower() or specialty.lower() in _source(r).lower()]
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
        out.append({"_key": key, "document_id": h.visit, "date": h.date, "source": _source(h)[:70],
                    "specialty": h.specialty, "hits": [h.id], "rows": ctx})
    for d in out:
        d.pop("_key")
    return out


__all__ = ["HybridIndex", "document_context", "expand_query", "ONTOLOGY", "_tokens"]
