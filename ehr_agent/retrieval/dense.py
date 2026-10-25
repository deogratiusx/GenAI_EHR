"""Dense retrieval with a local multilingual sentence encoder (multilingual-e5-small, CPU).

Passages and queries use the e5 prefixes ("passage: ", "query: "). Embeddings are cached on disk per
patient, keyed by a hash of the indexed texts, so a record is encoded once (≈30 s for a 100-page PDF).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

from ..config import DATA_DIR, MODELS_DIR
from ..sources import Record

EMB_DIR = DATA_DIR / "emb_cache"
EMB_MODEL = str(MODELS_DIR / "multilingual-e5-small")


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
            self.emb = self.model().encode(
                [f"passage: {t[:1200]}" for t in texts], batch_size=64, normalize_embeddings=True, show_progress_bar=False
            )
            np.save(path, self.emb)

    @classmethod
    def model(cls):
        if cls._model is None:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from sentence_transformers import SentenceTransformer

            cls._model = SentenceTransformer(EMB_MODEL, device="cpu")
        return cls._model

    def search(self, query: str, k: int) -> list[tuple[float, Record]]:
        qv = self.model().encode([f"query: {query}"], normalize_embeddings=True, show_progress_bar=False)[0]
        sims = self.emb @ qv
        order = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.docs[i]) for i in order]
