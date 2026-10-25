"""BM25 over the textual fields of one patient's record.

Text of an indexed row = document name + specialty + section header + field group + field + value.
Ultrasound protocols use short all-caps rows ("ЖЕЛЧНЫЙ ПУЗЫРЬ") as section headers: they are not
indexed themselves but prefix the rows below them. Numeric lab rows are left to the lab tools.
Russian morphology is approximated by 5-character prefix stemming; queries are expanded with
`SYNONYMS` before scoring.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from ..knowledge.lexicon import SYNONYMS
from ..sources import BOILERPLATE, Patient, Record

_TOKEN = re.compile(r"[a-zа-я0-9]+", re.I)


def stem(tok: str) -> str:
    tok = tok.lower().replace("ё", "е")
    return tok[:5] if len(tok) > 5 else tok


def tokens(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN.findall(text) if len(t) > 1]


def expand_synonyms(query: str) -> str:
    q = query.lower()
    extra = [" ".join(g) for g in SYNONYMS if any(term in q for term in g)]
    return " ".join([query, *extra])


class BM25Index:
    def __init__(self, p: Patient):
        self.docs: list[Record] = []
        self.texts: list[str] = []
        header, header_visit = "", ""
        for r in p.records:
            if not r.value or BOILERPLATE.match(r.param):
                continue
            if r.value.isupper() and len(r.value) < 40:
                header, header_visit = r.value, r.visit
                continue
            if r.visit != header_visit:
                header = ""
            if r.specialty == "лаборатория" and r.num is not None:
                continue
            if r.num is None and len(r.value) <= 2:
                continue
            self.docs.append(r)
            self.texts.append(f"{r.source} {r.specialty} {header} {r.group} {r.param} {r.value}")
        self.tf = [Counter(tokens(t)) for t in self.texts]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = sum(self.len) / max(1, len(self.len))
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, k: int, specialty: str | None = None) -> list[tuple[float, Record]]:
        q = list(dict.fromkeys(tokens(expand_synonyms(query))))
        scored = []
        for r, c, L in zip(self.docs, self.tf, self.len):
            if specialty and specialty.lower() not in r.specialty.lower() and specialty.lower() not in r.source.lower():
                continue
            s = 0.0
            for t in q:
                f = c.get(t, 0)
                if f:  # BM25 with k1 = 1.2, b = 0.75
                    s += self.idf.get(t, 0) * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * L / self.avg))
            if s > 0:
                scored.append((s, r))
        scored.sort(key=lambda x: -x[0])
        return scored[:k]
