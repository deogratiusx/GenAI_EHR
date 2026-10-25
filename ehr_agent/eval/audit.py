"""Audit of the automatic scorer: does its verdict match a reviewer's reading of the answer?

1. `sample` draws a stratified sample of final-run episodes (model × source × automatic verdict: wrong /
   partial / correct) and writes, for each, the question, the gold in words, the answer and the
   automatic score → data/audit_sample.jsonl (local: answers quote patient records).
2. A reviewer fills data/audit_judgments.json: {episode key: {"score": 0 | 0.5 | 1, "note": "…"}}.
3. `report` compares the two: exact agreement, agreement on correct-vs-not (score ≥ 0.5), Cohen's κ,
   and lists the disagreements by direction (scorer too strict / too lenient).

  python -m ehr_agent.eval.audit sample [n]
  python -m ehr_agent.eval.audit report
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict

from ..config import DATA_DIR, RUNS_DIR
from .analyze import load
from .stats import source

SAMPLE = DATA_DIR / "audit_sample.jsonl"
JUDGED = DATA_DIR / "audit_judgments.json"


def key(e: dict) -> str:
    return f"{e['model']}|{e['condition']}|{e['qid']}|{e.get('rep', 0)}"


def bucket(x: float) -> str:
    return "wrong" if x < 0.34 else "correct" if x > 0.99 else "partial"


def gold_words(q_gold: dict) -> str:
    g = {k: v for k, v in q_gold.items() if k not in ("family",)}
    return re.sub(r"\\[bswd]", "", json.dumps(g, ensure_ascii=False))[:600]


def sample(n: int = 48, seed: int = 7) -> None:
    eps = [
        e
        for e in load([str(p) for p in sorted(RUNS_DIR.glob("final_*.jsonl"))])
        if e["score"]["correct"] is not None and e.get("temperature", 0) == 0
    ]
    qs = {
        json.loads(line)["id"]: json.loads(line) for line in (DATA_DIR / "questions_layered.jsonl").read_text(encoding="utf-8").splitlines()
    }
    strata = defaultdict(list)
    for e in eps:
        strata[(e["model"], source(e), bucket(e["score"]["correct"]))].append(e)
    rnd = random.Random(seed)
    per = max(1, n // len(strata))
    picked = [e for s in sorted(strata) for e in rnd.sample(strata[s], min(per, len(strata[s])))]
    with SAMPLE.open("w", encoding="utf-8") as f:
        for e in picked[:n]:
            q = qs[e["qid"]]
            f.write(
                json.dumps(
                    {
                        "key": key(e),
                        "question": q["question"],
                        "category": q["category"],
                        "gold": gold_words(q["gold"]),
                        "answer": str(e["final"].get("answer", ""))[:2500],
                        "status": e["final"].get("status"),
                        "auto": e["score"]["correct"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"{min(n, len(picked))} episodes from {len(strata)} strata → {SAMPLE}")


def kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def report() -> None:
    items = [json.loads(line) for line in SAMPLE.read_text(encoding="utf-8").splitlines()]
    judged = json.loads(JUDGED.read_text(encoding="utf-8"))
    pairs = [(it, judged[it["key"]]) for it in items if it["key"] in judged]
    auto = [it["auto"] for it, _ in pairs]
    human = [j["score"] for _, j in pairs]
    exact = sum(abs(a - h) < 0.26 for a, h in zip(auto, human)) / len(pairs)
    ab, hb = [int(a >= 0.5) for a in auto], [int(h >= 0.5) for h in human]
    print(
        f"episodes: {len(pairs)}; agreement (±0.25): {exact:.2f}; correct-vs-not: "
        f"{sum(x == y for x, y in zip(ab, hb)) / len(pairs):.2f}; Cohen's κ: {kappa(ab, hb):.2f}"
    )
    print(
        f"scorer too strict: {sum(a < h - 0.26 for a, h in zip(auto, human))}; too lenient: {sum(a > h + 0.26 for a, h in zip(auto, human))}"
    )
    for it, j in pairs:
        if abs(it["auto"] - j["score"]) > 0.26:
            print(f"  {it['key']}: auto {it['auto']} vs review {j['score']} — {j.get('note', '')}")


if __name__ == "__main__":
    {"sample": lambda: sample(int(sys.argv[2]) if len(sys.argv) > 2 else 48), "report": report}[sys.argv[1]]()
