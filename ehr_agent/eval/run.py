"""Experiment runner: models × systems × questions → scored episodes appended to a JSONL file (resumable).

  python -m ehr_agent.eval.run --models GigaChat-2 qwen3.5-4b --systems tools code record --split test --sample 1
  EHR_REF_POLICY=layered python -m ehr_agent.eval.run ...   # reference policy of tools and gold

An episode that hit a backend error (e.g. HTTP 429) is not saved; rerunning the command retries it.
Local models run sequentially; `--workers` parallelizes API models (GigaChat personal tier: 1).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..agent import SYSTEMS, run_episode
from ..agent.llm import make_backend
from ..config import RUNS_DIR, ref_policy
from ..sources import get_patient
from .benchmark import OUT as QUESTIONS
from .score import score_episode

SPLITS = ["dev", "test", "para", "summary", "pdf", "pdf_summary", "report", "trend", "probe", "probe3", "probe4", "final", "all"]


def sparse(qs: list[dict], n: int) -> list[dict]:
    """Deterministic stratified subsample: n questions per (split, patient, category); rephrasings follow their base."""

    def h(q):
        return hashlib.sha1(q["id"].encode()).hexdigest()

    keep: set[str] = set()
    groups: dict[tuple, list[dict]] = {}
    for q in qs:
        if q["split"] != "para":
            groups.setdefault((q["split"], q["patient"], q["category"]), []).append(q)
    for g in groups.values():
        keep |= {q["id"] for q in sorted(g, key=h)[:n]}
    return [q for q in qs if q["id"] in keep or (q["split"] == "para" and q["base_id"] in keep)]


def _key(model: str, system: str, budget: int, qid: str, rep: int) -> str:
    return f"{model}|{system}|{budget}|{qid}|{rep}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--systems", nargs="+", default=["tools"], choices=sorted(SYSTEMS))
    ap.add_argument("--budgets", nargs="+", type=int, default=[4])
    ap.add_argument("--split", default="test", choices=SPLITS)
    ap.add_argument("--categories", nargs="*")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--reps", type=int, default=1, help="repeated runs per question (use with --temperature > 0)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--sample", type=int, default=0, help="at most N questions per (patient, category)")
    ap.add_argument("--levels", nargs="*", type=int, help="rephrasing levels to keep (2, 3)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=str(RUNS_DIR / "episodes.jsonl"))
    a = ap.parse_args()

    qs = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    if a.sample:
        qs = sparse(qs, a.sample)
    qs = [q for q in qs if a.split == "all" or q["split"] == a.split]
    if a.levels:
        qs = [q for q in qs if q.get("level", 1) == 1 or q["level"] in a.levels]
    if a.categories:
        qs = [q for q in qs if q["category"] in a.categories]
    if a.ids:
        qs = [q for q in qs if q["id"] in a.ids]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                done.add(_key(e["model"], e["condition"], e["budget"], e["qid"], e.get("rep", 0)))
    lock = threading.Lock()

    for model in a.models:
        backend = make_backend(model, temperature=a.temperature)
        jobs = []
        for s in a.systems:
            budgets = [0] if SYSTEMS[s].toolset == "none" else a.budgets  # no tools → no call budget
            jobs += [
                (s, b, q, rep) for b in budgets for q in qs for rep in range(a.reps) if _key(backend.model, s, b, q["id"], rep) not in done
            ]
        print(f"{model}: {len(jobs)} episodes", flush=True)

        def work(job, backend=backend):
            system, budget, q, rep = job
            ep = run_episode(backend, get_patient(q["patient"]), q["question"], system, budget=budget or 4)
            ep.update(
                qid=q["id"],
                category=q["category"],
                split=q["split"],
                rep=rep,
                budget=budget,
                ref_policy=ref_policy(),
                temperature=a.temperature,
            )
            if any(s["type"] == "backend_error" for s in ep["trace"]):
                raise RuntimeError(f"backend error on {q['id']} ({system}); not saved, rerun to retry")
            ep["score"] = score_episode(q, ep)
            with lock, out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(ep, ensure_ascii=False) + "\n")
            return ep

        workers = 1 if "qwen" in model.lower() else a.workers
        with ThreadPoolExecutor(workers) as ex:
            futs = [ex.submit(work, j) for j in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    ep = fut.result()
                    print(
                        f"[{i}/{len(jobs)}] {ep['model']} {ep['condition']} b{ep['budget']} {ep['qid']} "
                        f"correct={ep['score']['correct']} calls={ep['n_calls']} {ep['wall']}s",
                        flush=True,
                    )
                except Exception as exc:
                    print(f"[{i}/{len(jobs)}] FAILED: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
