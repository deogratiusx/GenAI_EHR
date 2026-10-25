"""Run the experiment grid and append scored episodes to runs/episodes.jsonl (resumable).

  python -m ehr_agent.run --models GigaChat-2-Max --conditions domain generic fullctx --split test
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .agent import run_episode
from .data import get_patient
from .llm import make_backend
from .questions import OUT as QUESTIONS
from .refs import ref_policy
from .score import score_episode

RUNS = Path(__file__).resolve().parent.parent / "runs"


def _key(model: str, cond: str, budget: int, qid: str, rep: int) -> str:
    return f"{model}|{cond}|{budget}|{qid}|{rep}"


def sparse(qs: list[dict], n: int) -> list[dict]:
    """Deterministic stratified subsample: the first n questions per (patient, category) by a hash of the id."""
    import hashlib

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--conditions", nargs="+", default=["domain", "generic", "fullctx"])
    ap.add_argument("--budgets", nargs="+", type=int, default=[4])
    ap.add_argument("--split", default="test", choices=["dev", "test", "probe", "para", "summary", "pdf", "pdf_summary", "probe3", "probe4", "all"])
    ap.add_argument("--categories", nargs="*")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--sample", type=int, default=0,
                    help="sparse run: at most N questions per (patient, category); paraphrases follow their base question")
    ap.add_argument("--levels", nargs="*", type=int, help="paraphrase levels to keep (2, 3)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=str(RUNS / "episodes.jsonl"))
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
        backend = make_backend(model)
        jobs = []
        for cond in a.conditions:
            for budget in (a.budgets if cond != "fullctx" else [0]):  # hybrid uses the budget
                for q in qs:
                    for rep in range(a.reps):
                        if _key(backend.model, cond, budget, q["id"], rep) not in done:
                            jobs.append((cond, budget, q, rep))
        print(f"{model}: {len(jobs)} episodes", flush=True)

        def work(job):
            cond, budget, q, rep = job
            ep = run_episode(backend, get_patient(q["patient"]), q["question"], cond, budget=budget or 4)
            ep.update(qid=q["id"], category=q["category"], split=q["split"], rep=rep, budget=budget, ref_policy=ref_policy())
            if any(s["type"] == "backend_error" for s in ep["trace"]):
                raise RuntimeError(f"backend error on {q['id']} ({cond}); not saved, rerun to retry")
            ep["score"] = score_episode(q, ep)
            with lock:
                with out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(ep, ensure_ascii=False) + "\n")
            return ep

        workers = 1 if "qwen" in model.lower() else a.workers
        with ThreadPoolExecutor(workers) as ex:
            futs = [ex.submit(work, j) for j in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    ep = fut.result()
                    print(f"[{i}/{len(jobs)}] {ep['model']} {ep['condition']} b{ep['budget']} {ep['qid']} "
                          f"correct={ep['score']['correct']} calls={ep['n_calls']} {ep['wall']}s", flush=True)
                except Exception as exc:  # keep the grid running; failures are visible in the log
                    print(f"[{i}/{len(jobs)}] FAILED: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
