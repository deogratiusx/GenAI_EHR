"""Collect what the hosted demo needs into demo_data/ (committed, shipped in the demo image).

Only the anonymized check-up records (source 1) leave this machine: the inpatient PDFs carry personal data
and stay local. From the final run the demo gets
  * runs/demo_runs.jsonl   — full episodes on check-up patients (for the trace view);
  * runs/runs_summary.json — counts only, over every final-run episode including the PDFs (for the dashboard).

  python scripts/build_demo_data.py
"""

from __future__ import annotations

import json
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ehr_agent.config import DATA_DIR, RUNS_DIR  # noqa: E402
from ehr_agent.eval.analyze import load  # noqa: E402

OUT = ROOT / "demo_data"


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "data" / "raw").mkdir(parents=True)
    (OUT / "runs").mkdir()
    for f in sorted((DATA_DIR / "raw").glob("P00*.xlsx")):
        shutil.copy2(f, OUT / "data" / "raw" / f.name)
    for name in ("assumed_ages.json", "nhanes_ri.json"):
        shutil.copy2(DATA_DIR / name, OUT / "data" / name)
    qs = [json.loads(line) for line in (DATA_DIR / "questions_layered.jsonl").read_text(encoding="utf-8").splitlines()]
    with (OUT / "data" / "questions_layered.jsonl").open("w", encoding="utf-8") as f:
        for q in qs:
            if q["patient"].startswith("P"):
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

    eps = [e for e in load([str(p) for p in sorted(RUNS_DIR.glob("final_*.jsonl"))]) if e.get("temperature", 0) == 0]
    with (OUT / "runs" / "demo_runs.jsonl").open("w", encoding="utf-8") as f:
        for e in eps:
            if e["patient"].startswith("P"):
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    cells: dict[tuple, list[dict]] = defaultdict(list)
    for e in eps:
        src = "summary" if e["category"] == "summary" else ("pdf" if e["patient"].startswith("K") else "table")
        cells[(e["model"], e["condition"], src)].append(e)
    rows = []
    for (model, system, src), g in sorted(cells.items()):
        tools, harness = Counter(), Counter()
        first = Counter()
        for e in g:
            calls = [s["name"] for s in e["trace"] if s["type"] == "call" and s["name"] != "final_answer"]
            tools.update(calls)
            if calls:
                first[calls[0]] += 1
            harness.update(s.get("action") for s in e["trace"] if s["type"] == "harness")
        scored = [e["score"]["correct"] for e in g if e["score"]["correct"] is not None]
        rows.append({
            "model": model, "system": system, "source": src, "episodes": len(g),
            "accuracy": round(statistics.mean(scored), 3) if scored else None,
            "not_applicable": len(g) - len(scored),
            "tool_calls": dict(tools), "first_tool": dict(first), "harness": dict(harness),
            "calls_per_episode": round(statistics.mean(e["n_calls"] for e in g), 2),
            "invalid_per_episode": round(statistics.mean(e["n_invalid"] for e in g), 3),
            "median_seconds": round(statistics.median(e["wall"] for e in g), 1),
            "median_prompt_tokens": int(statistics.median(e["usage"].get("prompt_tokens", 0) for e in g)),
        })  # fmt: skip
    (OUT / "runs" / "runs_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    n_demo = sum(1 for _ in (OUT / "runs" / "demo_runs.jsonl").open())
    print(f"demo_data: {len(list((OUT / 'data' / 'raw').glob('*.xlsx')))} check-ups, {n_demo} check-up episodes, {len(rows)} summary cells")


if __name__ == "__main__":
    main()
