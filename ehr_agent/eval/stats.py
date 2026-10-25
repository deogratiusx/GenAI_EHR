"""Final-run statistics: descriptive table and the pre-registered paired comparisons (docs/FINAL_RUN_PLAN.md).

Every comparison is paired over the same questions (temperature-0 run, rep 0). Test: two-sided sign-flip
permutation test of the mean difference; 95 % paired bootstrap CI; Holm correction over the whole
family H1–H4. Episodes with context_overflow are N/A and drop out of their pair.

  python -m ehr_agent.eval.stats            →  runs/final_stats.md
"""

from __future__ import annotations

import random
from collections import defaultdict

from ..config import RUNS_DIR
from .analyze import load, mean
from .labels import COND_RU, MODEL_ORDER, SHORT

OUT = RUNS_DIR / "final_stats.md"
LITE, MAX, QWEN, MED = MODEL_ORDER  # order in labels.py: Lite, Max, Qwen, Med-Qwen


def source(e: dict) -> str:
    if e["category"] == "summary":
        return "summary"
    return "pdf" if e["patient"].startswith("K") else "table"


def metric(e: dict, name: str):
    return e["score"]["correct"] if name == "correct" else e["score"]["detail"].get(name)


def scores(eps, model, system, src=None, name="correct", splits=None) -> dict[str, float]:
    return {
        e["qid"]: metric(e, name)
        for e in eps
        if e["model"] == model
        and e["condition"] == system
        and e.get("rep", 0) == 0
        and e.get("temperature", 0) == 0
        and (src is None or source(e) == src)
        and (splits is None or e["split"] in splits)
        and metric(e, name) is not None
    }


def paired(a: dict, b: dict, n_perm: int = 10_000, n_boot: int = 4_000, seed: int = 0) -> dict:
    keys = sorted(a.keys() & b.keys())
    d = [a[k] - b[k] for k in keys]
    if not d:
        return {"n": 0}
    rnd = random.Random(seed)
    obs = abs(sum(d))
    hits = sum(abs(sum(x if rnd.random() < 0.5 else -x for x in d)) >= obs - 1e-12 for _ in range(n_perm))
    boots = sorted(mean([rnd.choice(d) for _ in d]) for _ in range(n_boot))
    return {
        "n": len(d),
        "a": mean([a[k] for k in keys]),
        "b": mean([b[k] for k in keys]),
        "diff": mean(d),
        "ci": (boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]),
        "p": (hits + 1) / (n_perm + 1),
    }


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def comparisons(eps) -> list[tuple[str, str, dict]]:
    """(hypothesis, label, result) for the pre-registered family."""
    out = []
    summ = ("summary", "pdf_summary")  # clinician-style checklists; "report" has draft gold only
    for m in (MAX, LITE, QWEN):
        s = SHORT[m]
        out.append(("H1a", f"{s}: tools − code", paired(scores(eps, m, "tools"), scores(eps, m, "code"))))
        out.append(("H1b", f"{s}: tools − record (таблицы)", paired(scores(eps, m, "tools", "table"), scores(eps, m, "record", "table"))))
        out.append(
            ("H1b", f"{s}: tools − record_chunked (PDF)", paired(scores(eps, m, "tools", "pdf"), scores(eps, m, "record_chunked", "pdf")))
        )
    for m in (MAX, LITE):
        out.append(("H1c", f"{SHORT[m]}: tools − tools_plain", paired(scores(eps, m, "tools"), scores(eps, m, "tools_plain"))))
    for m in (MAX, LITE, QWEN, MED):
        s = SHORT[m]
        out.append(
            ("H2a", f"{s}: report − tools, чек-лист", paired(scores(eps, m, "report", splits=summ), scores(eps, m, "tools", splits=summ)))
        )
        out.append(
            (
                "H2a",
                f"{s}: report − tools, critical recall",
                paired(scores(eps, m, "report", "summary", "critical_recall"), scores(eps, m, "tools", "summary", "critical_recall")),
            )
        )
    for m in (MAX, LITE):
        out.append(
            (
                "H2b",
                f"{SHORT[m]}: report − report_nocheck, чек-лист",
                paired(scores(eps, m, "report", splits=summ), scores(eps, m, "report_nocheck", splits=summ)),
            )
        )
    point = {"table", "pdf"}

    def tools_point(m):
        return {k: v for src in point for k, v in scores(eps, m, "tools", src).items()}

    out.append(("H3", "Med-Qwen − Qwen (tools)", paired(tools_point(MED), tools_point(QWEN))))
    out.append(("H4", "Qwen − GigaChat Max (tools)", paired(tools_point(QWEN), tools_point(MAX))))
    out.append(("H4", "GigaChat Lite − GigaChat Max (tools)", paired(tools_point(LITE), tools_point(MAX))))
    return [c for c in out if c[2].get("n")]


def descriptive(eps) -> str:
    cells = defaultdict(list)
    for e in eps:
        if e.get("rep", 0) == 0 and e.get("temperature", 0) == 0:
            cells[(e["model"], e["condition"], source(e))].append(e["score"]["correct"])
    rows = ["| Модель | Система | Таблицы | PDF | Сводки |", "|---|---|---|---|---|"]
    for m in MODEL_ORDER:
        for c in sorted({k[1] for k in cells if k[0] == m}, key=lambda x: list(COND_RU).index(x) if x in COND_RU else 99):

            def cell(src, m=m, c=c):
                xs = cells.get((m, c, src), [])
                vals = [x for x in xs if x is not None]
                na = len(xs) - len(vals)
                return f"{mean(vals):.2f} ({len(vals)}{f', N/A {na}' if na else ''})" if xs else "—"

            rows.append(f"| {SHORT[m]} | {COND_RU.get(c, c)} | {cell('table')} | {cell('pdf')} | {cell('summary')} |")
    return "\n".join(rows)


def variance(eps) -> str:
    rows = ["| Модель | Прогонов | Средняя верность по прогонам | Доля вопросов с разными оценками |", "|---|---|---|---|"]
    for m in (MAX, LITE):
        by_rep, by_q = defaultdict(list), defaultdict(set)
        for e in eps:
            if e["model"] == m and e.get("temperature", 0) > 0 and e["score"]["correct"] is not None:
                by_rep[e.get("rep", 0)].append(e["score"]["correct"])
                by_q[e["qid"]].add(e["score"]["correct"])
        if by_rep:
            rows.append(
                f"| {SHORT[m]} | {len(by_rep)} | "
                + ", ".join(f"{mean(v):.2f}" for _, v in sorted(by_rep.items()))
                + f" | {mean([float(len(v) > 1) for v in by_q.values()]):.2f} |"
            )
    return "\n".join(rows)


def main() -> None:
    eps = load([str(p) for p in sorted(RUNS_DIR.glob("final_*.jsonl"))])
    comps = comparisons(eps)
    adj = holm([c[2]["p"] for c in comps])
    lines = [
        "# Финальный прогон: статистика",
        "",
        f"Эпизодов: {len(eps)}.",
        "",
        "## Верность по системам (среднее, n)",
        "",
        descriptive(eps),
        "",
        "## Предрегистрированные парные сравнения (Холм по всему семейству)",
        "",
        "| Гипотеза | Сравнение | n | A | B | A − B | 95 % ДИ | p | p (Холм) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for (h, label, r), p_adj in zip(comps, adj):
        lines.append(
            f"| {h} | {label} | {r['n']} | {r['a']:.2f} | {r['b']:.2f} | {r['diff']:+.2f} | "
            f"[{r['ci'][0]:+.2f}; {r['ci'][1]:+.2f}] | {r['p']:.3f} | {'**' if p_adj < 0.05 else ''}{p_adj:.3f}{'**' if p_adj < 0.05 else ''} |"
        )
    lines += ["", "## Разброс от прогона к прогону (T = 0.5, набор test, система tools)", "", variance(eps)]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
