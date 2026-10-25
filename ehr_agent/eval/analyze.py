"""Aggregate scored episodes into tables (Markdown) and figures (PNG).

python -m ehr_agent.eval.analyze runs/*.jsonl
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from ..config import RUNS_DIR as RUNS
from .labels import COLORS, COND_RU, CONDS, EXCLUDED_MODELS, MODEL_ORDER


def load(paths: list[str]) -> list[dict]:
    """Episodes re-scored with the current scorer against the gold of the reference policy they ran under."""
    from ..config import DATA_DIR
    from .score import score_episode

    gold: dict[str, dict] = {}

    def questions(policy: str) -> dict:
        if policy not in gold:
            f = DATA_DIR / ("questions.jsonl" if policy == "indicative" else f"questions_{policy}.jsonl")
            gold[policy] = (
                {json.loads(l)["id"]: json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()} if f.exists() else {}
            )
        return gold[policy]

    eps = []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if e["model"] in EXCLUDED_MODELS:
                continue
            qs = questions(e.get("ref_policy", "indicative"))
            q = qs.get(e["qid"])
            if q:
                e["score"] = score_episode(q, e)
                if e["final"].get("status") == "context_overflow":
                    e["score"]["correct"] = None  # condition not applicable (record exceeds the window); mean() skips None
                e["level"] = q.get("level", 1)
                e["gold_rows"] = q.get("support_rows") or q.get("gold_rows", [])
            eps.append(e)
    return eps


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def boot_ci(xs, n=2000, seed=0):
    xs = [x for x in xs if x is not None]
    if not xs:
        return (float("nan"), float("nan"))
    rnd = random.Random(seed)
    ms = sorted(mean([rnd.choice(xs) for _ in xs]) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n)]


def error_tags(e: dict) -> list[str]:
    tags = []
    s, f = e["score"], e["final"]
    if f.get("format") != "final_answer":
        tags.append("нет final_answer")
    if e["n_invalid"]:
        tags.append("невалидный вызов")
    if any(st.get("result", {}).get("truncated") for st in e["trace"] if isinstance(st.get("result"), dict)):
        tags.append("обрезанный вывод инструмента")
    if s["fabricated"]:
        tags.append("число не из карты")
    if s["detail"].get("hallucinated_value"):
        tags.append("значение вместо «нет данных»")
    if s["detail"].get("trapped"):
        tags.append("принял испорченные данные")
    calls = [st for st in e["trace"] if st.get("type") == "call" and st.get("name") != "final_answer"]
    if s["correct"] < 1 and calls and calls[-1].get("empty") and f.get("status") == "not_found" and e["category"] != "not_found":
        tags.append("сдался после пустого результата")
    if s["correct"] < 1 and e["category"] == "abnormal" and s["detail"].get("fp"):
        tags.append("ложное отклонение")
    if s["correct"] < 1 and not tags:
        tags.append("прочее (неполный/неверный ответ)")
    return tags


def summarize(eps: list[dict]) -> str:
    by = defaultdict(list)
    for e in eps:
        by[(e["model"], e["condition"], e["budget"])].append(e)
    models = [m for m in MODEL_ORDER if any(k[0] == m for k in by)] + sorted({k[0] for k in by} - set(MODEL_ORDER))
    lines = [
        "## Сводка (тестовый набор)",
        "",
        "| Модель | Условие | Бюджет | N | Верность | 95% ДИ | Цит. валидны | Цит. покрытие | Эп. с выдуманным числом | Невалидн. вызовы/эп. | Вызовов/эп. | Токенов/эп. | Время, с |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in models:
        for c in CONDS:
            for b in sorted({k[2] for k in by if k[0] == m and k[1] == c}):
                g = by[(m, c, b)]
                corr = [e["score"]["correct"] for e in g]
                lo, hi = boot_ci(corr)
                tok = mean([e["usage"].get("prompt_tokens", 0) + e["usage"].get("completion_tokens", 0) for e in g])
                lines.append(
                    f"| {m} | {COND_RU[c]} | {b or '—'} | {len(g)} | **{mean(corr):.2f}** | {lo:.2f}–{hi:.2f} | "
                    f"{mean([e['score']['cite_valid'] for e in g]):.2f} | {mean([e['score'].get('cite_recall') for e in g]):.2f} | "
                    f"{mean([1.0 if e['score']['fabricated'] else 0.0 for e in g]):.2f} | {mean([e['n_invalid'] for e in g]):.2f} | "
                    f"{mean([e['n_calls'] for e in g]):.1f} | {tok:,.0f} | {mean([e['wall'] for e in g]):.1f} |"
                )
    cats = sorted({e["category"] for e in eps})
    lines += [
        "",
        "## Верность по категориям вопросов",
        "",
        "| Модель | Условие | " + " | ".join(cats) + " |",
        "|---|---|" + "---|" * len(cats),
    ]
    for m in models:
        for c in CONDS:
            g = [e for e in eps if e["model"] == m and e["condition"] == c and e["budget"] in (0, 4)]
            if g:
                lines.append(
                    f"| {m} | {COND_RU[c]} | "
                    + " | ".join(f"{mean([e['score']['correct'] for e in g if e['category'] == k]):.2f}" for k in cats)
                    + " |"
                )
    lines += ["", "## Типы ошибок (число эпизодов с верностью < 1)", ""]
    tag_names = sorted({t for e in eps for t in error_tags(e) if e["score"]["correct"] < 1})
    lines += ["| Модель | Условие | " + " | ".join(tag_names) + " |", "|---|---|" + "---|" * len(tag_names)]
    for m in models:
        for c in CONDS:
            g = [e for e in eps if e["model"] == m and e["condition"] == c and e["budget"] in (0, 4) and e["score"]["correct"] < 1]
            if not any(e["model"] == m and e["condition"] == c for e in eps):
                continue
            cnt = defaultdict(int)
            for e in g:
                for t in error_tags(e):
                    cnt[t] += 1
            lines.append(f"| {m} | {COND_RU[c]} | " + " | ".join(str(cnt[t]) for t in tag_names) + " |")
    tools = defaultdict(lambda: defaultdict(int))
    for e in eps:
        if e["condition"] == "domain":
            for st in e["trace"]:
                if st.get("type") == "call":
                    tools[e["model"]][st["name"]] += 1
    if tools:
        names = sorted({n for d in tools.values() for n in d})
        lines += [
            "",
            "## Частота вызова инструментов (медицинский набор)",
            "",
            "| Модель | " + " | ".join(names) + " |",
            "|---|" + "---|" * len(names),
        ]
        for m in models:
            if m in tools:
                lines.append(f"| {m} | " + " | ".join(str(tools[m][n]) for n in names) + " |")
    return "\n".join(lines)


def plot(eps: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#c3c2b7",
            "axes.labelcolor": "#52514e",
            "xtick.color": "#52514e",
            "ytick.color": "#52514e",
        }
    )
    models = [m for m in MODEL_ORDER if any(e["model"] == m for e in eps)]
    fig, ax = plt.subplots(figsize=(8.5, 4.2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    w = 0.26
    for j, c in enumerate(CONDS):
        vals, errs_lo, errs_hi = [], [], []
        for m in models:
            xs = [e["score"]["correct"] for e in eps if e["model"] == m and e["condition"] == c and e["budget"] in (0, 4)]
            mu = mean(xs)
            lo, hi = boot_ci(xs)
            vals.append(mu)
            errs_lo.append(mu - lo)
            errs_hi.append(hi - mu)
        xs_pos = [i + (j - 1) * (w + 0.02) for i in range(len(models))]
        bars = ax.bar(
            xs_pos,
            vals,
            w,
            color=COLORS[c],
            label=COND_RU[c],
            yerr=[errs_lo, errs_hi],
            error_kw={"ecolor": "#52514e", "elinewidth": 1, "capsize": 2},
        )
        for b, v in zip(bars, vals):
            if v == v:
                ax.text(b.get_x() + b.get_width() / 2, 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color="#ffffff")
    ax.set_xticks(range(len(models)), models)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Доля верных ответов (тест)")
    ax.yaxis.grid(True, color="#e8e7e2", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(out / "accuracy_by_model.png", dpi=180)
    plt.close(fig)

    cats = ["lookup", "abnormal", "not_found", "freetext", "integrate", "icd", "anthropo", "quality"]
    fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3.6), sharey=True, facecolor="#fcfcfb")
    axes = axes if len(models) > 1 else [axes]
    for ax, m in zip(axes, models):
        ax.set_facecolor("#fcfcfb")
        for j, c in enumerate(CONDS):
            vals = [
                mean(
                    [
                        e["score"]["correct"]
                        for e in eps
                        if e["model"] == m and e["condition"] == c and e["category"] == k and e["budget"] in (0, 4)
                    ]
                )
                for k in cats
            ]
            ax.barh([i + (j - 1) * 0.27 for i in range(len(cats))], vals, 0.25, color=COLORS[c], label=COND_RU[c])
        ax.set_yticks(range(len(cats)), cats)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_title(m, fontsize=10, color="#0b0b0b")
        ax.xaxis.grid(True, color="#e8e7e2", linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.08), ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "accuracy_by_category.png", dpi=180)
    plt.close(fig)


def plot_budget(eps: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dom = [e for e in eps if e["condition"] == "domain"]
    models = [m for m in MODEL_ORDER if len({e["budget"] for e in dom if e["model"] == m}) > 1]
    if not models:
        return
    fig, ax = plt.subplots(figsize=(6, 3.6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    palette = ["#2a78d6", "#eb6834", "#1baf7a"]
    for i, m in enumerate(models):
        bs = sorted({e["budget"] for e in dom if e["model"] == m})
        ys = [mean([e["score"]["correct"] for e in dom if e["model"] == m and e["budget"] == b]) for b in bs]
        ax.plot(bs, ys, marker="o", markersize=6, linewidth=2, color=palette[i % 3], label=m)
        ax.annotate(m, (bs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color="#52514e")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    ax.set_xlabel("Лимит вызовов инструментов на вопрос")
    ax.set_ylabel("Доля верных ответов")
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, color="#e8e7e2", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "budget.png", dpi=180)
    plt.close(fig)


def levels_table(all_eps: list[dict]) -> str:
    """Accuracy by phrasing level. L1 = the original test question; L2/L3 = paraphrases with the same gold."""
    para = [e for e in all_eps if e["split"] == "para"]
    if not para:
        return ""
    base_ids = {e["qid"].rsplit("-L", 1)[0] for e in para}
    rows = [
        "",
        "## Устойчивость к формулировке (те же вопросы: L1 точно, L2 клинически, L3 размыто)",
        "",
        "| Модель | Условие | L1 | L2 | L3 | Падение L1→L3 |",
        "|---|---|---|---|---|---|",
    ]
    for m in MODEL_ORDER:
        for c in CONDS:
            l1 = [
                e["score"]["correct"]
                for e in all_eps
                if e["split"] == "test" and e["qid"] in base_ids and e["model"] == m and e["condition"] == c and e["budget"] in (0, 4)
            ]
            l2 = [e["score"]["correct"] for e in para if e["model"] == m and e["condition"] == c and e["level"] == 2]
            l3 = [e["score"]["correct"] for e in para if e["model"] == m and e["condition"] == c and e["level"] == 3]
            if l2 or l3:
                rows.append(f"| {m} | {COND_RU[c]} | {mean(l1):.2f} | {mean(l2):.2f} | {mean(l3):.2f} | {mean(l1) - mean(l3):+.2f} |")
    return "\n".join(rows)


def summary_table(all_eps: list[dict]) -> str:
    summ = [e for e in all_eps if e["split"] == "summary"]
    if not summ:
        return ""
    secs = ["labs", "studies", "diagnoses", "inconsistencies", "plan", "quality"]
    rows = [
        "",
        "## Сводка чекапа: полнота чек-листа по разделам (P003, P004, P006; P005 — распознание порчи данных)",
        "",
        "| Модель | Условие | Итого | " + " | ".join(secs) + " | Выдуманные числа | Цит. валидны | Вызовов | Токенов |",
        "|---|---|---|" + "---|" * len(secs) + "---|---|---|---|",
    ]
    for m in MODEL_ORDER:
        for c in CONDS:
            g = [e for e in summ if e["model"] == m and e["condition"] == c]
            if not g:
                continue
            per = {k: mean([e["score"]["detail"].get("sections", {}).get(k) for e in g]) for k in secs}
            tok = mean([e["usage"].get("prompt_tokens", 0) + e["usage"].get("completion_tokens", 0) for e in g])
            rows.append(
                f"| {m} | {COND_RU[c]} | **{mean([e['score']['correct'] for e in g]):.2f}** | "
                + " | ".join("—" if v != v else f"{v:.2f}" for v in per.values())
                + f" | {sum(len(e['score']['fabricated']) for e in g)} | {mean([e['score']['cite_valid'] for e in g]):.2f} | "
                f"{mean([e['n_calls'] for e in g]):.1f} | {tok:,.0f} |"
            )
    return "\n".join(rows)


if __name__ == "__main__":
    paths = sys.argv[1:] or [str(p) for p in RUNS.glob("*.jsonl") if "smoke" not in p.name]
    all_eps = load(paths)
    eps = [e for e in all_eps if e["split"] == "test"]
    md = summarize(eps) + levels_table(all_eps) + summary_table(all_eps)
    pdf = [e for e in all_eps if e["split"] == "pdf"]
    if pdf:
        md += "\n\n" + summarize(pdf).replace(
            "## Сводка (тестовый набор)", "## Источник 2: PDF историй болезни (форма 003/у, другая клиника)"
        )
    pdf_sum = [e for e in all_eps if e["split"] == "pdf_summary"]
    if pdf_sum:
        md += summary_table([{**e, "split": "summary"} for e in pdf_sum]).replace(
            "## Сводка чекапа: полнота чек-листа по разделам (P003, P004, P006; P005 — распознание порчи данных)",
            "## Сводка госпитализации по PDF: полнота чек-листа",
        )
    probe = [e for e in all_eps if e["split"] == "probe"]
    if probe:
        md += "\n\n" + summarize(probe).replace("## Сводка (тестовый набор)", "## Проверочный набор (вопросы после разбора v1)")
    (RUNS / "summary.md").write_text(md, encoding="utf-8")
    plot(eps, RUNS / "figs")
    plot_budget(eps, RUNS / "figs")
    print(md)
