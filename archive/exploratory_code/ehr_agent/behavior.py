"""Behavioural analysis of agent trajectories: how models use tools, not only whether they are right.

  python -m ehr_agent.behavior                 → runs/behavior.md (+ runs/figs/contrast.png)
  EHR_REF_POLICY=layered python -m ehr_agent.behavior runs/final_*.jsonl

Sections: contrasts (model × source × access), tool-use profile, trajectories, reaction to empty
and truncated results, format and status calibration, citations, failure exemplars.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .analyze import COND_RU, EXCLUDED_MODELS, MODEL_ORDER, RUNS, boot_ci, error_tags, load, mean

SOURCE = {"test": "таблица (чекап)", "pdf": "PDF (003/у)", "para": "таблица, перефраз.", "summary": "таблица: сводка",
          "pdf_summary": "PDF: сводка", "probe": "таблица: проверочный", "probe3": "проверочный v3", "probe4": "проверочный v4"}
SHORT = {"GigaChat-2": "Giga Lite", "GigaChat-2-Max": "Giga Max", "Qwen3.5-4B-4bit": "Qwen3.5-4B",
         "Med-Qwen-3.5-4B-it-4bit": "Med-Qwen-4B"}


def _calls(e: dict) -> list[dict]:
    return [s for s in e["trace"] if s.get("type") == "call" and s.get("name") != "final_answer"]


def _is_trunc(s: dict) -> bool:
    r = s.get("result")
    return isinstance(r, dict) and bool(r.get("truncated"))


def contrast_table(eps: list[dict]) -> str:
    rows = ["## 1. Контрасты: модель × источник × способ доступа к карте", "",
            "Верность (доля), 95% бутстреп-ДИ; в скобках число эпизодов. Одинаковые вопросы для всех моделей: разреженная выборка таблиц (22 вопроса: 1 на пару «пациент × категория») и все 17 вопросов PDF.", "",
            "| Модель | Условие | Таблица (чекап) | PDF (003/у) | Δ PDF − таблица |", "|---|---|---|---|---|"]
    for m in MODEL_ORDER:
        for c in COND_RU:
            t = [e["score"]["correct"] for e in eps if e["model"] == m and e["condition"] == c and e["split"] == "test" and e["budget"] in (0, 4)]
            p = [e["score"]["correct"] for e in eps if e["model"] == m and e["condition"] == c and e["split"] == "pdf"]
            if not t and not p:
                continue

            def fmt(x):
                if not x:
                    return "—"
                lo, hi = boot_ci(x)
                return f"**{mean(x):.2f}** [{lo:.2f}–{hi:.2f}] ({len(x)})"
            d = f"{mean(p) - mean(t):+.2f}" if t and p else "—"
            rows.append(f"| {SHORT.get(m, m)} | {COND_RU[c]} | {fmt(t)} | {fmt(p)} | {d} |")
    return "\n".join(rows)


def tool_profile(eps: list[dict]) -> str:
    rows = ["", "## 2. Профиль использования инструментов (медицинские инструменты, v1)", ""]
    groups = defaultdict(list)
    for e in eps:
        if e["condition"] in ("domain", "domain_v2") and e["split"] in ("test", "pdf"):
            groups[(e["model"], e["condition"], e["split"])].append(e)
    names = sorted({s["name"] for g in groups.values() for e in g for s in _calls(e)})
    rows += ["| Модель | Условие | Источник | Вызовов/эп. | Первый вызов (топ-2) | " + " | ".join(names) + " |",
             "|---|---|---|---|---|" + "---|" * len(names)]
    for (m, c, sp), g in sorted(groups.items(), key=lambda kv: (MODEL_ORDER.index(kv[0][0]) if kv[0][0] in MODEL_ORDER else 9, kv[0][1], kv[0][2])):
        cnt = Counter(s["name"] for e in g for s in _calls(e))
        first = Counter(_calls(e)[0]["name"] for e in g if _calls(e))
        tot = sum(cnt.values()) or 1
        rows.append(f"| {SHORT.get(m, m)} | {COND_RU[c]} | {SOURCE[sp]} | {mean([len(_calls(e)) for e in g]):.1f} | "
                    + ", ".join(f"{k} {v / len(g):.0%}" for k, v in first.most_common(2)) + " | "
                    + " | ".join(f"{cnt[n] / tot:.0%}" if cnt[n] else "·" for n in names) + " |")
    return "\n".join(rows)


def process_table(eps: list[dict]) -> str:
    rows = ["", "## 3. Процесс: формат, пустые результаты, обрезка, лимит, повторы", "",
            "- **Нет final_answer** — ответ пришёл простым текстом (после одного напоминания) или не пришёл вовсе.",
            "- **После пустого:** что модель сделала сразу после пустого/ошибочного результата инструмента: *переформулировала* (тот же инструмент, другие аргументы), *сменила инструмент*, *сдалась* (final с not_found), *ответила* (final с answered).",
            "- **Обрезка → уточнила:** доля случаев, когда после обрезанного вывода модель вызвала инструмент с фильтром, а не ответила.", "",
            "| Модель | Условие | Источник | N | Нет final_answer | Невалидн. вызовы | Пустых рез./эп. | После пустого: переформ. / смена / сдался / ответил | Обрезка → уточнила | Лимит исчерпан | Повтор вызова |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    groups = defaultdict(list)
    for e in eps:
        if e["split"] in ("test", "pdf") and e["condition"] in ("domain", "domain_v2", "generic"):
            groups[(e["model"], e["condition"], e["split"])].append(e)
    for (m, c, sp), g in sorted(groups.items(), key=lambda kv: (MODEL_ORDER.index(kv[0][0]) if kv[0][0] in MODEL_ORDER else 9, kv[0][1], kv[0][2])):
        react = Counter()
        trunc_total = trunc_refined = 0
        empties = 0
        for e in g:
            steps = [s for s in e["trace"] if s.get("type") in ("call", "text")]
            for i, s in enumerate(steps):
                if s.get("type") != "call" or s.get("name") == "final_answer":
                    continue
                if s.get("empty"):
                    empties += 1
                    nxt = steps[i + 1] if i + 1 < len(steps) else None
                    if nxt is None or nxt.get("type") == "text":
                        react["ответил"] += 1
                    elif nxt.get("name") == "final_answer":
                        st = (nxt.get("arguments") or {}).get("status") if isinstance(nxt.get("arguments"), dict) else None
                        react["сдался" if st == "not_found" else "ответил"] += 1
                    elif nxt.get("name") == s.get("name"):
                        react["переформ."] += 1
                    else:
                        react["смена"] += 1
                if _is_trunc(s):
                    trunc_total += 1
                    nxt = steps[i + 1] if i + 1 < len(steps) else None
                    if nxt is not None and nxt.get("type") == "call" and nxt.get("name") != "final_answer":
                        trunc_refined += 1
        tot_r = sum(react.values()) or 1
        rows.append(
            f"| {SHORT.get(m, m)} | {COND_RU[c]} | {SOURCE[sp]} | {len(g)} | "
            f"{mean([0.0 if e['final'].get('format') == 'final_answer' else 1.0 for e in g]):.0%} | "
            f"{mean([e['n_invalid'] for e in g]):.2f} | {empties / len(g):.2f} | "
            + " / ".join(f"{react[k] / tot_r:.0%}" for k in ("переформ.", "смена", "сдался", "ответил"))
            + f" | {(trunc_refined / trunc_total) if trunc_total else float('nan'):.0%} ({trunc_total}) | "
            f"{mean([1.0 if e['n_calls'] >= e['budget'] > 0 else 0.0 for e in g]):.0%} | "
            f"{mean([1.0 if any(s.get('repeat') for s in e['trace']) else 0.0 for e in g]):.0%} |")
    return "\n".join(rows)


def status_calibration(eps: list[dict]) -> str:
    """Does the model say 'not found' / 'uncertain' when it should?"""
    rows = ["", "## 4. Калибровка статуса ответа", "",
            "Строки — тип эталона; ячейки — доля эпизодов с данным статусом в final_answer (answered / not_found / uncertain / нет ответа).", "",
            "| Модель | Условие | Эталон «данные есть» | Эталон «нет данных» | Эталон «данные испорчены» |", "|---|---|---|---|---|"]
    kind = lambda e: "not_found" if e["category"] == "not_found" else "uncertain" if e["category"] == "quality" else "present"  # noqa: E731
    for m in MODEL_ORDER:
        for c in COND_RU:
            g = [e for e in eps if e["model"] == m and e["condition"] == c and e["split"] in ("test", "pdf") and e["budget"] in (0, 4)]
            if not g:
                continue
            cells = []
            for k in ("present", "not_found", "uncertain"):
                gg = [e for e in g if kind(e) == k]
                if not gg:
                    cells.append("—")
                    continue
                st = Counter(e["final"].get("status", "answered") if e["final"].get("format") != "none" else "none" for e in gg)
                cells.append(" / ".join(f"{st[s] / len(gg):.0%}" for s in ("answered", "not_found", "uncertain", "none")) + f" ({len(gg)})")
            rows.append(f"| {SHORT.get(m, m)} | {COND_RU[c]} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def grounding_table(eps: list[dict]) -> str:
    rows = ["", "## 5. Обоснованность и стоимость", "",
            "| Модель | Условие | Источник | Цитат/ответ | Валидные цитаты | Покрытие эталонных строк | Эп. с выдуманным числом | Длина ответа, симв. | Токенов/эп. | Время, с |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    groups = defaultdict(list)
    for e in eps:
        if e["split"] in ("test", "pdf") and e["budget"] in (0, 4):
            groups[(e["model"], e["condition"], e["split"])].append(e)
    for (m, c, sp), g in sorted(groups.items(), key=lambda kv: (MODEL_ORDER.index(kv[0][0]) if kv[0][0] in MODEL_ORDER else 9, kv[0][1], kv[0][2])):
        rows.append(f"| {SHORT.get(m, m)} | {COND_RU[c]} | {SOURCE[sp]} | {mean([e['score']['n_cites'] for e in g]):.1f} | "
                    f"{mean([e['score']['cite_valid'] for e in g]):.2f} | {mean([e['score'].get('cite_recall') for e in g]):.2f} | "
                    f"{mean([1.0 if e['score']['fabricated'] else 0.0 for e in g]):.2f} | {mean([len(str(e['final'].get('answer', ''))) for e in g]):.0f} | "
                    f"{mean([e['usage'].get('prompt_tokens', 0) + e['usage'].get('completion_tokens', 0) for e in g]):,.0f} | {mean([e['wall'] for e in g]):.1f} |")
    return "\n".join(rows)


def error_matrix(eps: list[dict]) -> str:
    rows = ["", "## 6. Типы ошибок по источникам", ""]
    g = [e for e in eps if e["split"] in ("test", "pdf") and e["budget"] in (0, 4) and e["score"]["correct"] < 1]
    tags = sorted({t for e in g for t in error_tags(e)})
    rows += ["| Модель | Условие | Источник | Ошибок | " + " | ".join(tags) + " |", "|---|---|---|---|" + "---|" * len(tags)]
    groups = defaultdict(list)
    for e in g:
        groups[(e["model"], e["condition"], e["split"])].append(e)
    for (m, c, sp), gg in sorted(groups.items(), key=lambda kv: (MODEL_ORDER.index(kv[0][0]) if kv[0][0] in MODEL_ORDER else 9, kv[0][1], kv[0][2])):
        cnt = Counter(t for e in gg for t in error_tags(e))
        rows.append(f"| {SHORT.get(m, m)} | {COND_RU[c]} | {SOURCE[sp]} | {len(gg)} | " + " | ".join(str(cnt[t]) if cnt[t] else "·" for t in tags) + " |")
    return "\n".join(rows)


# Tools that can answer each category; used to judge the first call (tool selection).
APT_TOOLS = {
    "lookup": {"get_lab", "list_abnormal_labs", "search_notes", "get_document"},
    "abnormal": {"list_abnormal_labs", "get_lab"},
    "not_found": {"get_lab", "list_abnormal_labs", "search_notes"},
    "icd": {"get_diagnoses", "get_conclusions"},
    "anthropo": {"check_consistency", "get_lab", "search_notes"},
    "freetext": {"search_notes", "get_conclusions", "get_document", "get_overview"},
    "integrate": {"get_conclusions", "search_notes", "get_lab", "list_abnormal_labs", "get_diagnoses", "get_document", "get_overview"},
    "quality": {"get_lab", "check_consistency", "list_abnormal_labs"},
}


def _seen_ids(e: dict) -> set[str]:
    import re as _re
    seen = set()
    for s in e["trace"]:
        if "result" in s:
            seen |= set(_re.findall(r"(?:P00\d|K0\d):(?:r\d+|p\d+:[bl]\d+)", json.dumps(s["result"], ensure_ascii=False)))
    return seen


def diagnosis_table(eps: list[dict]) -> str:
    """Failure locus, as in RAG evaluation: was the evidence retrieved, and was it used?"""
    rows = ["", "## 3a. Где ломается: поиск или использование (по эпизодам с эталонными строками)", "",
            "*Поиск*: эталонные строки ни разу не попали в вывод инструментов. *Частично*: попала часть. *Использование*: все эталонные строки модель видела, но ответ неверен или неполон. "
            "Для «вся карта» все строки в контексте по определению, поэтому любая ошибка — ошибка использования. *Выбор*: первый вызов не из подходящих для категории инструментов.", "",
            "| Модель | Условие | Источник | Ошибок | Поиск | Частично | Использование | Неудачный первый выбор (всех эп.) |", "|---|---|---|---|---|---|---|---|"]
    groups = defaultdict(list)
    for e in eps:
        if e["split"] in ("test", "pdf") and e["condition"] in ("domain", "domain_v2", "domain_v3", "fullctx", "hybrid"):
            groups[(e["model"], e["condition"], e["split"])].append(e)
    for (m, c, sp), g in sorted(groups.items(), key=lambda kv: (MODEL_ORDER.index(kv[0][0]) if kv[0][0] in MODEL_ORDER else 9, kv[0][1], kv[0][2])):
        loc = Counter()
        for e in g:
            if e["score"]["correct"] >= 1 or not e.get("gold_rows"):
                continue
            gold = set(e["gold_rows"])
            seen = gold if c in ("fullctx", "hybrid") else _seen_ids(e) & gold
            loc["поиск" if not seen else "использование" if seen == gold else "частично"] += 1
        bad_first = [e for e in g if c.startswith("domain") and _calls(e) and _calls(e)[0]["name"] not in APT_TOOLS.get(e["category"], set())]
        n_err = sum(loc.values())
        rows.append(f"| {SHORT.get(m, m)} | {COND_RU[c]} | {SOURCE[sp]} | {n_err} | {loc['поиск']} | {loc['частично']} | {loc['использование']} | "
                    + (f"{len(bad_first) / len(g):.0%}" if c.startswith("domain") else "—") + " |")
    return "\n".join(rows)


def v3_table(eps: list[dict]) -> str:
    rows = ["## 0. Итерация v3 (T1+T2+T8) против v1 и v2", "",
            "Доля верных ответов (число эпизодов). Проверочные v3 и v4 написаны до реализации соответствующей версии (v3: 5 таблица + 4 PDF; v4: 2 таблица + 8 PDF, назначения и исследования).", "",
            "| Модель | Версия | Таблица (22) | PDF (15) | Проверочный v3 (9) | Проверочный v4 (10) | Валидные цитаты PDF | Ошибки поиска / использования (таблица+PDF) | Вмешательства T5: текст / ссылки / «нет данных» |",
            "|---|---|---|---|---|---|---|---|---|"]
    for m in MODEL_ORDER:
        for c in ("domain", "domain_v2", "domain_v3", "domain_v3p", "domain_v3r", "domain_v4a", "domain_v4", "domain_v41a", "domain_v41"):
            def acc(sp):
                g = [e["score"]["correct"] for e in eps if e["model"] == m and e["condition"] == c and e["split"] == sp]
                return f"{mean(g):.2f} ({len(g)})" if g else "—"
            pdf = [e for e in eps if e["model"] == m and e["condition"] == c and e["split"] == "pdf"]
            if not any(e["model"] == m and e["condition"] == c for e in eps):
                continue
            loc = Counter()
            for e in eps:
                if e["model"] == m and e["condition"] == c and e["split"] in ("test", "pdf", "probe3", "probe4") and e["score"]["correct"] < 1 and e.get("gold_rows"):
                    seen = _seen_ids(e) & set(e["gold_rows"])
                    loc["поиск" if not seen else "использование"] += 1
            cv = mean([e["score"]["cite_valid"] for e in pdf]) if pdf else float("nan")
            hv = Counter()
            for e in eps:
                if e["model"] == m and e["condition"] == c:
                    for st in e["trace"]:
                        if st.get("type") == "harness":
                            msg = st.get("message") or {}
                            hv["text" if st.get("action") == "rescued_text_final" else "cite" if "invalid_ids" in msg else "absent" if "check" in msg else "other"] += 1
            rows.append(f"| {SHORT.get(m, m)} | { {'domain': 'v1', 'domain_v2': 'v2', 'domain_v3': 'v3', 'domain_v3p': 'v3+prefetch', 'domain_v3r': 'v3+routed', 'domain_v4a': 'v3+T5', 'domain_v4': 'v4 (T3+T5)', 'domain_v41a': 'v3+T5.1', 'domain_v41': 'v4.1 (T3+T5.1)'}[c] } | {acc('test')} | {acc('pdf')} | {acc('probe3')} | {acc('probe4')} | "
                        f"{'—' if cv != cv else f'{cv:.2f}'} | {loc['поиск']} / {loc['использование']} | "
                        + (f"{hv['text']} / {hv['cite']} / {hv['absent']}" if c in ('domain_v4a', 'domain_v4', 'domain_v41a', 'domain_v41') else "—") + " |")
    return "\n".join(rows)


def _traj(e: dict) -> str:
    parts = []
    for s in e["trace"]:
        if s.get("type") == "call":
            a = s.get("arguments")
            arg = ""
            if isinstance(a, dict) and s["name"] != "final_answer":
                arg = ", ".join(f"{k}={str(v)[:40]!r}" for k, v in a.items() if k != "code")
                if "code" in a:
                    arg = "code=" + repr(str(a["code"])[:120])
            mark = " ∅" if s.get("empty") else " ✂" if _is_trunc(s) else " ✗" if s.get("valid") is False else ""
            parts.append(f"`{s['name']}({arg})`{mark}")
        elif s.get("type") == "text":
            parts.append("*текст без final_answer*")
        else:
            parts.append(f"*{s.get('type')}*")
    return " → ".join(parts)


def exemplars(eps: list[dict], per_tag: int = 2) -> str:
    rows = ["", "## 7. Примеры провалов (траектории)", "", "∅ — пустой результат, ✂ — обрезанный вывод, ✗ — невалидный вызов.", ""]
    g = [e for e in eps if e["split"] in ("test", "pdf") and e["score"]["correct"] < 1 and e["budget"] in (0, 4)]
    by_tag = defaultdict(list)
    for e in g:
        for t in error_tags(e):
            by_tag[t].append(e)
    for t, es in sorted(by_tag.items(), key=lambda kv: -len(kv[1])):
        rows.append(f"### {t} ({len(es)})")
        seen = set()
        for e in es:
            key = (e["model"], e["condition"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(f"- **{SHORT.get(e['model'], e['model'])}, {COND_RU[e['condition']]}, {e['qid']}** — «{e['question'][:90]}»  \n"
                        f"  {_traj(e)}  \n  Ответ ({e['final'].get('status')}): {str(e['final'].get('answer'))[:220].replace(chr(10), ' ')}  \n"
                        f"  Оценка {e['score']['correct']}; {json.dumps(e['score']['detail'], ensure_ascii=False)[:160]}")
            if len(seen) >= per_tag:
                break
        rows.append("")
    return "\n".join(rows)


def plot_contrast(eps: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = [c for c in ("domain", "generic", "fullctx") if any(e["condition"] == c for e in eps)]
    models = [m for m in MODEL_ORDER if any(e["model"] == m for e in eps)]
    colors = {"domain": "#2a78d6", "generic": "#eb6834", "fullctx": "#1baf7a"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, facecolor="#fcfcfb")
    for ax, sp in zip(axes, ("test", "pdf")):
        ax.set_facecolor("#fcfcfb")
        for j, c in enumerate(conds):
            vals = [mean([e["score"]["correct"] for e in eps if e["model"] == m and e["condition"] == c and e["split"] == sp and e["budget"] in (0, 4)]) for m in models]
            xs = [i + (j - 1) * 0.27 for i in range(len(models))]
            ax.bar(xs, [0 if v != v else v for v in vals], 0.25, color=colors[c], label=COND_RU[c])
            for x, v in zip(xs, vals):
                ax.text(x, (0 if v != v else v) + 0.02, "—" if v != v else f"{v:.2f}", ha="center", fontsize=7, color="#52514e")
        ax.set_xticks(range(len(models)), [SHORT.get(m, m) for m in models], fontsize=9)
        ax.set_title(SOURCE[sp], fontsize=10, color="#0b0b0b")
        ax.set_ylim(0, 1.1)
        ax.yaxis.grid(True, color="#e8e7e2", linewidth=0.8)
        ax.set_axisbelow(True)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
    axes[0].set_ylabel("Доля верных ответов")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.18), ncol=3)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "contrast.png", dpi=180)
    plt.close(fig)


def comparable(eps: list[dict]) -> list[dict]:
    """Same questions for every model: table test episodes restricted to the sparse sample; budget 4 (0 for fullctx)."""
    from .questions import OUT as QUESTIONS
    from .run import sparse

    qs = [json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]
    keep = {q["id"] for q in sparse(qs, 1)}
    return [e for e in eps if e["budget"] in (0, 4) and (e["split"] not in ("test", "para", "summary") or e["qid"] in keep)]


if __name__ == "__main__":
    paths = sys.argv[1:] or [str(p) for p in RUNS.glob("*.jsonl") if "smoke" not in p.name]
    eps = comparable([e for e in load(paths) if e["model"] not in EXCLUDED_MODELS])
    md = "\n".join([f"# Поведение агентов ({len(eps)} эпизодов)\n", v3_table(eps), "", contrast_table(eps), tool_profile(eps), process_table(eps), diagnosis_table(eps),
                    status_calibration(eps), grounding_table(eps), error_matrix(eps), exemplars(eps)])
    (RUNS / "behavior.md").write_text(md, encoding="utf-8")
    plot_contrast(eps, RUNS / "figs")
    print(md[:6000])
