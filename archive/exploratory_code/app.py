"""Demo: ask a question about a patient's EHR, see tool calls, the answer and cited rows.

  .venv/bin/streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from ehr_agent.agent import run_episode
from ehr_agent.data import get_patient, list_patients
from ehr_agent.llm import make_backend

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
QUESTIONS = ROOT / "data" / "questions.jsonl"
COND_RU = {"domain": "Медицинские инструменты", "generic": "run_python (код по DataFrame)", "fullctx": "Вся карта в контексте"}
STATUS_RU = {"answered": "✅ ответ", "not_found": "∅ данных нет", "uncertain": "⚠️ данные недостоверны/противоречивы",
             "no_answer": "✖ нет ответа"}

st.set_page_config(page_title="EHR-агент", layout="wide")
st.title("EHR-агент: инструменты для LLM над медкартой")


@st.cache_resource
def backend(name: str):
    return make_backend(name)


def records_df(pid: str) -> pd.DataFrame:
    p = get_patient(pid)
    return pd.DataFrame([{"id": r.id, "дата": r.date, "специальность": r.specialty, "источник": r.visit_title or r.protocol,
                          "группа": r.group, "параметр": r.param, "значение": r.value} for r in p.records])


def show_episode(ep: dict) -> None:
    f = ep["final"]
    st.markdown(f"**Статус:** {STATUS_RU.get(f.get('status'), f.get('status'))} · "
                f"**вызовов:** {ep['n_calls']} (невалидных {ep['n_invalid']}) · **время:** {ep['wall']} с · "
                f"**токены:** {ep['usage'].get('prompt_tokens', 0) + ep['usage'].get('completion_tokens', 0):,}")
    st.success(f.get("answer") or "(пусто)")
    if "score" in ep:
        s = ep["score"]
        st.caption(f"Оценка: верность {s['correct']} · цитаты валидны {s['cite_valid']} · покрытие эталонных строк "
                   f"{s.get('cite_recall')} · числа не из карты: {s['fabricated'] or 'нет'} · детали {s['detail']}")
    st.subheader("Цепочка вызовов")
    for i, step in enumerate(ep["trace"], 1):
        if step["type"] == "call":
            icon = "🏁" if step["name"] == "final_answer" else ("❌" if step.get("valid") is False else ("∅" if step.get("empty") else "🔧"))
            with st.expander(f"{i}. {icon} {step['name']}({json.dumps(step.get('arguments'), ensure_ascii=False)[:120]})",
                             expanded=step["name"] != "final_answer" and i <= 2):
                if step.get("thought"):
                    st.caption(step["thought"])
                if step.get("error"):
                    st.error(step["error"])
                if step["name"] == "run_python" and isinstance(step.get("arguments"), dict):
                    st.code(step["arguments"].get("code", ""), language="python")
                if "result" in step:
                    st.json(step["result"], expanded=False)
        else:
            with st.expander(f"{i}. {step['type']}"):
                st.write(step.get("content") or step.get("error"))
    cites = [c for c in (f.get("citations") or [])]
    if cites:
        st.subheader("Строки карты, на которые ссылается ответ")
        df = records_df(ep["patient"])
        hit = df[df["id"].isin(cites)]
        missing = sorted(set(cites) - set(hit["id"]))
        st.dataframe(hit, hide_index=True, use_container_width=True)
        if missing:
            st.error(f"Несуществующие id в цитатах: {missing}")


tab_ask, tab_log, tab_res, tab_data = st.tabs(["Спросить", "Журнал прогонов", "Результаты", "Карта пациента"])

with tab_ask:
    c1, c2, c3, c4 = st.columns([1, 1.3, 1.6, 0.8])
    pid = c1.selectbox("Пациент", list_patients(), index=1)
    model = c2.selectbox("Модель", ["GigaChat-2-Max", "GigaChat-2", "qwen3.5-4b", "med-qwen3.5-4b"])
    cond = c3.selectbox("Условие", list(COND_RU), format_func=COND_RU.get)
    budget = c4.number_input("Лимит вызовов", 1, 8, 4)
    examples = []
    if QUESTIONS.exists():
        examples = [json.loads(line)["question"] for line in QUESTIONS.read_text(encoding="utf-8").splitlines()
                    if line.strip() and json.loads(line)["patient"] == pid]
    preset = st.selectbox("Пример вопроса", ["—"] + examples)
    q = st.text_area("Вопрос врача", value="" if preset == "—" else preset,
                     placeholder="Например: сведите все данные о печени: УЗИ, эластометрия, биохимия")
    if st.button("Спросить", type="primary", disabled=not q.strip()):
        with st.spinner("Агент работает…"):
            ep = run_episode(backend(model), get_patient(pid), q, cond, budget=int(budget))
        st.session_state["last"] = ep
    if "last" in st.session_state:
        show_episode(st.session_state["last"])

with tab_log:
    files = sorted(p for p in RUNS.glob("*.jsonl"))
    eps = []
    for p in files:
        eps += [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not eps:
        st.info("Прогонов пока нет.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        m = c1.selectbox("Модель ", sorted({e["model"] for e in eps}))
        c = c2.selectbox("Условие ", list(COND_RU), format_func=COND_RU.get)
        cat = c3.selectbox("Категория", ["все"] + sorted({e["category"] for e in eps}))
        only_bad = c4.checkbox("Только ошибки")
        sel = [e for e in eps if e["model"] == m and e["condition"] == c and (cat == "все" or e["category"] == cat)
               and (not only_bad or e["score"]["correct"] < 1)]
        if sel:
            labels = [f"{e['qid']} [{e['category']}, b{e['budget']}] {e['score']['correct']:.2f} — {e['question'][:70]}" for e in sel]
            i = st.selectbox("Эпизод", range(len(sel)), format_func=lambda k: labels[k])
            st.markdown(f"**Вопрос:** {sel[i]['question']}")
            # Same question under the other conditions, for side-by-side comparison.
            others = [e for e in eps if e["qid"] == sel[i]["qid"] and e["model"] == m and e["condition"] != c]
            if others:
                st.caption("Тот же вопрос в других условиях: " + "; ".join(
                    f"{COND_RU[e['condition']]} → {e['score']['correct']:.2f}" for e in others))
            show_episode(sel[i])
        else:
            st.info("Нет эпизодов под фильтр.")

with tab_res:
    summary = RUNS / "summary.md"
    for name in ("accuracy_by_model.png", "accuracy_by_category.png", "budget.png"):
        if (RUNS / "figs" / name).exists():
            st.image(str(RUNS / "figs" / name))
    if summary.exists():
        st.markdown(summary.read_text(encoding="utf-8"))

with tab_data:
    pid2 = st.selectbox("Пациент ", list_patients(), index=1)
    df = records_df(pid2)
    flt = st.text_input("Фильтр по тексту")
    if flt:
        df = df[df.apply(lambda r: flt.lower() in " ".join(map(str, r.values)).lower(), axis=1)]
    st.dataframe(df, hide_index=True, use_container_width=True, height=600)
