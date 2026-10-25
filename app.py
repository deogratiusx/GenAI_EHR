"""Demo: ask a question about a patient's record, see how the model uses the tools, the answer and cited rows.

The interface is in English (the defense is in English); records, questions and answers are Russian.
EHR_DEMO_PUBLIC=1 (the hosted demo) restricts it to the anonymized check-ups, the GigaChat models and the
demo snapshot of the final run (demo_data/, built by scripts/build_demo_data.py).

  .venv/bin/streamlit run app.py
"""

from __future__ import annotations

import json
import os
from collections import Counter

import altair as alt
import pandas as pd
import streamlit as st

from ehr_agent.agent import SYSTEMS, run_episode
from ehr_agent.agent.llm import make_backend
from ehr_agent.config import DATA_DIR, RUNS_DIR
from ehr_agent.eval.labels import EXCLUDED_MODELS, SHORT
from ehr_agent.sources import get_patient, list_patients

PUBLIC = os.environ.get("EHR_DEMO_PUBLIC") == "1"
QUESTIONS = DATA_DIR / "questions_layered.jsonl"
MODELS = ["GigaChat-2-Max", "GigaChat-2"] + ([] if PUBLIC else ["qwen3.5-4b", "med-qwen3.5-4b"])
PATIENTS = list_patients("checkup") if PUBLIC else list_patients()
SYSTEM_EN = {
    "tools": "Tools + harness (main system)",
    "tools_routed": "Tools routed by intent",
    "tools_prefetch": "Tools + prefetched intent tool",
    "tools_compact": "Compact tool set",
    "tools_plain": "Tools without harness",
    "report": "Report: dossier + completeness checks",
    "report_nocheck": "Report: dossier, no checks",
    "code": "Code interpreter (baseline)",
    "record": "Whole record in context (baseline)",
    "record_chunked": "Whole record, map-reduce (baseline)",
    "record+tools": "Whole record + tools",
}
CATEGORY_EN = {
    "lookup": "lab value", "abnormal": "out-of-range panel", "not_found": "not measured (trap)", "icd": "ICD-10 codes",
    "anthropo": "height/weight conflict", "freetext": "free-text fact", "integrate": "cross-document synthesis",
    "quality": "corrupted data (trap)", "summary": "whole-record summary", "dynamics": "dynamics",
}  # fmt: skip
STATUS_EN = {
    "answered": "✅ answered",
    "not_found": "∅ no data in the record",
    "uncertain": "⚠️ data unreliable or contradictory",
    "no_answer": "✖ no answer",
    "context_overflow": "✖ record does not fit the context window",
}
HARNESS_EN = {
    "format_repair": "harness: asked for a formatted final answer",
    "verify": "harness: answer check",
    "rescued_text_final": "harness: answer parsed from text",
}
SHOWCASE = ["P003-01", "P005-01", "P003-11", "P004-09", "P003-SUM"]  # first in the list of examples
MODEL_COLORS = {"GigaChat-2-Max": "#791cbe", "GigaChat-2": "#2a78d6", "Qwen3.5-4B-4bit": "#eb6834", "Med-Qwen-3.5-4B-it-4bit": "#1baf7a"}

st.set_page_config(page_title="AI-mnesis · EHR agent demo", layout="wide")
st.markdown(
    "<p style='color:#791cbe;font-weight:600;letter-spacing:2px;margin-bottom:0'>AI-MNESIS · COURSE PROJECT DEMO</p>",
    unsafe_allow_html=True,
)
st.title("LLM agent with tools over an electronic health record")
st.caption("Pick a question from our test set or write your own, run it, and inspect every tool call. Records and answers are in Russian.")


@st.cache_resource
def backend(name: str):
    return make_backend(name)


@st.cache_data
def questions() -> list[dict]:
    if not QUESTIONS.exists():
        return []
    qs = [
        q for q in map(json.loads, QUESTIONS.read_text(encoding="utf-8").splitlines()) if q["patient"] in PATIENTS and q["split"] != "dev"
    ]
    return sorted(qs, key=lambda q: (q["id"] not in SHOWCASE, SHOWCASE.index(q["id"]) if q["id"] in SHOWCASE else 0, q["id"]))


@st.cache_data
def episodes() -> list[dict]:
    files = [RUNS_DIR / "demo_runs.jsonl"] if PUBLIC else sorted(RUNS_DIR.glob("final_*.jsonl"))
    return [
        e
        for p in files
        if p.exists()
        for e in map(json.loads, p.read_text(encoding="utf-8").splitlines())
        if e.get("model") not in EXCLUDED_MODELS and "score" in e and e.get("temperature", 0) == 0
    ]


@st.cache_data
def summary() -> pd.DataFrame:
    """Per model × system × source counts: the shipped summary, or computed from the local final run."""
    f = RUNS_DIR / "runs_summary.json"
    if f.exists():
        return pd.DataFrame(json.loads(f.read_text(encoding="utf-8")))
    rows = []
    for e in episodes():
        src = "summary" if e["category"] == "summary" else ("pdf" if e["patient"].startswith("K") else "table")
        calls = [s["name"] for s in e["trace"] if s["type"] == "call" and s["name"] != "final_answer"]
        rows.append({"model": e["model"], "system": e["condition"], "source": src, "acc": e["score"]["correct"], "calls": calls,
                     "harness": [s.get("action") for s in e["trace"] if s["type"] == "harness"], "n_calls": e["n_calls"],
                     "wall": e["wall"], "n_invalid": e["n_invalid"]})  # fmt: skip
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    out = []
    for (m, s, src), g in df.groupby(["model", "system", "source"]):
        out.append({"model": m, "system": s, "source": src, "episodes": len(g), "accuracy": g["acc"].dropna().mean(),
                    "tool_calls": dict(Counter(c for cs in g["calls"] for c in cs)),
                    "first_tool": dict(Counter(cs[0] for cs in g["calls"] if cs)),
                    "harness": dict(Counter(h for hs in g["harness"] for h in hs)),
                    "calls_per_episode": g["n_calls"].mean(), "invalid_per_episode": g["n_invalid"].mean(),
                    "median_seconds": g["wall"].median()})  # fmt: skip
    return pd.DataFrame(out)


@st.cache_data
def records_df(pid: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": r.id,
                "date": r.date,
                "specialty": r.specialty,
                "document": r.source,
                "group": r.group,
                "field": r.param,
                "value": r.value,
            }
            for r in get_patient(pid).records
        ]
    )


def tool_path(ep: dict) -> None:
    """The episode as a path of steps with their latency: how the model moved through the tools."""
    steps = []
    for s in ep["trace"]:
        if s["type"] == "call":
            steps.append({"step": s["name"], "kind": "final answer" if s["name"] == "final_answer" else ("invalid call" if s.get("valid") is False else "tool call"),
                          "seconds": float(s.get("latency") or 0) + float(s.get("tool_time") or 0)})  # fmt: skip
        elif s["type"] in ("harness", "prefetch", "map"):
            steps.append({"step": s.get("action") or s.get("name") or f"chunk {s.get('chunk')}", "kind": "harness" if s["type"] == "harness" else s["type"],
                          "seconds": float(s.get("latency") or 0)})  # fmt: skip
    if not steps:
        return
    st.markdown(" → ".join(f"`{s['step']}`" for s in steps))
    df = pd.DataFrame(steps).reset_index(names="order")
    df["label"] = df["order"].astype(str) + ". " + df["step"]
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4, height=18)
        .encode(
            y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=300)),
            x=alt.X("seconds:Q", title="model latency, s"),
            color=alt.Color("kind:N", scale=alt.Scale(domain=["tool call", "final answer", "harness", "prefetch", "map", "invalid call"],
                                                       range=["#791cbe", "#1baf7a", "#eb6834", "#2a78d6", "#2a78d6", "#c62f2f"]), title=None),
            tooltip=["step", "kind", alt.Tooltip("seconds:Q", format=".2f")],
        )
        .properties(height=max(80, 26 * len(df)))
    )  # fmt: skip
    st.altair_chart(chart, use_container_width=True)


def show_episode(ep: dict) -> None:
    f = ep["final"] or {}
    usage = ep.get("usage") or {}
    st.markdown(
        f"**Status:** {STATUS_EN.get(f.get('status'), f.get('status'))} · **tool calls:** {ep['n_calls']} "
        f"(invalid {ep['n_invalid']}) · **time:** {ep['wall']} s · "
        f"**tokens:** {usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0):,}"
    )
    st.success(f.get("answer") or "(empty)")
    if "score" in ep:
        s = ep["score"]
        st.caption(
            f"Automatic score: correctness {s['correct']} · valid citations {s['cite_valid']} · numbers not in the record: {s['fabricated'] or 'none'}"
        )
    st.subheader("How the model used the tools")
    tool_path(ep)
    with st.expander("Full trace: arguments and tool outputs"):
        for i, step in enumerate(ep["trace"], 1):
            if step["type"] == "call":
                icon = (
                    "🏁"
                    if step["name"] == "final_answer"
                    else ("❌" if step.get("valid") is False else ("∅" if step.get("empty") else "🔧"))
                )
                st.markdown(f"**{i}. {icon} {step['name']}** `{json.dumps(step.get('arguments'), ensure_ascii=False)[:300]}`")
                if step.get("thought"):
                    st.caption(step["thought"])
                if step.get("error"):
                    st.error(step["error"])
                if step["name"] == "run_python" and isinstance(step.get("arguments"), dict):
                    st.code(step["arguments"].get("code", ""), language="python")
                if "result" in step:
                    st.json(step["result"], expanded=False)
            elif step["type"] == "harness":
                st.markdown(f"**{i}. 🛡 {HARNESS_EN.get(step.get('action'), step.get('action'))}**")
                st.json(step.get("message") or {k: v for k, v in step.items() if k not in ("type", "action")}, expanded=False)
            else:
                title = {"prefetch": f"prefetched {step.get('name')}", "map": f"map: chunk {step.get('chunk')}/{step.get('of')}"}.get(
                    step["type"], step["type"]
                )
                st.markdown(f"**{i}. 💬 {title}**")
                st.write(step.get("content") or step.get("error") or step.get("result"))
    cites = f.get("citations") or []
    if cites:
        st.subheader("Record rows cited by the answer")
        df = records_df(ep["patient"])
        hit = df[df["id"].isin(cites)]
        st.dataframe(hit, hide_index=True, use_container_width=True)
        missing = sorted(set(cites) - set(hit["id"]))
        if missing:
            st.error(f"Cited ids that do not exist: {missing}")


def patient_label(pid: str) -> str:
    return f"{pid} · {'check-up (table)' if pid.startswith('P') else 'inpatient record (PDF)'}"


def dashboard() -> None:
    df = summary()
    if df.empty:
        st.info("No final-run data in this deployment.")
        return
    df["model_name"] = df["model"].map(lambda m: SHORT.get(m, m))
    st.markdown("Aggregated over the final run (105 questions, 4 models, up to 11 systems). Accuracy is the automatic score, 0–1.")
    c1, c2 = st.columns([1, 1])
    src = c1.selectbox(
        "Questions",
        ["table", "pdf", "summary"],
        format_func={"table": "check-ups (table)", "pdf": "inpatient PDFs", "summary": "whole-record summaries"}.get,
    )
    acc = df[(df["source"] == src) & df["accuracy"].notna()].copy()
    acc["system_name"] = acc["system"].map(lambda s: SYSTEM_EN.get(s, s))
    st.subheader("Accuracy by system")
    st.altair_chart(
        alt.Chart(acc)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            y=alt.Y("system_name:N", title=None, sort=list(SYSTEM_EN.values()), axis=alt.Axis(labelLimit=360)),
            x=alt.X("accuracy:Q", scale=alt.Scale(domain=[0, 1]), title="accuracy"),
            yOffset="model_name:N",
            color=alt.Color("model_name:N", scale=alt.Scale(domain=[SHORT.get(m, m) for m in MODEL_COLORS], range=list(MODEL_COLORS.values())), title="model"),
            tooltip=["model_name", "system_name", alt.Tooltip("accuracy:Q", format=".2f"), "episodes"],
        )
        .properties(height=max(200, 34 * len(acc))),
        use_container_width=True,
    )  # fmt: skip
    system = c2.selectbox("System for tool use", [s for s in SYSTEM_EN if s in set(df["system"]) and s not in ("record", "record_chunked", "code")],
                          format_func=lambda s: SYSTEM_EN[s])  # fmt: skip
    use = df[(df["system"] == system) & (df["source"] == src)]
    rows = [{"model": r.model_name, "tool": t, "calls": n / r.episodes} for r in use.itertuples() for t, n in (r.tool_calls or {}).items()]
    left, right = st.columns(2)
    with left:
        st.subheader("Which tools the models call")
        if rows:
            st.altair_chart(
                alt.Chart(pd.DataFrame(rows))
                .mark_rect(cornerRadius=3)
                .encode(
                    x=alt.X("model:N", title=None),
                    y=alt.Y("tool:N", title=None, sort="-color"),
                    color=alt.Color("calls:Q", scale=alt.Scale(range=["#f3eeff", "#791cbe"]), title="calls per episode"),
                    tooltip=["model", "tool", alt.Tooltip("calls:Q", format=".2f")],
                )
                .properties(height=360),
                use_container_width=True,
            )  # fmt: skip
    with right:
        st.subheader("Harness interventions per episode")
        hrows = [{"model": r.model_name, "intervention": HARNESS_EN.get(h, h).replace("harness: ", ""), "per episode": n / r.episodes}
                 for r in use.itertuples() for h, n in (r.harness or {}).items() if h]  # fmt: skip
        if hrows:
            st.altair_chart(
                alt.Chart(pd.DataFrame(hrows))
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    y=alt.Y("intervention:N", title=None, axis=alt.Axis(labelLimit=300)),
                    x=alt.X("per episode:Q"),
                    yOffset="model:N",
                    color=alt.Color("model:N", scale=alt.Scale(domain=[SHORT.get(m, m) for m in MODEL_COLORS], range=list(MODEL_COLORS.values()))),
                    tooltip=["model", "intervention", alt.Tooltip("per episode:Q", format=".2f")],
                )
                .properties(height=360),
                use_container_width=True,
            )  # fmt: skip
        else:
            st.caption("No harness interventions for this system.")
    st.subheader("Effort and cost")
    table = use[["model_name", "episodes", "accuracy", "calls_per_episode", "invalid_per_episode", "median_seconds"]].rename(
        columns={
            "model_name": "model",
            "calls_per_episode": "tool calls / episode",
            "invalid_per_episode": "invalid calls / episode",
            "median_seconds": "median time, s",
        }
    )
    st.dataframe(table.round(2), hide_index=True, use_container_width=True)


tab_ask, tab_dash, tab_log, tab_data = st.tabs(["Ask", "Tool-use dashboard", "Recorded runs", "Patient record"])

with tab_ask:
    qs = questions()
    labels = {
        q[
            "id"
        ]: f"{'★ ' if q['id'] in SHOWCASE else ''}{q['patient']} · {CATEGORY_EN.get(q['category'], q['category'])} · {q['question'][:90]}"
        for q in qs
    }
    pick = st.selectbox(
        "Example from our test set (★ = showcase)",
        ["free"] + [q["id"] for q in qs],
        format_func=lambda k: "Your own question" if k == "free" else labels[k],
    )
    preset = next((q for q in qs if q["id"] == pick), None)
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.8, 0.8])
    pid = c1.selectbox("Patient", PATIENTS, index=PATIENTS.index(preset["patient"]) if preset else 0, format_func=patient_label)
    model = c2.selectbox("Model", MODELS)
    systems = [s for s in SYSTEMS if not (PUBLIC and s.startswith("record"))]
    default = "report" if preset and preset["category"] == "summary" else "tools"
    system = c3.selectbox("System", systems, index=systems.index(default), format_func=lambda s: SYSTEM_EN.get(s, s))
    budget = c4.number_input("Tool-call budget", 1, 8, 4)
    q = st.text_area(
        "Doctor’s question (Russian)", value=preset["question"] if preset else "", placeholder="Например: какой уровень ЛПНП у пациента?"
    )
    if st.button("Run", type="primary", disabled=not q.strip()):
        with st.spinner("The agent is working…"):
            st.session_state["last"] = run_episode(backend(model), get_patient(pid), q, system, budget=int(budget))
    if "last" in st.session_state:
        show_episode(st.session_state["last"])

with tab_dash:
    dashboard()

with tab_log:
    eps = episodes()
    if not eps:
        st.info("No recorded final-run episodes in this deployment.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        m = c1.selectbox("Model ", sorted({e["model"] for e in eps}), format_func=lambda x: SHORT.get(x, x))
        c = c2.selectbox("System ", sorted({e["condition"] for e in eps if e["model"] == m}), format_func=lambda x: SYSTEM_EN.get(x, x))
        cat = c3.selectbox("Category", ["all"] + sorted({e["category"] for e in eps}), format_func=lambda x: CATEGORY_EN.get(x, x))
        only_bad = c4.checkbox("Errors only")
        sel = [e for e in eps if e["model"] == m and e["condition"] == c and cat in ("all", e["category"])
               and (not only_bad or (e["score"]["correct"] or 0) < 1)]  # fmt: skip
        if sel:
            names = [
                f"{e['qid']} [{CATEGORY_EN.get(e['category'], e['category'])}] {e['score']['correct']} — {e['question'][:70]}" for e in sel
            ]
            i = st.selectbox("Episode", range(len(sel)), format_func=lambda k: names[k])
            st.markdown(f"**Question:** {sel[i]['question']}")
            # The same question under the other systems, for a side-by-side comparison.
            others = [e for e in eps if e["qid"] == sel[i]["qid"] and e["model"] == m and e["condition"] != c]
            if others:
                st.caption(
                    "Same question, other systems: "
                    + "; ".join(f"{SYSTEM_EN.get(e['condition'], e['condition'])} → {e['score']['correct']}" for e in others)
                )
            show_episode(sel[i])
        else:
            st.info("No episodes match the filter.")

with tab_data:
    pid2 = st.selectbox("Patient ", PATIENTS, format_func=patient_label)
    df = records_df(pid2)
    flt = st.text_input("Filter")
    if flt:
        df = df[df.apply(lambda r: flt.lower() in " ".join(map(str, r.values)).lower(), axis=1)]
    st.dataframe(df, hide_index=True, use_container_width=True, height=600)
