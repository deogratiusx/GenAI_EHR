"""The agent loop: the model picks tools and arguments; the harness validates, executes, checks, logs.

One episode = one question about one patient under one system configuration (`systems.py`).
Budget = maximum number of non-final tool calls; invalid calls count toward it (they cost a model
round-trip). When it is spent the backend is forced to call `final_answer`. Every step is written to
the trace (tool calls with arguments and results, text replies, harness interventions).
"""

from __future__ import annotations

import json
import time

from ..sources import Patient
from ..tools import build_toolset, is_empty, run_tool, validate_args
from ..tools.baselines import serialize_record
from ..tools.summary import REPORT_SECTIONS, merge_reports, render_report
from . import prompts
from .chunked import map_record
from .harness import check_answer, check_report, exposed_tools, keep_substantive, rescue_text_final, route_intent
from .systems import SystemConfig, get_system

CONTEXT_OVERFLOW = ("context too long", "maximum context", "context length")


def _history_text(text: str) -> str:
    """Assistant text echoed into the dialogue history. A text that mentions `final_answer` makes the
    GigaChat API reject the history ("undefined functions in dialog history"), so the name is replaced."""
    return text.replace("final_answer", "итоговый ответ")


EMPTY_HINT = (
    "Результат пуст. Это может означать неверный запрос или фильтр: проверьте формулировку или используйте "
    "другой инструмент. Если данных действительно нет — status=not_found."
)


def _system_prompt(cfg: SystemConfig, budget: int) -> str:
    if cfg.toolset == "code":
        note = prompts.NOTE_CODE
    elif cfg.chunked:
        note = prompts.NOTE_CHUNKED
    elif cfg.record_in_prompt:
        note = prompts.NOTE_RECORD_TOOLS if cfg.toolset != "none" else prompts.NOTE_RECORD
    else:
        note = prompts.NOTE_TOOLS
    note = note.format(budget=budget)
    if cfg.report:
        note += prompts.NOTE_REPORT
    if cfg.checks:
        note += prompts.NOTE_CHECKS
    if cfg.prefetch:
        note += prompts.NOTE_PREFETCH
    return prompts.SYSTEM + "\n" + note


def run_episode(backend, patient: Patient, question: str, system: str | SystemConfig = "tools", budget: int = 4) -> dict:
    cfg = get_system(system) if isinstance(system, str) else system
    only = None
    if cfg.routed_tools:
        from ..tools import TOOLSETS

        only = exposed_tools(question, TOOLSETS[cfg.toolset])
    specs, fns = build_toolset(cfg.toolset, patient, only=only)
    by_name = {s["name"]: s for s in specs}

    t_start = time.time()
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    trace: list[dict] = []
    map_error = None
    user = question
    seen_text = ""  # what the model has seen from the record (for the number check)
    if cfg.record_in_prompt:
        record = serialize_record(patient)
        user, seen_text = f"КАРТА ПАЦИЕНТА {patient.pid}:\n{record}\n\nВОПРОС: {question}", record
    elif cfg.chunked:
        # Map stage: one call per chunk; the reduce step below is the ordinary loop over the excerpts.
        notes, trace, usage, map_error = map_record(backend, patient, question)
        user, seen_text = f"ВЫПИСКИ ИЗ КАРТЫ ПАЦИЕНТА {patient.pid}:\n{notes}\n\nВОПРОС: {question}", notes
    messages: list[dict] = [{"role": "system", "content": _system_prompt(cfg, budget)}, {"role": "user", "content": user}]
    if cfg.report and "build_dossier" in fns:
        # Report systems start from the dossier: recall is the code's job, not the model's tool choice.
        dossier = run_tool(fns, "build_dossier", {})
        trace.append({"type": "prefetch", "name": "build_dossier", "arguments": {}, "result": dossier})
        seen_text += " " + json.dumps(dossier, ensure_ascii=False)
        messages[1]["content"] += "\n\nДОСЬЕ КАРТЫ (build_dossier):\n" + json.dumps(dossier, ensure_ascii=False)
    if cfg.prefetch:
        _, name, args = route_intent(question)
        if name in fns:
            pre = run_tool(fns, name, args)
            trace.append({"type": "prefetch", "name": name, "arguments": args, "result": pre})
            seen_text += " " + json.dumps(pre, ensure_ascii=False)
            messages[1]["content"] += f"\n\nПРЕДВАРИТЕЛЬНЫЙ РЕЗУЛЬТАТ ({name} {json.dumps(args, ensure_ascii=False)}):\n" + json.dumps(
                pre, ensure_ascii=False
            )

    calls = invalid = 0
    nudged = False
    checks_done: set[str] = set()
    first_final: dict | None = None
    cache: dict[str, dict] = {}
    final = None
    last_report: dict | None = None  # report systems: the latest sectioned report, merged across repairs
    tool_mode = bool(fns)

    def accept(candidate: dict, rescued: bool = False) -> tuple[dict, dict | None]:
        """Harness checks on a candidate answer → (answer to keep, message for the model or None if accepted)."""
        nonlocal first_final
        if not cfg.checks:
            return candidate, None
        candidate = keep_substantive(first_final, candidate)
        first_final = first_final or candidate
        msg = check_answer(candidate, patient, question, fns, seen_text, checks_done, rescued=rescued)
        if msg is None and cfg.report_checks and "build_dossier" in fns:
            msg = check_report(candidate, fns["build_dossier"].__self__, checks_done)
        return candidate, msg

    for _turn in range(0 if map_error else budget + 6):  # a failed map stage leaves its backend_error in the trace
        force = tool_mode and calls >= budget
        try:
            reply = backend.chat(messages, specs, force_final=force)
        except Exception as exc:
            if any(m in str(exc) for m in CONTEXT_OVERFLOW):
                # The record does not fit the model's window: the condition is not applicable, not a failure to retry.
                trace.append({"type": "context_overflow", "error": str(exc)[:300]})
                final = {"answer": "", "citations": [], "status": "context_overflow", "format": "none"}
            else:
                trace.append({"type": "backend_error", "error": str(exc)[:300]})
            break
        for k in usage:
            usage[k] += int(reply.usage.get(k, 0) or 0)
        tc = reply.tool_call

        # ---------------------------------------------------------------- text reply
        if tc is None:
            trace.append({"type": "text", "content": reply.content[:1500], "latency": round(reply.latency, 2)})
            messages.append({"role": "assistant", "content": _history_text(reply.content)})
            if cfg.format_repair and not nudged:
                # Reasoning and formatting decoupled: the text answer is converted by a forced final_answer call.
                nudged = True
                messages.append({"role": "user", "content": prompts.FORMAT_REPAIR})
                try:
                    rep = backend.chat(messages, specs, force_final=True)
                except Exception as exc:
                    trace.append({"type": "backend_error", "error": str(exc)[:300]})
                    break
                for k in usage:
                    usage[k] += int(rep.usage.get(k, 0) or 0)
                trace.append({"type": "harness", "action": "format_repair", "ok": bool(rep.tool_call)})
                if rep.tool_call and rep.tool_call.get("name") == "final_answer":
                    tc, reply = rep.tool_call, rep
            if tc is None:
                rescued = rescue_text_final(reply.content) if cfg.checks or cfg.format_repair else None
                if rescued:
                    kept, msg = accept(rescued, rescued=True)
                    trace.append({"type": "harness", "action": "rescued_text_final", "message": msg})
                    if msg is None:
                        final = {**kept, "format": "rescued_text"}
                        break
                    messages.append({"role": "user", "content": json.dumps(msg, ensure_ascii=False)})
                    continue
                if not nudged and reply.content.strip():
                    nudged = True
                    messages.append({"role": "user", "content": "Оформи ответ вызовом final_answer с citations и status."})
                    continue
                final = {"answer": reply.content, "citations": [], "status": "answered", "format": "plain_text"}
                break

        # ---------------------------------------------------------------- tool call
        name = tc.get("name") or ""
        step = {
            "type": "call",
            "name": name,
            "arguments": tc.get("arguments"),
            "latency": round(reply.latency, 2),
            "thought": (reply.content or "")[:400],
        }
        messages.append(
            {
                "role": "assistant",
                "content": _history_text(reply.content or ""),
                "tool_call": tc,
                "functions_state_id": tc.get("functions_state_id"),
            }
        )

        if name == "final_answer":
            args, err = validate_args(by_name["final_answer"], tc.get("arguments"))
            if err and not nudged:
                nudged = True
                invalid += 1
                step.update(valid=False, error=err)
                trace.append(step)
                messages.append({"role": "tool", "name": name, "content": json.dumps({"error": err}, ensure_ascii=False)})
                continue
            if err:
                raw = tc.get("arguments") if isinstance(tc.get("arguments"), dict) else {}
                args = {
                    **{k: str(raw[k]) for k in REPORT_SECTIONS if raw.get(k)},
                    "answer": str(raw.get("answer", "" if "key_findings" in raw else raw)),
                    "citations": raw.get("citations") or [],
                    "status": raw.get("status", "answered"),
                }
            if "key_findings" in args and not args.get("answer"):  # a sectioned report → one document
                args = merge_reports(last_report, {**args, "answer": render_report(args)})
                if not args["answer"]:  # JSON debris or placeholders only: not a report yet
                    invalid += 1
                    step.update(valid=False, error="empty report")
                    trace.append(step)
                    messages.append(
                        {
                            "role": "tool",
                            "name": name,
                            "content": json.dumps(
                                {
                                    "error": "Отчёт пуст: разделы не заполнены текстом.",
                                    "fix": "Заполните разделы отчёта по досье и вызовите final_answer снова.",
                                },
                                ensure_ascii=False,
                            ),
                        }
                    )
                    continue
                last_report = args
            step.update(valid=err is None)
            trace.append(step)
            kept, msg = accept(args)
            if msg is not None:
                trace.append({"type": "harness", "action": "verify", "message": msg})
                messages.append({"role": "tool", "name": name, "content": json.dumps(msg, ensure_ascii=False)})
                continue
            final = {**kept, "format": "final_answer"}
            break

        calls += 1
        if name not in fns:
            invalid += 1
            result = {"error": f"Инструмента '{name}' нет. Доступны: {sorted(fns)} и final_answer."}
            step.update(valid=False, error=result["error"])
        else:
            args, err = validate_args(by_name[name], tc.get("arguments"))
            if err:
                invalid += 1
                result = {"error": err}
                step.update(valid=False, error=err)
            else:
                key = name + json.dumps(args, ensure_ascii=False, sort_keys=True)
                if key in cache:
                    result = {**cache[key], "note_repeat": "Повторный вызов с теми же аргументами — используйте уже полученный результат."}
                    step["repeat"] = True
                else:
                    t0 = time.time()
                    result = cache[key] = run_tool(fns, name, args)
                    step["tool_time"] = round(time.time() - t0, 3)
                step.update(valid=True, arguments=args)
        step["empty"] = is_empty(result)
        if step["empty"] and "error" not in result:
            result = {**result, "hint": EMPTY_HINT}
        payload = json.dumps(result, ensure_ascii=False)
        seen_text += " " + payload
        step["result_chars"] = len(payload)
        step["result"] = result
        trace.append(step)
        messages.append({"role": "tool", "name": name, "content": payload})

    if final is None:
        final = {"answer": "", "citations": [], "status": "no_answer", "format": "none"}
    return {
        "patient": patient.pid,
        "question": question,
        "condition": cfg_name(system),
        "budget": budget,
        "model": getattr(backend, "model", "?"),
        "final": final,
        "trace": trace,
        "n_calls": calls,
        "n_invalid": invalid,
        "n_chunks": sum(s["type"] == "map" for s in trace),  # map-stage model calls (record_chunked only)
        "usage": usage,
        "wall": round(time.time() - t_start, 2),
    }


def cfg_name(system: str | SystemConfig) -> str:
    return system if isinstance(system, str) else repr(system)
