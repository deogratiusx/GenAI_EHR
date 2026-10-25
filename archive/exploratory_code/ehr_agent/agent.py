"""Tool-use loop: the model picks tools and arguments; the harness validates, executes, logs.

Budget = max number of non-final tool calls per question. Invalid calls count toward
the budget (as they cost a model round-trip). Once the budget is spent the backend
is forced to call ``final_answer``.
"""

from __future__ import annotations

import json
import re
import time

from .data import Patient
from .tools import build_toolset, run_tool, serialize_full, validate_args

SYSTEM = """Ты — ассистент врача частной клиники. Работаешь с электронной медицинской картой одного пациента (чекап: анализы, УЗИ, осмотры специалистов).
Правила:
1. Числа, даты, диагнозы бери только из карты. Ничего не придумывай и не бери нормы из памяти, если инструмент вернул интервал.
2. Каждое утверждение подкрепляй id строк карты (формат P00X:rN) в поле citations.
3. Если данных в карте нет — status="not_found". Если данные противоречат друг другу или выглядят недостоверными (например, сдвиг значений при выгрузке) — укажи это и поставь status="uncertain".
4. Отвечай кратко, по-русски, с единицами измерения. Ответ всегда завершай вызовом final_answer."""

# T8: no source-specific id format in the prompt (the v1 example "P00X:rN" leaked into PDF citations).
SYSTEM_V3 = SYSTEM.replace(
    "2. Каждое утверждение подкрепляй id строк карты (формат P00X:rN) в поле citations.",
    "2. Каждое утверждение подкрепляй id строк в поле citations. Копируй id точно в том виде, в каком он приведён в поле id "
    "вывода инструмента; не придумывай и не сокращай id.")
assert SYSTEM_V3 != SYSTEM

CONDITION_NOTE = {
    "domain": "У тебя есть медицинские инструменты для поиска по карте. Лимит: {budget} вызова(ов) инструментов, затем final_answer.",
    "domain_ri": "У тебя есть медицинские инструменты для поиска по карте; нормы учитывают пол и возраст пациента. Лимит: {budget} вызова(ов) инструментов, затем final_answer.",
    "domain_v2": "У тебя есть медицинские инструменты для поиска по карте. Лимит: {budget} вызова(ов) инструментов, затем final_answer.",
    "domain_v4a": "У тебя есть медицинские инструменты для поиска по карте. Лимит: {budget} вызова(ов) инструментов, затем final_answer. Ответ проверяется: ссылки должны существовать, «нет данных» перепроверяется.",
    "domain_v4": "У тебя есть медицинские инструменты для поиска по карте, включая назначения (get_medications) и исследования (get_studies). Лимит: {budget} вызова(ов) инструментов, затем final_answer. Ответ проверяется: ссылки должны существовать, «нет данных» перепроверяется.",
    "domain_v41a": "У тебя есть медицинские инструменты для поиска по карте. Лимит: {budget} вызова(ов) инструментов, затем final_answer. Ответ проверяется: ссылки должны существовать, «нет данных» перепроверяется.",
    "domain_v41": "У тебя есть медицинские инструменты для поиска по карте, включая назначения (get_medications) и исследования (get_studies). Лимит: {budget} вызова(ов) инструментов, затем final_answer. Ответ проверяется: ссылки должны существовать, «нет данных» перепроверяется.",
    "domain_v3r": "У тебя есть медицинские инструменты для поиска по карте. Результат подходящего инструмента по вопросу уже приведён ниже — проверь, отвечает ли он на вопрос, и при необходимости вызывай инструменты. Лимит: {budget} вызова(ов) инструментов, затем final_answer.",
    "domain_v3p": "У тебя есть медицинские инструменты для поиска по карте. Результат поиска по тексту вопроса уже приведён ниже — используй его и при необходимости вызывай инструменты. Лимит: {budget} вызова(ов) инструментов, затем final_answer.",
    "domain_v3": "У тебя есть медицинские инструменты для поиска по карте. Лимит: {budget} вызова(ов) инструментов, затем final_answer.",
    "generic": "У тебя есть инструмент run_python с DataFrame df (строки карты). Лимит: {budget} вызова(ов), затем final_answer.",
    "hybrid": "Вся карта пациента приведена ниже (формат: id | дата | источник | группа | параметр = значение). Дополнительно доступны медицинские инструменты (справочник норм, проверка согласованности и др.). Лимит: {budget} вызова(ов), затем final_answer.",
    "fullctx": "Вся карта пациента приведена ниже (формат: id | дата | источник | группа | параметр = значение). Сразу вызови final_answer.",
}


HARNESS_CONDITIONS = ("domain_v4a", "domain_v4")
_ID_RX = re.compile(r"(?:P00\d|K0\d):(?:r\d+|p\d+:[bl]\d+)")
_NOT_FOUND_TEXT = re.compile(r"нет\s+данн|не\s+найден|отсутству|не\s+указан|не\s+выполн|не\s+проводил|не\s+обнаружен", re.I)


def rescue_text_final(content: str) -> dict | None:
    """T5: a final answer written as text ('final_answer(answer=..., citations=[...])') is parsed, not discarded."""
    if "final_answer" not in content:
        return None
    body = content.split("final_answer", 1)[1]
    m = re.search(r"answer\s*[=:]\s*[\"'«]?(.+?)[\"'»]?\s*,\s*(?:\[|citations|\{)", body, re.S)
    answer = m.group(1) if m else re.sub(r"[(){}\[\]\"']", " ", body)
    status = "not_found" if "not_found" in body else "uncertain" if "uncertain" in body else "answered"
    return {"answer": answer.strip()[:1500], "citations": sorted(set(_ID_RX.findall(body))), "status": status}


def harness_check(final: dict, patient: Patient, question: str, fns: dict, done: set[str]) -> dict | None:
    """T5: verify citations and 'not found' before accepting the answer. Returns a message for the model or None."""
    if "cite" not in done:
        ids = {r.id for r in patient.records}
        cites = [str(c).strip() for c in (final.get("citations") or [])]
        bad = [c for c in cites if c not in ids]
        if bad or (not cites and final.get("status", "answered") == "answered"):
            done.add("cite")
            return {"error": "Проверка ссылок не пройдена.",
                    "invalid_ids": bad[:10], "missing": not cites,
                    "fix": "Укажите в citations id строк, скопированные точно из вывода инструментов (поле id), и вызовите final_answer снова."}
    absent = final.get("status") == "not_found" or bool(_NOT_FOUND_TEXT.search(str(final.get("answer", ""))))
    if absent and "absence" not in done:
        done.add("absence")
        found = {}
        name, args = route_intent(question)
        for n, a in {name: args, "search_notes": {"query": question, "top_k": 6}}.items():
            if n in fns:
                res = run_tool(fns, n, a)
                if not _is_empty(res):
                    found[n] = res
        if found:
            return {"check": "Перед ответом «нет данных» обвязка выполнила широкий поиск и нашла возможные данные.",
                    "results": found,
                    "fix": "Проверьте эти результаты. Если они отвечают на вопрос — дайте ответ по ним; если нет — подтвердите status=not_found."}
    return None


HARNESS_V41 = ("domain_v41a", "domain_v41")
_META_RX = re.compile(r"id\s+строк|укаж\w*\s+id|добав\w*\s+id|повторя|вызов\w*\s+final_answer|final_answer\s+снова", re.I)


def rescue_text_final_v41(content: str) -> dict | None:
    """T5.1: robust parse of a text-written final answer (special quote tokens, JSON, answer before the keyword)."""
    if "final_answer" not in content:
        return None
    t = content.replace("<|superquote|>", '"')
    before, after = t.split("final_answer", 1)
    m = re.search(r"\{.*\}", after, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and str(obj.get("answer", "")).strip():
                return {"answer": str(obj["answer"])[:1500], "citations": [str(c) for c in obj.get("citations") or [] if _ID_RX.fullmatch(str(c))],
                        "status": obj.get("status", "answered") if obj.get("status") in ("answered", "not_found", "uncertain") else "answered"}
        except json.JSONDecodeError:
            pass
    m = re.search(r"answer\s*[=:]\s*[\"'«]?(.+?)[\"'»]?\s*,\s*(?:\[|citations|\{|\")", after, re.S)
    answer = m.group(1) if m else ""
    if len(answer.strip()) < 15:
        answer = before if len(before.strip()) >= len(answer.strip()) else answer
    status = "not_found" if "not_found" in t else "uncertain" if "uncertain" in t else "answered"
    answer = re.sub(r"status\s*=\s*\w+|[{}\[\]\"]", " ", answer).strip()
    return {"answer": answer[:1500], "citations": sorted(set(_ID_RX.findall(t))), "status": status}


def merge_after_harness(first: dict | None, new: dict) -> dict:
    """T5.1: a reply to a harness message that only talks about ids keeps the first substantive answer."""
    if first and (_META_RX.search(str(new.get("answer", ""))) or len(str(new.get("answer", "")).strip()) < 20):
        return {**first, "citations": sorted(set(first.get("citations") or []) | set(new.get("citations") or [])),
                "status": first.get("status", "answered")}
    return new


def route_intent(question: str) -> tuple[str, dict]:
    """Deterministic intent router for the prefetch step; falls back to text search."""
    import re

    from .refs import CATALOG, resolve_analyte
    q = question.lower().replace("ё", "е")
    if re.search(r"рост|вес\b|масс\w* тела", q) and re.search(r"согласу|расхожд|разн|противореч", q):
        return "check_consistency", {}
    if re.search(r"мкб|код\w* диагноз|диагноз\w*.{0,20}код", q):
        return "get_diagnoses", {}
    if re.search(r"вне\s+нормы|за\s+пределы|отклон|референсн", q):
        panel = next((p for p in ("липид", "печен", "общего анализа крови", "щитовид", "почеч", "коагул") if p in q), None)
        return "list_abnormal_labs", ({"panel": panel} if panel else {})
    m = re.search(r"«([^»]+)»", question)
    cand = m.group(1) if m else None
    if not cand:
        names = sorted({n for a in CATALOG for n in (a.name.lower().split(" (")[0], *a.aliases) if len(n) > 2}, key=len, reverse=True)
        # names may carry a case ending in the question ("гемоглобина"); short abbreviations must match whole
        cand = next((n for n in names if re.search(rf"(?<![а-яa-z]){re.escape(n)}" + (r"[а-я]{0,3}(?![а-яa-z])" if len(n) > 4 else r"(?![а-яa-z])"), q)), None)
    if cand and resolve_analyte(cand) and re.search(r"уров|значен|какой|какая|сколько|показател", q):
        return "get_lab", {"analyte": cand}
    return "search_notes", {"query": question, "top_k": 6}


def _is_empty(result: dict) -> bool:
    if "error" in result or result.get("found") is False:
        return True
    for key in ("results", "items", "issues"):
        if key in result and not result[key]:
            return True
    if "stdout" in result and not str(result["stdout"]).strip():
        return True
    return False


def run_episode(backend, patient: Patient, question: str, condition: str, budget: int = 4) -> dict:
    specs, fns = build_toolset(condition, patient)
    by_name = {s["name"]: s for s in specs}
    system = (SYSTEM_V3 if condition.startswith(("domain_v3", "domain_v4")) else SYSTEM) + "\n" + CONDITION_NOTE[condition].format(budget=budget)
    user = question
    if condition in ("fullctx", "hybrid"):
        user = f"КАРТА ПАЦИЕНТА {patient.pid}:\n{serialize_full(patient)}\n\nВОПРОС: {question}"
    messages: list[dict] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    trace: list[dict] = []
    if condition == "domain_v3r":
        # Intent-routed prefetch (T6-lite): structured questions get the typed tool, text questions get search.
        name, args = route_intent(question)
        pre = run_tool(fns, name, args)
        trace.append({"type": "prefetch", "name": name, "arguments": args, "result": pre})
        messages[1]["content"] = (f"{question}\n\nПРЕДВАРИТЕЛЬНЫЙ РЕЗУЛЬТАТ ({name} {json.dumps(args, ensure_ascii=False)}):\n"
                                  + json.dumps(pre, ensure_ascii=False))
    if condition == "domain_v3p":
        # Prefetch: the harness runs search_notes on the question itself, so retrieval quality does not
        # depend on the model choosing the search tool (the weak-model bottleneck seen with v3).
        pre = run_tool(fns, "search_notes", {"query": question, "top_k": 6})
        trace.append({"type": "prefetch", "name": "search_notes", "arguments": {"query": question}, "result": pre})
        messages[1]["content"] = f"{question}\n\nРЕЗУЛЬТАТ ПОИСКА ПО КАРТЕ (search_notes по тексту вопроса):\n" + json.dumps(pre, ensure_ascii=False)

    calls, invalid, nudges = 0, 0, 0
    harness = condition in HARNESS_CONDITIONS
    harness41 = condition in HARNESS_V41
    first_final: dict | None = None
    checks: set[str] = set()
    seen: dict[str, dict] = {}
    final = None
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    t_start = time.time()
    max_turns = budget + 6 if harness or harness41 else budget + 4

    for _turn in range(max_turns):
        force = condition != "fullctx" and calls >= budget
        try:
            reply = backend.chat(messages, specs, force_final=force)
        except Exception as exc:
            trace.append({"type": "backend_error", "error": str(exc)[:300]})
            break
        for k in usage:
            usage[k] += int(reply.usage.get(k, 0) or 0)
        tc = reply.tool_call
        if tc is None:
            # Model answered in plain text. Nudge once to use final_answer, then accept the text.
            trace.append({"type": "text", "content": reply.content[:1500], "latency": round(reply.latency, 2)})
            if harness41:
                rescued = rescue_text_final_v41(reply.content)
                if rescued:
                    rescued = merge_after_harness(first_final, rescued)
                    first_final = first_final or rescued
                    msg = harness_check(rescued, patient, question, fns, checks | {"cite"})  # no citation loop for text answers
                    trace.append({"type": "harness", "action": "rescued_text_final", "message": msg})
                    if msg is None:
                        final = {**rescued, "format": "rescued_text"}
                        break
                    messages.append({"role": "assistant", "content": reply.content})
                    messages.append({"role": "user", "content": json.dumps(msg, ensure_ascii=False)})
                    continue
            if harness:
                rescued = rescue_text_final(reply.content)
                if rescued:
                    msg = harness_check(rescued, patient, question, fns, checks)
                    trace.append({"type": "harness", "action": "rescued_text_final", "message": msg})
                    if msg is None:
                        final = {**rescued, "format": "rescued_text"}
                        break
                    messages.append({"role": "assistant", "content": reply.content})
                    messages.append({"role": "user", "content": json.dumps(msg, ensure_ascii=False)})
                    continue
            if nudges == 0 and reply.content.strip():
                nudges += 1
                messages.append({"role": "assistant", "content": reply.content})
                messages.append({"role": "user", "content": "Оформи ответ вызовом final_answer с citations и status."})
                continue
            final = {"answer": reply.content, "citations": [], "status": "answered", "format": "plain_text"}
            break

        name = tc.get("name") or ""
        step = {"type": "call", "name": name, "arguments": tc.get("arguments"), "latency": round(reply.latency, 2),
                "thought": (reply.content or "")[:400]}
        messages.append({"role": "assistant", "content": reply.content or "", "tool_call": tc,
                         "functions_state_id": tc.get("functions_state_id")})

        if name == "final_answer":
            args, err = validate_args(by_name["final_answer"], tc.get("arguments"))
            if err and nudges == 0:
                nudges += 1
                invalid += 1
                step.update(valid=False, error=err)
                trace.append(step)
                messages.append({"role": "tool", "name": name, "content": json.dumps({"error": err}, ensure_ascii=False)})
                continue
            if err:
                raw = tc.get("arguments")
                raw = raw if isinstance(raw, dict) else {}
                args = {"answer": str(raw.get("answer", raw)), "citations": raw.get("citations") or [], "status": raw.get("status", "answered")}
            step.update(valid=err is None)
            trace.append(step)
            if harness:
                msg = harness_check(args, patient, question, fns, checks)
                if msg is not None:
                    trace.append({"type": "harness", "action": "verify", "message": msg})
                    messages.append({"role": "tool", "name": name, "content": json.dumps(msg, ensure_ascii=False)})
                    continue
            if harness41:
                args = merge_after_harness(first_final, args)
                first_final = first_final or args
                msg = harness_check(args, patient, question, fns, checks)
                if msg is not None:
                    trace.append({"type": "harness", "action": "verify", "message": msg})
                    messages.append({"role": "tool", "name": name, "content": json.dumps(msg, ensure_ascii=False)})
                    continue
            final = {**args, "format": "final_answer"}
            break

        calls += 1
        if name not in fns:
            err = f"Инструмента '{name}' нет. Доступны: {sorted(fns)} и final_answer."
            invalid += 1
            step.update(valid=False, error=err)
            result = {"error": err}
        else:
            args, err = validate_args(by_name[name], tc.get("arguments"))
            if err:
                invalid += 1
                step.update(valid=False, error=err)
                result = {"error": err}
            else:
                key = name + json.dumps(args, ensure_ascii=False, sort_keys=True)
                if key in seen:
                    result = {**seen[key], "note_repeat": "Повторный вызов с теми же аргументами — используйте уже полученный результат."}
                    step["repeat"] = True
                else:
                    t0 = time.time()
                    result = run_tool(fns, name, args)
                    step["tool_time"] = round(time.time() - t0, 3)
                    seen[key] = result
                step.update(valid=True, arguments=args)
        step["empty"] = _is_empty(result)
        payload = json.dumps(result, ensure_ascii=False)
        if step["empty"] and "error" not in result:
            result = {**result, "hint": "Результат пуст. Это может означать неверный запрос или фильтр: проверьте формулировку или используйте другой инструмент. Если данных действительно нет — status=not_found."}
            payload = json.dumps(result, ensure_ascii=False)
        step["result_chars"] = len(payload)
        step["result"] = result
        trace.append(step)
        messages.append({"role": "tool", "name": name, "content": payload})

    if final is None:
        final = {"answer": "", "citations": [], "status": "no_answer", "format": "none"}
    return {
        "patient": patient.pid, "question": question, "condition": condition, "budget": budget,
        "model": getattr(backend, "model", "?"), "final": final, "trace": trace,
        "n_calls": calls, "n_invalid": invalid, "usage": usage, "wall": round(time.time() - t_start, 2),
    }
