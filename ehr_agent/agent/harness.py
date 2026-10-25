"""The harness around the model: intent routing and answer checks (model-agnostic, deterministic).

* `route_intent` — classify the question (lab value / out-of-range panel / consistency / diagnoses /
  medications / studies / free text). Used to prefetch the right tool and to expose only the relevant
  tools to weak models (fewer choices → fewer selection errors).
* `check_answer` — before an answer is accepted: citations must exist, numbers must occur in the
  record or tool outputs, and a "not found" triggers one broad automatic search. Each check fires at
  most once per episode; its message is a repairable error for the model.
* `check_report` — for report systems: every `must_mention` item of the dossier is covered (or
  explicitly dismissed), and lab verdicts and trend directions agree with the dossier.
* `rescue_text_final` / `keep_substantive` — a final answer written as text is parsed (fallback after
  format repair), and a reply to a harness message that only talks about ids keeps the first
  substantive answer (a weak model otherwise replaces its answer by "here are the ids…").
"""

from __future__ import annotations

import json
import re

from ..knowledge.analytes import CATALOG, resolve_analyte
from ..knowledge.entities import STUDY_KINDS, drug_mentioned
from ..sources import Patient
from ..tools import run_tool
from ..tools.base import is_empty
from ..util import ID_RX, numbers_in

_NOT_FOUND_TEXT = re.compile(r"нет\s+данн|не\s+найден|отсутству|не\s+указан|не\s+выполн|не\s+проводил|не\s+обнаружен", re.I)
_MONTHS = "января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря"
_DATES = re.compile(
    rf"\b\d{{1,2}}[.\-/]\d{{1,2}}[.\-/]\d{{2,4}}\b|\b\d{{4}}-\d{{2}}-\d{{2}}\b|\b\d{{1,2}}\s+(?:{_MONTHS})(?:\s+\d{{4}})?", re.I
)
_LIST_MARKER = re.compile(r"(?m)(?:^|(?<=\s))\d{1,2}[.)](?=\s)")
_META_RX = re.compile(
    r"id\s+строк|укаж\w*\s+(?:\w+\s+)?id|неверн\w*\s+id|добав\w*\s+id|повторя|вызов\w*\s+final_answer|final_answer\s+снова", re.I
)
_ANALYTE_NAMES = sorted({n for a in CATALOG for n in (a.name.lower().split(" (")[0], *a.aliases) if len(n) > 2}, key=len, reverse=True)

# Tools exposed per intent (always plus search_notes and final_answer).
INTENT_TOOLS = {
    "lab": ["get_lab", "get_trend", "list_abnormal_labs", "check_consistency", "calculate"],
    "trend": ["get_trend", "get_lab", "list_abnormal_labs"],
    "abnormal": ["list_abnormal_labs", "get_lab", "check_consistency"],
    "consistency": ["check_consistency", "get_lab"],
    "diagnoses": ["get_diagnoses", "get_conclusions"],
    "medications": ["get_medications", "get_conclusions"],
    "studies": ["get_studies", "get_conclusions", "get_document"],
    "text": ["get_conclusions", "get_overview", "get_document", "get_medications", "get_studies", "get_lab"],
}


def route_intent(question: str) -> tuple[str, str, dict]:
    """(intent, tool to prefetch, its arguments). Deterministic rules; falls back to text search."""
    q = question.lower().replace("ё", "е")
    if re.search(r"рост|вес\b|масс\w* тела", q) and re.search(r"согласу|расхожд|разн|противореч", q):
        return "consistency", "check_consistency", {}
    if re.search(r"мкб|код\w* диагноз|диагноз\w*.{0,20}код", q):
        return "diagnoses", "get_diagnoses", {}
    if re.search(r"вне\s+нормы|за\s+пределы|отклон|референсн", q):
        panel = next((p for p in ("липид", "печен", "общего анализа крови", "щитовид", "почеч", "коагул") if p in q), None)
        return "abnormal", "list_abnormal_labs", ({"panel": panel} if panel else {})
    drug = drug_mentioned(question)
    if drug or re.search(r"препарат|терапи|назнач\w*.{0,20}(лекарств|препарат)|доз[аеуы]\b|лечени", q):
        return "medications", "get_medications", ({"drug": drug} if drug else {})
    trend_q = re.search(r"динамик|как\s+менял|изменял|тренд|нормализ|в\s+течение\s+госпитализ|за\s+госпитализ", q)
    m = re.search(r"«([^»]+)»", question)
    cand = (
        m.group(1)
        if m
        else next(
            (
                n
                for n in _ANALYTE_NAMES  # names may carry a case ending in the question ("гемоглобина")
                if re.search(rf"(?<![а-яa-z]){re.escape(n)}" + (r"[а-я]{0,3}(?![а-яa-z])" if len(n) > 4 else r"(?![а-яa-z])"), q)
            ),
            None,
        )
    )
    if cand and resolve_analyte(cand) and trend_q:
        return "trend", "get_trend", {"analyte": cand}
    if cand and resolve_analyte(cand) and re.search(r"уров|значен|какой|какая|сколько|показател", q):
        return "lab", "get_lab", {"analyte": cand}
    kind = next((k for k, rx in STUDY_KINDS.items() if re.search(rx, q)), None)
    if kind or re.search(r"фракци\w* выброса|\bфв\b", q):
        return "studies", "search_notes", {"query": question, "top_k": 6}
    return "text", "search_notes", {"query": question, "top_k": 6}


def exposed_tools(question: str, available: list[str]) -> list[str]:
    """Dynamic tool exposure: the intent's tools plus search, in the order of `available`."""
    intent, _, _ = route_intent(question)
    keep = set(INTENT_TOOLS[intent]) | {"search_notes"}
    return [n for n in available if n in keep]


# ---------------------------------------------------------------------- answer checks


def check_answer(
    final: dict, patient: Patient, question: str, fns: dict, seen_text: str, done: set[str], rescued: bool = False
) -> dict | None:
    """Return a repairable error for the model, or None to accept the answer.

    `seen_text` — everything the model has seen from the record (tool outputs, or the whole record).
    """
    answer = str(final.get("answer", ""))
    cites = [str(c).strip() for c in (final.get("citations") or [])]
    if "cite" not in done and not rescued:  # no citation loop on text-written answers (see module doc)
        ids = {r.id for r in patient.records}
        bad = [c for c in cites if c not in ids]
        if bad or (not cites and final.get("status", "answered") == "answered"):
            done.add("cite")
            return {
                "error": "Проверка ссылок не пройдена.",
                "invalid_ids": bad[:10],
                "missing": not cites,
                "fix": "Укажите в citations id строк, скопированные точно из вывода инструментов (поле id), и вызовите final_answer снова.",
            }
    if "numbers" not in done:
        by_id = patient.by_id()
        # List markers ("11. ОАК") and dates rewritten by the model ("29 апреля", "17-12-2025") are not values to ground.
        answer = _DATES.sub(" ", _LIST_MARKER.sub(" ", answer))
        source = seen_text + " " + " ".join(by_id[c].value for c in cites if c in by_id)
        unknown = sorted(n for n in numbers_in(answer) - numbers_in(source) if len(n.replace(".", "")) > 1)
        if unknown:
            done.add("numbers")
            return {
                "error": "В ответе есть числа, которых нет ни в процитированных строках, ни в выводе инструментов.",
                "numbers": unknown[:10],
                "fix": "Проверьте эти числа по карте, исправьте ответ и вызовите final_answer снова.",
            }
    absent = final.get("status") == "not_found" or bool(_NOT_FOUND_TEXT.search(answer))
    if absent and "absence" not in done:
        done.add("absence")
        found = {}
        _, name, args = route_intent(question)
        for n, a in {name: args, "search_notes": {"query": question, "top_k": 6}}.items():
            if n in fns:
                res = run_tool(fns, n, a)
                if not is_empty(res):
                    found[n] = res
        if found:
            return {
                "check": "Перед ответом «нет данных» обвязка выполнила широкий поиск и нашла возможные данные.",
                "results": found,
                "fix": "Проверьте эти результаты. Если они отвечают на вопрос — дайте ответ по ним; если нет — подтвердите status=not_found.",
            }
    return None


def rescue_text_final(content: str) -> dict | None:
    """Parse a final answer written as text: JSON after the keyword (GigaChat may emit <|superquote|>
    instead of quotes), `answer=…` arguments, or the text before the keyword."""
    if "final_answer" not in content:
        return None
    t = content.replace("<|superquote|>", '"')
    before, after = t.split("final_answer", 1)
    m = re.search(r"\{.*\}", after, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and str(obj.get("key_findings", "")).strip() and not obj.get("answer"):
                from ..tools.summary import REPORT_SECTIONS, render_report  # a sectioned report written as text

                obj = {
                    **{k: str(obj[k]) for k in REPORT_SECTIONS if obj.get(k)},
                    "answer": render_report(obj),
                    "citations": obj.get("citations"),
                    "status": obj.get("status"),
                }
            if isinstance(obj, dict) and str(obj.get("answer", "")).strip():
                status = obj.get("status") if obj.get("status") in ("answered", "not_found", "uncertain") else "answered"
                return {
                    **{k: v for k, v in obj.items() if k not in ("answer", "citations", "status")},
                    "answer": str(obj["answer"])[:6000],
                    "citations": [str(c) for c in obj.get("citations") or [] if ID_RX.fullmatch(str(c))],
                    "status": status,
                }
        except json.JSONDecodeError:
            pass
    m = re.search(r"answer\s*[=:]\s*[\"'«]?(.+?)[\"'»]?\s*,\s*(?:\[|citations|\{|\")", after, re.S)
    answer = m.group(1) if m else ""
    if len(answer.strip()) < 15:
        answer = before if len(before.strip()) >= len(answer.strip()) else answer
    status = "not_found" if "not_found" in t else "uncertain" if "uncertain" in t else "answered"
    answer = re.sub(r"status\s*=\s*\w+|[{}\[\]\"]", " ", answer).strip()
    if not answer:  # a bare "final_answer" keyword: nothing to rescue
        return None
    return {"answer": answer[:1500], "citations": sorted(set(ID_RX.findall(t))), "status": status}


def keep_substantive(first: dict | None, new: dict) -> dict:
    """A reply that only talks about ids keeps the first substantive answer (citations merged)."""
    if first and (_META_RX.search(str(new.get("answer", ""))) or len(str(new.get("answer", "")).strip()) < 20):
        return {
            **first,
            "citations": sorted(set(first.get("citations") or []) | set(new.get("citations") or [])),
            "status": first.get("status", "answered"),
        }
    return new


# ---------------------------------------------------------------------- report checks

_SAYS_NORMAL = re.compile(r"в\s+норм|в\s+пределах\s+норм|\bнорма\b|нормальн", re.I)
_SAYS_ABNORMAL = re.compile(r"\bвыше|\bниже|вне\s+норм|повыш|пониж|отклон|критич|нормализ|превыш", re.I)
_SAYS_UP = re.compile(r"\bрост|повысил|увелич|вырос|нараст", re.I)
_SAYS_DOWN = re.compile(r"сниж|снизил|уменьш|падени", re.I)


def report_text(final: dict) -> str:
    return " ".join(str(final.get(k) or "") for k in ("answer", "dismissed")) + " " + " ".join(map(str, final.get("citations") or []))


def coverage(final: dict, must: list[dict]) -> tuple[list[dict], list[dict]]:
    """(covered, missing) must-mention items: an item is covered when one of its ids is cited or its
    finding is named in the report (the detection regex of the item)."""
    text = report_text(final)
    low = text.lower()
    covered, missing = [], []
    for m in must:
        hit = any(i in text for i in m["ids"]) or re.search(rf"(?<![а-яa-z]){m['match']}", low, re.I)
        (covered if hit else missing).append(m)
    return covered, missing


def verdict_mismatches(final: dict, labs: list[dict]) -> list[str]:
    """Lab lines whose verdict or trend direction contradicts the dossier."""
    out = []
    lines = [line for line in str(final.get("answer", "")).splitlines() if line.strip()]
    for item in labs:
        stem = re.escape(item["analyte"].lower().split(" (")[0].split()[0][:6])
        for line in lines:
            if not re.search(rf"(?<![а-яa-z]){stem}", line.lower()):
                continue
            if item["flag"] in ("H", "L") and _SAYS_NORMAL.search(line) and not _SAYS_ABNORMAL.search(line):
                out.append(f"{item['analyte']}: в отчёте — норма, а последнее значение {item['value']:g} вне интервала ({item['flag']})")
            trend = item.get("trend", "")
            up, down = bool(_SAYS_UP.search(line)), bool(_SAYS_DOWN.search(line))
            if ", рост;" in trend and down and not up:
                out.append(f"{item['analyte']}: в отчёте — снижение, а по данным рост ({trend})")
            if ", снижение;" in trend and up and not down:
                out.append(f"{item['analyte']}: в отчёте — рост, а по данным снижение ({trend})")
    return list(dict.fromkeys(out))[:8]


def check_report(final: dict, tools, done: set[str]) -> dict | None:
    """Report-system checks against the dossier; each fires at most once per episode."""
    if "coverage" not in done:
        _, missing = coverage(final, tools.must_mention())
        if missing:
            done.add("coverage")
            return {
                "error": "Отчёт неполон: пропущены обязательные пункты досье (must_mention).",
                "missing": [{"label": m["label"], "ids": m["ids"], "section": m["section"]} for m in missing],
                "fix": "Добавьте эти пункты в соответствующие разделы с вердиктом и id или перечислите в dismissed с причиной, "
                "затем вызовите final_answer снова с полным отчётом.",
            }
    if "verdicts" not in done:
        bad = verdict_mismatches(final, tools.dossier()["labs"])
        if bad:
            done.add("verdicts")
            return {
                "error": "Вердикты в отчёте расходятся с данными досье.",
                "mismatches": bad,
                "fix": "Исправьте эти пункты и вызовите final_answer снова с полным отчётом.",
            }
    return None
