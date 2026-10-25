"""Shared machinery of the tool layer: JSON-schema specs, argument validation, output budgeting.

Every tool is a deterministic Python method that returns a JSON-serializable dict. The agent only
sees the spec (name, description, JSON schema) and the returned JSON; arithmetic, reference
intervals and record access never come from the model.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

DEFAULT_CHARS = 3500  # output budget per tool call (characters of JSON)
CHAR_LIMITS = {  # dense outputs get a larger window instead of being cut
    "get_conclusions": 7500,
    "search_notes": 6000,
    "get_medications": 6000,
    "get_studies": 6000,
    "get_trend": 5000,
    "build_dossier": 24000,  # prefetched once into report prompts; trimmed by section caps, not by fit()
}
LIST_KEYS = ("items", "results", "documents", "issues", "medications", "studies")


def spec(name: str, description: str, properties: dict | None = None, required: list[str] | None = None) -> dict:
    """Function-calling spec in the common (OpenAI / GigaChat) format."""
    return {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": properties or {}, "required": required or []},
    }


OFFSET = {"offset": {"type": "integer", "description": "Сдвиг для постраничной выдачи (из next_offset предыдущего ответа)"}}


def page(items: list, offset: int | None, size: int) -> tuple[list, dict]:
    """Slice a list for pagination; returns the page and the pagination fields to merge into the output."""
    offset = max(0, int(offset or 0))
    chunk = items[offset : offset + size]
    meta = {"total": len(items)}
    if offset + size < len(items):
        meta["next_offset"] = offset + size
    return chunk, meta


def fit(obj: Any, limit: int) -> Any:
    """Keep a tool output within its character budget by dropping trailing list items.

    The output says how many items were dropped and, when the tool supports it, which offset returns
    the rest — the model is told to narrow the request rather than silently losing information.
    """
    s = json.dumps(obj, ensure_ascii=False)
    if len(s) <= limit:
        return obj
    if isinstance(obj, dict):
        for key in LIST_KEYS:
            if isinstance(obj.get(key), list):
                items = list(obj[key])
                while items and len(json.dumps({**obj, key: items}, ensure_ascii=False)) > limit - 160:
                    items.pop()
                start = obj.get("offset", 0) or 0
                return {
                    **obj,
                    key: items,
                    "truncated": f"показано {len(items)} из {len(obj[key])}; уточните запрос",
                    "next_offset": start + len(items),
                }
    return {"text": s[: limit - 40] + "…", "truncated": True}


def validate_args(spec_: dict, args: Any) -> tuple[dict | None, str | None]:
    """Light JSON-schema check: object, required keys, primitive types (with coercion), enums.

    Unknown keys are dropped. Returns (clean_args, None) or (None, error message for the model).
    """
    if args is None:
        args = {}
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            return None, "Аргументы не являются корректным JSON-объектом."
    if not isinstance(args, dict):
        return None, "Аргументы должны быть объектом."
    props = spec_["parameters"].get("properties", {})
    for req in spec_["parameters"].get("required", []):
        if req not in args or args[req] in (None, ""):
            return None, f"Не передан обязательный аргумент '{req}'."
    clean = {}
    for k, v in args.items():
        if k not in props:
            continue
        typ = props[k].get("type")
        if typ == "number":
            try:
                v = float(str(v).replace(",", "."))
            except (TypeError, ValueError):
                return None, f"'{k}' должен быть числом."
        elif typ == "integer":
            try:
                v = int(v)
            except (TypeError, ValueError):
                return None, f"'{k}' должен быть целым числом."
        elif typ == "string" and not isinstance(v, str):
            v = str(v)
        elif typ == "array" and isinstance(v, str):
            v = [x.strip() for x in re.split(r"[,\s]+", v) if x.strip()]
        if "enum" in props[k] and v not in props[k]["enum"]:
            return None, f"'{k}' должен быть одним из {props[k]['enum']}."
        clean[k] = v
    return clean, None


def run_tool(fns: dict[str, Callable[..., dict]], name: str, args: dict) -> dict:
    """Execute a tool with its output budget; a tool bug becomes an error message, never a crash."""
    try:
        return fit(fns[name](**args), CHAR_LIMITS.get(name, DEFAULT_CHARS))
    except TypeError as exc:
        return {"error": f"Неверные аргументы: {exc}"}
    except Exception as exc:
        return {"error": f"Ошибка инструмента: {type(exc).__name__}: {exc}"}


def is_empty(result: dict) -> bool:
    """An empty or failed result (used by the agent loop and the harness)."""
    if "error" in result or result.get("found") is False:
        return True
    for key in ("results", "items", "issues", "documents", "medications", "studies"):
        if key in result and not result[key]:
            return True
    if "stdout" in result and not str(result["stdout"]).strip():
        return True
    return False
