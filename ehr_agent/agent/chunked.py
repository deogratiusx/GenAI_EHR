"""Map stage of the map-reduce baseline (`record_chunked`): the whole record, read in pieces, no tools.

A record longer than the context window is the case where "record in context" is not applicable.
The standard workaround without tools is map-reduce over chunks: the record is split into chunks
of `CHUNK_CHARS` characters on line boundaries; for each chunk the model names the lines that bear on
the question; the reduce step (the ordinary loop with no tools) answers from these excerpts alone.

Excerpts are grounded: a weak model often writes back a bare value ("4.06") or a reformatted line
without its id, which the reduce step can no longer interpret. So the harness puts the record's own
lines into the excerpt — the lines whose ids the model named, else the lines that contain what it
quoted — and keeps the model's text only when nothing matches. Every chunk costs one model call, so cost grows with the record, not with the
question — the price the tools are meant to avoid.
"""

from __future__ import annotations

import os
import re

from ..sources import Patient
from ..tools.baselines import serialize_record
from ..util import ID_RX
from . import prompts

# ~60k characters ≈ 22k tokens of Russian record text: fits every backend with room for the prompt.
CHUNK_CHARS = int(os.environ.get("EHR_CHUNK_CHARS", "60000"))
NOTHING = "НЕТ"
ENTRY_RX = re.compile(r"\n(?=" + ID_RX.pattern + r" \| )")
MAX_LINES = 40  # per chunk, so that one over-eager reply cannot flood the reduce prompt


def _entries(text: str) -> list[str]:
    """Serialized record → one entry per record (a value may span several lines)."""
    return [e for e in ENTRY_RX.split(text) if e.strip()]


def chunk_record(patient: Patient, size: int | None = None) -> list[str]:
    """Serialized record split into chunks of at most `size` (default CHUNK_CHARS) characters; a record is never cut."""
    size = size or CHUNK_CHARS
    chunks, cur, n = [], [], 0
    for line in _entries(serialize_record(patient)):
        if cur and n + len(line) + 1 > size:
            chunks.append("\n".join(cur))
            cur, n = [], 0
        cur.append(line)
        n += len(line) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def ground_excerpt(reply: str, chunk: str) -> list[str]:
    """Record lines a map reply refers to: by id, else by quoted words.

    A quoted fragment matches the lines that contain most of its words (at least 60 %), which tolerates
    the spacing and punctuation a model changes when it copies a line."""
    lines = _entries(chunk)
    low = [line.lower() for line in lines]
    by_id = {line.split(" | ", 1)[0]: line for line in lines}
    named = [by_id[i] for i in dict.fromkeys(ID_RX.findall(reply)) if i in by_id]
    if named:
        return named[:MAX_LINES]
    out: list[str] = []
    for frag in reply.splitlines():
        words = set(re.findall(r"[\w.,]{3,}", frag.lower().replace("ё", "е")))
        if not words:
            continue
        scores = [sum(w in line for w in words) for line in low]
        best = max(scores)
        if best >= max(1, 0.6 * len(words)):
            out += [lines[k] for k, sc in enumerate(scores) if sc == best][:3]
    return list(dict.fromkeys(out))[:MAX_LINES]


def map_record(backend, patient: Patient, question: str) -> tuple[str, list[dict], dict, str | None]:
    """Run the map stage → (excerpts joined for the reduce prompt, trace steps, token usage, error or None)."""
    chunks = chunk_record(patient)
    trace: list[dict] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    notes: list[str] = []
    for k, chunk in enumerate(chunks, 1):
        messages = [
            {"role": "system", "content": prompts.MAP_CHUNK.format(k=k, n=len(chunks), nothing=NOTHING)},
            {"role": "user", "content": f"ФРАГМЕНТ КАРТЫ {k}/{len(chunks)}:\n{chunk}\n\nВОПРОС ВРАЧА: {question}"},
        ]
        try:
            reply = backend.chat(messages, [])
        except Exception as exc:
            trace.append({"type": "backend_error", "error": str(exc)[:300], "chunk": k})
            return "", trace, usage, str(exc)
        for key in usage:
            usage[key] += int(reply.usage.get(key, 0) or 0)
        text = (reply.content or "").strip()
        empty = not text or text.strip(" .").upper() == NOTHING
        trace.append(
            {
                "type": "map",
                "chunk": k,
                "of": len(chunks),
                "chars": len(chunk),
                "content": text[:1500],
                "empty": empty,
                "latency": round(reply.latency, 2),
            }
        )
        grounded = [] if empty else ground_excerpt(text, chunk)
        trace[-1]["grounded"] = len(grounded)
        if grounded:
            notes.append(f"[фрагмент {k}/{len(chunks)}]\n" + "\n".join(grounded))
        elif not empty:  # nothing in the chunk matches: keep the model's own words, marked as such
            notes.append(f"[фрагмент {k}/{len(chunks)}, своими словами модели]\n{text}")
    return "\n\n".join(notes) or "(ни в одном фрагменте нет относящихся к вопросу записей)", trace, usage, None
