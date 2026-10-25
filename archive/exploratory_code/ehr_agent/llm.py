"""Chat backends with native function calling behind one interface.

``chat(messages, tools, force_final=False) -> Reply`` where messages use a neutral
format: {"role": "system"|"user"|"assistant"|"tool", "content": str,
"tool_call": {"name", "arguments"} (assistant), "name" (tool)}.

* GigaChat: REST ``functions`` API (Sber, data stays in RF).
* Qwen (MLX, local): the model's own chat template with ``tools=`` and a parser for
  both the JSON ``<tool_call>{...}</tool_call>`` and the XML
  ``<function=..><parameter=..>`` formats.
"""

from __future__ import annotations

import json
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class Reply:
    content: str = ""
    tool_call: dict | None = None  # {"name": str, "arguments": dict|str}
    raw: str = ""
    usage: dict = field(default_factory=dict)
    latency: float = 0.0


def _load_env() -> None:
    env = REPO_ROOT / ".env"
    if not env.is_file():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*)\s*$", line)
        if m and m.group(1) not in os.environ:
            os.environ[m.group(1)] = m.group(2).strip().strip('"').strip("'")


class GigaChat:
    OAUTH = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    CHAT = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

    def __init__(self, model: str = "GigaChat-2-Max", temperature: float = 0.0):
        _load_env()
        self.model = model
        self.temperature = temperature
        self.key = os.environ.get("SBER_AUTHORIZATION_KEY") or os.environ.get("GIGACHAT_AUTHORIZATION_KEY", "")
        self.scope = os.environ.get("SBER_GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
        self._token, self._exp = "", 0.0
        self.ctx = ssl.create_default_context()
        for name in ("russian_trusted_root_ca.crt", "russian_trusted_sub_ca.crt"):
            p = REPO_ROOT / "certs" / name
            if p.is_file():
                self.ctx.load_verify_locations(cafile=str(p))

    def _post(self, url: str, data: bytes, headers: dict, timeout: float = 180) -> dict:
        last = None
        for attempt in range(7):
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=timeout, context=self.ctx) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", "replace")[:500]
                if exc.code in (429, 500, 502, 503, 504) and attempt < 6:
                    time.sleep((4 if exc.code == 429 else 2) * (attempt + 1))
                    last = RuntimeError(f"HTTP {exc.code}: {body}")
                    continue
                raise RuntimeError(f"GigaChat HTTP {exc.code}: {body}") from None
            except (urllib.error.URLError, TimeoutError) as exc:
                last = exc
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"GigaChat unreachable: {last}")

    def _auth(self) -> str:
        if self._token and time.time() < self._exp - 60:
            return self._token
        payload = self._post(self.OAUTH, urllib.parse.urlencode({"scope": self.scope}).encode(), {
            "Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json",
            "RqUID": str(uuid.uuid4()), "Authorization": f"Basic {self.key}"}, timeout=30)
        self._token = payload["access_token"]
        exp = float(payload.get("expires_at", time.time() * 1000 + 1.5e6))
        self._exp = exp / 1000 if exp > 1e12 else exp
        return self._token

    def _messages(self, messages: list[dict]) -> list[dict]:
        out = []
        for m in messages:
            if m["role"] == "assistant" and m.get("tool_call"):
                args = m["tool_call"]["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                item = {"role": "assistant", "content": m.get("content") or "",
                        "function_call": {"name": m["tool_call"]["name"], "arguments": args}}
                if m.get("functions_state_id"):
                    item["functions_state_id"] = m["functions_state_id"]
                out.append(item)
            elif m["role"] == "tool":
                out.append({"role": "function", "name": m["name"], "content": m["content"]})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    def chat(self, messages: list[dict], tools: list[dict], force_final: bool = False) -> Reply:
        body: dict = {"model": self.model, "messages": self._messages(messages),
                      "temperature": self.temperature if self.temperature > 0 else 0.01,
                      "max_tokens": 1500}
        if tools:
            body["functions"] = tools
            body["function_call"] = {"name": "final_answer"} if force_final else "auto"
        t0 = time.time()
        resp = self._post(self.CHAT, json.dumps(body, ensure_ascii=False).encode("utf-8"), {
            "Content-Type": "application/json", "Accept": "application/json",
            "Authorization": f"Bearer {self._auth()}"})
        msg = resp["choices"][0]["message"]
        r = Reply(content=msg.get("content") or "", raw=json.dumps(msg, ensure_ascii=False),
                  usage=resp.get("usage", {}), latency=time.time() - t0)
        if msg.get("function_call"):
            r.tool_call = {"name": msg["function_call"]["name"], "arguments": msg["function_call"].get("arguments", {}),
                           "functions_state_id": msg.get("functions_state_id")}
        return r


_TOOL_JSON = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_TOOL_XML = re.compile(r"<function=([\w\-]+)>(.*?)</function>", re.S)
_PARAM_XML = re.compile(r"<parameter=([\w\-]+)>\n?(.*?)\n?</parameter>", re.S)


def parse_qwen_tool_call(text: str) -> dict | None:
    m = _TOOL_JSON.search(text)
    if m:
        try:
            obj = json.loads(m.group(1))
            return {"name": obj.get("name"), "arguments": obj.get("arguments", {})}
        except json.JSONDecodeError:
            return {"name": "__malformed__", "arguments": m.group(1)}
    m = _TOOL_XML.search(text)
    if m:
        args = {}
        for k, v in _PARAM_XML.findall(m.group(2)):
            v = v.strip()
            try:
                args[k] = json.loads(v) if v[:1] in "[{" or re.fullmatch(r"-?\d+", v) else v
            except json.JSONDecodeError:
                args[k] = v
        return {"name": m.group(1), "arguments": args}
    return None


class QwenMLX:
    _cache: dict = {}

    def __init__(self, repo: str = "mlx-community/Qwen3.5-4B-4bit", temperature: float = 0.0, max_tokens: int = 1200,
                 thinking: bool = False):
        # thinking=True for checkpoints trained to answer after a <think> block (Med-Qwen);
        # the instruct base runs with thinking disabled.
        self.repo, self.temperature, self.max_tokens, self.thinking = repo, temperature, max_tokens, thinking
        self.model = repo.rstrip("/").split("/")[-1]
        if repo not in self._cache:
            # Weights are already local; skip the hub round-trip that hangs behind the proxy.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from mlx_lm import load
            self._cache[repo] = load(repo)
        self.m, self.tok = self._cache[repo]

    def _messages(self, messages: list[dict]) -> list[dict]:
        out = []
        for m in messages:
            if m["role"] == "assistant" and m.get("tool_call"):
                args = m["tool_call"]["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                out.append({"role": "assistant", "content": m.get("content") or "",
                            "tool_calls": [{"type": "function", "function": {"name": m["tool_call"]["name"], "arguments": args}}]})
            elif m["role"] == "tool":
                out.append({"role": "tool", "name": m["name"], "content": m["content"]})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    def _generate(self, prompt: str, sampler) -> str:
        """Greedy generation with KV-cache reuse across agent turns.

        Qwen3.5 mixes linear-attention layers whose state cannot be trimmed, so the cache is only
        reused when the new prompt extends the tokens already in it; otherwise it is rebuilt.
        """
        from mlx_lm import stream_generate
        from mlx_lm.models.cache import make_prompt_cache

        ids = self.tok.encode(prompt, add_special_tokens=False) if hasattr(self.tok, "encode") else prompt
        cached = getattr(self, "_cached_ids", None)
        if cached and len(ids) > len(cached) and ids[: len(cached)] == cached:
            feed = ids[len(cached):]
            self.cache_hits = getattr(self, "cache_hits", 0) + 1
            self.reused_tokens = getattr(self, "reused_tokens", 0) + len(cached)
        else:
            self._cache = make_prompt_cache(self.m)
            feed = ids
        out_ids, chunks = [], []
        for resp in stream_generate(self.m, self.tok, prompt=feed, max_tokens=self.max_tokens, sampler=sampler,
                                    prompt_cache=self._cache):
            out_ids.append(resp.token)
            chunks.append(resp.text)
            # Some fine-tunes (Med-Qwen) do not mark <|im_end|> as EOS in their tokenizer config.
            if "<|im_end|>" in resp.text or "".join(chunks[-3:]).endswith("<|im_end|>"):
                break
        self._cached_ids = ids + out_ids
        return "".join(chunks).split("<|im_end|>")[0]

    def chat(self, messages: list[dict], tools: list[dict], force_final: bool = False) -> Reply:
        from mlx_lm.sample_utils import make_sampler

        msgs = self._messages(messages)
        if force_final:
            msgs.append({"role": "user", "content": "Лимит вызовов исчерпан. Сейчас вызови только final_answer."})
        tool_defs = [{"type": "function", "function": t} for t in tools] if tools else None
        kwargs = {"tools": tool_defs, "add_generation_prompt": True, "tokenize": False}
        try:
            prompt = self.tok.apply_chat_template(msgs, enable_thinking=self.thinking, **kwargs)
        except TypeError:
            prompt = self.tok.apply_chat_template(msgs, **kwargs)
        if self.thinking and not prompt.rstrip().endswith("<think>"):
            prompt += "<think>\n"
        t0 = time.time()
        text = self._generate(prompt, make_sampler(temp=self.temperature))
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        if self.thinking and "</think>" in text:
            text = text.split("</think>", 1)[1]
        text = text.strip()
        n_prompt = len(self.tok.encode(prompt))
        r = Reply(content=text, raw=text, latency=time.time() - t0,
                  usage={"prompt_tokens": n_prompt, "completion_tokens": len(self.tok.encode(text))})
        call = parse_qwen_tool_call(text)
        if call:
            r.tool_call = call
            r.content = text.split("<tool_call>")[0].strip()
        return r


def make_backend(name: str):
    if name.lower().startswith("gigachat"):
        return GigaChat(model=name)
    if name.lower().startswith(("qwen", "med-qwen")):
        local = Path(__file__).resolve().parent.parent / "models"
        repos = {"qwen3.5-4b": "mlx-community/Qwen3.5-4B-4bit",
                 "med-qwen3.5-4b": str(local / "Med-Qwen-3.5-4B-it-4bit"),
                 "qwen3-4b": "mlx-community/Qwen3-4B-Instruct-2507-4bit"}
        return QwenMLX(repos.get(name.lower(), name), thinking=name.lower().startswith("med-qwen"))
    raise ValueError(name)
