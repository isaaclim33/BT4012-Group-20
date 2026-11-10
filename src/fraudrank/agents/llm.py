"""LLM client with schema-constrained output, validation, retry and a response cache.

The agents run on Gemini; the key is read from GEMINI_API_KEY.

Design rules:
* output is constrained to a JSON schema (`response_json_schema`), then validated
  with pydantic; on a validation error the
  call is retried once with the error appended, and a second failure is raised,
  never silently dropped;
* every response is stored on disk, keyed by a hash of the full request, and
  `--replay` re-runs every agent experiment from it without a key;
* `effort` maps to Gemini's thinking level (low / high);
* Gemini records the exact model version that answered in each cached response,
  because aliases such as 'gemini-flash-latest' move over time.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Type

from pydantic import BaseModel, ValidationError

from ..config import CACHE_DIR, ROOT


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Read KEY=VALUE lines from the repo's .env into the environment.

    Values already set in the environment win. The file is git-ignored and is
    never copied into results or the output bundle.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv()
LLM_CACHE = CACHE_DIR / "llm"
LLM_CACHE.mkdir(parents=True, exist_ok=True)
DEFAULT_MODEL = os.environ.get("FRAUDRANK_LLM_MODEL", "gemini-3.5-flash-lite")
DEFAULT_EFFORT = os.environ.get("FRAUDRANK_LLM_EFFORT", "low")   # low: cheap, fast, enough for structured tasks
TEMPERATURE = os.environ.get("FRAUDRANK_LLM_TEMPERATURE")         # unset: the model's default
RETRIES = 7                                                       # rate limits and server overload
TIMEOUT_MS = 180_000                                              # per request; a stuck call is retried


API_KEY_VAR = "GEMINI_API_KEY"


class LLMError(RuntimeError):
    pass


class QuotaExhausted(RuntimeError):
    """The provider's daily quota is used up. Deliberately not an LLMError: steps record LLMErrors as
    failed items and carry on, but this must stop the step before it writes partial results."""


class LLMClient:
    """mode: 'live' calls the API and caches; 'replay' only reads the cache."""

    def __init__(self, model: str = DEFAULT_MODEL, mode: str = "live", max_tokens: int = 16000,
                 effort: str = DEFAULT_EFFORT):
        self.model, self.mode, self.max_tokens, self.effort = model, mode, max_tokens, effort
        self.calls = self.cache_hits = 0
        self.input_tokens = self.output_tokens = 0
        self.model_versions: set[str] = set()
        self._client = None

    # ------------------------------------------------------------------ internals
    def _api(self):
        if self._client is None:
            if not os.environ.get(API_KEY_VAR):
                raise LLMError(f"{API_KEY_VAR} is not set: copy .env.example to .env and paste your key, "
                               "or use --replay to run from the cache")
            from google import genai  # imported lazily so replay mode needs no key or SDK
            from google.genai import types
            self._client = genai.Client(api_key=os.environ[API_KEY_VAR], http_options=types.HttpOptions(
                timeout=TIMEOUT_MS, retry_options=types.HttpRetryOptions(attempts=1)))   # retries are ours, below
        return self._client

    def _key(self, payload: dict) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]

    def _cached(self, payload: dict):
        path = LLM_CACHE / f"{self._key(payload)}.json"
        if path.exists():
            cached = json.loads(path.read_text())
            self.cache_hits += 1
            self.input_tokens += cached.get("usage", {}).get("input", 0)
            self.output_tokens += cached.get("usage", {}).get("output", 0)
            if cached.get("model_version"):
                self.model_versions.add(cached["model_version"])
            return path, cached
        if self.mode == "replay":
            raise LLMError(f"replay mode: no cached response {path.name}")
        return path, None

    def _store(self, path: Path, payload: dict, output, stop_reason: str, usage: dict, version: str | None):
        self.calls += 1
        self.input_tokens += usage["input"]
        self.output_tokens += usage["output"]
        record = {"request": payload, "output": output, "stop_reason": stop_reason, "usage": usage}
        if version:
            self.model_versions.add(version)
            record["model_version"] = version
        path.write_text(json.dumps(record, indent=1))

    def _raw(self, system: str, messages: list[dict], schema: dict, effort: str | None = None) -> dict:
        eff = effort or self.effort
        payload = {"model": self.model, "system": system, "messages": messages, "schema": schema,
                   "effort": eff}
        path, cached = self._cached(payload)
        if cached is not None:
            return cached["output"]
        output, stop_reason, usage, version = self._gemini_json(system, messages, schema, eff)
        self._store(path, payload, output, stop_reason, usage, version)
        return output

    def _raw_with_tools(self, system: str, messages: list[dict], tools: list[dict],
                        effort: str | None = None) -> tuple[list[dict], str]:
        """API call with tool definitions. Returns (response content blocks, stop_reason)."""
        eff = effort or self.effort
        payload = {"model": self.model, "system": system, "messages": messages,
                   "tools": tools, "effort": eff}
        path, cached = self._cached(payload)
        if cached is not None:
            return cached["output"], cached.get("stop_reason", "end_turn")
        content, stop_reason, usage, version = self._gemini_tools(system, messages, tools, eff)
        self._store(path, payload, content, stop_reason, usage, version)
        return content, stop_reason

    # ------------------------------------------------------------------ Gemini
    def _gemini_config(self, system, eff, **extra):
        from google.genai import types
        cfg = {"system_instruction": system, "max_output_tokens": self.max_tokens,
               "thinking_config": types.ThinkingConfig(thinking_level=eff.upper()), **extra}
        if TEMPERATURE is not None:
            cfg["temperature"] = float(TEMPERATURE)
        return types.GenerateContentConfig(**cfg)

    def _gemini_call(self, contents, config):
        import httpx
        from google.genai import errors
        for attempt in range(RETRIES):
            try:
                return self._api().models.generate_content(model=self.model, contents=contents, config=config)
            except (errors.APIError, httpx.TransportError) as err:   # transport: timeouts, dropped connection
                code = getattr(err, "code", "network")
                if code == 429 and "per day" in str(err):   # daily quota: waiting minutes won't help
                    raise QuotaExhausted(f"Gemini daily quota used up for {self.model}; the cache keeps every "
                                   f"finished call, so re-run the same command after the reset. {str(err)[-400:]}") from err
                if code not in (429, 500, 503, 504, "network") or attempt == RETRIES - 1:
                    raise LLMError(f"Gemini API error {code}: {err}") from err
                wait = min(10 * 2 ** attempt, 120)          # 10, 20, 40, 80, 120, 120 s
                print(f"    Gemini {code} ({str(err)[:120]}); retrying in {wait}s", flush=True)
                time.sleep(wait)

    def _gemini_json(self, system, messages, schema, eff):
        resp = self._gemini_call(_to_gemini(messages), self._gemini_config(
            system, eff, response_mime_type="application/json", response_json_schema=schema))
        cand = _candidate(resp)
        text = "".join(p.text for p in (cand.content.parts or []) if p.text and not p.thought)
        return _parse_json(text), "end_turn", _gemini_usage(resp), resp.model_version

    def _gemini_tools(self, system, messages, tools, eff):
        from google.genai import types
        decls = [types.FunctionDeclaration(name=t["name"], description=t["description"],
                                           parameters_json_schema=t["input_schema"]) for t in tools]
        resp = self._gemini_call(_to_gemini(messages), self._gemini_config(
            system, eff, tools=[types.Tool(function_declarations=decls)],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)))
        content = []
        for i, p in enumerate(_candidate(resp).content.parts or []):
            if p.function_call:
                block = {"type": "tool_use", "id": p.function_call.id or f"call_{i}",
                         "native_id": p.function_call.id,     # None when Gemini sent no id
                         "name": p.function_call.name, "input": dict(p.function_call.args or {})}
                if p.thought_signature:                    # Gemini needs it back with the call
                    block["signature"] = base64.b64encode(p.thought_signature).decode()
                content.append(block)
            elif p.text and not p.thought:
                content.append({"type": "text", "text": p.text})
        stop = "tool_use" if any(b["type"] == "tool_use" for b in content) else "end_turn"
        return content, stop, _gemini_usage(resp), resp.model_version

    # ------------------------------------------------------------------ public
    def structured(self, system: str, user: str, out: Type[BaseModel], tool_name: str = "result",
                   effort: str | None = None) -> BaseModel:
        """tool_name is kept as a label for logging and tests; the API call uses a JSON schema."""
        schema = out.model_json_schema()
        messages = [{"role": "user", "content": user}]
        raw = self._raw(system, messages, schema, effort=effort)
        try:
            return out.model_validate(raw)
        except ValidationError as err:
            messages = messages + [
                {"role": "assistant", "content": json.dumps(raw)},
                {"role": "user", "content": f"Your output failed validation:\n{err}\nReturn a corrected result."},
            ]
            raw = self._raw(system, messages, schema, effort=effort)
            try:
                return out.model_validate(raw)
            except ValidationError as err2:
                raise LLMError(f"{out.__name__} failed validation twice: {err2}") from err2

    def react(self, system: str, user: str, tools: list[dict],
              tool_executor, out: Type[BaseModel],
              max_turns: int = 6, effort: str | None = None) -> tuple[BaseModel, list[dict]]:
        """ReAct loop: the model calls tools until it's ready, then a final
        structured() call produces the schema-validated output."""
        messages = [{"role": "user", "content": user}]
        trace = []
        for turn in range(max_turns):
            content, stop_reason = self._raw_with_tools(system, messages, tools, effort=effort)
            tool_calls = [b for b in content if b["type"] == "tool_use"]
            if not tool_calls:
                break
            messages.append({"role": "assistant", "content": content})
            tool_results = []
            for tc in tool_calls:
                result = tool_executor(tc["name"], tc["input"])
                trace.append({"tool": tc["name"], "input": tc["input"], "output": result})
                tool_results.append({"type": "tool_result", "tool_use_id": tc["id"], "name": tc["name"],
                                     "content": json.dumps(result)})   # Gemini matches results by function name
            messages.append({"role": "user", "content": tool_results})
        else:
            raise LLMError(f"react loop hit {max_turns} turns without finishing")
        summary = "\n".join(
            f"Tool {t['tool']}({json.dumps(t['input'])}): {json.dumps(t['output'])}"
            for t in trace
        )
        final_user = (f"{user}\n\n<investigation>\n{summary}\n</investigation>\n"
                      "Now write your final structured output using all the evidence above.")
        return self.structured(system, final_user, out, effort=effort), trace

    def usage(self) -> dict:
        return {"model": self.model, "model_versions": sorted(self.model_versions), "api_calls": self.calls,
                "cache_hits": self.cache_hits, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens}


def _parse_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError as err:
        raise LLMError(f"response was not valid JSON: {err}") from err


def _candidate(resp):
    if not resp.candidates:
        raise LLMError(f"no candidate returned; prompt feedback: {resp.prompt_feedback}")
    cand = resp.candidates[0]
    reason = getattr(cand.finish_reason, "name", cand.finish_reason)
    if reason not in ("STOP", None):
        raise LLMError(f"stopped with finish_reason={reason}; no complete response returned")
    if cand.content is None:
        raise LLMError("candidate has no content")
    return cand


def _gemini_usage(resp) -> dict:
    u = resp.usage_metadata
    return {"input": u.prompt_token_count or 0,
            "output": (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)}   # thinking is billed as output


def _to_gemini(messages: list[dict]) -> list:
    """Convert the cache's provider-neutral messages into Gemini contents."""
    from google.genai import types
    contents, native = [], {}
    for m in messages:
        role = "model" if m["role"] == "assistant" else "user"
        blocks = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
        parts = []
        for b in blocks:
            if b["type"] == "text":
                parts.append(types.Part(text=b["text"]))
            elif b["type"] == "tool_use":
                native[b["id"]] = b.get("native_id")
                sig = base64.b64decode(b["signature"]) if b.get("signature") else None
                parts.append(types.Part(function_call=types.FunctionCall(id=native[b["id"]], name=b["name"],
                                                                         args=b["input"]),
                                        thought_signature=sig))
            elif b["type"] == "tool_result":
                part = types.Part.from_function_response(name=b["name"], response={"result": json.loads(b["content"])})
                part.function_response.id = native.get(b["tool_use_id"])
                parts.append(part)
        contents.append(types.Content(role=role, parts=parts))
    return contents
