"""LLM client with schema-constrained output, validation, retry and a response cache.

Design rules from lectures 05-06, adapted to the current Claude API:
* output is constrained to a JSON schema through `output_config.format`, then
  validated with pydantic; on a validation error the call is retried once with the
  error appended, and a second failure is raised, never silently dropped;
* current models (e.g. Claude Sonnet 5.5) reject `temperature` and forced
  `tool_choice`, so temperature 0 is not available. Reproducibility therefore
  comes from the cache: every response is stored on disk, keyed by a hash of the
  full request, and `--replay` re-runs every agent experiment from it;
* constraints the API's schema subset does not accept (min/max, maxItems,
  length limits) are moved into field descriptions by the SDK's
  `transform_schema` and still enforced locally by pydantic.
"""
from __future__ import annotations

import hashlib
import json
import os
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
DEFAULT_MODEL = os.environ.get("FRAUDRANK_LLM_MODEL", "claude-sonnet-5-5")
DEFAULT_EFFORT = os.environ.get("FRAUDRANK_LLM_EFFORT", "low")   # low: cheap, fast, enough for structured tasks


class LLMError(RuntimeError):
    pass


class LLMClient:
    """mode: 'live' calls the API and caches; 'replay' only reads the cache."""

    def __init__(self, model: str = DEFAULT_MODEL, mode: str = "live", max_tokens: int = 16000,
                 effort: str = DEFAULT_EFFORT):
        self.model, self.mode, self.max_tokens, self.effort = model, mode, max_tokens, effort
        self.calls = self.cache_hits = 0
        self.input_tokens = self.output_tokens = 0
        self._client = None

    # ------------------------------------------------------------------ internals
    def _api(self):
        if self._client is None:
            import anthropic  # imported lazily so replay mode needs no key
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise LLMError("ANTHROPIC_API_KEY is not set: copy .env.example to .env and paste your key, "
                               "or use --replay to run from the cache")
            self._client = anthropic.Anthropic()
        return self._client

    def _key(self, payload: dict) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]

    def _raw(self, system: str, messages: list[dict], schema: dict) -> dict:
        payload = {"model": self.model, "system": system, "messages": messages, "schema": schema,
                   "effort": self.effort}
        path = LLM_CACHE / f"{self._key(payload)}.json"
        if path.exists():
            cached = json.loads(path.read_text())
            self.cache_hits += 1
            # token usage of the original call, so replayed runs report the same cost
            self.input_tokens += cached.get("usage", {}).get("input", 0)
            self.output_tokens += cached.get("usage", {}).get("output", 0)
            return cached["output"]
        if self.mode == "replay":
            raise LLMError(f"replay mode: no cached response {path.name}")
        resp = self._api().messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system, messages=messages,
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
        )
        self.calls += 1
        self.input_tokens += resp.usage.input_tokens
        self.output_tokens += resp.usage.output_tokens
        if resp.stop_reason not in ("end_turn", "stop_sequence"):
            raise LLMError(f"stopped with stop_reason={resp.stop_reason}; no complete JSON returned")
        text = "".join(b.text for b in resp.content if b.type == "text")
        try:
            output = json.loads(text)
        except json.JSONDecodeError as err:
            raise LLMError(f"response was not valid JSON: {err}") from err
        path.write_text(json.dumps({"request": payload, "output": output, "stop_reason": resp.stop_reason,
                                    "usage": {"input": resp.usage.input_tokens,
                                              "output": resp.usage.output_tokens}}, indent=1))
        return output

    # ------------------------------------------------------------------ public
    def structured(self, system: str, user: str, out: Type[BaseModel], tool_name: str = "result") -> BaseModel:
        """tool_name is kept as a label for logging and tests; the API call uses output_config."""
        from anthropic import transform_schema
        schema = transform_schema(out)
        messages = [{"role": "user", "content": user}]
        raw = self._raw(system, messages, schema)
        try:
            return out.model_validate(raw)
        except ValidationError as err:
            messages = messages + [
                {"role": "assistant", "content": json.dumps(raw)},
                {"role": "user", "content": f"Your output failed validation:\n{err}\nReturn a corrected result."},
            ]
            raw = self._raw(system, messages, schema)
            try:
                return out.model_validate(raw)
            except ValidationError as err2:
                raise LLMError(f"{out.__name__} failed validation twice: {err2}") from err2

    def usage(self) -> dict:
        return {"model": self.model, "api_calls": self.calls, "cache_hits": self.cache_hits,
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens}
