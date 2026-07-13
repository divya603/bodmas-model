"""OpenRouter client: one OpenAI-compatible endpoint, swap model via a string and params
via a dict. On-disk response cache (so reruns are free), tenacity retries, and a thread
pool for the independent-call path. Ported from the numberlink pipeline unchanged except
the parser, which reads the 6-point Likert {"rating": 1..6} instead of yes/no+confidence.

Env: OPENROUTER_API_KEY must be set. Optionally OPENROUTER_APP_URL / OPENROUTER_APP_NAME
for OpenRouter's attribution headers.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from . import RESULTS_DIR

BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_CACHE_DIR = RESULTS_DIR / "cache"


@dataclass
class LLMResponse:
    """Normalized result of one call."""
    answer_json: dict | None          # parsed {"rating": int} or None if unparseable
    raw_text: str                     # the model's message content
    reasoning: str | None             # thinking/reasoning text if the provider returned it
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    latency_ms: float
    cached: bool
    model: str
    error: str | None = None
    warning: str | None = None     # e.g. unsupported sampling params were dropped


@dataclass
class OpenRouterClient:
    model: str
    params: dict = field(default_factory=dict)         # e.g. {"max_tokens": 2048}
    cache_dir: Path = DEFAULT_CACHE_DIR
    api_key: str | None = None

    def __post_init__(self):
        self.api_key = self.api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        headers = {}
        if os.environ.get("OPENROUTER_APP_URL"):
            headers["HTTP-Referer"] = os.environ["OPENROUTER_APP_URL"]
        if os.environ.get("OPENROUTER_APP_NAME"):
            headers["X-Title"] = os.environ["OPENROUTER_APP_NAME"]
        self._client = OpenAI(base_url=BASE_URL, api_key=self.api_key, default_headers=headers or None)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── cache ────────────────────────────────────────────────────────────────────────
    def _cache_key(self, messages, response_format, effort) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "params": self.params,
                "messages": messages,
                "response_format": response_format,
                "effort": effort,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    # ── one call ─────────────────────────────────────────────────────────────────────
    def call(self, messages, response_format=None, effort: str = "direct") -> LLMResponse:
        """One completion. `effort` in {"direct","thinking"} toggles reasoning.

        Caches by sha256(model+params+messages+response_format+effort). A cache hit
        returns instantly and is not re-billed.
        """
        key = self._cache_key(messages, response_format, effort)
        cpath = self._cache_path(key)
        if cpath.exists():
            data = json.loads(cpath.read_text(encoding="utf-8"))
            return LLMResponse(cached=True, **data)

        resp = self._raw_call(messages, response_format, effort)

        # Persist only successful (non-error) responses so failures can be retried later.
        if resp.error is None:
            to_cache = {k: v for k, v in resp.__dict__.items() if k != "cached"}
            cpath.write_text(json.dumps(to_cache), encoding="utf-8")
        return resp

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _create(self, kwargs):
        """The actual network call, retried with exponential backoff on any exception.

        OpenRouter intermittently returns a 200 with an empty/None `choices` (a transient
        provider response, not an HTTP error). That's not an exception, so we raise here to
        let the retry decorator retry it (bounded at 5) instead of crashing downstream."""
        completion = self._client.chat.completions.create(**kwargs)
        if not getattr(completion, "choices", None):
            raise RuntimeError("empty 'choices' in completion (transient provider response)")
        return completion

    def _raw_call(self, messages, response_format, effort) -> LLMResponse:
        kwargs = dict(model=self.model, messages=messages, **self.params)
        if response_format is not None:
            kwargs["response_format"] = response_format
        # OpenRouter's unified reasoning control. Providers that don't support it ignore
        # the flag silently — so always log thinking_tokens downstream to detect when the
        # two effort levels collapsed for a given model.
        if effort == "thinking":
            kwargs["extra_body"] = {"reasoning": {"effort": "high"}}
        elif effort == "direct":
            # Try to turn reasoning OFF where supported. Reasoning-mandatory models reject
            # enabled:false — for those we fall back to the minimum allowed effort ("low")
            # in _fallback_create. thinking_tokens reveals when the arms collapsed.
            kwargs["extra_body"] = {"reasoning": {"enabled": False}}
        else:
            raise ValueError(f"unknown effort: {effort!r}")

        t0 = time.perf_counter()
        try:
            completion = self._create(kwargs)
        except Exception as e:  # noqa: BLE001
            completion, fb_warn, fb_err = self._fallback_create(kwargs, effort, str(e))
            latency_ms = (time.perf_counter() - t0) * 1000
            if fb_err is not None:
                return LLMResponse(
                    answer_json=None, raw_text="", reasoning=None,
                    prompt_tokens=None, completion_tokens=None, reasoning_tokens=None,
                    latency_ms=latency_ms, cached=False, model=self.model, error=fb_err,
                )
            return self._build_response(completion, latency_ms, warning=fb_warn)
        latency_ms = (time.perf_counter() - t0) * 1000
        return self._build_response(completion, latency_ms)

    def _fallback_create(self, kwargs, effort, err_msg):
        """Recover from model-specific 400s with a single adjusted retry.

        Handles two known cases, recording what was changed in a `warning`:
          1. reasoning-mandatory models reject `reasoning.enabled=false` (the direct arm) ->
             retry with the minimum allowed effort ("low").
          2. models that reject sampling params (temperature/top_p/seed) -> strip and retry.
        Returns (completion, warning, error) — exactly one of completion/error is set.
        """
        msg = err_msg.lower()
        warnings = []

        if effort == "direct" and "reasoning" in msg and (
            "mandatory" in msg or "cannot be disabled" in msg or "enabled" in msg
        ):
            kwargs = dict(kwargs)
            kwargs["extra_body"] = {"reasoning": {"effort": "low"}}
            warnings.append("direct arm: reasoning mandatory -> used effort=low (not off)")

        sampling = ("temperature", "top_p", "seed")
        if any(p in msg for p in sampling) and any(p in kwargs for p in sampling):
            kwargs = dict(kwargs)
            dropped = [p for p in sampling if kwargs.pop(p, None) is not None]
            warnings.append(f"dropped unsupported params: {dropped}")

        if not warnings:
            return None, None, f"unrecoverable: {err_msg}"

        try:
            completion = self._create(kwargs)
        except Exception as e2:  # noqa: BLE001 — surface after retries exhausted
            return None, None, f"{type(e2).__name__}: {e2}"
        return completion, "; ".join(warnings), None

    def _build_response(self, completion, latency_ms, warning=None) -> LLMResponse:
        # Defensive: if choices are still empty after retries, record an error row for this
        # one trial rather than crashing the whole run.
        if not getattr(completion, "choices", None):
            return LLMResponse(
                answer_json=None, raw_text="", reasoning=None,
                prompt_tokens=None, completion_tokens=None, reasoning_tokens=None,
                latency_ms=latency_ms, cached=False, model=self.model,
                error="empty choices after retries", warning=warning,
            )
        choice = completion.choices[0]
        raw_text = choice.message.content or ""
        reasoning = getattr(choice.message, "reasoning", None)
        usage = completion.usage
        reasoning_tokens = None
        if usage is not None:
            details = getattr(usage, "completion_tokens_details", None)
            if details is not None:
                reasoning_tokens = getattr(details, "reasoning_tokens", None)

        return LLMResponse(
            answer_json=_parse_rating(raw_text),
            raw_text=raw_text,
            reasoning=reasoning,
            prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            reasoning_tokens=reasoning_tokens,
            latency_ms=latency_ms,
            cached=False,
            model=self.model,
            warning=warning,
        )

    # ── parallel map (independent-call path) ──────────────────────────────────────────
    def map(self, jobs, response_format=None, effort="direct", max_workers=8):
        """Run many independent calls concurrently.

        `jobs` is a list of (key, messages). Returns {key: LLMResponse}. Cache hits make
        repeated runs effectively instant.
        """
        results: dict = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self.call, msgs, response_format, effort): k
                for k, msgs in jobs
            }
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
        return results


def _parse_rating(text: str) -> dict | None:
    """Extract the {"rating": 1..6} object from the model's text.

    With structured output this is the whole message; we still guard against providers
    that wrap it or emit stray prose. The rating is rounded to the nearest integer and
    clamped to 1..6 (the schema can't enforce min/max — Anthropic rejects them)."""
    if not text:
        return None
    text = text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict) or "rating" not in obj:
        return None
    try:
        rating = int(round(float(obj["rating"])))
    except (TypeError, ValueError):
        return None
    rating = max(1, min(6, rating))
    return {"rating": rating}
