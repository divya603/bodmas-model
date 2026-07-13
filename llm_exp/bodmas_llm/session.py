"""Trial delivery — the `delivery` factor.

independent : each trial is its own call (shared system prompt + that stimulus). No
              memory, fully parallel + cacheable. The i.i.d. default (the only arm we run
              to start; the numberlink pilot found independent beat append everywhere).
append      : one growing conversation per subject — kept for parity / a future memory
              contrast. Only the model's OWN parsed {rating} is echoed back; never the
              ground truth (humans got no per-trial feedback).

Both modes emit the same per-trial record shape so parse_results treats them identically.
"""

from __future__ import annotations

import json

from .bodmas_prompt import RESPONSE_FORMAT, build_messages, system_prompt, user_content
from .client import LLMResponse, OpenRouterClient


def _record(entry: dict, factors: dict, repeat: int, resp: LLMResponse) -> dict:
    """Flatten one trial's result into a raw row (parse_results maps to the human schema)."""
    aj = resp.answer_json or {}
    rating = aj.get("rating")
    statement_correct = entry["statement_correct"]
    # correct direction: agree (rating >= 4) iff the statement is actually true
    correct = None
    if rating is not None:
        correct = int((rating >= 4) == bool(statement_correct))
    return {
        **factors,
        "repeat": repeat,
        # entry identity + ground truth (for scoring / joining to human + Bayesian)
        "id": entry["id"],
        "category": entry["category"],
        "expression": entry["expression"],
        "misconceptions": entry["misconceptions"],
        "num_misconceptions": entry["num_misconceptions"],
        "probed_misconception": entry["probed_misconception"],
        "which_target": entry.get("which_target"),
        "statement_correct": statement_correct,
        "correct_answer": "agree" if statement_correct else "disagree",
        # response
        "response": rating,                                   # 1..6 Likert, or None
        "correct": correct,
        "raw_text": resp.raw_text,
        "reasoning": resp.reasoning,
        "prompt_tokens": resp.prompt_tokens,
        "completion_tokens": resp.completion_tokens,
        "thinking_tokens": resp.reasoning_tokens,
        "latency_ms": resp.latency_ms,
        "cached": resp.cached,
        "model": resp.model,
        "error": resp.error,
        "warning": resp.warning,
    }


def run_subject_independent(client, trials, effort, repeat, subject_id, worked_examples, max_workers=8):
    """Independent calls for all 24 trials (parallel)."""
    factors = _factor_dict(client, "independent", effort, worked_examples, subject_id)
    jobs = [(entry["id"], build_messages(entry, worked_examples)) for entry in trials]
    responses = client.map(jobs, response_format=RESPONSE_FORMAT, effort=effort, max_workers=max_workers)
    return [_record(entry, factors, repeat, responses[entry["id"]]) for entry in trials]


def run_subject_append(client, trials, effort, repeat, subject_id, worked_examples):
    """One growing conversation across the trials (serial). Only the model's own parsed
    rating is echoed back; no ground-truth feedback."""
    factors = _factor_dict(client, "append", effort, worked_examples, subject_id)
    messages = [{"role": "system", "content": system_prompt(worked_examples)}]
    rows = []
    for entry in trials:
        messages.append({"role": "user", "content": user_content(entry)})
        resp = client.call(messages, response_format=RESPONSE_FORMAT, effort=effort)
        rows.append(_record(entry, factors, repeat, resp))
        aj = resp.answer_json or {"rating": 3}  # fixed neutral placeholder on parse failure
        messages.append({"role": "assistant", "content": json.dumps(aj, separators=(",", ":"))})
    return rows


def run_subject(client, trials, *, delivery, effort, worked_examples, repeat, subject_id, max_workers=8):
    if delivery == "independent":
        return run_subject_independent(client, trials, effort, repeat, subject_id, worked_examples, max_workers)
    if delivery == "append":
        return run_subject_append(client, trials, effort, repeat, subject_id, worked_examples)
    raise ValueError(f"unknown delivery: {delivery!r}")


def _factor_dict(client, delivery, effort, worked_examples, subject_id) -> dict:
    return {
        "model": client.model,
        "delivery": delivery,
        "effort": effort,
        "practice": "examples" if worked_examples else "none",
        "subject_id": subject_id,
    }
