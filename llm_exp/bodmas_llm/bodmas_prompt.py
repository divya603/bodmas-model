"""Turn a BODMAS stimulus-pool entry into an OpenAI-style `messages` list.

Mirrors the human task 1:1 — the instructions are lifted verbatim from
src/user/components/trace_judgment/InstructionsView.vue, the stimulus is rendered
exactly as TraceJudgmentView.vue shows it (expression + the student's step-by-step
work + the belief statement), and the response is the same 6-point Likert rating.
The model does NOT solve the problem; it judges how well the belief statement
explains the student's work.

Answer-leak guard: the prompt may only contain `expression`, `trace`,
`belief_statement`, and `student_name`. The answer-revealing fields —
statement_correct, misconceptions, probed_misconception, which_target, category,
num_misconceptions, foil_status, io_foil_marginal — must NEVER appear.
`user_content` builds the prompt from only the allowed fields, and
`assert_no_leak` double-checks the rendered text.
"""

from __future__ import annotations

import json

from . import DATA_DIR

# Fields that would reveal the answer if they reached the prompt.
_LEAK_FIELDS = (
    "statement_correct",
    "misconceptions",
    "probed_misconception",
    "which_target",
    "category",
    "num_misconceptions",
    "foil_status",        # extended pool (extend_pool.py): refuted/unsupported
    "io_foil_marginal",
)

# ── Verbatim instructions from InstructionsView.vue ──────────────────────────────────
INSTRUCTIONS = (
    "In this task, you will see a math problem along with the step-by-step work a "
    "student produced while solving it.\n\n"
    "Below the work, you'll see a statement about what the student believes about the "
    "order of operations — for example, that they think addition should always come "
    "before multiplication.\n\n"
    "Your job is to judge how well that statement explains the work shown, using the "
    "student's steps as your evidence — not just whether their final answer happens "
    "to be right or wrong.\n\n"
    "For each problem, rate how much you agree with the statement on a 6-point scale "
    "from Strongly Disagree to Strongly Agree."
)

QUESTION = "How much do you agree that this is what the student believes?"

# ── Output contract (parsed downstream; no regex over prose) ─────────────────────────
ANSWER_INSTRUCTION = (
    "Respond with a single JSON object and nothing else: "
    '{"rating": an integer from 1 to 6}, where '
    "1 = Strongly Disagree, 2 = Disagree, 3 = Somewhat Disagree, "
    "4 = Somewhat Agree, 5 = Agree, 6 = Strongly Agree."
)

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "likert_rating",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                # NB: do NOT put minimum/maximum on rating — Anthropic models reject
                # them in structured output. The 1..6 range is stated in
                # ANSWER_INSTRUCTION and clamped in the parser.
                "rating": {"type": "integer"},
            },
            "required": ["rating"],
            "additionalProperties": False,
        },
    },
}


# ── Stimulus rendering (matches TraceJudgmentView.vue) ───────────────────────────────
def render_stimulus(entry: dict) -> str:
    """The expression + the student's work + the belief statement, exactly as a human
    participant sees it. The work is trace[1:] (the first element is the original
    expression, shown separately), each step prefixed with '='."""
    name = entry["student_name"]
    steps = "\n".join(f"= {s}" for s in entry["trace"][1:])
    return (
        f"Expression given to {name}:\n"
        f"{entry['expression']}\n\n"
        f"Here is the final answer {name} produced, along with their work:\n"
        f"{steps}\n\n"
        f'Statement: "{entry["belief_statement"]}"'
    )


# ── Leak guard ───────────────────────────────────────────────────────────────────────
def assert_no_leak(text: str, entry: dict) -> None:
    """Raise if any answer-revealing token from `entry` appears in the rendered stimulus.

    Guards the field NAMES themselves and the distinctive string VALUES (the
    misconception ID strings, and 'True'/'False' for statement_correct). Short/ambiguous
    values (category letters A-D, which_target 'first'/'second', num_misconceptions int)
    are NOT substring-matched — they collide with ordinary words/digits — but their field
    names are still guarded against an accidental full-entry dump."""
    tokens: list[str] = list(_LEAK_FIELDS)
    misc = entry.get("misconceptions") or []
    tokens.extend(str(m) for m in misc)                       # e.g. 'add_before_mul'
    if entry.get("probed_misconception"):
        tokens.append(str(entry["probed_misconception"]))
    if isinstance(entry.get("statement_correct"), bool):
        tokens.append(str(entry["statement_correct"]))        # 'True' / 'False'
    for tok in tokens:
        if tok and tok in text:
            raise AssertionError(f"answer leak: {tok!r} present in stimulus text")


# ── Worked examples (practice factor) ────────────────────────────────────────────────
# 3 constructed examples (data/practice_examples.json), SEPARATE from the 240-item pool.
# BODMAS humans had no worked practice, so these are built for the LLM: each shows the
# human-format stimulus + a short factual rationale (pointing only at the actual steps)
# + the target 1-6 rating. EX3 is a partial (2-misconception) match: its rationale
# explains why a full 6 is wrong even though the named misconception is present.


def _load_examples() -> list[dict]:
    path = DATA_DIR / "practice_examples.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


_WORKED = _load_examples()


def worked_examples_block() -> str:
    blocks = []
    for i, ex in enumerate(_WORKED, start=1):
        stim = render_stimulus(ex)
        assert_no_leak(stim, ex)
        out = json.dumps({"rating": ex["rating"]}, separators=(",", ":"))
        blocks.append(f"Example {i}:\n{stim}\n{QUESTION}\n{ex['rationale']}\nAnswer: {out}")
    return "\n\n".join(blocks)


# ── System prompt ────────────────────────────────────────────────────────────────────
def system_prompt(worked_examples: bool = False) -> str:
    parts = [INSTRUCTIONS, "", ANSWER_INSTRUCTION]
    if worked_examples:
        parts += ["", "Here are some worked examples:", "", worked_examples_block()]
    return "\n".join(parts)


def user_content(entry: dict) -> str:
    stim = render_stimulus(entry)
    assert_no_leak(stim, entry)
    return f"{stim}\n\n{QUESTION}"


def build_messages(entry: dict, worked_examples: bool = False) -> list[dict]:
    """Independent-call message list: one system turn + one user turn."""
    return [
        {"role": "system", "content": system_prompt(worked_examples)},
        {"role": "user", "content": user_content(entry)},
    ]
