"""Offline tests (no network): the prompt renders the human stimulus and never leaks
the answer. Run: `python -m pytest` from llm_exp/ (or `python tests/test_prompt.py`)."""

import json
from pathlib import Path

from bodmas_llm.bodmas_prompt import assert_no_leak, build_messages, render_stimulus, user_content
from bodmas_llm.sample_session import load_pool, sample_session

POOL = Path(__file__).resolve().parent.parent / "data" / "stimulus_pool.json"


def _items():
    return load_pool(POOL)


def test_no_answer_leak_across_pool():
    """The rendered stimulus never contains an answer-revealing field name or value."""
    for it in _items():
        stim = render_stimulus(it)
        assert_no_leak(stim, it)  # raises on leak
        # the misconception IDs and True/False must not appear verbatim
        for m in it["misconceptions"]:
            assert m not in stim, f"{m} leaked in {it['id']}"
        assert "statement_correct" not in stim
        assert "which_target" not in stim


def test_stimulus_contains_the_visible_fields():
    """Every stimulus shows the expression, the final trace step, and the statement."""
    for it in _items():
        stim = render_stimulus(it)
        assert it["expression"] in stim
        assert it["trace"][-1] in stim            # final answer shown
        assert it["belief_statement"] in stim
        assert it["student_name"] in stim


def test_messages_shape_and_rating_schema():
    it = _items()[0]
    msgs = build_messages(it)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "1 to 6" in msgs[0]["content"]         # scale stated
    assert user_content(it).endswith("what the student believes?")


def test_sample_session_is_balanced():
    """A sampled 24-trial form: 6 per category, 3 first / 3 second in C."""
    pool = _items()
    form = sample_session(pool, seed=1)
    assert len(form) == 24
    cats = {}
    for it in form:
        cats[it["category"]] = cats.get(it["category"], 0) + 1
    assert cats == {"A": 6, "B": 6, "C": 6, "D": 6}
    c_wt = [it["which_target"] for it in form if it["category"] == "C"]
    assert c_wt.count("first") == 3 and c_wt.count("second") == 3
    assert len({it["id"] for it in form}) == 24        # no repeated item


if __name__ == "__main__":
    test_no_answer_leak_across_pool()
    test_stimulus_contains_the_visible_fields()
    test_messages_shape_and_rating_schema()
    test_sample_session_is_balanced()
    print("all prompt tests passed")
