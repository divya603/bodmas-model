#!/usr/bin/env python3
"""
make_human_practice_items.py

Generate the 3 feedback practice trials for the HUMAN experiment and write them to
../src/user/data/practice_items.json. Participants answer each one exactly like a real
trial; after responding they see the erroneous step(s) highlighted plus a feedback
statement giving the right answer (never referencing their own choice).

  P1  add_before_div in trace, statement names add_before_div        -> agree
  P2  sub_before_mul in trace, statement names add_before_mul (foil) -> disagree
  P3  same_priority_rtl + outside_bracket_first in trace, statement
      names outside_bracket_first (partial match)                    -> agree, not strongly

Each item carries `error_steps`: the full-trace indices where a misconception actually
fired. Detection is state-local: a step is an error iff an expert standing at that
state would never produce it, and it is attributed to whichever SINGLE misconception's
learner (at that state) can produce it. Traces are rejection-sampled until every error
step is cleanly attributable and each misconception fires exactly once, so the
highlighting is unambiguous. Expressions are verified NOT to appear in the 240-item
pool, and student names come from outside STUDENT_NAMES (all 24 pool names are used
by sample_form, so reusing one could show a participant the "same student" twice).

Run from base-task/:
    python3 make_human_practice_items.py
"""

import json
import random
import re
import sys

sys.path.insert(0, ".")
from parser import build_dag
from misconceptions import dag_to_str
from generator import generate_expression
from traces import generate_traces, _next_dags
from distance import correct_answer
from stimulus_pool import IDS, STATEMENT_TEMPLATES, STUDENT_NAMES, _is_finished, _is_clean
from inference import posterior_over_profiles, marginal_rule_probability

OUT = "../src/user/data/practice_items.json"

pool = json.load(open("stimulus_pool.json"))
pool_exprs = {it["expression"] for it in pool}

PRACTICE_NAMES = {"P1": "Tara", "P2": "Sam", "P3": "Kira"}

STEP_NOTES = {
    "add_before_div":        "the student added before dividing here",
    "add_before_mul":        "the student added before multiplying here",
    "sub_before_mul":        "the student subtracted before multiplying here",
    "sub_before_div":        "the student subtracted before dividing here",
    "same_priority_rtl":     "the student worked right to left here",
    "outside_bracket_first": "the student worked outside the brackets before inside here",
}

FEEDBACK = {
    "P1": "In this trace, the student's error matches the belief statement: the highlighted "
          "step shows the student doing addition before division, exactly the misconception "
          "the statement describes. So the right answer would be to agree.",
    "P2": "Here the student did subtraction before multiplication (the highlighted step), "
          "but the statement said the student does addition before multiplication. The work "
          "does not match the statement, so the right answer would be to disagree.",
    "P3": "Here the belief statement partially matches the work. The two highlighted steps "
          "show two different misconceptions, and the statement names only one of them "
          "(working outside the brackets first). Strongly Agree would not be the best choice "
          "since the statement does not explain everything, but agreeing is still right "
          "because the statement partially matches.",
}


def _next_strs(state_str, misconceptions):
    """Canonical strings of every state a learner with `misconceptions` can reach
    in one step from `state_str` (re-parsed, which the parser supports)."""
    dag = build_dag(state_str)
    return {dag_to_str(d) for d in _next_dags(dag, list(misconceptions))}


def _error_steps(trace, misconceptions):
    """
    State-local error attribution. For each step s_t -> s_{t+1}: it is an error iff
    an expert at s_t would never produce s_{t+1}; it is attributed to the single
    misconception whose learner at s_t can produce it. Returns [(trace_index, mid)]
    or None if any error step is ambiguous (explainable by more than one of the
    item's misconceptions) or emergent (needs both flips at once) — those traces
    are unusable for clean highlighting.
    """
    steps = []
    for t in range(len(trace) - 1):
        nxt = trace[t + 1]
        if nxt in _next_strs(trace[t], []):
            continue
        owners = [m for m in misconceptions if nxt in _next_strs(trace[t], [m])]
        if len(owners) != 1:
            return None
        steps.append((t + 1, owners[0]))
    return steps


def _is_integer_only(trace):
    """Practice traces avoid decimals entirely (the pool allows up to 2 decimal
    places, but practice should not distract with decimal arithmetic)."""
    return not any("." in step for step in trace)


def _find_single(mid, required_ops, seedrange):
    """Bracket-free expression + trace where `mid` fires exactly once, cleanly
    attributable, with a final answer different from the expert's."""
    for s in seedrange:
        random.seed(s)
        expr = generate_expression(n_ops=4, bracket_prob=0.0)
        if expr in pool_exprs or not all(op in expr for op in required_ops):
            continue
        dag = build_dag(expr)
        expert_answer = correct_answer(generate_traces(dag, []))
        cands = [t for t in generate_traces(dag, [mid])
                 if _is_finished(t) and _is_clean(t) and _is_integer_only(t)
                 and t[-1] != expert_answer]
        for t in sorted(cands, key=lambda t: (len(t), t)):
            steps = _error_steps(t, [mid])
            if steps and len(steps) == 1:
                return expr, t, steps
    raise RuntimeError(f"no practice item found for {mid}")


def _changes_only_inside_bracket(before, after):
    """True if the step from `before` to `after` altered only the bracket interior
    (the outside parts, with bracket contents masked, are identical)."""
    mask = lambda s: re.sub(r"\([^()]*\)", "()", s)
    return "(" in before and "(" in after and mask(before) == mask(after)


def _find_pair(pair, seedrange):
    """
    Bracketed expression + trace where BOTH pair members (same_priority_rtl,
    outside_bracket_first) fire exactly once each, cleanly attributable, with a
    final answer different from the expert's. Two extra readability constraints,
    since this is the most complex practice item:
      - at least 4 distinct numbers, so steps are easy to follow visually;
      - the outside_bracket_first error is the FIRST step (the student visibly
        starts computing outside before touching the bracket) and the RTL error
        happens INSIDE the bracket, so the two highlights sit in visually
        separate places.
    """
    for s in seedrange:
        random.seed(s)
        expr = generate_expression(n_ops=4, bracket_prob=1.0)
        if expr in pool_exprs or len(set(re.findall(r"\d+", expr))) < 4:
            continue
        dag = build_dag(expr)
        expert_answer = correct_answer(generate_traces(dag, []))
        cands = [t for t in generate_traces(dag, list(pair))
                 if _is_finished(t) and _is_clean(t) and _is_integer_only(t)
                 and t[-1] != expert_answer]
        for t in sorted(cands, key=lambda t: (len(t), t)):
            steps = _error_steps(t, list(pair))
            if not (steps and len(steps) == 2 and {m for _, m in steps} == set(pair)):
                continue
            where = dict((m, i) for i, m in steps)
            if where["outside_bracket_first"] != 1:
                continue
            rtl_i = where["same_priority_rtl"]
            if not _changes_only_inside_bracket(t[rtl_i - 1], t[rtl_i]):
                continue
            return expr, t, steps
    raise RuntimeError(f"no practice item found for pair {pair}")


def _entry(pid, expr, trace, misconceptions, probed, statement_correct, steps):
    name = PRACTICE_NAMES[pid]
    return {
        "id": pid,
        "category": "practice",
        "expression": expr,
        "trace": trace,
        "misconceptions": list(misconceptions),
        "num_misconceptions": len(misconceptions),
        "probed_misconception": probed,
        "statement_correct": statement_correct,
        "student_name": name,
        "belief_statement": STATEMENT_TEMPLATES[probed].format(name=name),
        "error_steps": [{"trace_index": i, "misconception": m, "note": STEP_NOTES[m]}
                        for i, m in steps],
        "feedback": FEEDBACK[pid],
    }


def main():
    e1, t1, s1 = _find_single("add_before_div", ["+", "÷"], range(1, 5000))
    e2, t2, s2 = _find_single("sub_before_mul", ["+", "-", "×"], range(1, 5000))
    e3, t3, s3 = _find_pair(("same_priority_rtl", "outside_bracket_first"), range(1, 10000))

    items = [
        _entry("P1", e1, t1, ["add_before_div"], "add_before_div", True, s1),
        _entry("P2", e2, t2, ["sub_before_mul"], "add_before_mul", False, s2),
        _entry("P3", e3, t3, ["same_priority_rtl", "outside_bracket_first"],
               "outside_bracket_first", True, s3),
    ]

    for it in items:
        assert it["expression"] not in pool_exprs, f"leak: {it['expression']} is in the pool"
        assert it["student_name"] not in STUDENT_NAMES, f"name clash: {it['student_name']}"
        post = posterior_over_profiles(it["trace"])
        marg_str = "  ".join(f"{m}={marginal_rule_probability(post, m):.3f}" for m in IDS)
        print(f"{it['id']}  {it['expression']:28s} probed={it['probed_misconception']:22s}")
        print(f"    IO marginals (all rules): {marg_str}")
        for i, step in enumerate(it["trace"]):
            mark = "".join(f"   <- {e['note']}" for e in it["error_steps"]
                           if e["trace_index"] == i)
            prefix = "      " if i == 0 else "    = "
            print(f"{prefix}{step}{mark}")
        print(f"    statement: {it['belief_statement']}")

    with open(OUT, "w") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {len(items)} practice items to {OUT}")


if __name__ == "__main__":
    main()
