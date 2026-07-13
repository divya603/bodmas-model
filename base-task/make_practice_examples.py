#!/usr/bin/env python3
"""
make_practice_examples.py

Regenerate the 3 worked practice examples for the LLM experiment's `examples` arm and
write them to llm_exp/data/practice_examples.json. Each example is a real, valid trace
(reproducible from a fixed seed), verified NOT to appear in the 240-item pool, with a
hand-authored target rating + a short factual rationale that points only at the actual
steps. BODMAS humans had no worked practice, so these are constructed for the LLM.

  EX1  single misconception, statement matches           -> Strongly Agree (6)
  EX2  single misconception, statement is a foil          -> Strongly Disagree (1)
  EX3  two misconceptions, statement names one (partial)  -> Somewhat Agree (4)

Run from base-task/:
    python3 make_practice_examples.py
"""

import json
import random
import sys

sys.path.insert(0, ".")
from parser import build_dag
from generator import generate_expression
from traces import generate_traces
from stimulus_pool import _pick_trace, _pick_trace_for_target, STATEMENT_TEMPLATES
from distance import tree_edges

OUT = "../llm_exp/data/practice_examples.json"

pool = json.load(open("stimulus_pool.json"))
pool_exprs = {it["expression"] for it in pool}


def _both_fire(dag, trace, m1, m2):
    expert = tree_edges(generate_traces(dag, []))
    d1 = tree_edges(generate_traces(dag, [m1])) - expert
    d2 = tree_edges(generate_traces(dag, [m2])) - expert
    edges = {(trace[i], trace[i + 1]) for i in range(len(trace) - 1)}
    return bool(edges & d1) and bool(edges & d2)


def _find_single(mid, seedrange):
    for s in seedrange:
        random.seed(s)
        expr = generate_expression(n_ops=4, bracket_prob=0.3)
        if expr in pool_exprs:
            continue
        tr = _pick_trace(build_dag(expr), [mid])
        if tr and len(tr) <= 5:
            return expr, tr
    raise RuntimeError(f"no example found for {mid}")


def _find_single_ops(mid, required_ops, bracket_prob, seedrange):
    for s in seedrange:
        random.seed(s)
        expr = generate_expression(n_ops=4, bracket_prob=bracket_prob)
        if expr in pool_exprs or not all(op in expr for op in required_ops):
            continue
        tr = _pick_trace(build_dag(expr), [mid])
        if tr and len(tr) <= 5:
            return expr, tr
    raise RuntimeError(f"no example found for {mid} with ops {required_ops}")


def _find_pair(pair, target, seedrange):
    for s in seedrange:
        random.seed(s)
        expr = generate_expression(n_ops=4, bracket_prob=0.2)
        if expr in pool_exprs:
            continue
        dag = build_dag(expr)
        tr = _pick_trace_for_target(dag, pair, target)
        if tr and len(tr) <= 5 and _both_fire(dag, tr, *pair):
            return expr, tr
    raise RuntimeError(f"no pair example found for {pair}")


def _entry(expr, trace, probed, name, rating, rationale):
    return {
        "expression": expr,
        "trace": trace,
        "student_name": name,
        "probed_misconception": probed,
        "belief_statement": STATEMENT_TEMPLATES[probed].format(name=name),
        "rating": rating,
        "rationale": rationale,
    }


def main():
    e1, t1 = _find_single("add_before_mul", range(100, 400))
    e2, t2 = _find_single_ops("sub_before_mul", ["+", "-", "×"], 0.2, range(1, 2000))
    e3, t3 = _find_pair(("sub_before_div", "same_priority_rtl"), "sub_before_div", range(1, 3000))

    examples = [
        _entry(e1, t1, "add_before_mul", "Ava", 6,
               "The student computes 4 + 12 before the multiplication 1 × 4, so the work "
               "shows addition being done before multiplication — the statement explains it well."),
        _entry(e2, t2, "add_before_mul", "Leo", 1,
               "The student computes 2 - 6 before the multiplication, so the work shows "
               "subtraction — not addition — before multiplication; the statement is about "
               "addition, so it does not explain the work."),
        _entry(e3, t3, "sub_before_div", "Mia", 4,
               "A rating of 6 would mean the statement fully explains the work. It does not — "
               "the work also shows a second kind of error the statement doesn't mention — so "
               "it should not be the maximum. A moderate agreement is appropriate: the student "
               "does subtract before dividing (which the statement describes), but the "
               "explanation is incomplete."),
    ]

    # sanity: none of the example expressions may appear in the pool
    for ex in examples:
        assert ex["expression"] not in pool_exprs, f"leak: {ex['expression']} is in the pool"

    with open(OUT, "w") as f:
        json.dump(examples, f, indent=2)
    print(f"wrote {len(examples)} practice examples to {OUT}")
    for ex in examples:
        print(f"  {ex['expression']:24s} probed={ex['probed_misconception']:16s} rating={ex['rating']}")


if __name__ == "__main__":
    main()
