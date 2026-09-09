"""
find_pairs.py

Search for v3 matched item pairs: one expression that yields BOTH a step-1 and
a step-3 version of the same misconception, where each shown trace has exactly
one expert-illegal move and passes validate_trace().

This is the core sampler the v3 pool builder will call. Run directly to see
per-misconception yields.
"""

import random
import sys
from collections import Counter

from parser import build_dag
from traces import generate_traces
from distance import correct_answer
from learner import MISCONCEPTION_FLIPS
from generator_constrained import generate_expression, validate_trace, error_steps

IDS       = list(MISCONCEPTION_FLIPS.keys())
N_OPS     = 6
POSITIONS = (1, 3)


def pairs_for_expression(expr, misconception, positions=POSITIONS, n_ops=N_OPS):
    """
    {position: trace} for every requested position this expression supports.

    A trace qualifies when it finishes, reaches a DIFFERENT answer than the
    expert, is displayable (validate_trace), and its ONLY expert-illegal move
    is at the requested position.
    """
    try:
        dag     = build_dag(expr)
        expert  = generate_traces(dag, [])
        learner = generate_traces(dag, [misconception])
    except Exception:
        return {}
    if not expert:
        return {}

    right = correct_answer(expert)
    found = {}
    for t in learner:
        if len(t) != n_ops + 1 or len(t[-1].split()) != 1:
            continue
        if t[-1] == right or not validate_trace(t):
            continue
        errs = error_steps(t)
        if len(errs) == 1 and errs[0] in positions:
            found.setdefault(errs[0], t)
    return found


def find_matched_pairs(misconception, n_wanted, rng, max_draws=200_000):
    """Sample expressions until n_wanted matched pairs are found."""
    need_bracket = (misconception == 'outside_bracket_first')
    out, draws, seen = [], 0, set()
    while len(out) < n_wanted and draws < max_draws:
        draws += 1
        expr = generate_expression(
            n_ops=N_OPS, bracket_prob=1.0 if need_bracket else 0.4, rng=rng)
        if expr is None or expr in seen:
            continue
        seen.add(expr)
        got = pairs_for_expression(expr, misconception)
        if all(p in got for p in POSITIONS):
            out.append({'expression': expr,
                        'misconception': misconception,
                        'traces': {p: got[p] for p in POSITIONS}})
    return out, draws


if __name__ == '__main__':
    n_wanted = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    rng = random.Random(2026)
    print(f"v3 matched pairs, n_ops={N_OPS}, positions={POSITIONS}, "
          f"target {n_wanted} per misconception\n")
    print(f"{'misconception':24s} {'found':>6s} {'draws':>8s} {'hit rate':>9s}")
    all_pairs = {}
    for m in IDS:
        got, draws = find_matched_pairs(m, n_wanted, rng)
        all_pairs[m] = got
        rate = f"{len(got)/draws:.2%}" if draws else "n/a"
        print(f"{m:24s} {len(got):6d} {draws:8,d} {rate:>9s}")

    print("\n--- one matched pair per misconception ---")
    for m, ps in all_pairs.items():
        if not ps:
            print(f"\n{m}: NONE FOUND")
            continue
        p = ps[0]
        print(f"\n{m}   {p['expression']}")
        for pos in POSITIONS:
            print(f"   step-{pos}: " + " -> ".join(p['traces'][pos]))
