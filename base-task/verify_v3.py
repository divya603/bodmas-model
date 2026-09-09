"""
verify_v3.py

Independent checks on stimulus_pool_v3.json. Recomputes everything from the
model rather than trusting the builder: traces are re-derived from the
expression, error positions re-tested for expert legality, and foil marginals
re-run through the 22-hypothesis observer.

Run after any pool regeneration. Exits non-zero on any failure.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations

from parser import build_dag
from traces import generate_traces
from distance import correct_answer
from learner import MISCONCEPTION_FLIPS
from inference import posterior_over_profiles, marginal_rule_probability
from generator_v3 import validate_trace, error_steps
from pool_v3 import (HYPOTHESES, STATEMENT_TEMPLATES, POSITIONS, N_OPS,
                     REFUTED_MAX, UNSUPPORTED_MAX, TARGET_PER_CELL)

IDS = list(MISCONCEPTION_FLIPS.keys())
fails = []


def check(cond, msg):
    if not cond:
        fails.append(msg)


def main(path='stimulus_pool_v3.json'):
    items = json.load(open(path, encoding='utf-8'))
    print(f"verifying {len(items)} items from {path}\n")

    check(len({i['id'] for i in items}) == len(items), "duplicate item ids")

    by_pair = defaultdict(list)
    for i in items:
        by_pair[i['pair_id']].append(i)

    for it in items:
        tag = it['id']
        trace, expr = it['trace'], it['expression']
        true_m = it['misconceptions'][0]

        # structure
        check(it['num_misconceptions'] == 1 and len(it['misconceptions']) == 1,
              f"{tag}: not exactly one misconception")
        check(len(trace) == N_OPS + 1, f"{tag}: trace has {len(trace)} lines, want {N_OPS+1}")
        check(trace[0] == expr, f"{tag}: trace does not start at the expression")
        check(len(trace[-1].split()) == 1, f"{tag}: trace does not reduce to a number")

        # displayable arithmetic
        check(validate_trace(trace), f"{tag}: trace fails validate_trace ({trace})")

        # the trace is really a legal trace for a learner holding exactly true_m
        legal = generate_traces(build_dag(expr), [true_m])
        check(trace in legal, f"{tag}: trace is not generable by {true_m}")

        # exactly one expert-illegal move, at the declared position
        errs = error_steps(trace)
        check(errs == [it['error_position']],
              f"{tag}: error steps {errs}, declared position {it['error_position']}")

        # the misconception actually changes the answer
        expert = generate_traces(build_dag(expr), [])
        check(trace[-1] != correct_answer(expert),
              f"{tag}: learner answer equals the expert answer")

        # statement wiring
        probed = it['probed_misconception']
        check(it['belief_statement'] ==
              STATEMENT_TEMPLATES[probed].format(name=it['student_name']),
              f"{tag}: belief statement does not match probed rule / name")

        post = posterior_over_profiles(trace, profiles=HYPOTHESES)
        if it['category'] == 'A':
            check(probed == true_m and it['statement_correct'] is True,
                  f"{tag}: A item probed {probed} but trace holds {true_m}")
            check(marginal_rule_probability(post, probed) > 0.99,
                  f"{tag}: A item true-rule marginal too low")
            check('foil_status' not in it, f"{tag}: A item carries a foil_status")
        else:
            check(probed != true_m and it['statement_correct'] is False,
                  f"{tag}: B item probes its own true rule")
            marg = marginal_rule_probability(post, probed)
            check(abs(marg - it['io_foil_marginal']) < 1e-3,
                  f"{tag}: stored marginal {it['io_foil_marginal']} != recomputed {marg:.4f}")
            want = 'refuted' if marg < REFUTED_MAX else 'unsupported' if marg <= UNSUPPORTED_MAX else 'high'
            check(want == it['foil_status'],
                  f"{tag}: foil_status {it['foil_status']} but marginal {marg:.3f} is {want}")

    # matched pairs
    for pid, members in by_pair.items():
        check(len(members) == 2, f"{pid}: {len(members)} members, want 2")
        if len(members) != 2:
            continue
        a, b = members
        check(a['expression'] == b['expression'], f"{pid}: members differ in expression")
        check({a['error_position'], b['error_position']} == set(POSITIONS),
              f"{pid}: positions {a['error_position']}/{b['error_position']}")
        check(a['trace'] != b['trace'], f"{pid}: both members show the same trace")
        for f in ('category', 'probed_misconception', 'statement_correct'):
            check(a[f] == b[f], f"{pid}: members differ in {f}")
        if a['category'] == 'B':
            check(a['foil_status'] == b['foil_status'],
                  f"{pid}: members differ in foil_status (pair not matched on refutation)")

    # one expression is used by exactly one pair, so nobody can meet it twice
    per_expr = Counter(i['expression'] for i in items)
    check(all(v == 2 for v in per_expr.values()),
          f"expressions not used exactly twice: {[e for e,v in per_expr.items() if v!=2][:3]}")

    # cell balance
    ca = Counter((i['probed_misconception'], i['error_position']) for i in items if i['category']=='A')
    cb = Counter((i['probed_misconception'], i['foil_status'], i['error_position'])
                 for i in items if i['category']=='B')
    check(set(ca.values()) == {TARGET_PER_CELL}, f"A cells unbalanced: {sorted(set(ca.values()))}")
    check(set(cb.values()) == {TARGET_PER_CELL}, f"B cells unbalanced: {sorted(set(cb.values()))}")
    check(len(ca) == 12 and len(cb) == 24, f"cell count {len(ca)} A / {len(cb)} B, want 12 / 24")

    if fails:
        print(f"FAILED — {len(fails)} problem(s):")
        for f in fails[:25]:
            print("  -", f)
        if len(fails) > 25:
            print(f"  ... and {len(fails)-25} more")
        return 1

    vals = [i['io_foil_marginal'] for i in items if i['category'] == 'B']
    nums = [int(t) for i in items for s in i['trace']
            for t in re.sub(r'[()]', ' ', s).split() if t.isdigit()]
    print("ALL CHECKS PASSED")
    print(f"  {len(items)} items / {len(by_pair)} matched pairs / {len(per_expr)} expressions")
    print(f"  every trace: {N_OPS} steps, exactly 1 expert-illegal move, at step 1 or 3")
    print(f"  foil marginals: min {min(vals):.3f}  max {max(vals):.3f}  mean {sum(vals)/len(vals):.3f}")
    print(f"  numbers shown: min {min(nums)}  max {max(nums)}  (no negatives, no decimals, no zero)")
    return 0


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
