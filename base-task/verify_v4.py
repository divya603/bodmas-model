"""
verify_v4.py

Independent checks on stimulus_pool_v4.json. Recomputes everything from the
model rather than trusting the builder: traces are re-derived from the
expression, error positions re-tested for expert legality, and foil marginals
re-run through the 22-hypothesis observer.

The v4-specific checks are the balance ones. v4 exists to make the present x
named heatmap full, so this asserts the exact cell counts that guarantee it:
20 on each diagonal cell, 4 in each of the 30 off-diagonal cells, 0 empty.

It deliberately does NOT check refutation balance. Under v4 foil_status is
recorded but not a factor, and the per-foil counts are expected to be lopsided
(outside_bracket_first comes out about 2 refuted to 18 unsupported). What IS
checked is that the stored status matches a fresh recomputation, and that it is
the same at both positions of a pair.

Run after any pool regeneration. Exits non-zero on any failure.
"""

import json
import sys
from collections import Counter, defaultdict

from parser import build_dag
from traces import generate_traces
from distance import correct_answer
from learner import MISCONCEPTION_FLIPS
from inference import posterior_over_profiles, marginal_rule_probability
from generator_v3 import validate_trace, error_steps
from pool_v4 import (HYPOTHESES, STATEMENT_TEMPLATES, POSITIONS, N_OPS,
                     REFUTED_MAX, UNSUPPORTED_MAX, _status,
                     A_PAIRS_PER_RULE, B_PAIRS_PER_CELL)

IDS = list(MISCONCEPTION_FLIPS.keys())
fails = []


def check(cond, msg):
    if not cond:
        fails.append(msg)


def main(path='stimulus_pool_v4.json'):
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

        # the trace is really generable by a learner holding exactly true_m
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
            check(marg <= UNSUPPORTED_MAX,
                  f"{tag}: foil marginal {marg:.3f} above {UNSUPPORTED_MAX}, not a clean foil")
            # status is recorded, not balanced, but it must still be correct
            check(_status(marg) == it['foil_status'],
                  f"{tag}: foil_status {it['foil_status']} but marginal {marg:.3f} says {_status(marg)}")

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
        check(a['misconceptions'] == b['misconceptions'],
              f"{pid}: members differ in the present misconception")
        if a['category'] == 'B':
            # status is not a factor, but a flip inside a pair would put a
            # nuisance difference between the two positions being compared
            check(a['foil_status'] == b['foil_status'],
                  f"{pid}: foil_status differs between positions")

    # one expression is used by exactly one pair, so nobody can meet it twice
    per_expr = Counter(i['expression'] for i in items)
    check(all(v == 2 for v in per_expr.values()),
          f"expressions not used exactly twice: {[e for e,v in per_expr.items() if v!=2][:3]}")

    # ── v4 balance: the whole point of the design ──
    ca = Counter((i['misconceptions'][0], i['error_position'])
                 for i in items if i['category'] == 'A')
    check(len(ca) == 12 and set(ca.values()) == {A_PAIRS_PER_RULE},
          f"A cells (present x position) not all {A_PAIRS_PER_RULE}: {sorted(set(ca.values()))}")

    cb = Counter((i['misconceptions'][0], i['probed_misconception'], i['error_position'])
                 for i in items if i['category'] == 'B')
    check(len(cb) == 60 and set(cb.values()) == {B_PAIRS_PER_CELL},
          f"B cells (present x named x position) not all {B_PAIRS_PER_CELL}: "
          f"{len(cb)} cells, sizes {sorted(set(cb.values()))}")

    # per present rule: 40 items, 20 per position, half named half foiled
    for m in IDS:
        sub = [i for i in items if i['misconceptions'][0] == m]
        check(len(sub) == 40, f"{m}: {len(sub)} items, want 40")
        c = Counter((i['category'], i['error_position']) for i in sub)
        check(set(c.values()) == {10} and len(c) == 4,
              f"{m}: category x position split is {dict(c)}, want 10 each")

    # the heatmap has no empty boxes
    grid = Counter((i['misconceptions'][0], i['probed_misconception']) for i in items)
    empty = [(p, n) for p in IDS for n in IDS if grid[(p, n)] == 0]
    check(not empty, f"heatmap has {len(empty)} empty cells: {empty[:5]}")
    diag = {grid[(m, m)] for m in IDS}
    off = {grid[(p, n)] for p in IDS for n in IDS if p != n}
    check(diag == {20}, f"diagonal cells not all 20: {sorted(diag)}")
    check(off == {4}, f"off-diagonal cells not all 4: {sorted(off)}")

    # no foil is ever probed on a trace generated by itself
    for i in items:
        if i['category'] == 'B':
            check(i['probed_misconception'] != i['misconceptions'][0],
                  f"{i['id']}: foil equals the present rule")

    if fails:
        print(f"FAILED - {len(fails)} problem(s):")
        for f in fails[:25]:
            print("  -", f)
        if len(fails) > 25:
            print(f"  ... and {len(fails)-25} more")
        sys.exit(1)

    nums = [int(n) for i in items for line in i['trace']
            for n in line.replace('(', ' ').replace(')', ' ').split()
            if n.lstrip('-').isdigit()]
    marg = [i['io_foil_marginal'] for i in items if i['category'] == 'B']
    print("ALL CHECKS PASSED")
    print(f"  {len(items)} items / {len(by_pair)} matched pairs / {len(per_expr)} expressions")
    print(f"  every trace: {N_OPS} steps, exactly 1 expert-illegal move, at step "
          f"{POSITIONS[0]} or {POSITIONS[1]}")
    print(f"  present x named heatmap: 0 empty cells, diagonal 20, off-diagonal 4")
    print(f"  each rule present in 40 items: 10 x (A/B) x (pos{POSITIONS[0]}/pos{POSITIONS[1]})")
    print(f"  foil marginals: min {min(marg):.3f}  max {max(marg):.3f}  "
          f"mean {sum(marg)/len(marg):.3f}  (status recorded, NOT balanced)")
    print(f"  numbers shown: min {min(nums)}  max {max(nums)}  "
          f"(no negatives, no decimals, no zero)")


if __name__ == '__main__':
    main()
