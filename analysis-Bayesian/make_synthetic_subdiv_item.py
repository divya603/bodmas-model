#!/usr/bin/env python3
"""
make_synthetic_subdiv_item.py

The deployed 240-item pool contains NO category-B item where the sub<÷ foil
is refutable: in all 10 pool items naming sub_before_div, the trace never
reaches a subtraction-next-to-division decision point, so the ideal observer
can only leave the claim unsupported (~0.2), never refute it (~0).

This script searches the generator space for a SYNTHETIC B item that fills
that hole: a single-misconception trace that DOES pass through a sub/÷
decision point conventionally, so the sub_before_div foil is actively
refuted (ideal-observer marginal < 0.05). The item is written to
synthetic_items.json with a synthetic flag and is NOT added to the deployed
pool (humans never see it); it exists so the refuted-subset analysis can be
probed for the one foil the pool cannot test, on the Bayesian and LLM arms.

Run from repo root:
    python3 analysis-Bayesian/make_synthetic_subdiv_item.py
"""

import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_TASK = os.path.join(os.path.dirname(HERE), 'base-task')
sys.path.insert(0, BASE_TASK)
os.chdir(BASE_TASK)  # stimulus_pool imports expect to run from base-task/

from stimulus_pool import _build_item, IDS  # noqa: E402
from inference import posterior_over_profiles, marginal_rule_probability  # noqa: E402

FOIL = 'sub_before_div'
ACCEPT = 0.05
SEED = 20260715
MAX_TRIES = 300


def main():
    random.seed(SEED)
    presents = [m for m in IDS if m != FOIL]
    best = None
    for attempt in range(MAX_TRIES):
        mid = presents[attempt % len(presents)]
        item = _build_item(f'SYNB{attempt:03d}', 'B', [mid], FOIL, False,
                           student_name='Noah')
        if item is None:
            continue
        post = posterior_over_profiles(item['trace'])
        marg = marginal_rule_probability(post, FOIL)
        if best is None or marg < best[0]:
            best = (marg, item)
            print(f"attempt {attempt:3d}  present={mid:22s} marginal={marg:.3f}  <- new best")
        if marg < ACCEPT:
            break

    marg, item = best
    if marg >= ACCEPT:
        print(f"\nWARNING: best marginal {marg:.3f} did not reach the {ACCEPT} "
              f"refutation criterion after {MAX_TRIES} tries")
    item['synthetic'] = True
    item['synthetic_note'] = (
        'NOT in the deployed pool; generated 2026-07-15 (seed 20260715) because no '
        'pool B item makes the sub_before_div foil refutable. Ideal-observer '
        'marginal on the foil: %.4f' % marg)
    item['bayes_marginal_on_probe'] = round(marg, 4)

    out = os.path.join(HERE, 'synthetic_items.json')
    with open(out, 'w') as f:
        json.dump([item], f, indent=2)

    print(f"\nAccepted item ({item['id']}), present={item['misconceptions'][0]}, "
          f"marginal={marg:.4f}")
    print(f"  expression: {item['expression']}")
    for step in item['trace'][1:]:
        print(f"    = {step}")
    print(f"  statement: {item['belief_statement']}")
    print(f"\nWrote {out}")


if __name__ == '__main__':
    main()
