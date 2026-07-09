"""
misconception_difficulty.py

Bayesian ideal-observer baseline: for every item in stimulus_pool.json, runs
posterior_over_profiles() on its trace and reads off the marginal probability
placed on each ground-truth-present misconception (not the probed/statement
misconception -- what's actually true, regardless of what the belief
statement claims). Grouping these marginals by misconception gives a
difficulty ranking: which misconceptions the model can pin down confidently
from a typical trace, and which stay ambiguous. This is meant as a baseline
to compare against human participant accuracy per misconception, not a fix
for the profile-level tie.

Each misconception occurs in exactly 60 of the 240 pool items by
construction (10 alone via category A, 10 alone via B, 20 paired via C, 20
paired via D), so alone/paired/overall breakdowns are all balanced.

Run from base-task/:
    python3 misconception_difficulty.py
"""

import json
from collections import defaultdict

from inference import posterior_over_profiles, marginal_rule_probability
from learner import MISCONCEPTION_FLIPS

IDS = list(MISCONCEPTION_FLIPS.keys())


def main():
    pool = json.load(open('stimulus_pool.json'))

    alone  = defaultdict(list)   # rule -> list of marginals, item had 1 misconception present
    paired = defaultdict(list)   # rule -> list of marginals, item had 2 misconceptions present

    for it in pool:
        post = posterior_over_profiles(it['trace'])
        present = it['misconceptions']
        bucket = alone if len(present) == 1 else paired
        for r in present:
            bucket[r].append(marginal_rule_probability(post, r))

    def avg(xs):
        return sum(xs) / len(xs)

    print(f"{'misconception':24s} {'n':>4s} {'alone':>8s}   {'n':>4s} {'paired':>8s}   {'n':>4s} {'overall':>8s}")
    for r in IDS:
        a, p, o = alone[r], paired[r], alone[r] + paired[r]
        print(f"{r:24s} {len(a):4d} {avg(a):8.3f}   {len(p):4d} {avg(p):8.3f}   {len(o):4d} {avg(o):8.3f}")

    print()
    print("Ranked hardest -> easiest for the model (overall avg marginal on the true rule):")
    ranked = sorted(IDS, key=lambda r: avg(alone[r] + paired[r]))
    for r in ranked:
        print(f"  {r:24s} {avg(alone[r] + paired[r]):.3f}")

    with open('misconception_difficulty.json', 'w') as f:
        json.dump({
            r: {'alone': alone[r], 'paired': paired[r]}
            for r in IDS
        }, f, indent=2)
    print("\nWrote misconception_difficulty.json (raw per-item marginals, for later human-data comparison)")


if __name__ == '__main__':
    main()
