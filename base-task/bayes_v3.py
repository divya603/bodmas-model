"""
bayes_v3.py

Runs the ideal observer over every item in the v3 pool and saves its response,
the v3 replacement for analysis-Bayesian/b_item_marginals.json.

For each item the observer sees only what a participant sees (the trace) and
scores the probed rule under the 22 hypotheses (expert + 6 singletons +
15 pairs) at epsilon=0. Its answer to "does this statement explain the work?"
is the marginal probability that the student holds the named rule; the binary
judgment collapses that at 0.5, matching how human ratings are collapsed
(>=4 on the 1-6 scale = agree).

Use `probed_marginal` as the observer's response. `map_profile` is stored for
completeness but is NOT a good summary: on ~13% of items the MAP names two
rules for a one-misconception item, always pairing the true rule with
outside_bracket_first. That rule is the only one that removes options rather
than adding them (v2 blocks bracket recursion while outside work remains), so
on a trace that never enters its bracket early, the pair hypothesis gives the
observed path a higher likelihood than the true singleton does. It affects
neither correctness nor any marginal.
"""

import json
from collections import defaultdict

from inference import posterior_over_profiles, marginal_rule_probability
from pool_v3 import HYPOTHESES, IDS, POSITIONS

POOL = 'stimulus_pool_v3.json'
OUT  = 'bayes_per_item_v3.json'


def run(pool_path=POOL, out_path=OUT):
    items = json.load(open(pool_path, encoding='utf-8'))
    rows = []
    for it in items:
        post = posterior_over_profiles(it['trace'], profiles=HYPOTHESES)
        marg = marginal_rule_probability(post, it['probed_misconception'])
        rows.append({
            'id':                   it['id'],
            'pair_id':              it['pair_id'],
            'category':             it['category'],
            'error_position':       it['error_position'],
            'true_misconception':   it['misconceptions'][0],
            'probed_misconception': it['probed_misconception'],
            'foil_status':          it.get('foil_status'),
            'statement_correct':    it['statement_correct'],
            'probed_marginal':      round(marg, 6),
            'observer_agrees':      bool(marg > 0.5),
            'observer_correct':     bool((marg > 0.5) == it['statement_correct']),
            'map_profile':          list(max(post, key=post.get)),
            'map_posterior':        round(max(post.values()), 6),
        })
    json.dump(rows, open(out_path, 'w', encoding='utf-8'), indent=1)
    return rows


def _stats(vals):
    v = sorted(vals)
    return f"min {v[0]:.3f}  mean {sum(v)/len(v):.3f}  max {v[-1]:.3f}"


def summarise(rows):
    n = len(rows)
    print(f"ideal observer run on all {n} items (22 hypotheses, epsilon=0)\n")

    acc = sum(r['observer_correct'] for r in rows)
    print(f"OVERALL ACCURACY: {acc}/{n} = {acc/n:.1%}\n")

    for cat in ('A', 'B'):
        sub = [r for r in rows if r['category'] == cat]
        a = sum(r['observer_correct'] for r in sub)
        print(f"  {cat}: {a}/{len(sub)} correct ({a/len(sub):.0%})   "
              f"P(agree)={sum(r['observer_agrees'] for r in sub)/len(sub):.3f}   "
              f"marginal {_stats([r['probed_marginal'] for r in sub])}")
    print()

    print("A items, probed-rule marginal by rule x position:")
    for m in IDS:
        cells = []
        for p in POSITIONS:
            v = [r['probed_marginal'] for r in rows
                 if r['category'] == 'A' and r['probed_misconception'] == m
                 and r['error_position'] == p]
            cells.append(f"pos{p} {sum(v)/len(v):.3f}")
        print(f"    {m:24s} " + "   ".join(cells))
    print()

    print("B items, foil marginal by status x position (the refutation effect):")
    for st in ('refuted', 'unsupported'):
        for p in POSITIONS:
            v = [r['probed_marginal'] for r in rows if r['foil_status'] == st
                 and r['error_position'] == p]
            fa = sum(r['observer_agrees'] for r in rows if r['foil_status'] == st
                     and r['error_position'] == p) / len(v)
            print(f"    {st:12s} pos{p}:  n={len(v):3d}  {_stats(v)}   P(agree)={fa:.3f}")
    print()

    print("B items, foil marginal by probed rule x status:")
    for m in IDS:
        cells = []
        for st in ('refuted', 'unsupported'):
            v = [r['probed_marginal'] for r in rows
                 if r['probed_misconception'] == m and r['foil_status'] == st]
            cells.append(f"{st[:5]} {sum(v)/len(v):.3f}")
        print(f"    {m:24s} " + "   ".join(cells))
    print()

    mis = [r for r in rows if not r['observer_correct']]
    print(f"items the observer gets WRONG: {len(mis)}")
    for r in mis[:10]:
        print(f"    {r['id']} {r['category']} pos{r['error_position']} "
              f"probed={r['probed_misconception']} status={r['foil_status']} "
              f"marginal={r['probed_marginal']:.3f}")

    # MAP is reported for completeness only — see the note in the module docstring.
    hit      = sum(1 for r in rows if r['map_profile'] == [r['true_misconception']])
    contains = sum(1 for r in rows if r['true_misconception'] in r['map_profile'])
    two      = [r for r in rows if len(r['map_profile']) == 2]
    partners = {x for r in two for x in r['map_profile'] if x != r['true_misconception']}
    print(f"\nMAP hypothesis contains the true rule:          {contains}/{n} ({contains/n:.0%})")
    print(f"MAP hypothesis == the true single rule exactly: {hit}/{n} ({hit/n:.0%})")
    print(f"  the other {len(two)} have a 2-rule MAP, partner always one of {partners or '-'}.")
    print("  This is expected, not an error: outside_bracket_first is the only rule that REMOVES")
    print("  options (v2 blocks bracket recursion while outside work remains), so on a trace that")
    print("  never enters its bracket early, 'also holds outside_bracket_first' makes the observed")
    print("  path MORE likely and the pair outscores the singleton. It changes no item's")
    print("  correctness and no foil marginal. Use the marginal, not the MAP, as the observer's")
    print("  response.")


if __name__ == '__main__':
    rows = run()
    summarise(rows)
    print(f"\nwrote {OUT}")
