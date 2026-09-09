"""
bayes_hidden.py

Runs the ideal observer over every pool item under each HIDDEN-STEP condition
and saves its response, the hidden-trace counterpart of bayes.py.

Conditions per item:
  none        every line visible (the pilot-v4 baseline)
  s2          line s2 hidden
  s4          line s4 hidden
  error_line  the line PRODUCED BY the error is hidden (s1 if the error is at
              step 1, s3 if at step 3). This is the only condition that reliably
              moves the observer, because steps 1 and 3 carry 88% of all the
              evidence while steps 2, 4 and 5 carry almost none.

The observer sees only what a participant would see. With line s_k hidden the two
likelihood factors touching it collapse into a marginal over every value the
hidden state could have taken (see hidden.py).

Use `probed_marginal` as the observer's response, not `map_profile`, for the same
reason as in the fully observed case.
"""

import json
from collections import defaultdict

from hidden import hidden_posterior, visible_trace
from inference import posterior_over_profiles, marginal_rule_probability
from pool import HYPOTHESES, IDS, POSITIONS

POOL = 'stimulus_pool.json'
OUT = 'bayes_per_item_hidden.json'
HIDE_LINES = (2, 4)


def conditions(item):
    """{condition name: hidden line index or None}."""
    return {'none': None, 's2': 2, 's4': 4, 'error_line': item['error_position']}


def run(pool_path=POOL, out_path=OUT):
    items = json.load(open(pool_path, encoding='utf-8'))
    rows = []
    for it in items:
        probed = it['probed_misconception']
        base = None
        for cond, k in conditions(it).items():
            if k is None:
                post = posterior_over_profiles(it['trace'], profiles=HYPOTHESES)
                shown = it['trace']
            else:
                post = hidden_posterior(it['trace'], k, profiles=HYPOTHESES)
                shown = visible_trace(it['trace'], k)
            marg = marginal_rule_probability(post, probed)
            if cond == 'none':
                base = marg
            rows.append({
                'id': it['id'],
                'pair_id': it['pair_id'],
                'category': it['category'],
                'error_position': it['error_position'],
                'condition': cond,
                'hidden_line': k,
                'n_lines_shown': len(shown),
                'true_misconception': it['misconceptions'][0],
                'probed_misconception': probed,
                'statement_correct': it['statement_correct'],
                'probed_marginal': round(marg, 6),
                'delta_vs_full': round(marg - base, 6),
                'observer_agrees': bool(marg > 0.5),
                'observer_correct': bool((marg > 0.5) == it['statement_correct']),
            })
    json.dump(rows, open(out_path, 'w', encoding='utf-8'), indent=1)
    return rows


def _stats(v):
    v = sorted(v)
    return f"min {v[0]:.3f}  mean {sum(v)/len(v):.3f}  max {v[-1]:.3f}"


def summarise(rows):
    conds = ['none', 's2', 's4', 'error_line']
    by = defaultdict(list)
    for r in rows:
        by[(r['condition'], r['category'])].append(r)

    n_items = len({r['id'] for r in rows})
    print(f"ideal observer on {n_items} items x {len(conds)} hidden-step conditions "
          f"(22 hypotheses, epsilon=0)\n")

    print(f"{'condition':>12s} {'acc':>9s}   {'category A marginal':>34s}   {'category B marginal':>34s}")
    for c in conds:
        a, b = by[(c, 'A')], by[(c, 'B')]
        acc = sum(r['observer_correct'] for r in a + b)
        print(f"  {c:>10s} {acc:>4d}/{len(a)+len(b):<4d}   "
              f"{_stats([r['probed_marginal'] for r in a]):>34s}   "
              f"{_stats([r['probed_marginal'] for r in b]):>34s}")

    print("\nitems the hiding actually MOVES (|delta| > 1e-6), by category and error position")
    print(f"{'condition':>12s}" + "".join(f"  {c} pos{p}" for c in 'AB' for p in POSITIONS))
    for c in conds:
        if c == 'none':
            continue
        cells = []
        for cat in 'AB':
            for p in POSITIONS:
                grp = [r for r in by[(c, cat)] if r['error_position'] == p]
                cells.append(f"  {sum(1 for r in grp if abs(r['delta_vs_full']) > 1e-6):2d}/{len(grp):<3d}")
        print(f"  {c:>10s}" + "".join(cells))

    print("\ncategory A, mean marginal on the PRESENT rule by condition x error position")
    for c in conds:
        row = []
        for p in POSITIONS:
            v = [r['probed_marginal'] for r in by[(c, 'A')] if r['error_position'] == p]
            row.append(f"pos{p} {sum(v)/len(v):.3f}")
        print(f"  {c:>10s}  " + "   ".join(row))

    worst = sorted((r for r in rows if r['category'] == 'A'),
                   key=lambda r: r['probed_marginal'])[:8]
    print("\nlowest category-A marginals across all conditions (where hiding bites):")
    for r in worst:
        print(f"  {r['id']} {r['true_misconception']:>22s} err@{r['error_position']} "
              f"cond={r['condition']:>10s} -> {r['probed_marginal']:.3f}")


if __name__ == '__main__':
    rows = run()
    summarise(rows)
    print(f"\nwrote {OUT}")
