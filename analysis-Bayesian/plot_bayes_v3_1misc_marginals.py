#!/usr/bin/env python3
"""
plot_bayes_v3_1misc_marginals.py

v3 version of plot_bayes_1misc_marginals.py. The raw per-item posterior
P(named rule | trace), one dot per pool item, grouped by the misconception
PRESENT in the trace, with NO averaging. The item-to-item structure is the
point, and under v3 there are two structures worth seeing at once:

  - category A (blue) is a flat line of dots at exactly 1.000. The observer is
    a logical oracle on present rules, so this is a point mass, not a tight
    distribution. Drawn as dots rather than summarised so the ceiling is
    visibly a ceiling.
  - category B (orange) splits into two bands: refuted foils at ~0 and
    unsupported foils at ~0.17 to 0.33. That gap IS the refutation manipulation.

Each rule gets four dot columns: category x error position. Position is the new
v3 factor, so it is drawn everywhere rather than pooled away; the two positions
land on top of each other, which is the finding.

Dashed line at 0.5 = the agree/disagree decision boundary. Dotted line at 0.15 =
the refuted/unsupported cut used to assign foil_status. Dots within a column are
rank-ordered left to right for visibility only.

Run from repo root:
    python3 analysis-Bayesian/plot_bayes_v3_1misc_marginals.py
"""
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bayes_v3_common import (HERE, IDS, SHORT, BLUE, ORANGE, POSITIONS,
                             POS_MARKER, REFUTED_CUT, load_rows)

# (category, position) -> (x offset within the rule's slot, colour, marker)
SLOTS = [('A', 1, -0.285, BLUE), ('A', 3, -0.095, BLUE),
         ('B', 1, 0.095, ORANGE), ('B', 3, 0.285, ORANGE)]


def main():
    rows = load_rows()

    cells = {}
    for r in rows:
        key = (r['true_misconception'], r['category'], r['error_position'])
        cells.setdefault(key, []).append(r['probed_marginal'])

    for cat in 'AB':
        print(f"\ncategory {cat}: per-item P(named rule | trace), by present rule x position")
        for m in IDS:
            for pos in POSITIONS:
                vals = sorted(cells.get((m, cat, pos), []))
                lo, hi = (min(vals), max(vals)) if vals else (float('nan'),) * 2
                print(f"  {SHORT[m]:>10s} pos{pos}  n={len(vals):>2d}  "
                      f"min {lo:.3f}  max {hi:.3f}  mean {np.mean(vals):.3f}")

    fig, ax = plt.subplots(figsize=(12.4, 5.2))
    seen = set()
    for gi, m in enumerate(IDS):
        for cat, pos, off, color in SLOTS:
            vals = np.sort(cells.get((m, cat, pos), []))
            if not len(vals):
                continue
            xs = gi + off + np.linspace(-0.062, 0.062, len(vals))
            lab = f"category {cat}, error at step {pos}"
            ax.scatter(xs, vals, s=26, color=color, alpha=0.85,
                       marker=POS_MARKER[pos], edgecolors='white', linewidths=0.5,
                       zorder=3, label=None if lab in seen else lab)
            seen.add(lab)
        if gi:
            ax.axvline(gi - 0.5, color='#dcdad5', lw=0.8, zorder=0)

    ax.axhline(0.5, ls='--', lw=1, color='grey')
    ax.text(-0.44, 0.525, 'decision boundary', color='grey', fontsize=7, ha='left')
    ax.axhline(REFUTED_CUT, ls=':', lw=1, color='#8a6d3b')
    ax.text(-0.44, REFUTED_CUT + 0.025, f'refuted cut ({REFUTED_CUT})',
            color='#8a6d3b', fontsize=7, ha='left')

    ax.set_xticks(range(6))
    ax.set_xticklabels([SHORT[m] for m in IDS])
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel('P(named rule | trace)')
    ax.set_xlabel('misconception present in trace')
    ax.set_title('Bayesian ideal observer, v3 pool: per-item posterior on the named rule\n'
                 'one dot per item (12 per A column, 18 to 27 per B column); no averaging',
                 fontsize=11)
    ax.legend(fontsize=8, loc='center right', framealpha=0.95)
    fig.tight_layout()
    p = os.path.join(HERE, 'bayes_v3_1misc_marginals.png')
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"\nWrote {p}")


if __name__ == '__main__':
    main()
