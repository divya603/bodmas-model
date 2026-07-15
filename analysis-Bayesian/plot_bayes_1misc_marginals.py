#!/usr/bin/env python3
"""
plot_bayes_1misc_marginals.py

Bayesian ideal observer on the 1-misconception items (categories A and B):
the raw per-item posterior marginal P(named rule | trace), one dot per pool
item, grouped by the misconception PRESENT in the trace. No averaging: the
marginal varies item to item (tightly in A; in B it splits into "actively
refuted" items near 0 and "no evidence either way" items around 0.2), and
that structure is the point.

Blue dots: statement names the present misconception (category A, should be
high). Orange dots: statement names an absent one (category B, should be
low). Dashed line at 0.5 = the agree/disagree decision boundary. 10 items
per dot column; dots within a column are rank-ordered left to right for
visibility only.

Run from repo root:
    python3 analysis-Bayesian/plot_bayes_1misc_marginals.py
"""

import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_TASK = os.path.join(os.path.dirname(HERE), 'base-task')
sys.path.insert(0, BASE_TASK)

from inference import posterior_over_profiles, marginal_rule_probability  # noqa: E402

IDS = ['add_before_mul', 'add_before_div', 'sub_before_mul', 'sub_before_div',
       'same_priority_rtl', 'outside_bracket_first']
SHORT = {
    'add_before_mul': 'add<×', 'add_before_div': 'add<÷',
    'sub_before_mul': 'sub<×', 'sub_before_div': 'sub<÷',
    'same_priority_rtl': 'RTL', 'outside_bracket_first': 'outside()',
}
BLUE, ORANGE = '#2a78d6', '#eb6834'


def main():
    pool = json.load(open(os.path.join(BASE_TASK, 'stimulus_pool.json')))
    items = [it for it in pool if it['num_misconceptions'] == 1]

    cells = {}
    for it in items:
        post = posterior_over_profiles(it['trace'])
        marg = marginal_rule_probability(post, it['probed_misconception'])
        cells.setdefault((it['misconceptions'][0], it['category']), []).append(marg)

    for cat in 'AB':
        print(f"category {cat} — per-item P(named rule | trace)")
        for m in IDS:
            vals = sorted(cells[(m, cat)])
            print(f"  {SHORT[m]:>10s} " + " ".join(f"{v:.2f}" for v in vals))

    fig, ax = plt.subplots(figsize=(11, 4.8))
    for gi, m in enumerate(IDS):
        for off, cat, color in [(-0.19, 'A', BLUE), (0.19, 'B', ORANGE)]:
            vals = np.sort(cells[(m, cat)])
            xs = gi + off + np.linspace(-0.10, 0.10, len(vals))
            ax.scatter(xs, vals, s=34, color=color, alpha=0.85,
                       edgecolors='white', linewidths=0.6, zorder=3,
                       label={'A': 'statement matches (A)',
                              'B': 'statement is a foil (B)'}[cat] if gi == 0 else None)
    ax.axhline(0.5, ls='--', lw=1, color='grey')
    ax.text(-0.38, 0.53, 'decision boundary', color='grey', fontsize=7, ha='left')
    ax.set_xticks(range(6))
    ax.set_xticklabels([SHORT[m] for m in IDS])
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel('P(named rule | trace)')
    ax.set_xlabel('misconception present in trace')
    ax.set_title('Bayesian ideal observer, 1-misconception items: per-item posterior '
                 'on the named rule\n(one dot per pool item, 10 per column; no averaging)',
                 fontsize=11)
    ax.legend(fontsize=8, loc='center right')
    fig.tight_layout()
    p = os.path.join(HERE, 'bayes_1misc_marginals.png')
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"\nWrote {p}")


if __name__ == '__main__':
    main()
