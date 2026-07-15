#!/usr/bin/env python3
"""
plot_human_1misc_distributions.py

Human counterpart of analysis-Bayesian/plot_bayes_1misc_distributions.py:
distribution of the 1-6 Likert rating on the 1-misconception trials, same
panel layouts, with the decision boundary at 3.5 instead of 0.5.

  human_1misc_dist_A.png — category A (statement matches; agree correct).
      6 panels grouped by the misconception present (= named in A).
  human_1misc_dist_B.png — category B (statement is a foil; disagree
      correct). Row 1 grouped by the misconception PRESENT in the trace,
      row 2 by the misconception NAMED in the statement, row 3 = trials on
      the REFUTED-only items (the 12 B items whose ideal-observer marginal
      on the foil is < 0.15, from analysis-Bayesian/b_item_marginals.json),
      grouped by named foil. If humans use refutation evidence, row 3
      should sit lower than row 2.

Responses are shown BINARY: the 1-6 rating is collapsed to disagree (<=3)
vs agree (>=4), the same collapse used for accuracy scoring, and each panel
is the probability distribution over those two responses (fixed 0-1 y-axis
so peaks are comparable across panels; trial counts on the bars; dashed
line at 0.5). The full 1-6 scale proved too noisy at n=24 per cell to show
a picture. The Bayesian counterpart keeps KDE curves because its marginals
are continuous.

Usage (from repo root):
    python3 analysis_human/plot_human_1misc_distributions.py
"""

import argparse
import json
import os
from collections import Counter

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BAYES_MARGINALS = os.path.join(os.path.dirname(HERE), 'analysis-Bayesian',
                               'b_item_marginals.json')

IDS = ['add_before_mul', 'add_before_div', 'sub_before_mul', 'sub_before_div',
       'same_priority_rtl', 'outside_bracket_first']
SHORT = {
    'add_before_mul': 'add<×', 'add_before_div': 'add<÷',
    'sub_before_mul': 'sub<×', 'sub_before_div': 'sub<÷',
    'same_priority_rtl': 'RTL', 'outside_bracket_first': 'outside()',
}
GREEN, RED = '#008300', '#e34948'   # agree = green, disagree = red (heatmap convention)
XTICKLABELS = ['disagree\n(1–3)', 'agree\n(4–6)']


def panel(ax, vals, title):
    if len(vals) == 0:
        ax.text(0.5, 0.5, 'no refuted items\n(foil never testable)', ha='center',
                va='center', fontsize=8, color='#898781', transform=ax.transAxes)
        ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, 1)
        ax.set_xticks([0, 1]); ax.set_xticklabels(XTICKLABELS, fontsize=8)
        ax.set_yticks([0, 0.5, 1])
        ax.set_title(title, fontsize=10)
        return
    n = len(vals)
    yes = sum(v >= 4 for v in vals)
    props = [(n - yes) / n, yes / n]
    ax.bar([0, 1], props, width=0.6, color=[RED, GREEN], alpha=0.88)
    for x, (p, k) in enumerate(zip(props, [n - yes, yes])):
        ax.text(x, p + 0.03, str(k), ha='center', va='bottom', fontsize=8,
                color='#52514e')
    ax.axhline(0.5, ls='--', lw=0.8, color='grey')
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 1.12)
    ax.set_xticks([0, 1]); ax.set_xticklabels(XTICKLABELS, fontsize=8)
    ax.set_yticks([0, 0.5, 1])
    ax.set_title(title, fontsize=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/real-all-main-data.json')
    ap.add_argument('--out', default='analysis_human/plots')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    bay = json.load(open(BAYES_MARGINALS))
    refuted_ids = {i for i, v in bay['b_marginals'].items() if v < bay['refuted_cut']}

    data = json.load(open(args.data))
    trials = []
    for r in data:
        d = r['data']
        if d.get('done') is True and d.get('recruitmentService') == 'prolific':
            trials += [t for t in d['pageData_exp']['visit_0']['data']
                       if isinstance(t, dict) and 'response' in t
                       and t.get('num_misconceptions') == 1]

    a_by_present = {m: [] for m in IDS}
    b_by_present = {m: [] for m in IDS}
    b_by_named = {m: [] for m in IDS}
    b_refuted = {m: [] for m in IDS}
    for t in trials:
        if t['category'] == 'A':
            a_by_present[t['misconceptions'][0]].append(t['response'])
        else:
            b_by_present[t['misconceptions'][0]].append(t['response'])
            b_by_named[t['probed_misconception']].append(t['response'])
            if t['id'] in refuted_ids:
                b_refuted[t['probed_misconception']].append(t['response'])

    print(f"1-misc trials: {len(trials)}  "
          f"(A={sum(t['category'] == 'A' for t in trials)}, "
          f"B={sum(t['category'] == 'B' for t in trials)})")
    for name, cell in [('A by present', a_by_present), ('B by present', b_by_present),
                       ('B by named', b_by_named), ('B refuted-only', b_refuted)]:
        print(f"  {name:15s} " + "  ".join(f"{SHORT[m]}:{len(cell[m])}" for m in IDS))
        rates = "  ".join(
            f"{SHORT[m]}:{np.mean([v >= 4 for v in cell[m]]):.2f}" if cell[m] else f"{SHORT[m]}:-"
            for m in IDS)
        print(f"  {'  P(agree)':15s} " + rates)

    # ── figure 1: category A ──
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for ax, m in zip(axes.flat, IDS):
        panel(ax, a_by_present[m], f"{SHORT[m]}  (n={len(a_by_present[m])})")
    for ax in axes[1]:
        ax.set_xlabel('response', fontsize=9)
    fig.suptitle('Humans — category A (statement matches; agree correct):\n'
                 'probability of agree (green) vs disagree (red) by misconception present '
                 '(24 participants; counts on bars; correct answer = agree)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p1 = os.path.join(args.out, 'human_1misc_dist_A.png')
    fig.savefig(p1, dpi=140)
    plt.close(fig)

    # ── figure 2: category B — two groupings + refuted-only row ──
    fig, axes = plt.subplots(3, 6, figsize=(16.5, 8.6), sharex=True)
    for j, m in enumerate(IDS):
        panel(axes[0, j], b_by_present[m], f"{SHORT[m]}  (n={len(b_by_present[m])})")
        panel(axes[1, j], b_by_named[m], f"{SHORT[m]}  (n={len(b_by_named[m])})")
        panel(axes[2, j], b_refuted[m], f"{SHORT[m]}  (n={len(b_refuted[m])})")
        axes[2, j].set_xlabel('response', fontsize=8)
    axes[0, 0].set_ylabel('grouped by misconception\nPRESENT in trace', fontsize=9)
    axes[1, 0].set_ylabel('grouped by misconception\nNAMED in statement (foil)', fontsize=9)
    axes[2, 0].set_ylabel('REFUTED items only\n(ideal-obs P < 0.15), by named foil', fontsize=9)
    fig.suptitle('Humans — category B (statement is a foil; disagree correct): '
                 'probability of agree (green) vs disagree (red)\nsame trials: two groupings, then the '
                 'refutable-item subset (counts on bars; correct answer = disagree)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p2 = os.path.join(args.out, 'human_1misc_dist_B.png')
    fig.savefig(p2, dpi=140)
    plt.close(fig)

    print(f"\nWrote {p1}\nWrote {p2}")


if __name__ == '__main__':
    main()
