#!/usr/bin/env python3
"""
plot_bayes_hidden_dist_A.py

Category A (the statement names the rule that IS in the trace): the distribution
of the observer's posterior on that rule, one panel per misconception, with one
series per HIDDEN-STEP condition.

Conditions (see base-task/bayes_hidden.py):
  none        every line visible
  s2          line s2 hidden
  s4          line s4 hidden
  error_line  the line PRODUCED by the error is hidden (s1 if the error is at
              step 1, s3 if at step 3)

⚠️ Read the overlap correctly: `none`, `s2` and `s4` are EXACTLY equal on all 120
category-A items, so their three series sit on top of each other at 1.000 in
every panel. They are drawn with small horizontal offsets purely so you can see
that all three are present. Only `error_line` ever departs from 1.000, and only
for outside_bracket_first.

⚠️ Stems on the exact observed values, not KDEs. The marginals here take four
distinct values in total (1.000, 0.706, 0.571, 0.556); a Gaussian KDE would
invent a smooth density where the data is four spikes.

Run from repo root:
    python3 analysis-Bayesian/plot_bayes_hidden_dist_A.py
"""
import json
import os
from collections import Counter, defaultdict

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bayes_common import HERE, BASE_TASK, IDS, SHORT

ROWS = os.path.join(BASE_TASK, 'bayes_per_item_hidden.json')
CONDS = ['none', 's2', 's4', 'error_line']
LABEL = {'none': 'all lines shown', 's2': 'hide s2', 's4': 'hide s4',
         'error_line': "hide the error's own line"}
COLOR = {'none': '#3b3a36', 's2': '#2a78d6', 's4': '#0f9d8f', 'error_line': '#eb6834'}
MARKER = {'none': 'o', 's2': 's', 's4': '^', 'error_line': 'D'}
DX = 0.009
STEM_W = 2.0


def main():
    if not os.path.exists(ROWS):
        raise SystemExit(f"missing {ROWS}\nGenerate it first: "
                         f"cd base-task && python3 bayes_hidden.py")
    rows = [r for r in json.load(open(ROWS, encoding='utf-8')) if r['category'] == 'A']

    data = defaultdict(lambda: defaultdict(list))
    for r in rows:
        data[r['true_misconception']][r['condition']].append(r['probed_marginal'])

    fig, axes = plt.subplots(2, 3, figsize=(13.4, 6.6), sharex=True, sharey=True)
    for ax, m in zip(axes.flat, IDS):
        for ci, cond in enumerate(CONDS):
            vals = data[m][cond]
            if not vals:
                continue
            counts = Counter(round(v, 6) for v in vals)
            n = len(vals)
            xoff = (ci - 1.5) * DX
            for v, c in sorted(counts.items()):
                ax.vlines(v + xoff, 0, c / n, color=COLOR[cond], lw=STEM_W,
                          alpha=0.9, zorder=3)
                ax.plot(v + xoff, c / n, marker=MARKER[cond], ms=4.2,
                        color=COLOR[cond], zorder=4)
        ax.axvline(0.5, ls='--', lw=0.9, color='#b8b6b0', zorder=1)
        moved = [v for v in data[m]['error_line'] if v < 0.999]
        if moved:
            ax.text(0.33, 0.48, f'{len(moved)} of {len(data[m]["error_line"])} items\n'
                               f'fall to {min(moved):.2f}-{max(moved):.2f}',
                    fontsize=7.2, color=COLOR['error_line'], ha='center', va='center')
        ax.set_xlim(-0.02, 1.03)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.5, 1.0]); ax.set_yticklabels(['0', '.5', '1'], fontsize=7.5)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.tick_params(axis='x', labelsize=7.5)
        ax.set_title(SHORT[m], fontsize=10.5)
    for ax in axes[1]:
        ax.set_xlabel('P(named rule | what is shown)', fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel('proportion of items', fontsize=8.5)

    handles = [plt.Line2D([], [], color=COLOR[c], lw=2, marker=MARKER[c], ms=5,
                          label=LABEL[c]) for c in CONDS]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.915),
               ncol=4, fontsize=8.5, frameon=False)
    fig.suptitle('Bayesian ideal observer, category A (the statement names the rule that is there)\n'
                 'posterior on that rule, by misconception and hidden-step condition, '
                 '20 items per series\n'
                 'The first three conditions are EXACTLY equal everywhere and overlap at 1.000; '
                 'they are offset only to be visible.\n'
                 'Hiding never changes a judgement: the observer is 240/240 correct in all four '
                 'conditions.',
                 fontsize=10.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.855])
    p = os.path.join(HERE, 'bayes_hidden_dist_A.png')
    fig.savefig(p, dpi=140); plt.close(fig)

    print("category A, P(present rule | shown), by rule x condition")
    for m in IDS:
        cells = []
        for c in CONDS:
            v = data[m][c]
            cells.append(f"{c}={sum(v)/len(v):.3f}")
        print(f"  {SHORT[m]:>9s}  " + "   ".join(cells))
    print(f"\nWrote {p}")


if __name__ == '__main__':
    main()
