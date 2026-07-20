#!/usr/bin/env python3
"""
plot_bayes_1misc_heatmap.py

Present x named "confusion matrix" for the 1-misconception items (categories A
and B), Bayesian ideal observer. Rows = misconception PRESENT in the trace,
columns = misconception NAMED in the statement.

  - The DIAGONAL (present = named) is category A: the statement names the rule
    that is actually there, so agreeing is correct (green is ideal).
  - The OFF-DIAGONAL (present != named) is category B: the statement names a
    foil, so disagreeing is correct (red is ideal).

Cell value = mean posterior marginal P(named rule | trace) from
posterior_over_profiles(); 0.5 neutral, same diverging green/red scale and cell
style as plot_bayes_2misc_heatmap.py.

Two panels split the category-B off-diagonal by the foil's refutation status;
the diagonal category-A cells are identical in both (the shared agree reference):
  (a) foil REFUTED by the trace           (marginal collapses toward 0)
  (b) foil UNSUPPORTED (never had an opportunity to manifest; ~0.2 residual)

A perfect observer shows a green diagonal and a red off-diagonal in both panels.

Run from repo root:
    python3 analysis-Bayesian/plot_bayes_1misc_heatmap.py
"""
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

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
CMAP = LinearSegmentedColormap.from_list('divmarg',
                                         ['#e34948', '#f0efec', '#008300'])
NORM = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
DARK_AT = 0.28


def build_matrices(pool):
    """Return {'refuted': (mean,n), 'unsupported': (mean,n)}; each 6x6 with
    present on rows, named on cols. Diagonal = category A (shared)."""
    idx = {m: i for i, m in enumerate(IDS)}
    diag = {i: [] for i in range(6)}
    off = {'refuted': {}, 'unsupported': {}}
    for it in pool:
        if it['num_misconceptions'] != 1:
            continue
        marg = marginal_rule_probability(
            posterior_over_profiles(it['trace']), it['probed_misconception'])
        present, named = idx[it['misconceptions'][0]], idx[it['probed_misconception']]
        if it['category'] == 'A':
            diag[present].append(marg)
        else:                                   # category B foil
            st = it.get('foil_status')
            if st in off:
                off[st].setdefault((present, named), []).append(marg)
    mats = {}
    for st in ('refuted', 'unsupported'):
        mean, n = np.full((6, 6), np.nan), np.zeros((6, 6), int)
        for i in range(6):
            mean[i, i], n[i, i] = np.mean(diag[i]), len(diag[i])
        for (p, nm), vals in off[st].items():
            mean[p, nm], n[p, nm] = np.mean(vals), len(vals)
        mats[st] = (mean, n)
    return mats


def draw(ax, mean, n, title):
    shown = np.where(np.isnan(mean), 0.5, mean)
    im = ax.imshow(shown, cmap=CMAP, norm=NORM, aspect='equal')
    for i in range(6):
        for j in range(6):
            if n[i, j] == 0:
                ax.text(j, i, '·', ha='center', va='center', color='#898781')
                continue
            dark = abs(mean[i, j] - 0.5) > DARK_AT
            ax.text(j, i - 0.12, f"{mean[i, j]:.2f}", ha='center', va='center',
                    fontsize=8.5, fontweight='bold',
                    color='white' if dark else '#0b0b0b')
            ax.text(j, i + 0.28, f"n={n[i, j]}", ha='center', va='center',
                    fontsize=5.5, color='white' if dark else '#52514e')
    # outline the category-A diagonal so it reads as the agree reference
    for i in range(6):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor='#0b0b0b', lw=1.3))
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels([SHORT[m] for m in IDS], fontsize=8, rotation=30, ha='right')
    ax.set_yticklabels([SHORT[m] for m in IDS], fontsize=8)
    ax.set_xlabel('misconception NAMED in statement', fontsize=9)
    ax.set_ylabel('misconception PRESENT in trace', fontsize=9)
    ax.set_title(title, fontsize=10.5)
    return im


def main():
    pool = json.load(open(os.path.join(BASE_TASK, 'stimulus_pool.json')))
    mats = build_matrices(pool)
    for st in ('refuted', 'unsupported'):
        mean, n = mats[st]
        print(f"\n=== {st} panel (diagonal = A, off-diagonal = B {st}); mean marginal (n) ===")
        print(f"{'present \\ named':>15s} " + " ".join(f"{SHORT[m]:>10s}" for m in IDS))
        for i, p in enumerate(IDS):
            print(f"{SHORT[p]:>15s} " + " ".join(f"{mean[i, j]:>6.2f}({n[i, j]:>2d})" for j in range(6)))

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.0), gridspec_kw={'wspace': 0.30})
    fig.subplots_adjust(top=0.80)
    draw(axes[0], *mats['refuted'],
         '(a) category-B foils REFUTED by the trace\n(boxed diagonal = category A, agreeing is correct)')
    im = draw(axes[1], *mats['unsupported'],
              '(b) category-B foils UNSUPPORTED\n(boxed diagonal = category A, agreeing is correct)')
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02,
                        ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_label('posterior marginal P(named rule | trace)', fontsize=9)
    fig.suptitle('Bayesian ideal observer, one-misconception items: present (rows) × named (columns)',
                 fontsize=12.5, y=0.98)
    p = os.path.join(HERE, 'bayes_1misc_heatmap.png')
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"\nWrote {p}")


if __name__ == '__main__':
    main()
