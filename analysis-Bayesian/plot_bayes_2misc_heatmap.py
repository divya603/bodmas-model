#!/usr/bin/env python3
"""
plot_bayes_2misc_heatmap.py

Bayesian ideal-observer analog of analysis_human/plot_2misc_heatmap.py:
agreement structure on the 2-misconception items (categories C and D). The
observer's "agreement" with the belief statement is the posterior marginal it
places on the PROBED rule given the trace, P(rule in learner's policy |
trace), from posterior_over_profiles() over all 22 learner profiles. 0.5 is
the neutral point (the analog of 3.5 on the human/LLM 1-6 scale).

  (a) Category C (6x6 square): y = named rule (present), x = the OTHER
      present rule; (named, partner) uniquely identifies the pair.
  (b) Category D (6x15 rectangle): y = named foil (absent), x = the PAIR
      actually present. Only the 10 pairs not containing the foil are
      possible per row (gray = structurally impossible). One pool item per
      cell by construction.

Run from repo root:
    python3 analysis-Bayesian/plot_bayes_2misc_heatmap.py
"""

import json
import os
import sys
from itertools import combinations

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
PAIRS = list(combinations(range(6), 2))
PAIR_LABELS = [f"{SHORT[IDS[i]]}\n{SHORT[IDS[j]]}" for i, j in PAIRS]

# diverging: disagree (red) .. gray at 0.5 .. agree (green)
CMAP = LinearSegmentedColormap.from_list('divmarg',
                                         ['#e34948', '#f0efec', '#008300'])
NORM = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
DARK_AT = 0.28


def cell_stats(items):
    acc_c, acc_d = {}, {}
    for it in items:
        post = posterior_over_profiles(it['trace'])
        marg = marginal_rule_probability(post, it['probed_misconception'])
        p = IDS.index(it['probed_misconception'])
        pair = tuple(sorted(IDS.index(m) for m in it['misconceptions']))
        if it['category'] == 'C':
            partner = next(m for m in pair if m != p)
            acc_c.setdefault((p, partner), []).append(marg)
        else:
            acc_d.setdefault((p, PAIRS.index(pair)), []).append(marg)
    mean_c, n_c = np.full((6, 6), np.nan), np.zeros((6, 6), int)
    for (i, j), vals in acc_c.items():
        mean_c[i, j] = np.mean(vals); n_c[i, j] = len(vals)
    mean_d, n_d = np.full((6, 15), np.nan), np.zeros((6, 15), int)
    for (i, k), vals in acc_d.items():
        mean_d[i, k] = np.mean(vals); n_d[i, k] = len(vals)
    return (mean_c, n_c), (mean_d, n_d)


def impossible_c():
    return np.eye(6, dtype=bool)


def impossible_d():
    imp = np.zeros((6, 15), bool)
    for k, (i, j) in enumerate(PAIRS):
        imp[i, k] = imp[j, k] = True
    return imp


def draw(ax, mean, n, impossible, xlabels, title, xlabel):
    shown = np.ma.masked_where(impossible, np.where(np.isnan(mean), 0.5, mean))
    cm = CMAP.copy()
    cm.set_bad('#d8d7d2')
    im = ax.imshow(shown, cmap=cm, norm=NORM, aspect='equal')
    rows, cols = mean.shape
    for i in range(rows):
        for j in range(cols):
            if impossible[i, j]:
                continue
            if n[i, j] == 0:
                ax.text(j, i, '·', ha='center', va='center', color='#898781')
                continue
            dark = abs(mean[i, j] - 0.5) > DARK_AT
            ax.text(j, i - 0.12, f"{mean[i, j]:.2f}", ha='center', va='center',
                    fontsize=8.5, fontweight='bold',
                    color='white' if dark else '#0b0b0b')
            ax.text(j, i + 0.27, f"n={n[i, j]}", ha='center', va='center',
                    fontsize=5.5, color='white' if dark else '#52514e')
    ax.set_xticks(range(cols)); ax.set_yticks(range(rows))
    ax.set_xticklabels(xlabels, fontsize=7.5)
    ax.set_yticklabels([SHORT[m] for m in IDS], fontsize=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel('misconception NAMED in statement', fontsize=9)
    ax.set_title(title, fontsize=10.5)
    return im


def main():
    pool = json.load(open(os.path.join(BASE_TASK, 'stimulus_pool.json')))
    items = [it for it in pool if it['num_misconceptions'] == 2]
    n_c = sum(it['category'] == 'C' for it in items)
    print(f"2-misconception items: {len(items)} (C={n_c}, D={len(items) - n_c})")
    c_mats, d_mats = cell_stats(items)

    mean_c, n_cm = c_mats
    print("\ncategory C — named rule PRESENT (should be > 0.5); mean marginal (n)")
    print(f"{'named \\ partner':>16s} " + " ".join(f"{SHORT[m]:>10s}" for m in IDS))
    for i, p in enumerate(IDS):
        cells = " ".join("         ·" if n_cm[i, j] == 0 else
                         f"{mean_c[i, j]:>6.2f}({n_cm[i, j]:>2d})" for j in range(6))
        print(f"{SHORT[p]:>16s} {cells}")
    mean_d, n_dm = d_mats
    print("\ncategory D — named rule ABSENT (should be < 0.5); marginal (n) by present pair")
    for i, p in enumerate(IDS):
        cells = [f"{PAIR_LABELS[k].replace(chr(10), '+')}={mean_d[i, k]:.2f}({n_dm[i, k]})"
                 for k in range(15) if n_dm[i, k] > 0]
        print(f"  {SHORT[p]:>10s}: " + "  ".join(cells))

    fig, axes = plt.subplots(1, 2, figsize=(17.5, 4.9),
                             gridspec_kw={'width_ratios': [6, 15.4], 'wspace': 0.14})
    draw(axes[0], *c_mats, impossible_c(), [SHORT[m] for m in IDS],
         '(a) Category C: statement names a PRESENT rule\n'
         'ideal observer should agree (green)',
         'OTHER misconception present in trace')
    im = draw(axes[1], *d_mats, impossible_d(), PAIR_LABELS,
              '(b) Category D: statement names an ABSENT rule\n'
              'ideal observer should disagree (red)',
              'PAIR of misconceptions present in trace')
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.015,
                        ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_label('posterior marginal P(named rule | trace)', fontsize=9)

    fig.suptitle('Bayesian ideal observer, two-misconception items: present × shown × agreement',
                 fontsize=12.5)
    p = os.path.join(HERE, 'bayes_2misc_heatmap.png')
    fig.savefig(p, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"\nWrote {p}")


if __name__ == '__main__':
    main()
