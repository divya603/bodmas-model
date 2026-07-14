#!/usr/bin/env python3
"""
make_llm_2misc_heatmap.py

LLM analog of analysis_human/plot_2misc_heatmap.py: agreement structure on
the 2-misconception items (categories C and D), one row of panels per regime
(haiku thinking, haiku direct, gpt-4o direct). Each regime rated every pool
item once (temp=0).

  (a) Category C (6x6 square): y = named misconception (present), x = the
      OTHER present one; (named, partner) uniquely identifies the pair.
  (b) Category D (6x15 rectangle): y = named foil (absent), x = the PAIR
      actually present. Only the 10 pairs not containing the foil are
      possible per row (gray = structurally impossible). One pool item per
      cell by construction.

Reads the full-pool runs from results/raw_*.jsonl (subject_id contains
"all_items"); results/parsed_all.parquet holds only the 24-item factorial
pilot sessions.

Run from llm_exp/:
    python3 make_llm_2misc_heatmap.py
"""

import glob
import json
import os
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, 'plots')

IDS = ['add_before_mul', 'add_before_div', 'sub_before_mul', 'sub_before_div',
       'same_priority_rtl', 'outside_bracket_first']
SHORT = {
    'add_before_mul': 'add<×', 'add_before_div': 'add<÷',
    'sub_before_mul': 'sub<×', 'sub_before_div': 'sub<÷',
    'same_priority_rtl': 'RTL', 'outside_bracket_first': 'outside()',
}
PAIRS = list(combinations(range(6), 2))
PAIR_LABELS = [f"{SHORT[IDS[i]]}\n{SHORT[IDS[j]]}" for i, j in PAIRS]
REGIMES = [
    ('haiku (thinking)', 'anthropic/claude-haiku-4.5', 'thinking'),
    ('haiku (direct)', 'anthropic/claude-haiku-4.5', 'direct'),
    ('gpt-4o (direct)', 'openai/gpt-4o', 'direct'),
]

# diverging: disagree (red) .. gray boundary .. agree (green)
CMAP = LinearSegmentedColormap.from_list('divratings',
                                         ['#e34948', '#f0efec', '#008300'])
NORM = TwoSlopeNorm(vmin=1.0, vcenter=3.5, vmax=6.0)
DARK_AT = 1.4


def cell_stats(rows):
    acc_c, acc_d = {}, {}
    for r in rows.itertuples():
        p = IDS.index(r.probed_misconception)
        pair = tuple(sorted(IDS.index(m) for m in r.misconceptions))
        if r.category == 'C':
            partner = next(m for m in pair if m != p)
            acc_c.setdefault((p, partner), []).append(r.response)
        else:
            acc_d.setdefault((p, PAIRS.index(pair)), []).append(r.response)
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
    shown = np.ma.masked_where(impossible, np.where(np.isnan(mean), 3.5, mean))
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
            dark = abs(mean[i, j] - 3.5) > DARK_AT
            ax.text(j, i - 0.12, f"{mean[i, j]:.1f}", ha='center', va='center',
                    fontsize=8, fontweight='bold',
                    color='white' if dark else '#0b0b0b')
            ax.text(j, i + 0.27, f"n={n[i, j]}", ha='center', va='center',
                    fontsize=5.5, color='white' if dark else '#52514e')
    ax.set_xticks(range(cols)); ax.set_yticks(range(rows))
    ax.set_xticklabels(xlabels, fontsize=7)
    ax.set_yticklabels([SHORT[m] for m in IDS], fontsize=7.5)
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.set_ylabel('misconception NAMED in statement', fontsize=8.5)
    ax.set_title(title, fontsize=10)
    return im


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rows = []
    for path in sorted(glob.glob(os.path.join(HERE, 'results', 'raw_*.jsonl'))):
        for line in open(path):
            r = json.loads(line)
            if 'all_items' in r['subject_id'] and r.get('error') is None \
                    and r.get('response') is not None:
                rows.append(r)
    df = pd.DataFrame(rows)
    df = df[df.num_misconceptions == 2]
    df = df.drop_duplicates(subset=['model', 'effort', 'id'], keep='first')

    fig, axes = plt.subplots(3, 2, figsize=(17.5, 15),
                             gridspec_kw={'width_ratios': [6, 15.4],
                                          'wspace': 0.14, 'hspace': 0.5})
    fig.subplots_adjust(top=0.90, bottom=0.05, left=0.07, right=0.88)
    im = None
    for row, (label, model, effort) in enumerate(REGIMES):
        sub = df[(df.model == model) & (df.effort == effort)]
        print(f"{label}: {len(sub)} two-misconception items "
              f"(C={int((sub.category == 'C').sum())}, D={int((sub.category == 'D').sum())})")
        c_mats, d_mats = cell_stats(sub)
        draw(axes[row, 0], *c_mats, impossible_c(), [SHORT[m] for m in IDS],
             '(a) statement names a PRESENT misconception\n(correct = agree)',
             'OTHER misconception present in trace')
        im = draw(axes[row, 1], *d_mats, impossible_d(), PAIR_LABELS,
                  '(b) statement names an ABSENT misconception\n(correct = disagree)',
                  'PAIR of misconceptions present in trace')
        mid = (axes[row, 0].get_position().x1 + axes[row, 1].get_position().x0) / 2
        fig.text(mid, axes[row, 0].get_position().y1 + 0.038, label,
                 ha='center', va='bottom', fontsize=13, fontweight='bold')

    cax = fig.add_axes([0.92, 0.30, 0.015, 0.40])
    cbar = fig.colorbar(im, cax=cax, ticks=[1, 2, 3, 3.5, 4, 5, 6])
    cbar.ax.set_yticklabels(['1 (SD)', '2', '3', '3.5', '4', '5', '6 (SA)'], fontsize=8)
    cbar.set_label('mean rating (>3.5 = agree side)', fontsize=9)
    fig.suptitle('LLM two-misconception items: present × shown × agreement (all 120 C/D items)',
                 fontsize=13.5, y=0.99)
    p = os.path.join(OUTDIR, 'llm_2misc_heatmap.png')
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"Wrote {p}")


if __name__ == '__main__':
    main()
