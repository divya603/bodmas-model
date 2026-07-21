#!/usr/bin/env python3
"""
plot_2misc_heatmap_dots.py

Raw-value version of plot_2misc_heatmap.py (which it leaves untouched). Same
two panels, same cells, same diverging colour for the CELL MEAN, but each cell
also shows the individual trials behind that mean:

    within a cell, a dot's HORIZONTAL position is its 1-6 rating
    (1 at the left edge .. 6 at the right edge), the dashed centre line is the
    3.5 agree/disagree boundary, and dots stack vertically when a rating repeats.

Why: cells hold 1-7 trials, so a mean of 3.5 can be "two 3s and two 4s" or
"a 1 and a 6" and the heatmap cannot tell you which. Too few values per cell
for a density curve (a KDE through 2 points is theatre), so the raw values are
plotted directly. Dots carry no colour or label of their own: position is the
value, and the cell background already encodes the mean.

Usage (from repo root):
    python3 analysis_human/plot_2misc_heatmap_dots.py --cohort practice
"""

import argparse
import os
import sys
from itertools import combinations

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plot_2misc_heatmap import (  # noqa: E402
    IDS, SHORT, PAIRS, PAIR_LABELS, CMAP, NORM, DARK_AT,
    load, impossible_c, impossible_d,
)

# ── cell geometry (axis units; one cell spans 1.0 x 1.0) ──────────────
STRIP_HALF = 0.40      # rating 1 sits at -0.40, rating 6 at +0.40
DOT_DY = 0.115         # vertical gap between stacked repeats
STACK_MAX_H = 0.52     # stacks taller than this get compressed
BASE_Y = 0.13          # vertical centre of the dot cloud, below the mean text


def cell_values(trials):
    """C: (named, partner) -> [ratings].  D: (named foil, pair index) -> [ratings]."""
    vals_c, vals_d = {}, {}
    for t in trials:
        p = IDS.index(t['probed_misconception'])
        pair = tuple(sorted(IDS.index(m) for m in t['misconceptions']))
        if t['category'] == 'C':
            partner = next(m for m in pair if m != p)
            vals_c.setdefault((p, partner), []).append(t['response'])
        else:
            vals_d.setdefault((p, PAIRS.index(pair)), []).append(t['response'])
    return vals_c, vals_d


def to_matrices(vals, shape):
    mean = np.full(shape, np.nan)
    n = np.zeros(shape, int)
    for (i, j), v in vals.items():
        mean[i, j] = np.mean(v)
        n[i, j] = len(v)
    return mean, n


def rating_x(j, r):
    """x position of rating r (1..6) inside the cell centred on column j."""
    return j + (r - 3.5) * (STRIP_HALF * 2 / 5)


def draw(ax, vals, shape, impossible, xlabels, title, xlabel):
    mean, n = to_matrices(vals, shape)
    shown = np.ma.masked_where(impossible, np.where(np.isnan(mean), 3.5, mean))
    cm = CMAP.copy()
    cm.set_bad('#d8d7d2')
    im = ax.imshow(shown, cmap=cm, norm=NORM, aspect='equal')

    rows, cols = shape
    for i in range(rows):
        for j in range(cols):
            if impossible[i, j]:
                continue
            if n[i, j] == 0:
                ax.text(j, i, 'no\ntrials', ha='center', va='center',
                        fontsize=6, color='#898781', linespacing=1.1)
                continue
            dark = abs(mean[i, j] - 3.5) > DARK_AT
            fg = 'white' if dark else '#0b0b0b'

            # mean + n on one line, top of the cell
            ax.text(j, i - 0.34, f"{mean[i, j]:.1f}  n={n[i, j]}", ha='center',
                    va='center', fontsize=6.4, fontweight='bold', color=fg)

            # 3.5 boundary + strip baseline with a tick under each rating
            ax.plot([j, j], [i - 0.16, i + 0.44], ls=(0, (1.6, 1.6)), lw=0.8,
                    color=fg, alpha=0.6, zorder=2)
            ax.plot([j - STRIP_HALF, j + STRIP_HALF], [i + 0.44, i + 0.44],
                    lw=0.7, color=fg, alpha=0.35, zorder=2)
            for r in range(1, 7):
                ax.plot([rating_x(j, r)] * 2, [i + 0.44, i + 0.40], lw=0.7,
                        color=fg, alpha=0.35, zorder=2)

            # dots: x = rating, stacked upward from the baseline on repeats
            counts = {r: sum(1 for v in vals[(i, j)] if v == r) for r in range(1, 7)}
            tallest = max(counts.values())
            dy = min(DOT_DY, STACK_MAX_H / tallest) if tallest > 1 else DOT_DY
            for r, c in counts.items():
                if not c:
                    continue
                offs = (np.arange(c) - (c - 1) / 2) * dy
                ax.scatter([rating_x(j, r)] * c, i + BASE_Y - offs, s=17,
                           facecolor='white', edgecolor='#161616', linewidth=0.65,
                           zorder=5, clip_on=False)

    ax.set_xticks(range(cols)); ax.set_yticks(range(rows))
    ax.set_xticklabels(xlabels, fontsize=7.5)
    ax.set_yticklabels([SHORT[m] for m in IDS], fontsize=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel('misconception NAMED in statement', fontsize=9)
    ax.set_title(title, fontsize=10.5)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/real-all-main-data.json')
    ap.add_argument('--out', default='analysis_human/plots')
    ap.add_argument('--cohort', choices=['all', 'new', 'practice'], default='practice')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    trials, n_part = load(args.data, args.cohort)
    n_c = sum(t['category'] == 'C' for t in trials)
    print(f"2-misconception trials: {len(trials)} (C={n_c}, D={len(trials) - n_c}), "
          f"cohort={args.cohort}, participants={n_part}")

    vals_c, vals_d = cell_values(trials)
    sizes = sorted(len(v) for v in list(vals_c.values()) + list(vals_d.values()))
    print(f"trials per occupied cell: min {sizes[0]}, median {sizes[len(sizes)//2]}, "
          f"max {sizes[-1]}  ({len(sizes)} occupied of {30 + 60} possible)")

    fig, axes = plt.subplots(1, 2, figsize=(26, 9),
                             gridspec_kw={'width_ratios': [6, 15.4], 'wspace': 0.10})
    draw(axes[0], vals_c, (6, 6), impossible_c(), [SHORT[m] for m in IDS],
         '(a) Category C: statement names a PRESENT misconception\ncorrect = agree (green)',
         'OTHER misconception present in trace')
    im = draw(axes[1], vals_d, (6, 15), impossible_d(), PAIR_LABELS,
              '(b) Category D: statement names an ABSENT misconception\ncorrect = disagree (red)',
              'PAIR of misconceptions present in trace')

    cbar = fig.colorbar(im, ax=axes, fraction=0.014, pad=0.012,
                        ticks=[1, 2, 3, 3.5, 4, 5, 6])
    cbar.ax.set_yticklabels(['1 (SD)', '2', '3', '3.5', '4', '5', '6 (SA)'], fontsize=8)
    cbar.set_label('cell MEAN rating (>3.5 = agree side)', fontsize=9)

    who = {'all': 'all prolific', 'new': 'rebalanced-pool',
           'practice': 'practice-task'}[args.cohort]
    fig.suptitle(f'Two-misconception trials: cell mean (colour) with every individual '
                 f'trial shown ({n_part} {who} participants)', fontsize=13, y=0.965)
    fig.text(0.5, 0.035,
             'Within each cell a dot is one trial: horizontal position = that trial\'s rating '
             '(1 at the left edge .. 6 at the right edge), dashed line = the 3.5 agree/disagree '
             'boundary, dots stack vertically when a rating repeats. '
             'Dots left of the dashed line are disagree responses, right of it are agree.',
             ha='center', fontsize=8.5, color='#52514e')

    suffix = {'all': '', 'new': '_new', 'practice': '_practice'}[args.cohort]
    p = os.path.join(args.out, f'human_2misc_heatmap_dots{suffix}.png')
    fig.savefig(p, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"\nWrote {p}")


if __name__ == '__main__':
    main()
