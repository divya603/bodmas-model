#!/usr/bin/env python3
"""
make_llm_2misc_heatmap_dots.py

Raw-value version of make_llm_2misc_heatmap.py (which it leaves untouched).
Same 3 regime rows x 2 panels, same cells, same diverging colour for the CELL
MEAN, but each cell also shows the individual items behind that mean:

    within a cell, a dot's HORIZONTAL position is its 1-6 rating
    (1 at the left edge .. 6 at the right edge), the dashed centre line is the
    3.5 agree/disagree boundary, and dots stack vertically when a rating repeats.

Why: category-C cells hold 3-6 items and category-D cells hold exactly 2, so a
cell mean of 3.5 can be "two 3s and two 4s" or "a 1 and a 6" and the heatmap
cannot tell you which. Too few values per cell for a density curve, so the raw
values are plotted directly. Dots carry no colour or label of their own:
position is the value, and the cell background already encodes the mean.

The figure is deliberately large (3 regimes x a 15-column D panel with a
6-point strip in every cell). Expect ~26x26 inches.

Run from repo root:
    python3 llm_exp/make_llm_2misc_heatmap_dots.py
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, 'plots')
sys.path.insert(0, HERE)

from make_llm_2misc_heatmap import (  # noqa: E402
    IDS, SHORT, PAIRS, PAIR_LABELS, REGIMES, CMAP, NORM, DARK_AT,
    impossible_c, impossible_d,
)

# ── cell geometry (axis units; one cell spans 1.0 x 1.0) ──────────────
STRIP_HALF = 0.40      # rating 1 sits at -0.40, rating 6 at +0.40
DOT_DY = 0.115         # vertical gap between stacked repeats
STACK_MAX_H = 0.52     # stacks taller than this get compressed
BASE_Y = 0.13          # vertical centre of the dot cloud, below the mean text


def cell_values(rows):
    """C: (named, partner) -> [ratings].  D: (named foil, pair index) -> [ratings]."""
    vals_c, vals_d = {}, {}
    for r in rows.itertuples():
        p = IDS.index(r.probed_misconception)
        pair = tuple(sorted(IDS.index(m) for m in r.misconceptions))
        if r.category == 'C':
            partner = next(m for m in pair if m != p)
            vals_c.setdefault((p, partner), []).append(r.response)
        else:
            vals_d.setdefault((p, PAIRS.index(pair)), []).append(r.response)
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
                ax.text(j, i, 'no\nitems', ha='center', va='center',
                        fontsize=6, color='#898781', linespacing=1.1)
                continue
            dark = abs(mean[i, j] - 3.5) > DARK_AT
            fg = 'white' if dark else '#0b0b0b'

            ax.text(j, i - 0.34, f"{mean[i, j]:.1f}  n={n[i, j]}", ha='center',
                    va='center', fontsize=6.4, fontweight='bold', color=fg)

            ax.plot([j, j], [i - 0.16, i + 0.44], ls=(0, (1.6, 1.6)), lw=0.8,
                    color=fg, alpha=0.6, zorder=2)
            ax.plot([j - STRIP_HALF, j + STRIP_HALF], [i + 0.44, i + 0.44],
                    lw=0.7, color=fg, alpha=0.35, zorder=2)
            for r in range(1, 7):
                ax.plot([rating_x(j, r)] * 2, [i + 0.44, i + 0.40], lw=0.7,
                        color=fg, alpha=0.35, zorder=2)

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
    ax.set_xticklabels(xlabels, fontsize=7)
    ax.set_yticklabels([SHORT[m] for m in IDS], fontsize=7.5)
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.set_ylabel('misconception NAMED in statement', fontsize=8.5)
    ax.set_title(title, fontsize=10)
    return im


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    pool_ids = {it['id'] for it in json.load(open(os.path.join(HERE, 'data', 'stimulus_pool.json')))}
    rows = []
    for path in sorted(glob.glob(os.path.join(HERE, 'results', 'raw_*.jsonl'))):
        for line in open(path):
            r = json.loads(line)
            if 'all_items' in r['subject_id'] and r.get('error') is None \
                    and r.get('response') is not None and r['id'] in pool_ids:
                rows.append(r)
    df = pd.DataFrame(rows)
    df = df[df.num_misconceptions == 2]
    df = df.drop_duplicates(subset=['model', 'effort', 'id'], keep='first')
    n_cd = df['id'].nunique()

    fig, axes = plt.subplots(3, 2, figsize=(26, 26),
                             gridspec_kw={'width_ratios': [6, 15.4],
                                          'wspace': 0.10, 'hspace': 0.30})
    fig.subplots_adjust(top=0.93, bottom=0.05, left=0.05, right=0.90)
    im = None
    for row, (label, model, effort) in enumerate(REGIMES):
        sub = df[(df.model == model) & (df.effort == effort)]
        vals_c, vals_d = cell_values(sub)
        sizes = sorted(len(v) for v in list(vals_c.values()) + list(vals_d.values()))
        print(f"{label}: {len(sub)} two-misconception items; items per occupied cell "
              f"min {sizes[0]}, median {sizes[len(sizes)//2]}, max {sizes[-1]} "
              f"({len(sizes)} occupied of {30 + 60})")
        draw(axes[row, 0], vals_c, (6, 6), impossible_c(), [SHORT[m] for m in IDS],
             '(a) statement names a PRESENT misconception\n(correct = agree)',
             'OTHER misconception present in trace')
        im = draw(axes[row, 1], vals_d, (6, 15), impossible_d(), PAIR_LABELS,
                  '(b) statement names an ABSENT misconception\n(correct = disagree)',
                  'PAIR of misconceptions present in trace')
        mid = (axes[row, 0].get_position().x1 + axes[row, 1].get_position().x0) / 2
        fig.text(mid, axes[row, 0].get_position().y1 + 0.012, label,
                 ha='center', va='bottom', fontsize=14, fontweight='bold')

    cax = fig.add_axes([0.93, 0.35, 0.010, 0.30])
    cbar = fig.colorbar(im, cax=cax, ticks=[1, 2, 3, 3.5, 4, 5, 6])
    cbar.ax.set_yticklabels(['1 (SD)', '2', '3', '3.5', '4', '5', '6 (SA)'], fontsize=8)
    cbar.set_label('cell MEAN rating (>3.5 = agree side)', fontsize=9)

    fig.suptitle(f'LLM two-misconception items: cell mean (colour) with every individual '
                 f'item shown (all {n_cd} C/D items)', fontsize=15, y=0.965)
    fig.text(0.5, 0.028,
             'Within each cell a dot is one item: horizontal position = that item\'s rating '
             '(1 at the left edge .. 6 at the right edge), dashed line = the 3.5 agree/disagree '
             'boundary, dots stack vertically when a rating repeats. '
             'Dots left of the dashed line are disagree responses, right of it are agree.',
             ha='center', fontsize=10, color='#52514e')

    p = os.path.join(OUTDIR, 'llm_2misc_heatmap_dots.png')
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\nWrote {p}")


if __name__ == '__main__':
    main()
