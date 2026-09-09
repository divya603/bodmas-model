#!/usr/bin/env python3
"""
plot_bayes_1misc_heatmap.py

Present x named "confusion matrix" for the Bayesian ideal observer on the v4
pool. Rows = misconception PRESENT in the trace, columns = misconception NAMED
in the statement.

  - The DIAGONAL (present = named) is category A: the statement names the rule
    that is there, so agreeing is correct.
  - The OFF-DIAGONAL (present != named) is category B: the statement names a
    foil, so disagreeing is correct.

The v4 pool was built so this figure has NO EMPTY BOXES. Every present rule
supplies 40 items: 10 in each (category x position) cell, with the 20 category-B
items spread evenly over all 5 foils. So the panels below are complete whether
they are drawn pooled or split by position:

    pooled by position   diagonal 20, off-diagonal 4
    split by position    diagonal 10, off-diagonal 2

Panels here split by ERROR POSITION, which is the v4 factor. Do NOT split this
figure by refutation status: under v4 status is recorded but not balanced, so a
status split reintroduces exactly the holes v4 was built to remove (that is the
v3 heatmap's problem, see HANDOFF 7b).

Run from repo root:
    python3 analysis-Bayesian/plot_bayes_1misc_heatmap.py
"""
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bayes_common import (HERE, BASE_TASK, ROWS_PATH, IDS, SHORT, CMAP, NORM,
                          DARK_AT, POSITIONS, load_rows)


def build_matrix(rows):
    """(mean 6x6, n 6x6) with present on rows and named on columns."""
    idx = {m: i for i, m in enumerate(IDS)}
    vals = {}
    for r in rows:
        vals.setdefault((idx[r['true_misconception']],
                         idx[r['probed_misconception']]), []).append(r['probed_marginal'])
    mean, n = np.full((6, 6), np.nan), np.zeros((6, 6), int)
    for (i, j), v in vals.items():
        mean[i, j], n[i, j] = np.mean(v), len(v)
    return mean, n


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
    rows = load_rows(ROWS_PATH, expect=240)

    for pos in POSITIONS:
        mean, n = build_matrix([r for r in rows if r['error_position'] == pos])
        empty = int((n == 0).sum())
        print(f"\n=== error at step {pos}; mean marginal (n) ===  empty cells: {empty}")
        print(f"{'present \\ named':>15s} " + " ".join(f"{SHORT[m]:>10s}" for m in IDS))
        for i, p in enumerate(IDS):
            print(f"{SHORT[p]:>15s} " + " ".join(
                f"{mean[i, j]:>6.2f}({n[i, j]:>2d})" for j in range(6)))

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.0), gridspec_kw={'wspace': 0.30})
    fig.subplots_adjust(top=0.80)
    im = None
    for ax, pos in zip(axes, POSITIONS):
        mean, n = build_matrix([r for r in rows if r['error_position'] == pos])
        im = draw(ax, mean, n,
                  f'({"ab"[POSITIONS.index(pos)]}) error at step {pos}\n'
                  '(boxed diagonal = category A, agreeing is correct)')
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02,
                        ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_label('posterior marginal P(named rule | trace)', fontsize=9)
    fig.suptitle('Bayesian ideal observer, v4 pool (240 items): present (rows) × named (columns)\n'
                 'split by error position; every cell occupied by design',
                 fontsize=12.5, y=0.98)
    p = os.path.join(HERE, 'bayes_1misc_heatmap.png')
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"\nWrote {p}")

    mean, n = build_matrix(rows)
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    im = draw(ax, mean, n,
              'Bayesian ideal observer, v4 pool (positions pooled)\n'
              'diagonal = category A (agree); off-diagonal = category B foils (disagree)')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_label('posterior marginal P(named rule | trace)', fontsize=9)
    pc = os.path.join(HERE, 'bayes_1misc_heatmap_combined.png')
    fig.savefig(pc, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {pc}")


if __name__ == '__main__':
    main()
