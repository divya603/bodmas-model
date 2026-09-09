#!/usr/bin/env python3
"""
plot_bayes_v3_1misc_heatmap.py

v3 version of plot_bayes_1misc_heatmap.py. Present x named "confusion matrix"
for the Bayesian ideal observer. Rows = misconception PRESENT in the trace,
columns = misconception NAMED in the statement.

  - The DIAGONAL (present = named) is category A: the statement names the rule
    that is there, so agreeing is correct. Under v3 every one of these cells is
    exactly 1.00, at both error positions, for all six rules.
  - The OFF-DIAGONAL (present != named) is category B: the statement names a
    foil, so disagreeing is correct.

Cell value = mean recorded marginal P(named rule | trace); 0.5 neutral, same
diverging green/red scale and cell style as the v2 script.

Two panels split the off-diagonal by the foil's refutation status, with the
category-A diagonal shared between them as the agree reference:
  (a) foil REFUTED     (the trace passed a decision point and the student
                        visibly acted against the foil; marginal collapses to ~0)
  (b) foil UNSUPPORTED  (the foil never had an opportunity; ~0.25 residual)

Positions are POOLED here. Splitting the 30 off-diagonal cells four ways leaves
26 of 120 empty and drops the median cell to 3 items, which draws sparsity
rather than signal. Position is carried by the marginals and distribution
figures, where each group holds 24 or more items. The position null is printed
to stdout by this script so it is on the record either way.

Run from repo root:
    python3 analysis-Bayesian/plot_bayes_v3_1misc_heatmap.py
"""
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bayes_v3_common import (HERE, IDS, SHORT, CMAP, NORM, DARK_AT, POSITIONS,
                             load_rows, split_ab)


def build_matrices(rows):
    """{'refuted'|'unsupported'|'combined': (mean 6x6, n 6x6)}, present on rows,
    named on cols. The diagonal is category A and is shared by every panel."""
    idx = {m: i for i, m in enumerate(IDS)}
    a_rows, b_rows = split_ab(rows)

    diag = {i: [] for i in range(6)}
    for r in a_rows:
        diag[idx[r['true_misconception']]].append(r['probed_marginal'])

    off = {'refuted': {}, 'unsupported': {}, 'combined': {}}
    for r in b_rows:
        p, nm = idx[r['true_misconception']], idx[r['probed_misconception']]
        st = r['foil_status']
        if st in ('refuted', 'unsupported'):
            off[st].setdefault((p, nm), []).append(r['probed_marginal'])
            off['combined'].setdefault((p, nm), []).append(r['probed_marginal'])

    mats = {}
    for st in ('refuted', 'unsupported', 'combined'):
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


def report_position(rows):
    """Print the position contrast the pooled panels deliberately hide."""
    _, b_rows = split_ab(rows)
    print("\n=== position check (pooled into the panels above) ===")
    for st in ('refuted', 'unsupported'):
        line = []
        for pos in POSITIONS:
            v = [r['probed_marginal'] for r in b_rows
                 if r['foil_status'] == st and r['error_position'] == pos]
            line.append(f"pos{pos} n={len(v):>3d} mean {np.mean(v):.3f}")
        print(f"  {st:>12s}: " + "   ".join(line))


def main():
    rows = load_rows()
    mats = build_matrices(rows)

    for st in ('refuted', 'unsupported'):
        mean, n = mats[st]
        print(f"\n=== {st} panel (diagonal = A, off-diagonal = B {st}); mean marginal (n) ===")
        print(f"{'present \\ named':>15s} " + " ".join(f"{SHORT[m]:>10s}" for m in IDS))
        for i, p in enumerate(IDS):
            print(f"{SHORT[p]:>15s} " + " ".join(
                f"{mean[i, j]:>6.2f}({n[i, j]:>2d})" if n[i, j] else f"{'.':>6s}({0:>2d})"
                for j in range(6)))
    report_position(rows)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.0), gridspec_kw={'wspace': 0.30})
    fig.subplots_adjust(top=0.80)
    draw(axes[0], *mats['refuted'],
         '(a) category-B foils REFUTED by the trace\n(boxed diagonal = category A, agreeing is correct)')
    im = draw(axes[1], *mats['unsupported'],
              '(b) category-B foils UNSUPPORTED\n(boxed diagonal = category A, agreeing is correct)')
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02,
                        ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_label('posterior marginal P(named rule | trace)', fontsize=9)
    fig.suptitle('Bayesian ideal observer, v3 pool (432 items, one misconception each): '
                 'present (rows) × named (columns)\nerror positions pooled',
                 fontsize=12.5, y=0.98)
    p = os.path.join(HERE, 'bayes_v3_1misc_heatmap.png')
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"\nWrote {p}")

    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    im = draw(ax, *mats['combined'],
              'Bayesian ideal observer, v3 pool\n'
              'diagonal = category A (agree); off-diagonal = category B foils (disagree)')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_label('posterior marginal P(named rule | trace)', fontsize=9)
    pc = os.path.join(HERE, 'bayes_v3_1misc_heatmap_combined.png')
    fig.savefig(pc, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {pc}")


if __name__ == '__main__':
    main()
