#!/usr/bin/env python3
"""
plot_bayes_v3_1misc_distributions.py

v3 version of plot_bayes_1misc_distributions.py. Distribution of the Bayesian
ideal observer's per-item posterior P(named rule | trace), split by the new v3
error-position factor throughout.

  bayes_v3_1misc_dist_A.png: category A (statement matches). 6 panels by the
      misconception present (= named; in A they coincide by design). Every
      panel is a single spike at 1.000: the observer is a logical oracle on
      present rules and there is no distribution to draw. The figure exists to
      show that, and to show it holds at both error positions.
  bayes_v3_1misc_dist_B.png: category B (statement is a foil). Three rows over
      the same 288 items, because present and named are decoupled:
      row 1 by the misconception PRESENT in the trace ("does the student's
      actual bug make the work confusable?"), row 2 by the misconception NAMED
      in the statement ("are some claims inherently easier to rule out?"),
      row 3 the REFUTED subset only, by named foil.

⚠️ NOT KDEs, unlike the v2 script. The v3 B marginals take only 11 distinct
values in [0, 0.333] and category A takes exactly one, so these are discrete
distributions and a Gaussian KDE would invent shape between the spikes and
smear the point mass. Each panel plots the EXACT observed values as stems whose
height is the proportion of that group's items sitting on that value, so the
figure never claims more than the data does. The v2 KDE remains correct for the
v2 pool, where the marginals were genuinely continuous.

Run from repo root:
    python3 analysis-Bayesian/plot_bayes_v3_1misc_distributions.py
"""
import os
from collections import Counter

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bayes_v3_common import (HERE, IDS, SHORT, POSITIONS, POS_COLOR, POS_MARKER,
                             POS_LABEL, REFUTED_CUT, load_rows, split_ab)

DX = 0.007          # x offset between the two position stems (category A, full axis)
B_DX = 0.0028       # ditto on the zoomed category-B axis
STEM_W = 1.9
B_XMAX = 0.4        # no category-B item exceeds 0.333


def stem_panel(ax, by_pos, title, show_counts=True, xmax=1.0, n_side='right'):
    """Draw exact-value stems for {position: [marginals]}. Height = proportion
    of that position's items on that value, so the two positions stay
    comparable when their n differ.

    `xmax` < 1 zooms the axis. Category B never exceeds 0.333, so plotting it
    on [0, 1] pushes the whole distribution into the left third and hides the
    refuted/unsupported gap, which is the structure the figure is for. The
    0.5 decision boundary is then off-scale and the caption says so."""
    zoomed = xmax < 1.0
    ticks = ([0, 0.1, 0.2, 0.3, 0.4] if zoomed else [0, 0.5, 1])
    any_vals = any(len(v) for v in by_pos.values())
    if not any_vals:
        ax.text(0.5, 0.5, 'no items', ha='center', va='center',
                fontsize=8, color='#898781', transform=ax.transAxes)
        ax.set_xlim(-0.02 * xmax, xmax * 1.02); ax.set_ylim(0, 1.08)
        ax.set_yticks([]); ax.set_xticks(ticks)
        ax.set_title(title, fontsize=9.5)
        return

    for k, pos in enumerate(POSITIONS):
        vals = by_pos.get(pos, [])
        if not len(vals):
            continue
        counts = Counter(round(v, 6) for v in vals)
        n = len(vals)
        xoff = (k - 0.5) * 2 * (B_DX if zoomed else DX)
        for v, c in sorted(counts.items()):
            ax.vlines(v + xoff, 0, c / n, color=POS_COLOR[pos], lw=STEM_W,
                      alpha=0.9, zorder=3)
            ax.plot(v + xoff, c / n, marker=POS_MARKER[pos], ms=3.4,
                    color=POS_COLOR[pos], zorder=4)

    if zoomed:
        ax.axvline(REFUTED_CUT, ls=':', lw=1.0, color='#8a6d3b', zorder=1)
    else:
        ax.axvline(0.5, ls='--', lw=0.8, color='grey', zorder=1)
    pad = 0.02 * xmax
    ax.set_xlim(-pad, xmax + pad)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(['0', '.5', '1'], fontsize=7)
    ax.set_xticks(ticks)
    ax.tick_params(axis='x', labelsize=7)
    if show_counts:
        # sit the label on whichever side the distribution is not using
        ns = " / ".join(str(len(by_pos.get(p, []))) for p in POSITIONS)
        x, ha = (0.97, 'right') if n_side == 'right' else (0.03, 'left')
        ax.text(x, 0.93, f"n={ns}", transform=ax.transAxes, ha=ha,
                va='top', fontsize=6.5, color='#52514e')
    ax.set_title(title, fontsize=9.5)


def by_position(rows, key):
    """{rule: {position: [marginals]}} keyed by row field `key`."""
    out = {m: {p: [] for p in POSITIONS} for m in IDS}
    for r in rows:
        out[r[key]][r['error_position']].append(r['probed_marginal'])
    return out


def legend(fig, loc='upper right', anchor=(0.99, 0.985)):
    handles = [plt.Line2D([], [], color=POS_COLOR[p], lw=2,
                          marker=POS_MARKER[p], ms=4.5, label=POS_LABEL[p])
               for p in POSITIONS]
    fig.legend(handles=handles, loc=loc, fontsize=8.5,
               ncol=2, frameon=False, bbox_to_anchor=anchor)


def main():
    rows = load_rows()
    a_rows, b_rows = split_ab(rows)

    # ── figure 1: category A ──
    a_present = by_position(a_rows, 'true_misconception')
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for ax, m in zip(axes.flat, IDS):
        stem_panel(ax, a_present[m], SHORT[m], n_side='left')
    for ax in axes[1]:
        ax.set_xlabel('P(named rule | trace)', fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel('proportion of items', fontsize=8.5)
    fig.suptitle('Bayesian ideal observer, v3, category A (statement matches)\n'
                 'the posterior on the named rule is a POINT MASS at 1.000: both error '
                 'positions, all six rules,\n12 items per position per panel',
                 fontsize=11.5, y=0.985)
    legend(fig, loc='upper center', anchor=(0.5, 0.90))
    fig.tight_layout(rect=[0, 0, 1, 0.855])
    p1 = os.path.join(HERE, 'bayes_v3_1misc_dist_A.png')
    fig.savefig(p1, dpi=140)
    plt.close(fig)

    # ── figure 2: category B, two groupings + refuted-only row ──
    b_present = by_position(b_rows, 'true_misconception')
    b_named = by_position(b_rows, 'probed_misconception')
    ref_rows = [r for r in b_rows if r['foil_status'] == 'refuted']
    b_refuted = by_position(ref_rows, 'probed_misconception')

    print("refuted items per named foil (foil_status == 'refuted'): " +
          "  ".join(f"{SHORT[m]}={sum(len(v) for v in b_refuted[m].values())}" for m in IDS))
    mism = [r for r in b_rows
            if (r['probed_marginal'] < REFUTED_CUT) != (r['foil_status'] == 'refuted')]
    print(f"items where the {REFUTED_CUT} cut and the stored foil_status disagree: {len(mism)}")

    fig, axes = plt.subplots(3, 6, figsize=(16.5, 8.6), sharex=True)
    for j, m in enumerate(IDS):
        stem_panel(axes[0, j], b_present[m], SHORT[m], xmax=B_XMAX)
        stem_panel(axes[1, j], b_named[m], SHORT[m], xmax=B_XMAX)
        stem_panel(axes[2, j], b_refuted[m], SHORT[m], xmax=B_XMAX)
        axes[2, j].set_xlabel('P(named rule | trace)', fontsize=8)
    axes[0, 0].set_ylabel('grouped by misconception\nPRESENT in trace', fontsize=9)
    axes[1, 0].set_ylabel('grouped by misconception\nNAMED in statement (foil)', fontsize=9)
    axes[2, 0].set_ylabel(f'REFUTED subset only,\nby named foil', fontsize=9)
    fig.suptitle('Bayesian ideal observer, v3, category B (statement is a foil): '
                 'posterior on the named rule\n'
                 f'same {len(b_rows)} items, two groupings then the refuted subset; '
                 'stems sit on the exact observed values (n = step 1 / step 3)\n'
                 'x axis zoomed to [0, 0.4]: no foil item exceeds 0.333, so the 0.5 '
                 'decision boundary is off-scale. Dotted line = the 0.15 refuted cut.',
                 fontsize=11.5)
    legend(fig)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    p2 = os.path.join(HERE, 'bayes_v3_1misc_dist_B.png')
    fig.savefig(p2, dpi=140)
    plt.close(fig)

    print(f"Wrote {p1}\nWrote {p2}")


if __name__ == '__main__':
    main()
