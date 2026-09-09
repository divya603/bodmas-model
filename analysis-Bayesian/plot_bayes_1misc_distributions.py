#!/usr/bin/env python3
"""
plot_bayes_1misc_distributions.py

Distributions of the Bayesian ideal observer's per-item posterior, split by the
error-position factor throughout.

An important property of this observer, which these figures are built around:
`posterior_over_profiles()` takes ONLY THE TRACE. The belief statement is never
an input. The named rule enters afterwards purely as an index, picking one entry
out of a posterior that was already computed. So category A vs B is not a
difference in the observer's computation, only a difference in WHICH of the six
marginals gets read off. Every trace carries all six.

Three figures:

  bayes_1misc_dist_A.png  category A (statement names the present rule). Six
      panels by rule. Every panel is a single spike at 1.000: at epsilon 0 the
      observer is a logical oracle on a rule that is actually there, and it does
      not care where in the trace the error fell. There is no distribution to
      draw and the figure exists to show that.

  bayes_1misc_dist_B.png  category B (statement names a foil). Two rows over the
      same 120 items. Row 2 is the clean one: it groups by the NAMED rule, so the
      grouping variable and the plotted marginal are the same rule and each panel
      asks exactly one question ("how confidently can this claim be ruled out
      when it is false?").
      ⚠️ Row 1 groups by the rule PRESENT in the trace while still plotting the
      marginal on whichever rule the statement named, so a single panel pools
      five different marginals. The panel titled "trace contains add<mul" never
      plots P(add<mul | trace) at all. It is a task-level summary of the B trials
      that came from those traces, NOT a property of the observer, and it is
      titled and captioned to say so. For the observer-level version of the
      trace-side question use bayes_1misc_profile.png, which keeps every rule's
      marginal separate.

  bayes_1misc_profile.png  the figure the indexing point above motivates. For
      every trace it plots ALL SIX marginals, not just the queried one, grouped
      by the misconception present. This is the observer's full read of a trace:
      the present rule pinned at 1.000 and the other five mostly low. It is
      where all the structure that dist_A cannot show lives, including one thing
      no other figure can reveal: on 36 of 1200 trace-by-rule combinations an
      ABSENT rule scores above 0.35, and on 15 of them above 0.5 (max 0.871).
      Those are all outside_bracket_first. The pool's foil selection drops any
      foil above 0.35 as "not a clean foil", so those hardest cases are
      systematically absent from category B. This figure is the only place they
      are visible.

⚠️ NOT KDEs. The B marginals take only 8 distinct values in [0, 0.333] and the A
marginals take exactly one, so these are discrete distributions; a Gaussian KDE
would invent shape between the spikes and smear the point mass. Stems sit on the
exact observed values, with height = the proportion of that group on that value.

⚠️ Nothing here is split by `foil_status`. Refutation is recorded but not
balanced in this pool, so splitting on it produces lopsided and empty cells.

Run from repo root:
    python3 analysis-Bayesian/plot_bayes_1misc_distributions.py
"""
import os
from collections import Counter

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bayes_common import (HERE, IDS, SHORT, POSITIONS, POS_COLOR, POS_MARKER,
                          POS_LABEL, load_rows, split_ab)

DX, B_DX = 0.007, 0.0028   # stem offset between positions, full vs zoomed axis
STEM_W = 1.9
B_XMAX = 0.4               # no category-B marginal exceeds 0.333


def stem_panel(ax, by_pos, title, xmax=1.0, n_side='right'):
    """Exact-value stems for {position: [marginals]}. Height = proportion of
    that position's items on that value, so the positions stay comparable when
    their n differ."""
    zoomed = xmax < 1.0
    ticks = [0, 0.1, 0.2, 0.3, 0.4] if zoomed else [0, 0.5, 1]
    if not any(len(v) for v in by_pos.values()):
        ax.text(0.5, 0.5, 'no items', ha='center', va='center', fontsize=8,
                color='#898781', transform=ax.transAxes)
    else:
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
        ns = " / ".join(str(len(by_pos.get(p, []))) for p in POSITIONS)
        x, ha = (0.97, 'right') if n_side == 'right' else (0.03, 'left')
        ax.text(x, 0.93, f"n={ns}", transform=ax.transAxes, ha=ha, va='top',
                fontsize=6.5, color='#52514e')

    if not zoomed:
        ax.axvline(0.5, ls='--', lw=0.8, color='grey', zorder=1)
    pad = 0.02 * xmax
    ax.set_xlim(-pad, xmax + pad)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.5, 1.0]); ax.set_yticklabels(['0', '.5', '1'], fontsize=7)
    ax.set_xticks(ticks); ax.tick_params(axis='x', labelsize=7)
    ax.set_title(title, fontsize=9.5)


def by_position(rows, key):
    out = {m: {p: [] for p in POSITIONS} for m in IDS}
    for r in rows:
        out[r[key]][r['error_position']].append(r['probed_marginal'])
    return out


def legend(fig, loc='upper right', anchor=(0.99, 0.985)):
    handles = [plt.Line2D([], [], color=POS_COLOR[p], lw=2, marker=POS_MARKER[p],
                          ms=4.5, label=POS_LABEL[p]) for p in POSITIONS]
    fig.legend(handles=handles, loc=loc, fontsize=8.5, ncol=2, frameon=False,
               bbox_to_anchor=anchor)


def main():
    rows = load_rows()
    a_rows, b_rows = split_ab(rows)

    # ── figure 1: category A, the point mass ──
    a_present = by_position(a_rows, 'true_misconception')
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for ax, m in zip(axes.flat, IDS):
        stem_panel(ax, a_present[m], SHORT[m], n_side='left')
    for ax in axes[1]:
        ax.set_xlabel('P(named rule | trace)', fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel('proportion of items', fontsize=8.5)
    fig.suptitle('Bayesian ideal observer, category A (statement names the rule that is there)\n'
                 'the posterior on that rule is a POINT MASS at 1.000: both error positions, '
                 'all six rules,\n10 items per position per panel',
                 fontsize=11.5, y=0.985)
    legend(fig, loc='upper center', anchor=(0.5, 0.90))
    fig.tight_layout(rect=[0, 0, 1, 0.855])
    p1 = os.path.join(HERE, 'bayes_1misc_dist_A.png')
    fig.savefig(p1, dpi=140); plt.close(fig)

    # ── figure 2: category B, two groupings ──
    b_present = by_position(b_rows, 'true_misconception')
    b_named = by_position(b_rows, 'probed_misconception')
    fig, axes = plt.subplots(2, 6, figsize=(16.5, 6.6), sharex=True)
    for j, m in enumerate(IDS):
        # ⚠️ The two rows do NOT plot the same quantity, and the panel titles say so.
        # Row 1 groups by the rule in the TRACE while the value plotted is the marginal
        # on whichever rule that item's statement named, so one panel pools five
        # different marginals. It is a task-level summary, not a property of the
        # observer. For the observer-level version of the same question, use
        # bayes_1misc_profile.png, which plots each rule's marginal separately.
        # Row 2 groups by the NAMED rule, so grouping variable and plotted quantity
        # coincide and each panel asks exactly one question.
        stem_panel(axes[0, j], b_present[m], f'trace contains {SHORT[m]}', xmax=B_XMAX)
        stem_panel(axes[1, j], b_named[m], f'statement names {SHORT[m]}', xmax=B_XMAX)
        axes[1, j].set_xlabel('P(named rule | trace)', fontsize=8)
    axes[0, 0].set_ylabel('by rule PRESENT in trace\n(pools 5 different named rules)', fontsize=8.5)
    axes[1, 0].set_ylabel('by rule NAMED in statement\n(one question per panel)', fontsize=8.5)
    fig.suptitle('Bayesian ideal observer, category B (statement names a foil): '
                 'posterior on the named rule\n'
                 f'same {len(b_rows)} items under two groupings, 20 per panel '
                 '(n = step 1 / step 3); stems sit on the exact observed values\n'
                 '⚠️ ROW 1 IS A MIXTURE: it groups by the rule in the trace, but plots the '
                 'marginal on whichever rule the statement named,\nso each panel pools five '
                 'different marginals. Row 2 groups and plots the same rule, so each panel asks '
                 'one question.\nx axis zoomed to [0, 0.4]: no probed foil exceeds 0.333, so the '
                 '0.5 decision boundary is off-scale',
                 fontsize=10)
    legend(fig)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    p2 = os.path.join(HERE, 'bayes_1misc_dist_B.png')
    fig.savefig(p2, dpi=140); plt.close(fig)

    # ── figure 3: the full six-marginal profile per trace ──
    # The observer computes one posterior per trace; the statement only picks an
    # index into it. So plot every entry, not just the queried one.
    import json
    from bayes_common import BASE_TASK
    import sys
    sys.path.insert(0, BASE_TASK)
    from inference import posterior_over_profiles, marginal_rule_probability
    from pool import HYPOTHESES

    pool = json.load(open(os.path.join(BASE_TASK, 'stimulus_pool.json'), encoding='utf-8'))
    # one posterior per TRACE; a trace appears once in the pool
    profile = {m: {q: {p: [] for p in POSITIONS} for q in IDS} for m in IDS}
    for it in pool:
        post = posterior_over_profiles(it['trace'], profiles=HYPOTHESES)
        present, pos = it['misconceptions'][0], it['error_position']
        for q in IDS:
            profile[present][q][pos].append(marginal_rule_probability(post, q))

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.4), sharey=True)
    for ax, present in zip(axes.flat, IDS):
        for gi, q in enumerate(IDS):
            for k, pos in enumerate(POSITIONS):
                vals = np.sort(profile[present][q][pos])
                if not len(vals):
                    continue
                xs = gi + (k - 0.5) * 0.34 + np.linspace(-0.08, 0.08, len(vals))
                ax.scatter(xs, vals, s=13, color=POS_COLOR[pos],
                           marker=POS_MARKER[pos], alpha=0.75,
                           edgecolors='white', linewidths=0.35, zorder=3)
            if gi:
                ax.axvline(gi - 0.5, color='#e6e4df', lw=0.8, zorder=0)
        ax.axhline(0.5, ls='--', lw=0.9, color='grey', zorder=1)
        ax.set_xticks(range(6))
        ax.set_xticklabels([SHORT[q] for q in IDS], fontsize=7.5, rotation=30, ha='right')
        ax.set_xlim(-0.5, 5.5); ax.set_ylim(-0.04, 1.04)
        ax.set_title(f'trace contains {SHORT[present]}', fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel('P(rule | trace)', fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel('marginal read off for which rule', fontsize=8.5)
    fig.suptitle('What the ideal observer actually extracts from a trace: ALL SIX marginals\n'
                 'The statement is never an input to inference, it only picks which entry gets read. '
                 'One dot per item, 40 per panel per rule, 20 per position.\n'
                 'The present rule is pinned at 1.000. Absent rules mostly sit below 0.35, but '
                 'outside() is the exception: it crosses 0.5 on 15 of 1200\n'
                 'trace-by-rule combinations (max 0.871). Those high-marginal foils are EXCLUDED '
                 'from the pool, so no category-B item ever probes one.',
                 fontsize=10.5)
    legend(fig, loc='upper right', anchor=(0.995, 0.93))
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    p3 = os.path.join(HERE, 'bayes_1misc_profile.png')
    fig.savefig(p3, dpi=140); plt.close(fig)

    # stdout summary
    print("category A marginals, distinct values:",
          sorted({r['probed_marginal'] for r in a_rows}))
    print("category B marginals, distinct values:",
          sorted({round(r['probed_marginal'], 3) for r in b_rows}))
    print("\nprofile: max NON-present marginal per present rule")
    for present in IDS:
        mx = max(max(v) for q, d in profile[present].items() if q != present
                 for v in d.values())
        print(f"  trace contains {SHORT[present]:>9s}: {mx:.3f}")
    print(f"\nWrote {p1}\nWrote {p2}\nWrote {p3}")


if __name__ == '__main__':
    main()
