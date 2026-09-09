#!/usr/bin/env python3
"""
plot_bayes_1misc_profile.py

Every marginal the ideal observer extracts from a trace, not just the queried
one, grouped by the misconception PRESENT in the trace.

Why this figure exists: `posterior_over_profiles()` takes ONLY the trace. The
belief statement enters afterwards purely as an index into an already-computed
posterior. So each trace carries all six marginals, and plotting only the one
the statement happened to name throws away five sixths of the observer's read.

Companion to plot_bayes_1misc_by_rule.py, which is the transpose: that one asks
"for rule X, present versus absent"; this one asks "for a trace containing X,
what does the observer make of every rule at once".

What it shows that no category A/B figure can: on 36 of 1200 (trace, absent
rule) combinations an ABSENT rule scores above 0.35, and on 15 above 0.5
(max 0.871). All 15 are outside_bracket_first, the one rule that removes rather
than adds options, so a trace that never enters its bracket early reads as
positive evidence FOR it. `pool.py: foil_options()` drops any foil above 0.35 as
"not a clean foil", so no category-B item ever probes one of these.

Run from repo root:
    python3 analysis-Bayesian/plot_bayes_1misc_profile.py
"""
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bayes_common import (HERE, BASE_TASK, IDS, SHORT, POSITIONS, POS_COLOR,
                          POS_MARKER, POS_LABEL)

sys.path.insert(0, BASE_TASK)
from inference import posterior_over_profiles, marginal_rule_probability  # noqa: E402
from pool import HYPOTHESES, UNSUPPORTED_MAX                              # noqa: E402


def collect():
    """{present rule: {queried rule: {position: [marginals]}}}."""
    pool = json.load(open(os.path.join(BASE_TASK, 'stimulus_pool.json'), encoding='utf-8'))
    out = {m: {q: {p: [] for p in POSITIONS} for q in IDS} for m in IDS}
    for it in pool:
        post = posterior_over_profiles(it['trace'], profiles=HYPOTHESES)
        present, pos = it['misconceptions'][0], it['error_position']
        for q in IDS:
            out[present][q][pos].append(marginal_rule_probability(post, q))
    return out


def main():
    profile = collect()

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.6), sharey=True)
    for ax, present in zip(axes.flat, IDS):
        ax.axhspan(UNSUPPORTED_MAX, 0.94, color='#f3f0e8', zorder=0)
        for gi, q in enumerate(IDS):
            for k, pos in enumerate(POSITIONS):
                vals = np.sort(profile[present][q][pos])
                if not len(vals):
                    continue
                xs = gi + (k - 0.5) * 0.34 + np.linspace(-0.08, 0.08, len(vals))
                ax.scatter(xs, vals, s=13, color=POS_COLOR[pos], marker=POS_MARKER[pos],
                           alpha=0.75, edgecolors='white', linewidths=0.35, zorder=3)
            if gi:
                ax.axvline(gi - 0.5, color='#e6e4df', lw=0.8, zorder=1)
        ax.axhline(0.5, ls='--', lw=0.9, color='#b8b6b0', zorder=2)
        ax.set_xticks(range(6))
        ax.set_xticklabels([SHORT[q] for q in IDS], fontsize=7.5, rotation=30, ha='right')
        ax.set_xlim(-0.5, 5.5); ax.set_ylim(-0.04, 1.04)
        ax.set_title(f'trace contains {SHORT[present]}', fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel('P(rule | trace)', fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel('marginal read off for which rule', fontsize=8.5)

    handles = [plt.Line2D([], [], color=POS_COLOR[p], lw=0, marker=POS_MARKER[p],
                          ms=5.5, label=POS_LABEL[p]) for p in POSITIONS]
    handles.append(plt.Rectangle((0, 0), 1, 1, color='#f3f0e8',
                                 label=f'an ABSENT rule above {UNSUPPORTED_MAX} is '
                                       f'never used as a foil'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.925),
               ncol=3, fontsize=8.5, frameon=False)
    fig.suptitle('What the ideal observer extracts from a trace: ALL SIX marginals\n'
                 'The statement is never an input to inference, it only picks which entry gets '
                 'read. One dot per item, 40 per panel per rule.\n'
                 'The present rule is pinned at 1.000. Absent rules mostly sit below 0.35, except '
                 'outside(), which crosses 0.5 on 15 of 1200\ntrace-by-rule combinations '
                 '(max 0.871). Those high-marginal foils are excluded from the pool, so no '
                 'category-B item ever probes one.',
                 fontsize=10.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    p = os.path.join(HERE, 'bayes_1misc_profile.png')
    fig.savefig(p, dpi=140); plt.close(fig)

    print("max marginal on an ABSENT rule, per present rule:")
    for present in IDS:
        mx = max(max(v) for q, d in profile[present].items() if q != present for v in d.values())
        print(f"  trace contains {SHORT[present]:>9s}: {mx:.3f}")
    print(f"\nWrote {p}")


if __name__ == '__main__':
    main()
