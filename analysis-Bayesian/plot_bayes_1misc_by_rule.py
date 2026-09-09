#!/usr/bin/env python3
"""
plot_bayes_1misc_by_rule.py

The Bayesian ideal observer, one panel per misconception, showing the only
question the observer actually answers about that rule:

    P(rule X | trace), when X IS in the trace   vs   when X is ABSENT

This replaces the old dist_A / dist_B pair. Those split the same question by
category, which made a figure out of a distinction the observer does not have:
`posterior_over_profiles()` takes ONLY the trace, and the belief statement enters
afterwards purely as an index into an already-computed posterior. So category A
was the present half and category B row 2 was the absent half of one question,
while dist_B row 1 pooled five incommensurable marginals into one panel.

Two deliberate choices:

1. **Every trace counts, not only the ones the pool probed.** For rule X the
   present group is all 40 traces containing X and the absent group is all 200
   traces that do not, rather than the 20 A items and 20 B items whose statement
   happened to name X. That is the honest denominator once you accept the
   statement is not an input, and it matters: `pool.py` refuses any foil whose
   marginal exceeds 0.35, so restricting to probed items would silently hide
   every hard case.
2. **The excluded band is drawn.** Anything above 0.35 is shaded, because no
   category-B item ever probes a foil in that band. For five rules the band is
   empty. For `outside_bracket_first` it is not, and that asymmetry is the point.

Stems sit on the exact observed values; heights are proportions WITHIN each
(condition, position) group, so present and absent stay comparable despite the
5x difference in group size. Not KDEs: the marginals are discrete.

Run from repo root:
    python3 analysis-Bayesian/plot_bayes_1misc_by_rule.py
"""
import json
import os
import sys
from collections import Counter

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bayes_common import (HERE, BASE_TASK, IDS, SHORT, POSITIONS, POS_COLOR,
                          POS_MARKER, POS_LABEL)

sys.path.insert(0, BASE_TASK)
from inference import posterior_over_profiles, marginal_rule_probability  # noqa: E402
from pool import HYPOTHESES, UNSUPPORTED_MAX                              # noqa: E402

DX = 0.006          # stem offset between the two positions
STEM_W = 2.0


def collect():
    """{rule: {'present'|'absent': {position: [marginals]}}} over every trace."""
    pool = json.load(open(os.path.join(BASE_TASK, 'stimulus_pool.json'), encoding='utf-8'))
    out = {m: {c: {p: [] for p in POSITIONS} for c in ('present', 'absent')} for m in IDS}
    for it in pool:
        post = posterior_over_profiles(it['trace'], profiles=HYPOTHESES)
        true_m, pos = it['misconceptions'][0], it['error_position']
        for rule in IDS:
            cond = 'present' if rule == true_m else 'absent'
            out[rule][cond][pos].append(marginal_rule_probability(post, rule))
    return out


def panel(ax, data, rule):
    # The band the pool refuses to use as a foil. It stops short of 1.0 so it
    # cannot be read as covering the PRESENT spike: foil selection only ever
    # applies to a rule that is absent.
    ax.axvspan(UNSUPPORTED_MAX, 0.94, color='#f3f0e8', zorder=0)
    ax.axvline(0.5, ls='--', lw=0.9, color='#b8b6b0', zorder=1)

    for cond in ('absent', 'present'):
        for k, pos in enumerate(POSITIONS):
            vals = data[cond][pos]
            if not vals:
                continue
            counts = Counter(round(v, 6) for v in vals)
            n = len(vals)
            xoff = (k - 0.5) * 2 * DX
            for v, c in sorted(counts.items()):
                ax.vlines(v + xoff, 0, c / n, color=POS_COLOR[pos], lw=STEM_W,
                          alpha=0.9, zorder=3)
                ax.plot(v + xoff, c / n, marker=POS_MARKER[pos], ms=3.6,
                        color=POS_COLOR[pos], zorder=4)

    n_pres = sum(len(v) for v in data['present'].values())
    n_abs = sum(len(v) for v in data['absent'].values())
    ax.text(0.985, 1.055, f'PRESENT\nn={n_pres}', ha='right', va='top',
            fontsize=6.8, color='#0b0b0b', fontweight='bold')
    ax.text(0.02, 1.055, f'ABSENT  n={n_abs}', ha='left', va='top',
            fontsize=6.8, color='#52514e')

    # how much absent mass sits in the excluded band
    hi = [v for p in POSITIONS for v in data['absent'][p] if v > UNSUPPORTED_MAX]
    if hi:
        ax.text(0.66, 0.62, f'{len(hi)} of {n_abs}\nabove {UNSUPPORTED_MAX}\nmax {max(hi):.2f}',
                fontsize=6.6, color='#8a6d3b', ha='center', va='center')

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.16)
    ax.set_yticks([0, 0.5, 1.0]); ax.set_yticklabels(['0', '.5', '1'], fontsize=7)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.tick_params(axis='x', labelsize=7)
    ax.set_title(SHORT[rule], fontsize=10.5)


def main():
    data = collect()

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.8), sharex=True)
    for ax, rule in zip(axes.flat, IDS):
        panel(ax, data[rule], rule)
    for ax in axes[1]:
        ax.set_xlabel('P(this rule | trace)', fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel('proportion within group', fontsize=8.5)

    handles = [plt.Line2D([], [], color=POS_COLOR[p], lw=2, marker=POS_MARKER[p],
                          ms=4.5, label=POS_LABEL[p]) for p in POSITIONS]
    handles.append(plt.Rectangle((0, 0), 1, 1, color='#f3f0e8',
                                 label=f'an ABSENT rule above {UNSUPPORTED_MAX} is '
                                       f'never used as a foil'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.915),
               ncol=3, fontsize=8.5, frameon=False)
    fig.suptitle('Bayesian ideal observer: P(rule | trace) when the rule IS in the trace '
                 'versus when it is ABSENT\n'
                 'One panel per misconception, over all 240 traces (40 present, 200 absent per '
                 'rule). The statement is never an input,\nso this is the whole question. '
                 'Present is a point mass at 1.000 everywhere; absent is where the variation is.',
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.865])
    p = os.path.join(HERE, 'bayes_1misc_by_rule.png')
    fig.savefig(p, dpi=140); plt.close(fig)

    print("per rule: P(rule | trace)")
    for rule in IDS:
        pres = [v for p in POSITIONS for v in data[rule]['present'][p]]
        abst = [v for p in POSITIONS for v in data[rule]['absent'][p]]
        hi = [v for v in abst if v > UNSUPPORTED_MAX]
        print(f"  {SHORT[rule]:>9s}  present n={len(pres):3d} all={set(pres)}   "
              f"absent n={len(abst):3d} max={max(abst):.3f}  "
              f"above {UNSUPPORTED_MAX}: {len(hi)}")
    print(f"\nWrote {p}")


if __name__ == '__main__':
    main()
