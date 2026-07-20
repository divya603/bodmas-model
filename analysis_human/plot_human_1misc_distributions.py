#!/usr/bin/env python3
"""
plot_human_1misc_distributions.py

Human counterpart of analysis-Bayesian/plot_bayes_1misc_distributions.py:
distribution of the 1-6 Likert rating on the 1-misconception trials, same
panel layouts, with the decision boundary at 3.5 instead of 0.5.

  human_1misc_dist_A.png — category A (statement matches; agree correct).
      6 panels grouped by the misconception present (= named in A).
  human_1misc_dist_B.png — category B (statement is a foil; disagree
      correct). Rows 1-2 are category-B (1-misconception) trials grouped by
      the misconception PRESENT and NAMED. Row 3 is the REFUTED foils only,
      POOLED across categories B and D (foil_status == 'refuted' in the pool;
      each rule is refuted exactly once per participant across B+D, so pooling
      doubles the density vs a B-only view), grouped by named rule. If humans
      use refutation evidence, row 3 should sit lower than row 2.

Responses are shown BINARY: the 1-6 rating is collapsed to disagree (<=3)
vs agree (>=4), the same collapse used for accuracy scoring, and each panel
is the probability distribution over those two responses (fixed 0-1 y-axis
so peaks are comparable across panels; trial counts on the bars; dashed
line at 0.5). The full 1-6 scale proved too noisy at n=24 per cell to show
a picture. The Bayesian counterpart keeps KDE curves because its marginals
are continuous.

Usage (from repo root):
    python3 analysis_human/plot_human_1misc_distributions.py
"""

import argparse
import json
import os
from collections import Counter

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
POOL = os.path.join(os.path.dirname(HERE), 'base-task', 'stimulus_pool.json')

IDS = ['add_before_mul', 'add_before_div', 'sub_before_mul', 'sub_before_div',
       'same_priority_rtl', 'outside_bracket_first']
SHORT = {
    'add_before_mul': 'add<×', 'add_before_div': 'add<÷',
    'sub_before_mul': 'sub<×', 'sub_before_div': 'sub<÷',
    'same_priority_rtl': 'RTL', 'outside_bracket_first': 'outside()',
}
GREEN, RED = '#008300', '#e34948'   # agree = green, disagree = red (heatmap convention)
XTICKLABELS = ['disagree\n(1–3)', 'agree\n(4–6)']

EXPECTED_TRIALS = 24   # a complete form is 24 task trials


def task_trials(d):
    return [t for t in d.get('pageData_exp', {}).get('visit_0', {}).get('data', [])
            if isinstance(t, dict) and 'response' in t]


def is_complete(d):
    """Incomplete participants (abandoned / no completion code) are dropped up
    front: require the Smile done flag AND a full 24-trial form. The 2-minute
    'NO CODE' Prolific returns come through as done=False with 0 task trials."""
    return d.get('done') is True and len(task_trials(d)) >= EXPECTED_TRIALS


def panel(ax, vals, title, empty_text='no refuted items\n(foil never testable)'):
    if len(vals) == 0:
        ax.text(0.5, 0.5, empty_text, ha='center',
                va='center', fontsize=8, color='#898781', transform=ax.transAxes)
        ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, 1)
        ax.set_xticks([0, 1]); ax.set_xticklabels(XTICKLABELS, fontsize=8)
        ax.set_yticks([0, 0.5, 1])
        ax.set_title(title, fontsize=10)
        return
    n = len(vals)
    yes = sum(v >= 4 for v in vals)
    props = [(n - yes) / n, yes / n]
    ax.bar([0, 1], props, width=0.6, color=[RED, GREEN], alpha=0.88)
    for x, (p, k) in enumerate(zip(props, [n - yes, yes])):
        ax.text(x, p + 0.03, str(k), ha='center', va='bottom', fontsize=8,
                color='#52514e')
    ax.axhline(0.5, ls='--', lw=0.8, color='grey')
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 1.12)
    ax.set_xticks([0, 1]); ax.set_xticklabels(XTICKLABELS, fontsize=8)
    ax.set_yticks([0, 0.5, 1])
    ax.set_title(title, fontsize=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/real-all-main-data.json')
    ap.add_argument('--out', default='analysis_human/plots')
    ap.add_argument('--cohort', choices=['all', 'practice'], default='all',
                    help="'practice' = only participants who did the new practice-trial "
                         "task (identified by a pageData_practice block); the intermediate "
                         "new-data view")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # authoritative per-item refutation status (the extended pool has foil_status
    # on every B and D foil); join by id so it works for old and new pool ids
    foil_status = {it['id']: it.get('foil_status') for it in json.load(open(POOL))}

    data = json.load(open(args.data))
    trials = []        # 1-misconception trials (categories A, B) for rows 1-2
    foil_trials = []   # all foil trials (B + D) for the pooled refuted row 3
    n_part = 0
    n_incomplete = 0
    for r in data:
        d = r['data']
        if args.cohort == 'practice' and not d.get('pageData_practice'):
            continue
        if not is_complete(d):          # very first filter: drop incompletes
            if d.get('pageData_practice'):
                n_incomplete += 1
            continue
        if d.get('recruitmentService') != 'prolific':
            continue
        n_part += 1
        tt = task_trials(d)
        trials += [t for t in tt if t.get('num_misconceptions') == 1]
        foil_trials += [t for t in tt if t['category'] in ('B', 'D')]
    print(f"cohort={args.cohort}: {n_part} complete participants "
          f"(dropped {n_incomplete} incomplete)")
    empty_ref = 'no trials yet' if args.cohort == 'practice' else \
                'no refuted items\n(foil never testable)'

    a_by_present = {m: [] for m in IDS}
    b_by_present = {m: [] for m in IDS}
    b_by_named = {m: [] for m in IDS}
    for t in trials:
        if t['category'] == 'A':
            a_by_present[t['misconceptions'][0]].append(t['response'])
        else:
            b_by_present[t['misconceptions'][0]].append(t['response'])
            b_by_named[t['probed_misconception']].append(t['response'])

    # refuted row pools refuted foils from BOTH B and D (each rule is refuted
    # exactly once per participant across B+D, doubling the density vs a B-only
    # view), grouped by the named foil rule
    b_refuted = {m: [] for m in IDS}
    for t in foil_trials:
        if foil_status.get(t['id']) == 'refuted':
            b_refuted[t['probed_misconception']].append(t['response'])

    print(f"1-misc trials: {len(trials)}  "
          f"(A={sum(t['category'] == 'A' for t in trials)}, "
          f"B={sum(t['category'] == 'B' for t in trials)})")
    for name, cell in [('A by present', a_by_present), ('B by present', b_by_present),
                       ('B by named', b_by_named), ('refuted B+D', b_refuted)]:
        print(f"  {name:15s} " + "  ".join(f"{SHORT[m]}:{len(cell[m])}" for m in IDS))
        rates = "  ".join(
            f"{SHORT[m]}:{np.mean([v >= 4 for v in cell[m]]):.2f}" if cell[m] else f"{SHORT[m]}:-"
            for m in IDS)
        print(f"  {'  P(agree)':15s} " + rates)

    tag = ('PRACTICE-TASK cohort' if args.cohort == 'practice'
           else 'all prolific participants')
    # ── figure 1: category A ──
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for ax, m in zip(axes.flat, IDS):
        panel(ax, a_by_present[m], f"{SHORT[m]}  (n={len(a_by_present[m])})")
    for ax in axes[1]:
        ax.set_xlabel('response', fontsize=9)
    fig.suptitle('Humans — category A (statement matches; agree correct):\n'
                 'probability of agree (green) vs disagree (red) by misconception present '
                 f'({n_part} participants, {tag}; counts on bars; correct answer = agree)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p1 = os.path.join(args.out, 'human_1misc_dist_A.png')
    fig.savefig(p1, dpi=140)
    plt.close(fig)

    # ── figure 2: category B — two groupings + refuted-only row ──
    fig, axes = plt.subplots(3, 6, figsize=(16.5, 8.6), sharex=True)
    for j, m in enumerate(IDS):
        panel(axes[0, j], b_by_present[m], f"{SHORT[m]}  (n={len(b_by_present[m])})")
        panel(axes[1, j], b_by_named[m], f"{SHORT[m]}  (n={len(b_by_named[m])})")
        panel(axes[2, j], b_refuted[m], f"{SHORT[m]}  (n={len(b_refuted[m])})",
              empty_text=empty_ref)
        axes[2, j].set_xlabel('response', fontsize=8)
    axes[0, 0].set_ylabel('grouped by misconception\nPRESENT in trace', fontsize=9)
    axes[1, 0].set_ylabel('grouped by misconception\nNAMED in statement (foil)', fontsize=9)
    axes[2, 0].set_ylabel('REFUTED foils only\n(B + D pooled), by named rule', fontsize=9)
    fig.suptitle('Humans — category B (statement is a foil; disagree correct): '
                 'probability of agree (green) vs disagree (red)\n'
                 f'{n_part} participants ({tag}): B by present, B by named, then REFUTED '
                 'foils pooled across B+D (counts on bars; correct answer = disagree)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p2 = os.path.join(args.out, 'human_1misc_dist_B.png')
    fig.savefig(p2, dpi=140)
    plt.close(fig)

    print(f"\nWrote {p1}\nWrote {p2}")


if __name__ == '__main__':
    main()
