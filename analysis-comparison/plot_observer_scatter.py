#!/usr/bin/env python3
"""
plot_observer_scatter.py

Cross-observer comparison on a single common scale. Humans rate 1-6, LLMs rate
1-6, the Bayesian ideal observer emits a posterior probability P(named rule |
trace); those do not share a scale, so we reduce every observer to ONE quantity:

    P(Acc | trace) = probability the observer assigns to the CORRECT answer
                   =      P_hat(agree)   when the statement is correct (rule
                                         present: categories A, C)
                   =  1 - P_hat(agree)   when the statement is a foil (rule
                                         absent: categories B, D)

where P_hat(agree) is:
    Bayes  : the posterior marginal P(named rule | trace)          [native prob]
    human  : the 1-6 rating read as confidence, (rating - 1) / 5   [graded]
    LLM    : same graded map on its 1-6 rating

(--scoring binary instead collapses the rating to agree iff >= 4, so P(Acc)
becomes plain accuracy; kept for cross-check against the d'/accuracy figures.)

A point is one (named misconception x category) group. Humans have n=21 trials
per group, Bayes/LLM ~20 items per group, so every point is well estimated
(per-item human data is unusable: median 1 rating per item). The scatter plots
one observer's P(Acc) against another's over the same 24 groups:

    (a) Bayes (y) vs Human (x)                 -- the left panel of the sketch
    (b) LLM   (y) vs Human (x)  x 3 regimes
    (c) LLM   (y) vs Bayes (x)  x 3 regimes

The dashed y = x line is perfect agreement; a point above it means the y-axis
observer scored the correct answer higher on that group. Bayes is the ideal, so
in (a) points sit at/above the diagonal and in (c) at/below it.

Run from repo root:
    python3 analysis-comparison/plot_observer_scatter.py
    python3 analysis-comparison/plot_observer_scatter.py --scoring binary
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'analysis_human'))
from plot_2misc_heatmap import is_complete, task_trials  # noqa: E402

IDS = ['add_before_mul', 'add_before_div', 'sub_before_mul', 'sub_before_div',
       'same_priority_rtl', 'outside_bracket_first']
SHORT = {
    'add_before_mul': 'add<×', 'add_before_div': 'add<÷',
    'sub_before_mul': 'sub<×', 'sub_before_div': 'sub<÷',
    'same_priority_rtl': 'RTL', 'outside_bracket_first': 'outside()',
}
# fixed colour per misconception, fixed marker per category
MISC_COLOR = {
    'add_before_mul': '#d62728', 'add_before_div': '#ff7f0e',
    'sub_before_mul': '#2ca02c', 'sub_before_div': '#1f77b4',
    'same_priority_rtl': '#9467bd', 'outside_bracket_first': '#111111',
}
CAT_MARKER = {'A': 'o', 'B': 's', 'C': '^', 'D': 'D'}
CAT_DESC = {'A': 'A  present, 1-misc', 'B': 'B  absent, 1-misc',
            'C': 'C  present, 2-misc', 'D': 'D  absent, 2-misc'}
REGIMES = [
    ('haiku (thinking)', 'anthropic/claude-haiku-4.5', 'thinking'),
    ('haiku (direct)', 'anthropic/claude-haiku-4.5', 'direct'),
    ('gpt-4o (direct)', 'openai/gpt-4o', 'direct'),
]


def pool_info():
    """id -> (probed_misconception, category, statement_correct)."""
    pool = json.load(open(os.path.join(ROOT, 'base-task/stimulus_pool.json')))
    return {it['id']: (it['probed_misconception'], it['category'],
                       bool(it['statement_correct'])) for it in pool}


def p_agree(rating, scoring):
    return (rating >= 4) * 1.0 if scoring == 'binary' else (rating - 1) / 5.0


def group_scores(items, info, scoring, is_prob=False):
    """
    items: iterable of (id, value). value is a 1-6 rating unless is_prob (Bayes,
    already P_hat(agree)). Returns {(misc, cat): (mean_pacc, sem, n)}.
    """
    buckets = defaultdict(list)
    for iid, val in items:
        if iid not in info:
            continue
        probed, cat, correct = info[iid]
        phat = val if is_prob else p_agree(val, scoring)
        pacc = phat if correct else 1.0 - phat
        buckets[(probed, cat)].append(pacc)
    out = {}
    for key, vals in buckets.items():
        v = np.array(vals)
        sem = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        out[key] = (v.mean(), sem, len(v))
    return out


def bayes_items():
    b = json.load(open(os.path.join(ROOT, 'dashboard/bayes_per_item.json')))
    return [(iid, m) for iid, m in b.items()]


def human_items():
    data = json.load(open(os.path.join(ROOT, 'data/real-all-main-data.json')))
    items = []
    n_part = 0
    for r in data:
        d = r['data']
        if not d.get('pageData_practice'):          # practice-task cohort
            continue
        if not is_complete(d) or d.get('recruitmentService') != 'prolific':
            continue
        n_part += 1
        for t in task_trials(d):
            if t.get('num_misconceptions') in (1, 2) and 'response' in t:
                items.append((t['id'], t['response']))
    return items, n_part


def llm_items():
    pool_ids = {it['id'] for it in json.load(
        open(os.path.join(ROOT, 'llm_exp/data/stimulus_pool.json')))}
    by_regime = {lab: [] for lab, _, _ in REGIMES}
    seen = {lab: set() for lab, _, _ in REGIMES}
    lookup = {(m, e): lab for lab, m, e in REGIMES}
    for path in sorted(glob.glob(os.path.join(ROOT, 'llm_exp/results/raw_*.jsonl'))):
        for line in open(path):
            r = json.loads(line)
            if 'all_items' not in r['subject_id'] or r.get('error') is not None:
                continue
            if r.get('response') is None or r['id'] not in pool_ids:
                continue
            lab = lookup.get((r['model'], r['effort']))
            if lab is None or r['id'] in seen[lab]:
                continue
            seen[lab].add(r['id'])
            by_regime[lab].append((r['id'], r['response']))
    return by_regime


def draw_scatter(ax, gx, gy, xlabel, ylabel, title):
    """gx, gy: {(misc,cat): (mean,sem,n)}. One point per shared group."""
    ax.plot([0, 1], [0, 1], ls='--', lw=1.0, color='#999', zorder=1)
    for key in sorted(set(gx) & set(gy)):
        misc, cat = key
        mx, sx, _ = gx[key]
        my, sy, _ = gy[key]
        ax.errorbar(mx, my, xerr=sx, yerr=sy, fmt='none',
                    ecolor=MISC_COLOR[misc], elinewidth=0.7, alpha=0.5, zorder=2)
        ax.scatter(mx, my, marker=CAT_MARKER[cat], s=55,
                   facecolor=MISC_COLOR[misc], edgecolor='white', linewidth=0.6,
                   zorder=3)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_aspect('equal')
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.set_title(title, fontsize=9.5)
    ax.tick_params(labelsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scoring', choices=['graded', 'binary'], default='graded')
    ap.add_argument('--out', default='analysis-comparison')
    args = ap.parse_args()
    os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)

    info = pool_info()
    g_bayes = group_scores(bayes_items(), info, args.scoring, is_prob=True)
    h_items, n_part = human_items()
    g_human = group_scores(h_items, info, args.scoring)
    g_llm = {lab: group_scores(items, info, args.scoring)
             for lab, items in llm_items().items()}

    # console table
    print(f"scoring={args.scoring}  (human practice cohort n={n_part})")
    print(f"{'group':22s} {'Bayes':>7s} {'Human':>7s} " +
          " ".join(f"{lab.split()[0][:6]:>7s}" for lab, _, _ in REGIMES))
    for cat in 'ABCD':
        for m in IDS:
            k = (m, cat)
            if k not in g_bayes:
                continue
            row = f"{SHORT[m]+'/'+cat:22s} {g_bayes[k][0]:7.3f} {g_human[k][0]:7.3f} "
            row += " ".join(f"{g_llm[lab][k][0]:7.3f}" for lab, _, _ in REGIMES)
            print(row)

    # figure: row 1 = [Bayes-vs-Human | LLM-vs-Human x3], row 2 = [legend | LLM-vs-Bayes x3]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9.2))
    draw_scatter(axes[0, 0], g_human, g_bayes,
                 'Human  P(Acc | trace)', 'Bayes  P(Acc | trace)',
                 '(a) Bayes vs Human')
    for c, (lab, _, _) in enumerate(REGIMES):
        draw_scatter(axes[0, c + 1], g_human, g_llm[lab],
                     'Human  P(Acc | trace)', f'{lab}  P(Acc)',
                     f'(b) {lab} vs Human')
        # Bayes stays on the y-axis in every panel it appears (same as panel a),
        # so the ideal is always the ceiling / above the diagonal; the LLM is the
        # one observer that flips axis between (b) and (c).
        draw_scatter(axes[1, c + 1], g_llm[lab], g_bayes,
                     f'{lab}  P(Acc)', 'Bayes  P(Acc | trace)',
                     f'(c) Bayes vs {lab}')

    # legend cell
    lg = axes[1, 0]; lg.axis('off')
    misc_h = [plt.Line2D([], [], marker='o', ls='', color=MISC_COLOR[m],
                         markeredgecolor='white', markersize=8, label=SHORT[m])
              for m in IDS]
    cat_h = [plt.Line2D([], [], marker=CAT_MARKER[c], ls='', color='#555',
                        markeredgecolor='white', markersize=8, label=CAT_DESC[c])
             for c in 'ABCD']
    diag_h = [plt.Line2D([], [], ls='--', color='#999', label='y = x (agreement)')]
    lg.legend(handles=misc_h + cat_h + diag_h, loc='center', fontsize=8.5,
              frameon=False, title='colour = misconception,  marker = category',
              title_fontsize=9)

    score_desc = ('graded: rating→(r−1)/5' if args.scoring == 'graded'
                  else 'binary: agree iff rating≥4 (= accuracy)')
    fig.suptitle('Observer comparison on a common scale  P(Acc | trace) = probability placed on '
                 'the correct answer\n'
                 f'one point per (misconception × category), {score_desc}; '
                 'above the dashed line = the y-axis observer is more accurate on that group',
                 fontsize=12.5, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(ROOT, args.out, f'observer_scatter_{args.scoring}.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nWrote {p}")


if __name__ == '__main__':
    main()
