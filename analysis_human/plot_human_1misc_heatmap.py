#!/usr/bin/env python3
"""
plot_human_1misc_heatmap.py

Human counterpart of analysis-Bayesian/plot_bayes_1misc_heatmap.py: the
present x named "confusion matrix" for the 1-misconception items (categories A
and B), practice cohort. Rows = misconception PRESENT in the trace, columns =
misconception NAMED in the statement.

  - DIAGONAL (present = named) = category A (agreeing is correct; green ideal).
  - OFF-DIAGONAL (present != named) = category B foils (disagreeing is correct;
    red ideal), split into two panels by the foil's refutation status.

Cell value = mean 1--6 rating (centered at 3.5), same diverging green/red scale
as the 2-misconception human heatmap. At n=21 the off-diagonal cells are sparse
(a dot marks a cell no participant drew); they fill in as the cohort grows.

Run from repo root:
    python3 analysis_human/plot_human_1misc_heatmap.py
    python3 analysis_human/plot_human_1misc_heatmap.py --cohort all
"""
import argparse
import json
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
POOL = os.path.join(os.path.dirname(HERE), 'base-task', 'stimulus_pool.json')

IDS = ['add_before_mul', 'add_before_div', 'sub_before_mul', 'sub_before_div',
       'same_priority_rtl', 'outside_bracket_first']
SHORT = {
    'add_before_mul': 'add<×', 'add_before_div': 'add<÷',
    'sub_before_mul': 'sub<×', 'sub_before_div': 'sub<÷',
    'same_priority_rtl': 'RTL', 'outside_bracket_first': 'outside()',
}
CMAP = LinearSegmentedColormap.from_list('divratings',
                                         ['#e34948', '#f0efec', '#008300'])
NORM = TwoSlopeNorm(vmin=1.0, vcenter=3.5, vmax=6.0)
DARK_AT = 1.4
EXPECTED_TRIALS = 24


def task_trials(d):
    return [t for t in d.get('pageData_exp', {}).get('visit_0', {}).get('data', [])
            if isinstance(t, dict) and 'response' in t]


def is_complete(d):
    return d.get('done') is True and len(task_trials(d)) >= EXPECTED_TRIALS


def build_matrices(data, cohort):
    foil_status = {it['id']: it.get('foil_status')
                   for it in json.load(open(POOL))}
    idx = {m: i for i, m in enumerate(IDS)}
    diag = {i: [] for i in range(6)}
    off = {'refuted': {}, 'unsupported': {}}
    n_part = 0
    for r in data:
        d = r['data']
        if cohort == 'practice' and not d.get('pageData_practice'):
            continue
        if not is_complete(d) or d.get('recruitmentService') != 'prolific':
            continue
        n_part += 1
        for t in task_trials(d):
            if t.get('num_misconceptions') != 1:
                continue
            present, named = idx[t['misconceptions'][0]], idx[t['probed_misconception']]
            if t['category'] == 'A':
                diag[present].append(t['response'])
            else:
                st = foil_status.get(t['id'])
                if st in off:
                    off[st].setdefault((present, named), []).append(t['response'])
    mats = {}
    for st in ('refuted', 'unsupported'):
        mean, n = np.full((6, 6), np.nan), np.zeros((6, 6), int)
        for i in range(6):
            if diag[i]:
                mean[i, i], n[i, i] = np.mean(diag[i]), len(diag[i])
        for (p, nm), vals in off[st].items():
            mean[p, nm], n[p, nm] = np.mean(vals), len(vals)
        mats[st] = (mean, n)
    return mats, n_part


def draw(ax, mean, n, title):
    shown = np.where(np.isnan(mean), 3.5, mean)
    im = ax.imshow(shown, cmap=CMAP, norm=NORM, aspect='equal')
    for i in range(6):
        for j in range(6):
            if n[i, j] == 0:
                ax.text(j, i, '·', ha='center', va='center', color='#898781')
                continue
            dark = abs(mean[i, j] - 3.5) > DARK_AT
            ax.text(j, i - 0.12, f"{mean[i, j]:.1f}", ha='center', va='center',
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/real-all-main-data.json')
    ap.add_argument('--out', default='analysis_human/plots')
    ap.add_argument('--cohort', choices=['all', 'practice'], default='practice')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    data = json.load(open(args.data))
    mats, n_part = build_matrices(data, args.cohort)
    print(f"cohort={args.cohort}: {n_part} participants")

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.0), gridspec_kw={'wspace': 0.30})
    fig.subplots_adjust(top=0.80)
    draw(axes[0], *mats['refuted'],
         '(a) category-B foils REFUTED by the trace\n(boxed diagonal = category A, agreeing is correct)')
    im = draw(axes[1], *mats['unsupported'],
              '(b) category-B foils UNSUPPORTED\n(boxed diagonal = category A, agreeing is correct)')
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, ticks=[1, 2, 3, 3.5, 4, 5, 6])
    cbar.ax.set_yticklabels(['1 (SD)', '2', '3', '3.5', '4', '5', '6 (SA)'], fontsize=8)
    cbar.set_label('mean rating (>3.5 = agree side)', fontsize=9)
    fig.suptitle(f'Humans (practice cohort, n={n_part}), one-misconception items: '
                 'present (rows) × named (columns)', fontsize=12.5, y=0.98)
    suffix = '_with_practice' if args.cohort == 'practice' else ''
    p = os.path.join(args.out, f'human_1misc_heatmap{suffix}.png')
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"Wrote {p}")


if __name__ == '__main__':
    main()
