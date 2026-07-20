#!/usr/bin/env python3
"""
make_llm_1misc_heatmap.py

LLM counterpart of analysis-Bayesian/plot_bayes_1misc_heatmap.py and
analysis_human/plot_human_1misc_heatmap.py: the present x named "confusion
matrix" for the 1-misconception items (categories A and B). Rows =
misconception PRESENT in the trace, columns = misconception NAMED in the
statement; DIAGONAL = category A (agree correct), OFF-DIAGONAL = category B
foils (disagree correct), split into refuted vs unsupported panels.

Three regimes, one row each (haiku thinking, haiku direct, gpt-4o direct); two
columns (refuted, unsupported). Cell value = mean 1--6 rating, centered at 3.5,
same diverging green/red scale as the human and 2-misc figures. Reads the
full-pool runs from results/raw_*.jsonl (subject_id contains "all_items"),
filtered to the current pool.

Run from llm_exp/:
    python3 make_llm_1misc_heatmap.py
"""
import glob
import json
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, 'plots')
POOL = os.path.join(HERE, 'data', 'stimulus_pool.json')

IDS = ['add_before_mul', 'add_before_div', 'sub_before_mul', 'sub_before_div',
       'same_priority_rtl', 'outside_bracket_first']
SHORT = {
    'add_before_mul': 'add<×', 'add_before_div': 'add<÷',
    'sub_before_mul': 'sub<×', 'sub_before_div': 'sub<÷',
    'same_priority_rtl': 'RTL', 'outside_bracket_first': 'outside()',
}
REGIMES = [('haiku (thinking)', 'anthropic/claude-haiku-4.5', 'thinking'),
           ('haiku (direct)', 'anthropic/claude-haiku-4.5', 'direct'),
           ('gpt-4o (direct)', 'openai/gpt-4o', 'direct')]
CMAP = LinearSegmentedColormap.from_list('divratings',
                                         ['#e34948', '#f0efec', '#008300'])
NORM = TwoSlopeNorm(vmin=1.0, vcenter=3.5, vmax=6.0)
DARK_AT = 1.4


def load_rows():
    pool = {it['id']: it for it in json.load(open(POOL))}
    rows = []
    for path in sorted(glob.glob(os.path.join(HERE, 'results', 'raw_*.jsonl'))):
        for line in open(path):
            r = json.loads(line)
            if ('all_items' in r['subject_id'] and r.get('error') is None
                    and r.get('response') is not None and r['id'] in pool
                    and r['num_misconceptions'] == 1):
                r['foil_status'] = pool[r['id']].get('foil_status')
                rows.append(r)
    return rows


def matrices(rows, model, effort):
    idx = {m: i for i, m in enumerate(IDS)}
    sub = [r for r in rows if r['model'] == model and r['effort'] == effort]
    seen, diag, off = set(), {i: [] for i in range(6)}, {'refuted': {}, 'unsupported': {}}
    for r in sub:
        if r['id'] in seen:
            continue
        seen.add(r['id'])
        present, named = idx[r['misconceptions'][0]], idx[r['probed_misconception']]
        if r['category'] == 'A':
            diag[present].append(r['response'])
        else:
            st = r['foil_status']
            if st in off:
                off[st].setdefault((present, named), []).append(r['response'])
    out = {}
    for st in ('refuted', 'unsupported'):
        mean, n = np.full((6, 6), np.nan), np.zeros((6, 6), int)
        for i in range(6):
            if diag[i]:
                mean[i, i], n[i, i] = np.mean(diag[i]), len(diag[i])
        for (p, nm), vals in off[st].items():
            mean[p, nm], n[p, nm] = np.mean(vals), len(vals)
        out[st] = (mean, n)
    return out


def draw(ax, mean, n, title, ylabel):
    shown = np.where(np.isnan(mean), 3.5, mean)
    im = ax.imshow(shown, cmap=CMAP, norm=NORM, aspect='equal')
    for i in range(6):
        for j in range(6):
            if n[i, j] == 0:
                ax.text(j, i, '·', ha='center', va='center', color='#898781')
                continue
            dark = abs(mean[i, j] - 3.5) > DARK_AT
            ax.text(j, i - 0.1, f"{mean[i, j]:.1f}", ha='center', va='center',
                    fontsize=8, fontweight='bold', color='white' if dark else '#0b0b0b')
            ax.text(j, i + 0.29, f"n={n[i, j]}", ha='center', va='center',
                    fontsize=5, color='white' if dark else '#52514e')
    for i in range(6):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor='#0b0b0b', lw=1.2))
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels([SHORT[m] for m in IDS], fontsize=7, rotation=30, ha='right')
    ax.set_yticklabels([SHORT[m] for m in IDS], fontsize=7)
    if ylabel:
        ax.set_ylabel(ylabel + '\n\npresent in trace', fontsize=9, fontweight='bold')
    if title:
        ax.set_title(title, fontsize=10)
    return im


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rows = load_rows()
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 15.5),
                             gridspec_kw={'wspace': 0.05, 'hspace': 0.28})
    fig.subplots_adjust(top=0.93, bottom=0.05)
    im = None
    for row, (label, model, effort) in enumerate(REGIMES):
        mats = matrices(rows, model, effort)
        top = ('category-B foils REFUTED' if row == 0 else '',
               'category-B foils UNSUPPORTED' if row == 0 else '')
        im = draw(axes[row, 0], *mats['refuted'], top[0], label)
        draw(axes[row, 1], *mats['unsupported'], top[1], '')
        for ax in axes[row]:
            ax.set_xlabel('named in statement', fontsize=8)
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, ticks=[1, 2, 3, 3.5, 4, 5, 6])
    cbar.ax.set_yticklabels(['1 (SD)', '2', '3', '3.5', '4', '5', '6 (SA)'], fontsize=8)
    cbar.set_label('mean rating (>3.5 = agree side)', fontsize=9)
    fig.suptitle('LLMs, one-misconception items: present (rows) × named (columns)\n'
                 'boxed diagonal = category A (agree correct); off-diagonal = category B foils '
                 '(disagree correct), refuted (left) vs unsupported (right)', fontsize=12.5)
    p = os.path.join(OUTDIR, 'llm_1misc_heatmap.png')
    fig.savefig(p, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {p}")


if __name__ == '__main__':
    main()
