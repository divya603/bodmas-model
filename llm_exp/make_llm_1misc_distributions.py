#!/usr/bin/env python3
"""
make_llm_1misc_distributions.py

LLM counterpart of the human/Bayesian 1-misconception distribution figures:
distribution of the 1-6 rating on the 1-misconception items, one curve per
regime (haiku thinking / haiku direct / gpt-4o direct) overlaid per panel.

  llm_1misc_dist_A.png — category A (statement matches), 6 panels grouped by
      the misconception present (= named in A). 10 items per regime per panel.
  llm_1misc_dist_B.png — category B (foil): row 1 grouped by misconception
      PRESENT, row 2 by misconception NAMED, row 3 = the REFUTED-only items
      (ideal-observer marginal < 0.15, from
      analysis-Bayesian/b_item_marginals.json), grouped by named foil.

Reads the full-pool runs from results/raw_*.jsonl (subject_id contains
"all_items"). Ratings are discrete (1..6); each regime is a SMOOTH curve
drawn with shape-preserving (PCHIP) interpolation through the EXACT
proportion at each rating: the curve passes through every true value, stays
at literal 0 across ratings a regime never used (no overshoot, no invented
bumps), and just rounds the corners so it reads like the Bayesian KDE
figures. KDE itself was rejected earlier (it borrowed mass into ratings the
models never chose). Dots mark the six exact values. Fixed 0-1 y-axis.

Run from llm_exp/:
    python3 make_llm_1misc_distributions.py
"""

import glob
import json
import os
from collections import Counter

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, 'plots')
BAYES_MARGINALS = os.path.join(os.path.dirname(HERE), 'analysis-Bayesian',
                               'b_item_marginals.json')

IDS = ['add_before_mul', 'add_before_div', 'sub_before_mul', 'sub_before_div',
       'same_priority_rtl', 'outside_bracket_first']
SHORT = {
    'add_before_mul': 'add<×', 'add_before_div': 'add<÷',
    'sub_before_mul': 'sub<×', 'sub_before_div': 'sub<÷',
    'same_priority_rtl': 'RTL', 'outside_bracket_first': 'outside()',
}
REGIMES = [
    ('haiku (thinking)', 'anthropic/claude-haiku-4.5', 'thinking', '#2a78d6', -0.24),
    ('haiku (direct)', 'anthropic/claude-haiku-4.5', 'direct', '#eda100', 0.0),
    ('gpt-4o (direct)', 'openai/gpt-4o', 'direct', '#4a3aa7', 0.24),
]


def panel(ax, by_regime, title):
    """by_regime: list of (label, vals, color, xoff); exact frequency-polygon
    curves — through the true proportion at each rating, through 0 where a
    rating was never used."""
    nonempty = [(lab, v, c, o) for lab, v, c, o in by_regime if len(v)]
    if not nonempty:
        ax.text(0.5, 0.5, 'no refuted items\n(foil never testable)', ha='center',
                va='center', fontsize=8, color='#898781', transform=ax.transAxes)
        ax.axvline(3.5, ls='--', lw=0.8, color='grey')
        ax.set_xlim(0.7, 6.3); ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.5, 1]); ax.set_xticks(range(1, 7))
        ax.set_title(title, fontsize=10)
        return
    grid = np.linspace(1, 6, 300)
    for _, vals, color, _ in nonempty:
        counts = Counter(vals)
        props = [counts.get(r, 0) / len(vals) for r in range(1, 7)]
        smooth = np.clip(PchipInterpolator(range(1, 7), props)(grid), 0, None)
        ax.plot(grid, smooth, color=color, lw=1.8, zorder=3)
        ax.fill_between(grid, smooth, color=color, alpha=0.12, lw=0)
        ax.scatter(range(1, 7), props, s=13, color=color, zorder=4)
    ax.axvline(3.5, ls='--', lw=0.8, color='grey')
    ax.set_xlim(0.7, 6.3)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticks(range(1, 7))
    ax.set_title(title, fontsize=10)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    bay = json.load(open(BAYES_MARGINALS))
    refuted_ids = {i for i, v in bay['b_marginals'].items() if v < bay['refuted_cut']}

    rows = []
    for path in sorted(glob.glob(os.path.join(HERE, 'results', 'raw_*.jsonl'))):
        for line in open(path):
            r = json.loads(line)
            if 'all_items' in r['subject_id'] and r.get('error') is None \
                    and r.get('response') is not None and r['num_misconceptions'] == 1:
                rows.append(r)

    def cells(regime_model, regime_effort):
        sub = [r for r in rows if r['model'] == regime_model and r['effort'] == regime_effort]
        seen = set()
        a_p = {m: [] for m in IDS}; b_p = {m: [] for m in IDS}
        b_n = {m: [] for m in IDS}; b_r = {m: [] for m in IDS}
        for r in sub:
            if r['id'] in seen:
                continue
            seen.add(r['id'])
            if r['category'] == 'A':
                a_p[r['misconceptions'][0]].append(r['response'])
            else:
                b_p[r['misconceptions'][0]].append(r['response'])
                b_n[r['probed_misconception']].append(r['response'])
                if r['id'] in refuted_ids:
                    b_r[r['probed_misconception']].append(r['response'])
        return a_p, b_p, b_n, b_r

    per_regime = {}
    for label, model, effort, color, xoff in REGIMES:
        per_regime[label] = cells(model, effort)
        a_p = per_regime[label][0]
        print(f"{label}: A items {sum(len(v) for v in a_p.values())}, "
              f"B items {sum(len(v) for v in per_regime[label][1].values())}")

    def collect(idx, m):
        return [(label, per_regime[label][idx][m], color, xoff)
                for label, _, _, color, xoff in REGIMES]

    handles = [plt.Line2D([], [], color=c, lw=2, label=lab)
               for lab, _, _, c, _ in REGIMES]

    # ── figure 1: category A ──
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), sharex=True)
    for ax, m in zip(axes.flat, IDS):
        panel(ax, collect(0, m), SHORT[m])
    for ax in axes[1]:
        ax.set_xlabel('rating (1=SD .. 6=SA)', fontsize=9)
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9, frameon=False)
    fig.suptitle('LLMs — category A (statement matches; agree correct):\n'
                 'distribution of ratings by misconception present '
                 '(10 items per regime per panel; exact proportion at each rating)', fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    p1 = os.path.join(OUTDIR, 'llm_1misc_dist_A.png')
    fig.savefig(p1, dpi=140)
    plt.close(fig)

    # supplementary items (analysis-Bayesian/synthetic_items.json, e.g. the
    # sub<÷ refutable case; slated for the pool when it grows) join each
    # regime's refuted-subset cells as ordinary items, using the ratings
    # collected by run_synthetic_item.py
    extra_path = os.path.join(os.path.dirname(HERE), 'analysis-Bayesian',
                              'synthetic_items.json')
    n_extra = {m: 0 for m in IDS}
    if os.path.exists(extra_path):
        for it in json.load(open(extra_path)):
            m = it['probed_misconception']
            ratings = it.get('llm_ratings') or {}
            if all(ratings.get(label) is not None for label, *_ in REGIMES):
                for label, *_ in REGIMES:
                    per_regime[label][3][m].append(ratings[label])
                n_extra[m] += 1

    # ── figure 2: category B ──
    fig, axes = plt.subplots(3, 6, figsize=(16.5, 9), sharex=True)
    for j, m in enumerate(IDS):
        panel(axes[0, j], collect(1, m), SHORT[m])
        panel(axes[1, j], collect(2, m), SHORT[m])
        n_ref = len(per_regime[REGIMES[0][0]][3][m])
        n_lab = f"n={n_ref} items" if n_extra[m] else f"n={n_ref}/10 items"
        panel(axes[2, j], collect(3, m), f"{SHORT[m]}  ({n_lab})")
        axes[2, j].set_xlabel('rating (1=SD .. 6=SA)', fontsize=8)
    axes[0, 0].set_ylabel('grouped by misconception\nPRESENT in trace', fontsize=9)
    axes[1, 0].set_ylabel('grouped by misconception\nNAMED in statement (foil)', fontsize=9)
    axes[2, 0].set_ylabel('REFUTED items only\n(ideal-obs P < 0.15), by named foil', fontsize=9)
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9, frameon=False)
    fig.suptitle('LLMs — category B (statement is a foil; disagree correct): '
                 'distribution of ratings\ntwo groupings, then the refutable-item subset '
                 '(exact proportion at each rating)', fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    p2 = os.path.join(OUTDIR, 'llm_1misc_dist_B.png')
    fig.savefig(p2, dpi=140)
    plt.close(fig)

    print(f"\nWrote {p1}\nWrote {p2}")


if __name__ == '__main__':
    main()
