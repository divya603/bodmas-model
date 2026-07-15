#!/usr/bin/env python3
"""
plot_bayes_1misc_distributions.py

Distribution view of the Bayesian ideal observer's per-item posterior
P(named rule | trace) on the 1-misconception items, 10 items per curve.

  bayes_1misc_dist_A.png — category A (statement matches). 6 panels grouped
      by the misconception present (= named; in A they coincide by design,
      so no other grouping exists).
  bayes_1misc_dist_B.png — category B (statement is a foil). Three rows over
      the same 60 items, because present and named are decoupled:
      row 1 groups by the misconception PRESENT in the trace ("does the
      student's actual bug make the work confusable?"), row 2 by the
      misconception NAMED in the statement ("are some claims inherently
      easier to rule out?"), row 3 shows ONLY the refuted subset (marginal
      < 0.15: the trace contained a decision point for the named rule and
      the student visibly acted against it), grouped by named foil. Refuted
      items should pile up near 0; foils that never get an opportunity
      (sub<÷) have an empty panel.

Curves are Gaussian KDEs with reflection at the [0,1] boundaries; the dots
underneath are the actual 10 item values (the curve never claims more than
they do).

Run from repo root:
    python3 analysis-Bayesian/plot_bayes_1misc_distributions.py
"""

import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_TASK = os.path.join(os.path.dirname(HERE), 'base-task')
sys.path.insert(0, BASE_TASK)

from inference import posterior_over_profiles, marginal_rule_probability  # noqa: E402

IDS = ['add_before_mul', 'add_before_div', 'sub_before_mul', 'sub_before_div',
       'same_priority_rtl', 'outside_bracket_first']
SHORT = {
    'add_before_mul': 'add<×', 'add_before_div': 'add<÷',
    'sub_before_mul': 'sub<×', 'sub_before_div': 'sub<÷',
    'same_priority_rtl': 'RTL', 'outside_bracket_first': 'outside()',
}
BLUE, ORANGE = '#2a78d6', '#eb6834'
GRID = np.linspace(0.0, 1.0, 400)


REFUTED_CUT = 0.15


def kde_bounded(vals, grid=GRID):
    """Gaussian KDE with reflection at 0 and 1 (marginals are bounded)."""
    vals = np.asarray(vals, float)
    n = len(vals)
    sd = vals.std(ddof=1) if n > 1 else 0.0
    bw = max(1.06 * sd * n ** (-1 / 5), 0.015)
    dens = np.zeros_like(grid)
    for v in vals:
        for center in (v, -v, 2 - v):          # data + reflections
            dens += np.exp(-0.5 * ((grid - center) / bw) ** 2)
    return dens / (n * bw * np.sqrt(2 * np.pi))


def panel(ax, vals, color, title):
    if len(vals) == 0:
        ax.text(0.5, 0.5, 'no refuted items\n(foil never testable)', ha='center',
                va='center', fontsize=8, color='#898781', transform=ax.transAxes)
        ax.axvline(0.5, ls='--', lw=0.8, color='grey')
        ax.set_xlim(0, 1); ax.set_yticks([]); ax.set_xticks([0, 0.5, 1])
        ax.set_title(title, fontsize=10)
        return
    dens = kde_bounded(vals)
    ax.fill_between(GRID, dens, color=color, alpha=0.25, lw=0)
    ax.plot(GRID, dens, color=color, lw=1.6)
    ymax = dens.max()
    ax.scatter(vals, np.full(len(vals), -0.055 * ymax), s=16, color=color,
               marker='|', linewidths=1.2, clip_on=False)
    ax.axvline(0.5, ls='--', lw=0.8, color='grey')
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.09 * ymax, 1.12 * ymax)
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1])
    ax.set_title(title, fontsize=10)


def main():
    pool = json.load(open(os.path.join(BASE_TASK, 'stimulus_pool.json')))
    items = [it for it in pool if it['num_misconceptions'] == 1]

    a_by_present = {m: [] for m in IDS}
    b_by_present = {m: [] for m in IDS}
    b_by_named = {m: [] for m in IDS}
    b_marginals = {}
    for it in items:
        post = posterior_over_profiles(it['trace'])
        marg = marginal_rule_probability(post, it['probed_misconception'])
        if it['category'] == 'A':
            a_by_present[it['misconceptions'][0]].append(marg)
        else:
            b_by_present[it['misconceptions'][0]].append(marg)
            b_by_named[it['probed_misconception']].append(marg)
            b_marginals[it['id']] = marg

    # per-item B marginals for the human/LLM scripts (same refuted-item split)
    with open(os.path.join(HERE, 'b_item_marginals.json'), 'w') as f:
        json.dump({'refuted_cut': REFUTED_CUT, 'b_marginals': b_marginals}, f, indent=1)

    # ── figure 1: category A ──
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for ax, m in zip(axes.flat, IDS):
        panel(ax, a_by_present[m], BLUE, SHORT[m])
    for ax in axes[1]:
        ax.set_xlabel('P(named rule | trace)', fontsize=9)
    fig.suptitle('Bayesian ideal observer — category A (statement matches):\n'
                 'distribution of the posterior on the named rule, by misconception '
                 '(10 items per curve; ticks = items)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p1 = os.path.join(HERE, 'bayes_1misc_dist_A.png')
    fig.savefig(p1, dpi=140)
    plt.close(fig)

    # ── figure 2: category B — two groupings + refuted-only row ──
    b_refuted = {m: [v for v in b_by_named[m] if v < REFUTED_CUT] for m in IDS}
    print("refuted items per named foil (P < %.2f): " % REFUTED_CUT +
          "  ".join(f"{SHORT[m]}={len(b_refuted[m])}" for m in IDS))

    # supplementary items (analysis-Bayesian/synthetic_items.json, e.g. the
    # sub<÷ refutable case; slated for the pool when it grows) join the
    # refuted-subset row as ordinary items
    n_extra = {m: 0 for m in IDS}
    extra_path = os.path.join(HERE, 'synthetic_items.json')
    if os.path.exists(extra_path):
        for it in json.load(open(extra_path)):
            post = posterior_over_profiles(it['trace'])
            marg = marginal_rule_probability(post, it['probed_misconception'])
            print(f"supplementary item {it['id']} (foil {SHORT[it['probed_misconception']]}): "
                  f"marginal {marg:.4f}")
            if marg < REFUTED_CUT:
                b_refuted[it['probed_misconception']].append(marg)
                n_extra[it['probed_misconception']] += 1

    fig, axes = plt.subplots(3, 6, figsize=(16.5, 8.6), sharex=True)
    for j, m in enumerate(IDS):
        panel(axes[0, j], b_by_present[m], ORANGE, SHORT[m])
        panel(axes[1, j], b_by_named[m], ORANGE, SHORT[m])
        n_lab = (f"n={len(b_refuted[m])}" if n_extra[m]
                 else f"n={len(b_refuted[m])}/10")
        panel(axes[2, j], b_refuted[m], ORANGE, f"{SHORT[m]}  ({n_lab})")
        axes[2, j].set_xlabel('P(named rule | trace)', fontsize=8)
    axes[0, 0].set_ylabel('grouped by misconception\nPRESENT in trace', fontsize=9)
    axes[1, 0].set_ylabel('grouped by misconception\nNAMED in statement (foil)', fontsize=9)
    axes[2, 0].set_ylabel(f'REFUTED subset only\n(P < {REFUTED_CUT}), by named foil', fontsize=9)
    fig.suptitle('Bayesian ideal observer — category B (statement is a foil): '
                 'distribution of the posterior on the named rule\n'
                 'same 60 items: two groupings, then the refuted subset '
                 '(10 per curve in rows 1–2; ticks = items)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p2 = os.path.join(HERE, 'bayes_1misc_dist_B.png')
    fig.savefig(p2, dpi=140)
    plt.close(fig)

    print(f"Wrote {p1}\nWrote {p2}")


if __name__ == '__main__':
    main()
