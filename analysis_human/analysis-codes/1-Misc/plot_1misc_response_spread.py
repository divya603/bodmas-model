#!/usr/bin/env python3
"""
plot_1misc_response_spread.py

Two strip plots for the 1-MISCONCEPTION trials (categories A and B), showing
the spread of every trial's 6-point Likert response across the 6 misconception
types — one dot per trial per participant.

  Graph 1  (category A): statement MATCHES the present misconception
                         -> correct answer is AGREE
  Graph 2  (category B): statement is a FOIL (names an absent misconception)
                         -> correct answer is DISAGREE

Axes (both graphs):
  x = the misconception named in the belief statement (probed); for A this is
      also the one actually present.
  y = Likert response, 1 = Strongly Disagree ... 6 = Strongly Agree.
Colour = the response value on a fixed dark-green(agree) -> dark-red(disagree)
scale, identical in both graphs. (To colour by *correctness* instead, flip the
value->colour mapping for graph 2 only — see COLOR_BY_CORRECTNESS below.)

Run from repo root:
    python3 analysis_human/analysis-codes/1-Misc/plot_1misc_response_spread.py
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── config ──────────────────────────────────────────────────────────
DATA = 'data/real-all-main-data.json'
OUTDIR = 'analysis_human/plots/1-misc-trials'
COLOR_BY_CORRECTNESS = False   # False = colour by response (green=agree); see docstring

IDS = ['add_before_mul', 'add_before_div', 'sub_before_mul', 'sub_before_div',
       'same_priority_rtl', 'outside_bracket_first']
SHORT = {'add_before_mul': 'add<×', 'add_before_div': 'add<÷',
         'sub_before_mul': 'sub<×', 'sub_before_div': 'sub<÷',
         'same_priority_rtl': 'RTL', 'outside_bracket_first': 'outside()'}
LIKERT = ['Strongly\nDisagree', 'Disagree', 'Somewhat\nDisagree',
          'Somewhat\nAgree', 'Agree', 'Strongly\nAgree']

# dark-red (Strongly Disagree=1) -> light -> dark-green (Strongly Agree=6)
CMAP = LinearSegmentedColormap.from_list(
    'darkRG', ['#8B0000', '#E8674F', '#F5D76E', '#A9D66B', '#3FA34D', '#0B5D1E'])
DISC = [CMAP(i / 5) for i in range(6)]   # 6 discrete colours, index 0..5 = response 1..6


# ── data ────────────────────────────────────────────────────────────
def load_prolific(path):
    data = json.load(open(path))
    recs = [r['data'] for r in data]
    return [d for d in recs if d.get('done') is True and d.get('recruitmentService') == 'prolific']


def trials_of(d):
    exp = d.get('pageData_exp', {}).get('visit_0', {}).get('data', [])
    return [t for t in exp if isinstance(t, dict) and 'response' in t and 'misconceptions' in t]


def collect(parts, category):
    """Return {misconception_id: [responses...]} for the given category,
    grouped by the misconception actually PRESENT in the trace (so the x-axis
    reads trace difficulty per misconception). For category A the present
    misconception is also the one named; for B it is the one that generated
    the trace, not the foil in the statement."""
    out = {m: [] for m in IDS}
    for d in parts:
        for t in trials_of(d):
            if t['category'] == category:
                out[t['misconceptions'][0]].append(t['response'])
    return out


# ── plot ────────────────────────────────────────────────────────────
def strip_plot(by_misc, title, correct_side, outpath, rng):
    fig, ax = plt.subplots(figsize=(10, 6))

    # shade the correct half of the scale
    if correct_side == 'agree':
        ax.axhspan(3.5, 6.7, color='#3FA34D', alpha=0.06)
    else:
        ax.axhspan(0.3, 3.5, color='#8B0000', alpha=0.06)
    ax.axhline(3.5, ls='--', lw=1, color='grey', zorder=1)

    for xi, m in enumerate(IDS):
        resp = by_misc[m]
        for r in resp:
            jx = xi + rng.uniform(-0.14, 0.14)   # small horizontal spread only
            jy = r                               # exact Likert height (no vertical jitter)
            if COLOR_BY_CORRECTNESS:
                correct = (r >= 4) if correct_side == 'agree' else (r <= 3)
                color = '#0B5D1E' if correct else '#8B0000'
            else:
                color = DISC[r - 1]
            ax.scatter(jx, jy, s=22, color=color, alpha=0.85,
                       edgecolors='white', linewidths=0.3, zorder=3)
        ax.text(xi, 6.85, f"n={len(resp)}", ha='center', fontsize=8, color='#555')

    ax.set_xlim(-0.6, len(IDS) - 0.4)
    ax.set_ylim(0.4, 7.0)
    ax.set_xticks(range(len(IDS)))
    ax.set_xticklabels([SHORT[m] for m in IDS], fontsize=11)
    ax.set_yticks(range(1, 7))
    ax.set_yticklabels(LIKERT, fontsize=9)
    ax.set_xlabel('Misconception present in the trace', fontsize=11)
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    return outpath


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    parts = load_prolific(DATA)
    rng = np.random.default_rng(0)
    n = len(parts)

    a = collect(parts, 'A')
    b = collect(parts, 'B')

    p1 = strip_plot(
        a, f'1-misconception trials — statement MATCHES the misconception\n'
           f'(correct answer = AGREE) · {n} participants',
        'agree', os.path.join(OUTDIR, '1misc_statement_matches_agree.png'), rng)
    p2 = strip_plot(
        b, f'1-misconception trials — statement is a FOIL\n'
           f'(correct answer = DISAGREE) · {n} participants',
        'disagree', os.path.join(OUTDIR, '1misc_statement_foil_disagree.png'), rng)

    print(f"{n} participants")
    print("Graph 1 (A) responses per misconception:",
          {SHORT[m]: len(a[m]) for m in IDS})
    print("Graph 2 (B) responses per misconception:",
          {SHORT[m]: len(b[m]) for m in IDS})
    print("Wrote:\n  " + p1 + "\n  " + p2)


if __name__ == '__main__':
    main()
