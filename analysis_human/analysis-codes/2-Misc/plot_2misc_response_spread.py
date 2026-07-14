#!/usr/bin/env python3
"""
plot_2misc_response_spread.py

Two strip plots for the 2-MISCONCEPTION trials where the statement correctly
names one of the two present misconceptions (category C, correct answer =
AGREE), split by whether the statement points to the 1st or the 2nd
misconception of the pair (which_target).

  Graph 1  (C, which_target='first')  : statement points to the 1st misconception
  Graph 2  (C, which_target='second') : statement points to the 2nd misconception

Axes (both):
  x = the misconception the statement points to (the target) — it is present in
      the trace alongside its pair partner.
  y = Likert response, 1 = Strongly Disagree ... 6 = Strongly Agree.
Colour = response value on a fixed dark-green(agree) -> dark-red(disagree) scale.
Correct answer is AGREE in both, so the agree half of the scale is shaded green.

Run from repo root:
    python3 analysis_human/analysis-codes/2-Misc/plot_2misc_response_spread.py
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

DATA = 'data/real-all-main-data.json'
OUTDIR = 'analysis_human/plots/2-misc-trials'
COLOR_BY_CORRECTNESS = False

IDS = ['add_before_mul', 'add_before_div', 'sub_before_mul', 'sub_before_div',
       'same_priority_rtl', 'outside_bracket_first']
SHORT = {'add_before_mul': 'add<×', 'add_before_div': 'add<÷',
         'sub_before_mul': 'sub<×', 'sub_before_div': 'sub<÷',
         'same_priority_rtl': 'RTL', 'outside_bracket_first': 'outside()'}
LIKERT = ['Strongly\nDisagree', 'Disagree', 'Somewhat\nDisagree',
          'Somewhat\nAgree', 'Agree', 'Strongly\nAgree']

CMAP = LinearSegmentedColormap.from_list(
    'darkRG', ['#8B0000', '#E8674F', '#F5D76E', '#A9D66B', '#3FA34D', '#0B5D1E'])
DISC = [CMAP(i / 5) for i in range(6)]


def load_prolific(path):
    data = json.load(open(path))
    recs = [r['data'] for r in data]
    return [d for d in recs if d.get('done') is True and d.get('recruitmentService') == 'prolific']


def trials_of(d):
    exp = d.get('pageData_exp', {}).get('visit_0', {}).get('data', [])
    return [t for t in exp if isinstance(t, dict) and 'response' in t and 'misconceptions' in t]


def collect_C(parts, which):
    """Category-C responses grouped by the target (probed) misconception, for
    trials where the statement points to the given position ('first'/'second')."""
    out = {m: [] for m in IDS}
    for d in parts:
        for t in trials_of(d):
            if t['category'] == 'C' and t.get('which_target') == which:
                out[t['probed_misconception']].append(t['response'])
    return out


def strip_plot(by_misc, title, outpath, rng):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.axhspan(3.5, 6.7, color='#3FA34D', alpha=0.06)   # agree half (correct)
    ax.axhline(3.5, ls='--', lw=1, color='grey', zorder=1)

    for xi, m in enumerate(IDS):
        resp = by_misc[m]
        for r in resp:
            jx = xi + rng.uniform(-0.14, 0.14)
            jy = r
            if COLOR_BY_CORRECTNESS:
                color = '#0B5D1E' if r >= 4 else '#8B0000'
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
    ax.set_xlabel('Misconception the statement points to (present in the trace)', fontsize=11)
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

    first = collect_C(parts, 'first')
    second = collect_C(parts, 'second')

    p1 = strip_plot(
        first, f'2-misconception trials — statement points to the 1st misconception of the pair\n'
               f'(correct answer = AGREE) · {n} participants',
        os.path.join(OUTDIR, '2misc_C_points_to_first.png'), rng)
    p2 = strip_plot(
        second, f'2-misconception trials — statement points to the 2nd misconception of the pair\n'
                f'(correct answer = AGREE) · {n} participants',
        os.path.join(OUTDIR, '2misc_C_points_to_second.png'), rng)

    print(f"{n} participants")
    print("points-to-1st responses per target misconception:", {SHORT[m]: len(first[m]) for m in IDS},
          " total", sum(len(v) for v in first.values()))
    print("points-to-2nd responses per target misconception:", {SHORT[m]: len(second[m]) for m in IDS},
          " total", sum(len(v) for v in second.values()))
    print("Wrote:\n  " + p1 + "\n  " + p2)


if __name__ == '__main__':
    main()
