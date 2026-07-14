#!/usr/bin/env python3
"""
plot_2misc_heatmap.py

Agreement structure on the 2-misconception trials (categories C and D):
what is PRESENT in the trace x what the statement SHOWS, colored by how much
participants agreed (mean 1-6 rating; 3.5 is the agree/disagree boundary).

  (a) Category C: the statement names a misconception that IS present.
      6x6 square: y = the named (probed) misconception, x = the OTHER present
      one; (named, partner) uniquely identifies the item's pair. Correct
      response is agree, so green rows mean the target survives its partner;
      red cells show the partner masking it.
  (b) Category D: the statement names a misconception that is ABSENT.
      6x15 rectangle: y = the named foil, x = the PAIR actually present in the
      trace. Per foil row only the 10 pairs not containing the foil are
      possible; the rest are structurally impossible (gray). Correct response
      is disagree, so red is good and green cells are seductive confusions.

Usage (from repo root):
    python3 analysis_human/plot_2misc_heatmap.py
    python3 analysis_human/plot_2misc_heatmap.py --cohort new   # rebalanced-pool participants only
"""

import argparse
import json
import os
from datetime import datetime, timezone
from itertools import combinations

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

IDS = ['add_before_mul', 'add_before_div', 'sub_before_mul', 'sub_before_div',
       'same_priority_rtl', 'outside_bracket_first']
SHORT = {
    'add_before_mul': 'add<×', 'add_before_div': 'add<÷',
    'sub_before_mul': 'sub<×', 'sub_before_div': 'sub<÷',
    'same_priority_rtl': 'RTL', 'outside_bracket_first': 'outside()',
}
PAIRS = list(combinations(range(6), 2))                      # 15 unordered pairs
PAIR_LABELS = [f"{SHORT[IDS[i]]}\n{SHORT[IDS[j]]}" for i, j in PAIRS]
NEW_POOL_CUTOFF = datetime(2026, 7, 13, tzinfo=timezone.utc).timestamp()

# diverging: disagree (red) .. gray boundary .. agree (green)
CMAP = LinearSegmentedColormap.from_list('divratings',
                                         ['#e34948', '#f0efec', '#008300'])
NORM = TwoSlopeNorm(vmin=1.0, vcenter=3.5, vmax=6.0)
DARK_AT = 1.4          # |value - center| beyond which cell text turns white


def load(path, cohort):
    data = json.load(open(path))
    trials = []
    for r in data:
        d = r['data']
        if d.get('done') is not True or d.get('recruitmentService') != 'prolific':
            continue
        if cohort == 'new' and d['endtime']['_seconds'] < NEW_POOL_CUTOFF:
            continue
        trials += [t for t in d['pageData_exp']['visit_0']['data']
                   if isinstance(t, dict) and 'response' in t
                   and t.get('num_misconceptions') == 2]
    return trials


def cell_stats(trials):
    """C: (named, partner) 6x6.  D: (named foil, present pair) 6x15."""
    acc_c, acc_d = {}, {}
    for t in trials:
        p = IDS.index(t['probed_misconception'])
        pair = tuple(sorted(IDS.index(m) for m in t['misconceptions']))
        if t['category'] == 'C':
            partner = next(m for m in pair if m != p)
            acc_c.setdefault((p, partner), []).append(t['response'])
        else:
            acc_d.setdefault((p, PAIRS.index(pair)), []).append(t['response'])
    mean_c, n_c = np.full((6, 6), np.nan), np.zeros((6, 6), int)
    for (i, j), vals in acc_c.items():
        mean_c[i, j] = np.mean(vals); n_c[i, j] = len(vals)
    mean_d, n_d = np.full((6, 15), np.nan), np.zeros((6, 15), int)
    for (i, k), vals in acc_d.items():
        mean_d[i, k] = np.mean(vals); n_d[i, k] = len(vals)
    return (mean_c, n_c), (mean_d, n_d)


def impossible_c():
    return np.eye(6, dtype=bool)


def impossible_d():
    imp = np.zeros((6, 15), bool)
    for k, (i, j) in enumerate(PAIRS):
        imp[i, k] = imp[j, k] = True
    return imp


def draw(ax, mean, n, impossible, xlabels, title, xlabel, value_fmt='{:.1f}',
         center=3.5, dark_at=DARK_AT, norm=NORM, cmap=CMAP):
    shown = np.ma.masked_where(impossible, np.where(np.isnan(mean), center, mean))
    cm = cmap.copy()
    cm.set_bad('#d8d7d2')
    im = ax.imshow(shown, cmap=cm, norm=norm, aspect='equal')
    rows, cols = mean.shape
    for i in range(rows):
        for j in range(cols):
            if impossible[i, j]:
                continue
            if n[i, j] == 0:
                ax.text(j, i, '·', ha='center', va='center', color='#898781')
                continue
            dark = abs(mean[i, j] - center) > dark_at
            ax.text(j, i - 0.12, value_fmt.format(mean[i, j]), ha='center',
                    va='center', fontsize=8.5, fontweight='bold',
                    color='white' if dark else '#0b0b0b')
            ax.text(j, i + 0.27, f"n={n[i, j]}", ha='center', va='center',
                    fontsize=5.5, color='white' if dark else '#52514e')
    ax.set_xticks(range(cols)); ax.set_yticks(range(rows))
    ax.set_xticklabels(xlabels, fontsize=7.5)
    ax.set_yticklabels([SHORT[m] for m in IDS], fontsize=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel('misconception NAMED in statement', fontsize=9)
    ax.set_title(title, fontsize=10.5)
    return im


def print_tables(c_mats, d_mats):
    mean_c, n_c = c_mats
    print("\ncategory C — named misconception PRESENT (correct = agree); mean rating (n)")
    print(f"{'named \\ partner':>16s} " + " ".join(f"{SHORT[m]:>10s}" for m in IDS))
    for i, p in enumerate(IDS):
        cells = " ".join("         ·" if n_c[i, j] == 0 else
                         f"{mean_c[i, j]:>6.2f}({n_c[i, j]:>2d})" for j in range(6))
        print(f"{SHORT[p]:>16s} {cells}")
    mean_d, n_d = d_mats
    print("\ncategory D — named misconception ABSENT (correct = disagree); mean rating (n) by present pair")
    for i, p in enumerate(IDS):
        cells = [f"{PAIR_LABELS[k].replace(chr(10), '+')}={mean_d[i, k]:.2f}({n_d[i, k]})"
                 for k in range(15) if n_d[i, k] > 0]
        print(f"  {SHORT[p]:>10s}: " + "  ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/real-all-main-data.json')
    ap.add_argument('--out', default='analysis_human/plots')
    ap.add_argument('--cohort', choices=['all', 'new'], default='all',
                    help="'new' = only the 9 rebalanced-pool participants")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    trials = load(args.data, args.cohort)
    n_c = sum(t['category'] == 'C' for t in trials)
    print(f"2-misconception trials: {len(trials)} (C={n_c}, D={len(trials) - n_c}), "
          f"cohort={args.cohort}")
    c_mats, d_mats = cell_stats(trials)
    print_tables(c_mats, d_mats)

    fig, axes = plt.subplots(1, 2, figsize=(17.5, 4.9),
                             gridspec_kw={'width_ratios': [6, 15.4], 'wspace': 0.14})
    draw(axes[0], *c_mats, impossible_c(), [SHORT[m] for m in IDS],
         '(a) Category C: statement names a PRESENT misconception\n'
         'correct = agree (green)',
         'OTHER misconception present in trace')
    im = draw(axes[1], *d_mats, impossible_d(), PAIR_LABELS,
              '(b) Category D: statement names an ABSENT misconception\n'
              'correct = disagree (red)',
              'PAIR of misconceptions present in trace')
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.015,
                        ticks=[1, 2, 3, 3.5, 4, 5, 6])
    cbar.ax.set_yticklabels(['1 (SD)', '2', '3', '3.5', '4', '5', '6 (SA)'], fontsize=8)
    cbar.set_label('mean rating (>3.5 = agree side)', fontsize=9)

    who = 'all 24 participants' if args.cohort == 'all' else '9 rebalanced-pool participants'
    fig.suptitle(f'Two-misconception trials: present × shown × agreement ({who})',
                 fontsize=12.5)
    suffix = '' if args.cohort == 'all' else '_new'
    p = os.path.join(args.out, f'human_2misc_heatmap{suffix}.png')
    fig.savefig(p, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"\nWrote {p}")


if __name__ == '__main__':
    main()
