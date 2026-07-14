#!/usr/bin/env python3
"""
plot_human_rt.py

Reaction-time screening for the human trace-judgment data: are low-d'
participants just waiting out the 3-second answer lock and clicking through?

  (a) per-trial RTs (dots) per participant, sorted by d' — with the 3 s lock
      marked, so click-through profiles (a pile-up just above 3 s) are obvious.
  (b) median RT vs d' — engagement vs performance in one look.

Usage (from repo root):
    python3 analysis_human/plot_human_rt.py
"""

import argparse
import json
import os
from statistics import NormalDist

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

_N = NormalDist()
BLUE, ORANGE = '#2a78d6', '#eb6834'
LOCK_S = 3.0


def z_corrected(k, n):
    p = k / n
    p = 1 / (2 * n) if p == 0 else (1 - 1 / (2 * n) if p == 1 else p)
    return _N.inv_cdf(p)


def load(path):
    data = json.load(open(path))
    out = []
    for r in data:
        d = r['data']
        if d.get('done') is not True or d.get('recruitmentService') != 'prolific':
            continue
        ts = [t for t in d['pageData_exp']['visit_0']['data']
              if isinstance(t, dict) and 'response' in t and 'misconceptions' in t]
        dp = (z_corrected(sum(t['response'] >= 4 for t in ts if t['category'] in 'AC'), 12)
              - z_corrected(sum(t['response'] >= 4 for t in ts if t['category'] in 'BD'), 12))
        out.append(dict(seed=d['seedID'][:8], dprime=dp,
                        rts=np.array([t['rt'] for t in ts], float) / 1000.0))
    out.sort(key=lambda s: s['dprime'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/real-all-main-data.json')
    ap.add_argument('--out', default='analysis_human/plots')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    parts = load(args.data)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── (a) per-trial RTs, participants sorted by d' ──
    ax = axes[0]
    y = np.arange(len(parts))
    for i, s in enumerate(parts):
        ax.scatter(s['rts'], np.full_like(s['rts'], i), s=14, color=BLUE,
                   alpha=0.45, edgecolors='none', zorder=2)
        ax.scatter([np.median(s['rts'])], [i], s=60, color=ORANGE, marker='|',
                   linewidths=2.2, zorder=3)
    ax.axvline(LOCK_S, ls='--', lw=1, color='grey')
    ax.text(LOCK_S * 1.05, len(parts) - 0.4, '3 s answer lock', color='grey',
            fontsize=7, va='top')
    ax.set_xscale('log')
    ax.set_xticks([3, 5, 10, 20, 60, 120, 300])
    ax.set_xticklabels(['3s', '5s', '10s', '20s', '1m', '2m', '5m'])
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s['seed']}  (d'={s['dprime']:+.2f})" for s in parts],
                       fontsize=7, color='#52514e')
    ax.set_xlabel('per-trial RT (log scale) — dots: trials, tick: median')
    ax.set_title("(a) Reaction times per participant, sorted by d'", fontsize=11)

    # ── (b) median RT vs d' ──
    ax = axes[1]
    meds = [np.median(s['rts']) for s in parts]
    dps = [s['dprime'] for s in parts]
    ax.axvline(0, ls='--', lw=1, color='grey')
    ax.scatter(dps, meds, s=70, color=BLUE, edgecolors='white', linewidths=0.9,
               zorder=3)
    for s, m in zip(parts, meds):
        if s['dprime'] < -0.5 or m > 40 or m < 7:
            ax.annotate(s['seed'], (s['dprime'], m), xytext=(6, 4),
                        textcoords='offset points', fontsize=7, color='#52514e')
    rk = lambda v: np.argsort(np.argsort(v))
    rho = np.corrcoef(rk(dps), rk(meds))[0, 1]
    ax.set_xlabel("d'")
    ax.set_ylabel('median RT (s)')
    ax.set_title(f"(b) Median RT vs d'  (Spearman rho = {rho:.2f})", fontsize=11)

    fig.suptitle(f'Human reaction times — {len(parts)} participants, 24 trials each',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(args.out, 'human_rt.png')
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"Wrote {p}")


if __name__ == '__main__':
    main()
