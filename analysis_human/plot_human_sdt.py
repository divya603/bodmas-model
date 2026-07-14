#!/usr/bin/env python3
"""
plot_human_sdt.py

Per-participant signal-detection analysis of the human trace-judgment data,
mirroring the LLM figure (llm_exp/make_llm_figures.py fig_signal):

  (a) per-participant d' distribution (sorted dot plot, pooled d' marked)
  (b) ROC space (FA vs hit) with iso-d' curves — one dot per participant,
      plus the pooled-over-all-trials point.

Signal = A/C (agree correct), noise = B/D (disagree correct); agree = response
>= 4. Hit/FA rates use the Macmillan-Creelman 1/(2N) correction before
z-scoring, exactly as in the LLM analysis, so d' values are comparable.

Usage (from repo root):
    python3 analysis_human/plot_human_sdt.py
"""

import argparse
import json
import os
from datetime import datetime, timezone
from statistics import NormalDist

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

_N = NormalDist()
BLUE, ORANGE = '#2a78d6', '#eb6834'   # validated categorical pair
NEW_POOL_CUTOFF = datetime(2026, 7, 13, tzinfo=timezone.utc).timestamp()


def z_corrected(k, n):
    """Rate with the Macmillan-Creelman 1/(2N) correction, then z-scored."""
    p = k / n
    if p == 0:
        p = 1 / (2 * n)
    elif p == 1:
        p = 1 - 1 / (2 * n)
    return p, _N.inv_cdf(p)


def sdt(trials):
    sig = [t for t in trials if t['category'] in 'AC']
    noi = [t for t in trials if t['category'] in 'BD']
    hit, zh = z_corrected(sum(t['response'] >= 4 for t in sig), len(sig))
    fa, zf = z_corrected(sum(t['response'] >= 4 for t in noi), len(noi))
    acc = np.mean([(t['response'] >= 4) == (t['statement_correct'] is True)
                   for t in trials])
    return dict(hit=hit, fa=fa, dprime=zh - zf, criterion=-0.5 * (zh + zf),
                accuracy=acc, n=len(trials))


def load(path):
    data = json.load(open(path))
    parts = [r['data'] for r in data
             if r['data'].get('done') is True
             and r['data'].get('recruitmentService') == 'prolific']
    out = []
    for d in parts:
        ts = [t for t in d['pageData_exp']['visit_0']['data']
              if isinstance(t, dict) and 'response' in t and 'misconceptions' in t]
        s = sdt(ts)
        s['seed'] = d['seedID'][:8]
        s['cohort'] = 'new' if d['endtime']['_seconds'] >= NEW_POOL_CUTOFF else 'old'
        s['trials'] = ts
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/real-all-main-data.json')
    ap.add_argument('--out', default='analysis_human/plots')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    parts = load(args.data)
    parts.sort(key=lambda s: s['dprime'])
    pooled = sdt([t for s in parts for t in s['trials']])

    print(f"{'seed':10s} {'cohort':7s} {'acc':>5s} {'hit':>6s} {'FA':>6s} {'dprime':>7s} {'crit':>6s}")
    for s in parts:
        print(f"{s['seed']:10s} {s['cohort']:7s} {s['accuracy']:>5.2f} "
              f"{s['hit']:>6.2f} {s['fa']:>6.2f} {s['dprime']:>7.2f} {s['criterion']:>+6.2f}")
    dps = [s['dprime'] for s in parts]
    print(f"\npooled: hit={pooled['hit']:.2f} FA={pooled['fa']:.2f} "
          f"d'={pooled['dprime']:.2f} c={pooled['criterion']:+.2f}")
    print(f"individual d': mean={np.mean(dps):.2f}  median={np.median(dps):.2f}  "
          f"range=[{min(dps):.2f}, {max(dps):.2f}]  above 0: {sum(d > 0 for d in dps)}/{len(dps)}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── (a) sorted per-participant d' ──
    y = np.arange(len(parts))
    axes[0].axvline(0, ls='--', lw=1, color='grey')
    axes[0].axvline(pooled['dprime'], lw=1.4, color=ORANGE)
    axes[0].text(pooled['dprime'] + 0.05, len(parts) - 0.4,
                 f"pooled d'={pooled['dprime']:.2f}", color=ORANGE, fontsize=8, va='top')
    axes[0].hlines(y, 0, dps, color='#e1e0d9', lw=1, zorder=1)
    axes[0].scatter(dps, y, s=42, color=BLUE, zorder=3)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([s['seed'] for s in parts], fontsize=7, color='#52514e')
    axes[0].set_xlabel("d'")
    axes[0].set_title("(a) Per-participant d' (sorted)", fontsize=11)
    axes[0].text(0.02, -2.1, 'chance', color='grey', fontsize=7, ha='left')

    # ── (b) ROC space with iso-d' curves ──
    ax = axes[1]
    fa_grid = np.linspace(0.001, 0.999, 200)
    zs = np.array([_N.inv_cdf(v) for v in fa_grid])
    for d in (-1.0, 1.0, 2.0, 3.0):
        ax.plot(fa_grid, [_N.cdf(zv + d) for zv in zs], color='#ccc', lw=0.8, zorder=1)
    ax.plot([0, 1], [0, 1], ls='--', color='grey', lw=1, zorder=1)
    for d, (lx, va) in [(-1.0, (0.30, 'top')), (1.0, (0.02, 'bottom')),
                        (2.0, (0.02, 'bottom')), (3.0, (0.02, 'bottom'))]:
        ax.text(lx, _N.cdf(_N.inv_cdf(lx) + d), f"d'={d:g}", color='#999',
                fontsize=7, va=va)

    # deterministic micro-offset so identical (FA, hit) pairs stay visible
    seen = {}
    for s in parts:
        key = (round(s['fa'], 3), round(s['hit'], 3))
        k = seen.get(key, 0)
        seen[key] = k + 1
        off = 0.010 * k
        ax.scatter(s['fa'] + off, s['hit'] + off, s=70, color=BLUE, alpha=0.9,
                   edgecolors='white', linewidths=0.9, zorder=3,
                   label='participant' if s is parts[0] else None)
    ax.scatter(pooled['fa'], pooled['hit'], s=170, color=ORANGE, marker='D',
               edgecolors='white', linewidths=1.2, zorder=4, label='all trials pooled')
    ax.annotate(f"pooled\nd'={pooled['dprime']:.2f}", (pooled['fa'], pooled['hit']),
                xytext=(10, -18), textcoords='offset points', fontsize=8, color=ORANGE)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('false-alarm rate  P(agree | B/D)')
    ax.set_ylabel('hit rate  P(agree | A/C)')
    ax.set_title("(b) Signal detection (iso-d' curves)", fontsize=11)
    ax.legend(fontsize=8, loc='lower right')

    fig.suptitle(f'Human signal detection — {len(parts)} participants '
                 f'(12 signal / 12 noise trials each)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(args.out, 'human_signal_detection.png')
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"\nWrote {p}")


if __name__ == '__main__':
    main()
