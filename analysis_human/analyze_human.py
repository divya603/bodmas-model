#!/usr/bin/env python3
"""
analyze_human.py

Sanity-check + first-pass analysis of the human trace-judgment data pulled
from the Smile study (`data/real-all-main-data.json`).

Two jobs:
  1. SANITY — confirm the data is actually all there: how many completed
     participants, whether each has 24 answered trials, no missing responses,
     a bonus block, demographics, feedback, and mouse data; plus category /
     misconception coverage.
  2. PLOTS — accuracy grouped by 1-misconception vs 2-misconception, and by
     misconception type (alone vs paired), with Wilson 95% CIs so the small
     pilot n is shown honestly. Built to line up with the Bayesian
     `misconception_difficulty.py` table for the eventual 3-way comparison.

Accuracy is recomputed from raw Likert responses against the answer key
(statement_correct: response >= 4 counts as "agree"), not trusted from the
client-stored is_correct.

Usage (from repo root):
    python3 analysis_human/analyze_human.py
    python3 analysis_human/analyze_human.py --include-web      # also include pre-fix 'web' completes
    python3 analysis_human/analyze_human.py --data <path> --out <dir>
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

IDS = ['add_before_mul', 'add_before_div', 'sub_before_mul', 'sub_before_div',
       'same_priority_rtl', 'outside_bracket_first']
SHORT = {
    'add_before_mul': 'add<×', 'add_before_div': 'add<÷',
    'sub_before_mul': 'sub<×', 'sub_before_div': 'sub<÷',
    'same_priority_rtl': 'RTL', 'outside_bracket_first': 'outside()',
}
CATS = list('ABCD')


# ── helpers ─────────────────────────────────────────────────────────

def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion. Returns (p, lo, hi)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def trials_of(d):
    exp = d.get('pageData_exp', {}).get('visit_0', {}).get('data', [])
    return [t for t in exp if isinstance(t, dict) and 'response' in t and 'misconceptions' in t]


def is_correct(t):
    return (t['response'] >= 4) == (t['statement_correct'] is True)


def load(path, include_web):
    data = json.load(open(path))
    recs = [r['data'] for r in data]
    done = [d for d in recs if d.get('done') is True]
    keep = {'prolific'} | ({'web'} if include_web else set())
    parts = [d for d in done if d.get('recruitmentService') in keep]
    return recs, done, parts


# ── sanity ──────────────────────────────────────────────────────────

def sanity(recs, done, parts):
    print("=" * 78)
    print("SANITY CHECK")
    print("=" * 78)
    svc_done = defaultdict(int)
    for d in done:
        svc_done[d.get('recruitmentService')] += 1
    print(f"records in file: {len(recs)}   completed: {len(done)}   "
          f"({dict(svc_done)})")
    print(f"participants included in analysis: {len(parts)}\n")

    hdr = f"{'seed':10s} {'svc':9s} {'nTrials':>7s} {'missing':>7s} {'bonus':>6s} {'demo':>5s} {'feedbk':>6s} {'mousePts':>8s} {'quizTry':>7s}"
    print(hdr)
    for d in parts:
        ts = trials_of(d)
        missing = sum(1 for t in ts if t.get('response') is None)
        exp = d.get('pageData_exp', {}).get('visit_0', {}).get('data', [])
        has_bonus = any(isinstance(x, dict) and x.get('phase') == 'traceJudgmentBonus' for x in exp)
        demo = d.get('pageData_demograph', {}).get('visit_0', {}).get('data', [])
        fb = d.get('pageData_feedback', {}).get('visit_0', {}).get('data', [])
        mouse_pts = sum(len(t.get('mouse', [])) for t in ts)
        quiz = d.get('pageData_quiz', {}).get('visit_0', {}).get('data', [])
        quiz_try = quiz[0].get('persist', {}).get('attempts', '?') if quiz else '?'
        print(f"{d.get('seedID','?')[:8]:10s} {str(d.get('recruitmentService')):9s} "
              f"{len(ts):>7d} {missing:>7d} {'yes' if has_bonus else 'NO':>6s} "
              f"{'yes' if demo else 'NO':>5s} {'yes' if fb else 'NO':>6s} "
              f"{mouse_pts:>8d} {str(quiz_try):>7s}")

    # coverage
    cat_cov = defaultdict(int)
    misc_cov = defaultdict(int)
    for d in parts:
        for t in trials_of(d):
            cat_cov[t['category']] += 1
            for m in t['misconceptions']:
                misc_cov[m] += 1
    print(f"\ncategory coverage (trials): " + "  ".join(f"{c}={cat_cov[c]}" for c in CATS))
    print(f"misconception coverage (present-instances): " +
          "  ".join(f"{SHORT[m]}={misc_cov[m]}" for m in IDS))
    print()


# ── metrics ─────────────────────────────────────────────────────────

def compute(parts):
    overall = {'alone': [0, 0], 'paired': [0, 0]}
    by_cat = {c: [0, 0] for c in CATS}
    by_misc = {m: {'alone': [0, 0], 'paired': [0, 0]} for m in IDS}
    likert = defaultdict(int)
    per_part = {}

    for d in parts:
        pc = [0, 0]
        for t in trials_of(d):
            c = is_correct(t)
            grp = 'alone' if t['num_misconceptions'] == 1 else 'paired'
            overall[grp][0] += c; overall[grp][1] += 1
            by_cat[t['category']][0] += c; by_cat[t['category']][1] += 1
            likert[t['response']] += 1
            pc[0] += c; pc[1] += 1
            for m in t['misconceptions']:
                by_misc[m][grp][0] += c; by_misc[m][grp][1] += 1
        per_part[d.get('seedID', '?')[:8]] = pc
    return overall, by_cat, by_misc, likert, per_part


# ── plots ───────────────────────────────────────────────────────────

def _acc_bar(ax, labels, cells, title, ylabel='accuracy'):
    """cells: list of [correct, total]; draws bars with Wilson CIs + n labels."""
    ps, los, his, ns = [], [], [], []
    for k, n in cells:
        p, lo, hi = wilson(k, n)
        ps.append(p); los.append(max(0.0, p - lo)); his.append(max(0.0, hi - p)); ns.append(n)
    x = np.arange(len(labels))
    ax.bar(x, ps, yerr=[los, his], capsize=4, color='#4C78A8', alpha=0.85)
    ax.axhline(0.5, ls='--', lw=1, color='grey')
    ax.text(len(labels) - 0.5, 0.51, 'chance', color='grey', fontsize=7, ha='right')
    for i, (p, n) in enumerate(zip(ps, ns)):
        ax.text(i, min(p + his[i] + 0.03, 1.02), f"n={n}", ha='center', fontsize=7, color='#555')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=11)


def plot_sanity(parts, likert, per_part, by_cat, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    # trials per participant
    names = list(per_part.keys())
    counts = [per_part[k][1] for k in names]
    axes[0, 0].bar(names, counts, color='#54A24B')
    axes[0, 0].axhline(24, ls='--', color='grey', lw=1)
    axes[0, 0].set_title('Trials answered per participant (target = 24)', fontsize=11)
    axes[0, 0].set_ylabel('n trials'); axes[0, 0].tick_params(axis='x', rotation=30)

    # response distribution
    vals = [likert.get(i, 0) for i in range(1, 7)]
    axes[0, 1].bar(range(1, 7), vals, color='#E45756')
    axes[0, 1].set_xticks(range(1, 7))
    axes[0, 1].set_xticklabels(['1\nSD', '2\nD', '3\nsD', '4\nsA', '5\nA', '6\nSA'], fontsize=8)
    axes[0, 1].set_title('Likert response distribution (all trials)', fontsize=11)
    axes[0, 1].set_ylabel('count')

    # trials per category
    axes[1, 0].bar(CATS, [by_cat[c][1] for c in CATS], color='#72B7B2')
    axes[1, 0].set_title('Trials per category', fontsize=11); axes[1, 0].set_ylabel('n trials')

    # per-participant accuracy
    accs = [per_part[k][0] / per_part[k][1] if per_part[k][1] else 0 for k in names]
    axes[1, 1].bar(names, accs, color='#B279A2')
    axes[1, 1].axhline(0.5, ls='--', color='grey', lw=1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title('Overall accuracy per participant', fontsize=11)
    axes[1, 1].set_ylabel('accuracy'); axes[1, 1].tick_params(axis='x', rotation=30)

    fig.suptitle(f'Data sanity dashboard — {len(parts)} participant(s)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(outdir, 'sanity.png')
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def plot_accuracy(overall, by_cat, by_misc, n_parts, outdir):
    # 1-misc vs 2-misc + by category
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    _acc_bar(axes[0], ['1 misconception\n(alone)', '2 misconceptions\n(paired)'],
             [overall['alone'], overall['paired']], 'Accuracy: 1-misc vs 2-misc')
    _acc_bar(axes[1], CATS, [by_cat[c] for c in CATS],
             'Accuracy by category\n(A/C: agree correct · B/D: disagree correct)')
    fig.suptitle(f'Human accuracy overview — {n_parts} participant(s)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p1 = os.path.join(outdir, 'accuracy_group_category.png')
    fig.savefig(p1, dpi=130); plt.close(fig)

    # by misconception: alone vs paired grouped bars
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(IDS)); w = 0.38
    for off, grp, color in [(-w / 2, 'alone', '#4C78A8'), (w / 2, 'paired', '#F58518')]:
        ps, los, his, ns = [], [], [], []
        for m in IDS:
            k, n = by_misc[m][grp]
            p, lo, hi = wilson(k, n)
            ps.append(p); los.append(max(0.0, p - lo)); his.append(max(0.0, hi - p)); ns.append(n)
        ax.bar(x + off, ps, w, yerr=[los, his], capsize=3, label=f'{grp}', color=color, alpha=0.85)
        for i, (p, n) in enumerate(zip(ps, ns)):
            ax.text(x[i] + off, min(p + his[i] + 0.02, 1.03), f"{n}", ha='center', fontsize=6, color='#555')
    ax.axhline(0.5, ls='--', lw=1, color='grey')
    ax.set_xticks(x); ax.set_xticklabels([SHORT[m] for m in IDS])
    ax.set_ylim(0, 1.15); ax.set_ylabel('accuracy')
    ax.set_title(f'Accuracy by misconception (alone vs paired) — {n_parts} participant(s)', fontsize=12)
    ax.legend(title='condition', loc='lower right')
    fig.tight_layout()
    p2 = os.path.join(outdir, 'accuracy_by_misconception.png')
    fig.savefig(p2, dpi=130); plt.close(fig)
    return p1, p2


# ── main ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/real-all-main-data.json')
    ap.add_argument('--out', default='analysis_human/plots')
    ap.add_argument('--include-web', action='store_true',
                    help="also include pre-URL-fix 'web' completions (likely your own testing)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    recs, done, parts = load(args.data, args.include_web)

    sanity(recs, done, parts)
    if not parts:
        print("No participants to analyze.")
        return

    overall, by_cat, by_misc, likert, per_part = compute(parts)

    print("=" * 78)
    print("ACCURACY SUMMARY")
    print("=" * 78)
    for g in ('alone', 'paired'):
        k, n = overall[g]
        print(f"  {g:7s}: {k}/{n} = {k / n:.2f}" if n else f"  {g}: n/a")
    print("  by category: " + "  ".join(f"{c}={by_cat[c][0]}/{by_cat[c][1]}" for c in CATS))
    print()

    sp = plot_sanity(parts, likert, per_part, by_cat, args.out)
    p1, p2 = plot_accuracy(overall, by_cat, by_misc, len(parts), args.out)
    print("Wrote plots:")
    for p in (sp, p1, p2):
        print(f"  {p}")


if __name__ == '__main__':
    main()
