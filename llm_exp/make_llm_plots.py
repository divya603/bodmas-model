#!/usr/bin/env python3
"""
make_llm_plots.py

Plots for the LLM experiment on the full 480-item pool (the --all-items runs).
Three configs: haiku (thinking), haiku (direct), gpt-4o (direct).

  1. accuracy overview  — overall accuracy per config + accuracy by category (A/B/C/D)
  2. accuracy by misconception — per-misconception accuracy per config (comparable to
     the Bayesian difficulty ranking), plus alone vs paired for the best config.

Accuracy = binary direction match (rating >= 4 vs statement_correct), same as the human
scoring. Wilson 95% CIs shown. Outputs to llm_exp/plots/.

Run from llm_exp/:
    python3 make_llm_plots.py
"""

import glob
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = "plots"
IDS = ["add_before_mul", "add_before_div", "sub_before_mul", "sub_before_div",
       "same_priority_rtl", "outside_bracket_first"]
SHORT = {"add_before_mul": "add<×", "add_before_div": "add<÷", "sub_before_mul": "sub<×",
         "sub_before_div": "sub<÷", "same_priority_rtl": "RTL", "outside_bracket_first": "outside()"}
CATS = list("ABCD")

# config order + colours
CONFIGS = ["haiku (thinking)", "haiku (direct)", "gpt-4o (direct)"]
COLORS = {"haiku (thinking)": "#2E7D32", "haiku (direct)": "#4C78A8", "gpt-4o (direct)": "#E4820B"}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    c = (p + z * z / (2 * n)) / denom
    h = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, max(0.0, p - max(0.0, c - h)), max(0.0, min(1.0, c + h) - p)  # p, lo_err, hi_err


def load_all_items():
    # filter to the current 480-item pool (raw logs hold 487; 7 ambiguous dropped)
    pool_ids = {it["id"] for it in json.load(open("data/stimulus_pool.json"))}
    rows, seen = [], set()
    for f in glob.glob("results/raw_*.jsonl"):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if not r["subject_id"].endswith("all_items|r0") or r.get("response") is None:
                continue
            if r["id"] not in pool_ids:
                continue
            lab = _cfg_label(r)
            if (lab, r["id"]) in seen:
                continue
            seen.add((lab, r["id"]))
            rows.append({**r, "config": lab})
    return rows


def _cfg_label(r):
    m = r["model"].split("/")[-1]
    m = "haiku" if "haiku" in m else ("gpt-4o" if "gpt-4o" in m else m)
    return f"{m} ({r['effort']})"


def _acc(cells, ax, labels, title, ylabel="accuracy"):
    ps, los, his, ns = [], [], [], []
    for k, n in cells:
        p, lo, hi = wilson(k, n)
        ps.append(p); los.append(lo); his.append(hi); ns.append(n)
    x = np.arange(len(labels))
    colors = [COLORS.get(l, "#888") for l in labels]
    ax.bar(x, ps, yerr=[los, his], capsize=4, color=colors, alpha=0.9)
    ax.axhline(0.5, ls="--", lw=1, color="grey")
    for i, (p, n) in enumerate(zip(ps, ns)):
        ax.text(i, min(p + his[i] + 0.03, 1.04), f"{p:.2f}\n(n={n})", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.15); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=11)


def plot_overview(rows):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # overall per config
    agg = {c: [0, 0] for c in CONFIGS}
    for r in rows:
        a = agg[r["config"]]; a[0] += r["correct"]; a[1] += 1
    _acc([agg[c] for c in CONFIGS], axes[0], CONFIGS, "Overall accuracy on all 480 items")

    # by category, grouped by config
    bycat = {c: {cat: [0, 0] for cat in CATS} for c in CONFIGS}
    for r in rows:
        cell = bycat[r["config"]][r["category"]]; cell[0] += r["correct"]; cell[1] += 1
    x = np.arange(len(CATS)); w = 0.26
    for j, cfg in enumerate(CONFIGS):
        ps, los, his = [], [], []
        for cat in CATS:
            p, lo, hi = wilson(*bycat[cfg][cat]); ps.append(p); los.append(lo); his.append(hi)
        axes[1].bar(x + (j - 1) * w, ps, w, yerr=[los, his], capsize=2,
                    label=cfg, color=COLORS[cfg], alpha=0.9)
    axes[1].axhline(0.5, ls="--", lw=1, color="grey")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["A\nagree", "B\ndisagree", "C\nagree", "D\ndisagree"], fontsize=9)
    axes[1].set_ylim(0, 1.15); axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy by category (correct = agree for A/C, disagree for B/D)", fontsize=11)
    axes[1].legend(fontsize=8, loc="lower center", ncol=3)

    fig.suptitle("LLM accuracy  (all 480 items)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUTDIR, "llm_accuracy_overview.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    return p


def plot_by_misconception(rows):
    # per-misconception accuracy per config (present misconception, both alone & paired)
    agg = {c: {m: [0, 0] for m in IDS} for c in CONFIGS}
    aln = {c: {m: [0, 0] for m in IDS} for c in CONFIGS}
    par = {c: {m: [0, 0] for m in IDS} for c in CONFIGS}
    for r in rows:
        bucket = aln if r["num_misconceptions"] == 1 else par
        for m in r["misconceptions"]:
            agg[r["config"]][m][0] += r["correct"]; agg[r["config"]][m][1] += 1
            bucket[r["config"]][m][0] += r["correct"]; bucket[r["config"]][m][1] += 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # left: per-misconception overall, grouped by config
    x = np.arange(len(IDS)); w = 0.26
    for j, cfg in enumerate(CONFIGS):
        ps, los, his = [], [], []
        for m in IDS:
            p, lo, hi = wilson(*agg[cfg][m]); ps.append(p); los.append(lo); his.append(hi)
        axes[0].bar(x + (j - 1) * w, ps, w, yerr=[los, his], capsize=2,
                    label=cfg, color=COLORS[cfg], alpha=0.9)
    axes[0].axhline(0.5, ls="--", lw=1, color="grey")
    axes[0].set_xticks(x); axes[0].set_xticklabels([SHORT[m] for m in IDS], fontsize=9)
    axes[0].set_ylim(0, 1.15); axes[0].set_ylabel("accuracy")
    axes[0].set_title("Accuracy by misconception (all configs)", fontsize=11)
    axes[0].legend(fontsize=8, loc="lower center", ncol=3)

    # right: alone vs paired for the best config (haiku thinking)
    best = "haiku (thinking)"
    xa = np.arange(len(IDS)); w2 = 0.38
    for off, bucket, lab, col in [(-w2 / 2, aln, "alone (1-misc)", "#4C78A8"),
                                  (w2 / 2, par, "paired (2-misc)", "#F58518")]:
        ps, los, his = [], [], []
        for m in IDS:
            p, lo, hi = wilson(*bucket[best][m]); ps.append(p); los.append(lo); his.append(hi)
        axes[1].bar(xa + off, ps, w2, yerr=[los, his], capsize=2, label=lab, color=col, alpha=0.9)
    axes[1].axhline(0.5, ls="--", lw=1, color="grey")
    axes[1].set_xticks(xa); axes[1].set_xticklabels([SHORT[m] for m in IDS], fontsize=9)
    axes[1].set_ylim(0, 1.15); axes[1].set_ylabel("accuracy")
    axes[1].set_title(f"{best}: alone vs paired", fontsize=11)
    axes[1].legend(fontsize=8, loc="lower center")

    fig.suptitle("LLM accuracy by misconception  (all 480 items)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUTDIR, "llm_accuracy_by_misconception.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    return p


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rows = load_all_items()
    n_cfg = defaultdict(int)
    for r in rows:
        n_cfg[r["config"]] += 1
    print("configs:", dict(n_cfg))
    p1 = plot_overview(rows)
    p2 = plot_by_misconception(rows)
    print("wrote:\n ", p1, "\n ", p2)


if __name__ == "__main__":
    main()
