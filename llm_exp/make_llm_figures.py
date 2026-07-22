#!/usr/bin/env python3
"""
make_llm_figures.py

Report figures for the BODMAS LLM experiment on the full 480-item pool
(--all-items runs), mirroring the numberlink results report but re-framed for a
6-point Likert judgment over misconceptions instead of a binary solvability call.

Three regimes: haiku (thinking), haiku (direct), gpt-4o (direct).

Signal-detection framing (agree = rating >= 4 = the "yes" response):
  signal present  = A/C (statement names a truly-present misconception -> agree correct)
  signal absent   = B/D (statement is a foil -> disagree correct)
  hit  = P(agree | A/C)      false alarm = P(agree | B/D)
  d'   = z(hit) - z(FA)      criterion c = -0.5[z(hit)+z(FA)]  (c>0 = disagree-biased)

Figures are written to plots/. Run from llm_exp/:
    python3 make_llm_figures.py                 # all figures built so far
    python3 make_llm_figures.py signal          # just the signal-detection fig
"""

import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

OUTDIR = "plots"
CONFIGS = ["haiku (thinking)", "haiku (direct)", "gpt-4o (direct)"]
COLORS = {"haiku (thinking)": "#2E7D32", "haiku (direct)": "#4C78A8", "gpt-4o (direct)": "#E4820B"}
AGREE_CATS = {"A", "C"}       # statement true -> agree correct (signal present)


# ── data ────────────────────────────────────────────────────────────
def _cfg_label(r):
    m = r["model"].split("/")[-1]
    m = "haiku" if "haiku" in m else ("gpt-4o" if "gpt-4o" in m else m)
    return f"{m} ({r['effort']})"


def load_all_items():
    # filter to the current 480-item pool (the raw logs hold 487: the 7
    # never-sampled ambiguous items were dropped, see base-task/drop_ambiguous.py)
    pool_ids = {it["id"] for it in json.load(open("data/stimulus_pool.json"))}
    rows, seen = [], set()
    for f in glob.glob("results/raw_*.jsonl"):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if not r["subject_id"].endswith("all_items|r0") or r.get("response") is None:
                continue
            if r["id"] not in pool_ids:
                continue
            r["config"] = _cfg_label(r)
            if (r["config"], r["id"]) in seen:
                continue
            seen.add((r["config"], r["id"]))
            r["agree"] = r["response"] >= 4
            rows.append(r)
    return rows


# ── signal detection ────────────────────────────────────────────────
def _z_corrected(k, n):
    """Hit/FA rate with the Macmillan-Creelman 1/(2N) correction, then z-scored."""
    p = k / n
    if p == 0:
        p = 1 / (2 * n)
    elif p == 1:
        p = 1 - 1 / (2 * n)
    return p, norm.ppf(p)


def sdt_by_config(rows):
    out = {}
    for cfg in CONFIGS:
        rc = [r for r in rows if r["config"] == cfg]
        signal = [r for r in rc if r["category"] in AGREE_CATS]   # A/C: should agree
        noise = [r for r in rc if r["category"] not in AGREE_CATS]  # B/D: should disagree
        hit_k = sum(r["agree"] for r in signal)
        fa_k = sum(r["agree"] for r in noise)
        hit, zh = _z_corrected(hit_k, len(signal))
        fa, zf = _z_corrected(fa_k, len(noise))
        acc = np.mean([r["correct"] for r in rc])
        out[cfg] = dict(
            n=len(rc), accuracy=acc,
            acc_agree=np.mean([r["correct"] for r in signal]),
            acc_disagree=np.mean([r["correct"] for r in noise]),
            hit=hit, fa=fa, dprime=zh - zf, criterion=-0.5 * (zh + zf),
        )
    return out


def fig_signal(rows):
    sdt = sdt_by_config(rows)

    # ── table to stdout ──
    print(f"\n{'regime':22s} {'acc':>5s} {'agree':>6s} {'disagree':>9s} {'hit':>6s} {'FA':>6s} {'d-prime':>8s} {'criterion':>10s}")
    for cfg in CONFIGS:
        s = sdt[cfg]
        print(f"{cfg:22s} {s['accuracy']:>5.3f} {s['acc_agree']:>6.3f} {s['acc_disagree']:>9.3f} "
              f"{s['hit']:>6.3f} {s['fa']:>6.3f} {s['dprime']:>8.2f} {s['criterion']:>+10.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) accuracy by signal type, grouped by config
    groups = ["overall", "agree items\n(A/C)", "disagree items\n(B/D)"]
    keys = ["accuracy", "acc_agree", "acc_disagree"]
    x = np.arange(len(groups)); w = 0.26
    for j, cfg in enumerate(CONFIGS):
        vals = [sdt[cfg][k] for k in keys]
        axes[0].bar(x + (j - 1) * w, vals, w, label=cfg, color=COLORS[cfg], alpha=0.9)
    axes[0].axhline(0.5, ls="--", lw=1, color="grey")
    axes[0].set_xticks(x); axes[0].set_xticklabels(groups, fontsize=9)
    axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("accuracy")
    axes[0].set_title("(a) Accuracy by signal type", fontsize=11)
    axes[0].legend(fontsize=8, loc="lower center", ncol=1)

    # (b) ROC / signal-detection space with iso-d' curves
    fa = np.linspace(0.001, 0.999, 200)
    for d in (0.5, 1.0, 1.5, 2.0, 2.5):
        axes[1].plot(fa, norm.cdf(norm.ppf(fa) + d), color="#ccc", lw=0.8, zorder=1)
        axes[1].text(0.02, norm.cdf(norm.ppf(0.02) + d), f"d'={d}", color="#999", fontsize=7, va="bottom")
    axes[1].plot([0, 1], [0, 1], ls="--", color="grey", lw=1, zorder=1)  # chance
    for cfg in CONFIGS:
        s = sdt[cfg]
        axes[1].scatter(s["fa"], s["hit"], s=120, color=COLORS[cfg], label=cfg,
                        edgecolors="white", linewidths=0.8, zorder=3)
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("false-alarm rate  P(agree | B/D)")
    axes[1].set_ylabel("hit rate  P(agree | A/C)")
    axes[1].set_title("(b) Signal detection (iso-d' curves)", fontsize=11)
    axes[1].legend(fontsize=8, loc="lower right")

    fig.suptitle("LLM signal detection (all 480 items)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUTDIR, "llm_signal_detection.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    return p


# ── thinking tokens as a reaction-time analog (haiku thinking only) ──
def fig_tokens(rows):
    from scipy.stats import mannwhitneyu, spearmanr

    rc = [r for r in rows if r["config"] == "haiku (thinking)" and r.get("thinking_tokens")]
    tok = np.array([r["thinking_tokens"] for r in rc], float)
    npar = np.array([r["num_misconceptions"] for r in rc])
    agree_item = np.array([r["category"] in AGREE_CATS for r in rc])   # A/C should-agree
    correct = np.array([r["correct"] for r in rc])
    cats = np.array([r["category"] for r in rc])

    rho, prho = spearmanr(npar, tok)
    # do the agree-items (detect the misconception) cost more than disagree-items?
    u_dir, p_dir = mannwhitneyu(tok[agree_item], tok[~agree_item])
    # do errors cost more than correct answers?
    u_err, p_err = mannwhitneyu(tok[correct == 0], tok[correct == 1]) if (correct == 0).sum() else (np.nan, np.nan)
    print(f"\nhaiku-thinking tokens (n={len(rc)}): "
          f"Spearman(tokens, #misconceptions)={rho:.2f} p={prho:.1e} | "
          f"agree {np.median(tok[agree_item]):.0f} vs disagree {np.median(tok[~agree_item]):.0f} tok (p={p_dir:.1e}) | "
          f"error {np.median(tok[correct==0]):.0f} vs correct {np.median(tok[correct==1]):.0f} tok (p={p_err:.1e})")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.3))

    # (a) tokens by category
    axes[0].boxplot([tok[cats == c] for c in list("ABCD")], tick_labels=list("ABCD"), showfliers=False)
    axes[0].set_title("(a) Tokens by category", fontsize=10)
    axes[0].set_ylabel("thinking tokens"); axes[0].set_xlabel("category")

    # (b) tokens: agree-items (A/C) vs disagree-items (B/D)
    axes[1].boxplot([tok[agree_item], tok[~agree_item]],
                    tick_labels=["agree\n(A/C)", "disagree\n(B/D)"], showfliers=False,
                    patch_artist=True, boxprops=dict(facecolor="#A9D66B"))
    axes[1].set_title(f"(b) Tokens by direction\n(p={p_dir:.0e})", fontsize=10)
    axes[1].set_ylabel("thinking tokens")

    # (c) tokens by outcome, split by direction
    data, labs, cols = [], [], []
    for mask, lab in [(agree_item, "agree"), (~agree_item, "disagree")]:
        for cv, on, col in [(1, "correct", "#54A24B"), (0, "error", "#E4574C")]:
            sel = tok[mask & (correct == cv)]
            if len(sel):
                data.append(sel); labs.append(f"{lab}\n{on}"); cols.append(col)
    bp = axes[2].boxplot(data, tick_labels=labs, showfliers=False, patch_artist=True)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    axes[2].set_title("(c) Tokens by outcome", fontsize=10); axes[2].set_ylabel("thinking tokens")

    # (d) accuracy by token quartile
    q = np.quantile(tok, [0, .25, .5, .75, 1.0])
    qi = np.clip(np.digitize(tok, q[1:-1]), 0, 3)
    ps, los, his, labs2 = [], [], [], []
    for k in range(4):
        m = qi == k; n = m.sum(); c = correct[m].sum(); p = c / n
        # wilson
        z = 1.96; den = 1 + z * z / n; cen = (p + z * z / (2 * n)) / den
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        ps.append(p); los.append(max(0, p - max(0, cen - h))); his.append(max(0, min(1, cen + h) - p))
        labs2.append(f"Q{k+1}\n(~{int(np.median(tok[m]))} tok)")
    axes[3].errorbar(range(4), ps, yerr=[los, his], marker="o", color="#2E7D32", capsize=4)
    axes[3].axhline(0.5, ls="--", lw=1, color="grey")
    axes[3].set_xticks(range(4)); axes[3].set_xticklabels(labs2, fontsize=8)
    axes[3].set_ylim(0, 1.05); axes[3].set_ylabel("accuracy")
    axes[3].set_title("(d) Accuracy by token quartile", fontsize=10)

    fig.suptitle("haiku (thinking): thinking tokens as a reaction-time analog (all 480 items)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(OUTDIR, "llm_thinking_tokens.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    return p


# ── confidence = Likert magnitude |rating - 3.5| ─────────────────────
def fig_confidence(rows):
    from scipy.stats import mannwhitneyu

    def conf(r):
        return abs(r["response"] - 3.5)   # 0.5 (rating 3/4) .. 2.5 (rating 1/6)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) mean confidence by outcome, per config (+ AUC of confidence -> correctness)
    print(f"\nconfidence calibration (|rating-3.5| discriminating correct vs error):")
    x = np.arange(len(CONFIGS)); w = 0.36
    for cv, off, col, lab in [(0, -w / 2, "#E4574C", "error"), (1, w / 2, "#54A24B", "correct")]:
        means = []
        for cfg in CONFIGS:
            rc = [r for r in rows if r["config"] == cfg and r["correct"] == cv]
            means.append(np.mean([conf(r) for r in rc]) if rc else 0)
        axes[0].bar(x + off, means, w, color=col, alpha=0.85, label=lab)
    for j, cfg in enumerate(CONFIGS):
        rc = [r for r in rows if r["config"] == cfg]
        cc = np.array([conf(r) for r in rc]); ok = np.array([r["correct"] for r in rc])
        if ok.sum() and (ok == 0).sum():
            u, _ = mannwhitneyu(cc[ok == 1], cc[ok == 0])
            auc = u / (ok.sum() * (ok == 0).sum())
        else:
            auc = float("nan")
        axes[0].text(j, 0.05, f"AUC\n{auc:.2f}", ha="center", fontsize=8, color="#333")
        print(f"  {cfg:22s} AUC={auc:.2f}")
    axes[0].set_xticks(x); axes[0].set_xticklabels(CONFIGS, fontsize=9)
    axes[0].set_ylabel("mean confidence  |rating - 3.5|")
    axes[0].set_title("(a) Confidence by outcome\n(gap => calibrated)", fontsize=11)
    axes[0].legend(fontsize=8, loc="upper right")

    # (b) partial-match test: among AGREE responses, is C (pair, partial) rated lower than A (single, full)?
    print(f"\npartial-match magnitude (mean rating among agree responses, rating>=4):")
    x2 = np.arange(len(CONFIGS)); w2 = 0.36
    for cat, off, col in [("A", -w2 / 2, "#2E7D32"), ("C", w2 / 2, "#9BD79B")]:
        means, errs = [], []
        for cfg in CONFIGS:
            vals = [r["response"] for r in rows
                    if r["config"] == cfg and r["category"] == cat and r["response"] >= 4]
            means.append(np.mean(vals) if vals else 0)
            errs.append(np.std(vals) / np.sqrt(len(vals)) if vals else 0)
        axes[1].bar(x2 + off, means, w2, yerr=errs, capsize=3, color=col, alpha=0.9,
                    label=f"{cat} ({'single' if cat == 'A' else 'pair, partial'})")
        for cfg, m in zip(CONFIGS, means):
            print(f"  {cfg:22s} {cat}: mean agree-rating {m:.2f}")
    axes[1].set_xticks(x2); axes[1].set_xticklabels(CONFIGS, fontsize=9)
    axes[1].set_ylim(3.5, 6.1); axes[1].set_ylabel("mean rating | agree (4-6)")
    axes[1].set_title("(b) Partial-match: A (full) vs C (partial)\nlower on C => perceives incompleteness", fontsize=11)
    axes[1].legend(fontsize=8, loc="upper right")

    fig.suptitle("LLM confidence (Likert magnitude) (all 480 items)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(OUTDIR, "llm_confidence.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    return p


# ── response style (the "surface cue" analog) ────────────────────────
def fig_response(rows):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) rating distribution 1-6 per config
    x = np.arange(1, 7); w = 0.26
    for j, cfg in enumerate(CONFIGS):
        counts = np.array([sum(1 for r in rows if r["config"] == cfg and r["response"] == v) for v in range(1, 7)])
        axes[0].bar(x + (j - 1) * w, counts / counts.sum(), w, label=cfg, color=COLORS[cfg], alpha=0.9)
    axes[0].axvline(3.5, ls="--", lw=1, color="grey")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["1\nSD", "2\nD", "3\nsD", "4\nsA", "5\nA", "6\nSA"], fontsize=8)
    axes[0].set_ylabel("share of responses")
    axes[0].set_title("(a) Rating distribution\n(direct = extremes; thinking = graded)", fontsize=11)
    axes[0].legend(fontsize=8, loc="upper center")

    # (b) agree-rate per config vs the true agree-rate (50%: half the pool is agree-correct)
    print("\nagree-rate per config (true agree-rate = 0.50):")
    rates = []
    for cfg in CONFIGS:
        rc = [r for r in rows if r["config"] == cfg]
        rate = np.mean([r["response"] >= 4 for r in rc]); rates.append(rate)
        print(f"  {cfg:22s} agrees {rate:.0%} of the time")
    axes[1].bar(range(len(CONFIGS)), rates, color=[COLORS[c] for c in CONFIGS], alpha=0.9)
    axes[1].axhline(0.5, ls="--", lw=1, color="grey")
    axes[1].text(len(CONFIGS) - 0.5, 0.52, "true rate (50%)", ha="right", color="grey", fontsize=8)
    axes[1].set_xticks(range(len(CONFIGS))); axes[1].set_xticklabels(CONFIGS, fontsize=9)
    axes[1].set_ylim(0, 1); axes[1].set_ylabel("P(agree)")
    axes[1].set_title("(b) Overall agree-rate\n(below 50% = disagree-biased)", fontsize=11)

    fig.suptitle("LLM response style (all 480 items)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(OUTDIR, "llm_response_style.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    return p


# ── cost and latency ─────────────────────────────────────────────────
def fig_cost(rows):
    print(f"\n{'regime':22s} {'n':>4s} {'med latency':>11s} {'prompt tok':>11s} {'compl tok':>10s} {'reason tok':>11s}")
    for cfg in CONFIGS:
        rc = [r for r in rows if r["config"] == cfg]
        lat = np.median([r["latency_ms"] for r in rc if r.get("latency_ms")]) / 1000
        pt = sum(r.get("prompt_tokens") or 0 for r in rc)
        ct = sum(r.get("completion_tokens") or 0 for r in rc)
        rt = sum(r.get("thinking_tokens") or 0 for r in rc)
        print(f"{cfg:22s} {len(rc):>4d} {lat:>9.1f}s {pt:>11,d} {ct:>10,d} {rt:>11,d}")
    return None   # table only, no figure


# ── per-misconception detection vs rejection (grouped by PROBED misconception) ──
IDS = ["add_before_mul", "add_before_div", "sub_before_mul", "sub_before_div",
       "same_priority_rtl", "outside_bracket_first"]
SHORT = {"add_before_mul": "add<×", "add_before_div": "add<÷", "sub_before_mul": "sub<×",
         "sub_before_div": "sub<÷", "same_priority_rtl": "RTL", "outside_bracket_first": "outside()"}


def fig_dprime_by_misconception(rows):
    """Per-misconception signal detection, grouped by the misconception NAMED in the
    statement (probed). Detection = P(agree | X named & present)  [A + C-target trials];
    rejection failure (FA) = P(agree | X named & absent) [B/D foil trials]. This asks
    'when asked about X, can the model recognize it / reject it?' and gives a
    per-misconception d' directly comparable to the Bayesian difficulty ranking."""
    stats = {}   # (cfg, X) -> dict
    print(f"\n{'config':20s} {'probed':12s} {'hit':>6s} {'n':>3s} {'FA':>6s} {'n':>3s} {'dprime':>7s}")
    for cfg in CONFIGS:
        for X in IDS:
            sig = [r for r in rows if r["config"] == cfg and r["probed_misconception"] == X
                   and r["statement_correct"]]
            noi = [r for r in rows if r["config"] == cfg and r["probed_misconception"] == X
                   and not r["statement_correct"]]
            hit, zh = _z_corrected(sum(r["agree"] for r in sig), len(sig))
            fa, zf = _z_corrected(sum(r["agree"] for r in noi), len(noi))
            stats[(cfg, X)] = dict(hit=hit, fa=fa, dprime=zh - zf, n_sig=len(sig), n_noi=len(noi))
            print(f"{cfg:20s} {SHORT[X]:12s} {hit:>6.2f} {len(sig):>3d} {fa:>6.2f} {len(noi):>3d} "
                  f"{zh - zf:>7.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) per-misconception d' per config
    x = np.arange(len(IDS)); w = 0.26
    for j, cfg in enumerate(CONFIGS):
        vals = [stats[(cfg, X)]["dprime"] for X in IDS]
        axes[0].bar(x + (j - 1) * w, vals, w, label=cfg, color=COLORS[cfg], alpha=0.9)
    axes[0].axhline(0, color="grey", lw=1)
    axes[0].set_xticks(x); axes[0].set_xticklabels([SHORT[m] for m in IDS], fontsize=9)
    axes[0].set_ylabel("d'  (per probed misconception)")
    axes[0].set_title("(a) Sensitivity by probed misconception", fontsize=11)
    axes[0].legend(fontsize=8, loc="upper right")

    # (b) the components for haiku (thinking): detection vs false alarm per misconception
    best = "haiku (thinking)"
    for off, key, col, lab in [(-0.19, "hit", "#2E7D32", "detection  P(agree | named & present)"),
                               (0.19, "fa", "#C4442A", "false alarm  P(agree | named & absent)")]:
        vals = [stats[(best, X)][key] for X in IDS]
        axes[1].bar(x + off, vals, 0.38, color=col, alpha=0.9, label=lab)
    axes[1].axhline(0.5, ls="--", lw=1, color="grey")
    axes[1].set_xticks(x); axes[1].set_xticklabels([SHORT[m] for m in IDS], fontsize=9)
    axes[1].set_ylim(0, 1.05); axes[1].set_ylabel("P(agree)")
    axes[1].set_title(f"(b) {best}: detection vs false alarm", fontsize=11)
    axes[1].legend(fontsize=8, loc="upper right")

    fig.suptitle("Per-misconception signal detection (grouped by the misconception named in the statement)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(OUTDIR, "llm_dprime_by_misconception.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    return p


# ── foil-confusion structure ─────────────────────────────────────────
def fig_confusion(rows):
    """On foil trials (B/D: the named misconception is absent), which named foil gets
    (wrongly) agreed to, as a function of which misconception is actually present?
    Structured near-miss confusions (e.g. sub<÷ named while sub<× is present) indicate
    partial reasoning; a flat map indicates noise. D items contribute one judgment to
    each of their two present misconceptions."""
    def confusion(cfg_filter, label, ax, ylab=True):
        agree = np.zeros((len(IDS), len(IDS))); count = np.zeros((len(IDS), len(IDS)))
        for r in rows:
            if not cfg_filter(r["config"]) or r["statement_correct"]:
                continue
            yj = IDS.index(r["probed_misconception"])
            for X in r["misconceptions"]:
                xi = IDS.index(X)
                count[xi, yj] += 1
                agree[xi, yj] += r["agree"]
        rate = np.divide(agree, count, out=np.full_like(agree, np.nan), where=count > 0)
        im = ax.imshow(rate, vmin=0, vmax=1, cmap="Reds")
        for i in range(len(IDS)):
            for j in range(len(IDS)):
                if count[i, j] > 0:
                    ax.text(j, i, f"{rate[i, j]:.2f}\n({int(count[i, j])})",
                            ha="center", va="center", fontsize=7,
                            color="white" if rate[i, j] > 0.5 else "#333")
                else:
                    ax.text(j, i, "–", ha="center", va="center", color="#bbb")
        ax.set_xticks(range(len(IDS))); ax.set_xticklabels([SHORT[m] for m in IDS], fontsize=8, rotation=30)
        ax.set_yticks(range(len(IDS))); ax.set_yticklabels([SHORT[m] for m in IDS], fontsize=8)
        ax.set_xlabel("misconception NAMED in statement (absent foil)")
        if ylab:
            ax.set_ylabel("misconception PRESENT in trace")
        ax.set_title(label, fontsize=10)
        return im

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), layout="constrained")
    confusion(lambda c: c == "haiku (thinking)", "(a) haiku (thinking)", axes[0])
    im = confusion(lambda c: c in ("haiku (direct)", "gpt-4o (direct)"),
                   "(b) direct regimes (pooled)", axes[1], ylab=False)
    cbar = fig.colorbar(im, ax=axes, shrink=0.8)
    cbar.set_label("false-alarm rate  P(agree | foil named)", fontsize=9)
    fig.suptitle("Foil confusion: which absent misconception gets wrongly endorsed, given what is present",
                 fontsize=12)
    p = os.path.join(OUTDIR, "llm_foil_confusion.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    return p


# ── early vs late error (category C which_target) ────────────────────
def fig_error_position(rows):
    """Category C only: the statement names one of the two present misconceptions, and
    which_target says whether that misconception's error appears FIRST or LATER in the
    student's work. Does detection depend on where the evidence sits in the trace?"""
    C = [r for r in rows if r["category"] == "C"]
    positions = ["first", "second"]

    print(f"\n{'config':20s} {'position':8s} {'acc':>6s} {'n':>3s} {'mean rating':>12s}")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) accuracy (= agree rate; agree is correct in C) by position per config
    x = np.arange(len(CONFIGS)); w = 0.36
    for k, (pos, col) in enumerate(zip(positions, ["#4C78A8", "#F58518"])):
        ps, los, his = [], [], []
        for cfg in CONFIGS:
            rc = [r for r in C if r["config"] == cfg and r["which_target"] == pos]
            n = len(rc); kk = sum(r["correct"] for r in rc)
            p = kk / n
            z = 1.96; den = 1 + z * z / n; cen = (p + z * z / (2 * n)) / den
            h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
            ps.append(p); los.append(max(0, p - max(0, cen - h))); his.append(max(0, min(1, cen + h) - p))
            mr = np.mean([r["response"] for r in rc])
            print(f"{cfg:20s} {pos:8s} {p:>6.2f} {n:>3d} {mr:>12.2f}")
        axes[0].bar(x + (k - 0.5) * w, ps, w, yerr=[los, his], capsize=3,
                    label=f"error {pos}", color=col, alpha=0.9)
    axes[0].axhline(0.5, ls="--", lw=1, color="grey")
    axes[0].set_xticks(x); axes[0].set_xticklabels(CONFIGS, fontsize=9)
    axes[0].set_ylim(0, 1.1); axes[0].set_ylabel("accuracy (agree rate)")
    axes[0].set_title("(a) Category C accuracy by position of the named error", fontsize=11)
    axes[0].legend(fontsize=8, loc="lower right")

    # (b) mean rating by position per config (graded version of the same question)
    for k, (pos, col) in enumerate(zip(positions, ["#4C78A8", "#F58518"])):
        means, errs = [], []
        for cfg in CONFIGS:
            vals = [r["response"] for r in C if r["config"] == cfg and r["which_target"] == pos]
            means.append(np.mean(vals)); errs.append(np.std(vals) / np.sqrt(len(vals)))
        axes[1].bar(x + (k - 0.5) * w, means, w, yerr=errs, capsize=3,
                    label=f"error {pos}", color=col, alpha=0.9)
    axes[1].axhline(3.5, ls="--", lw=1, color="grey")
    axes[1].set_xticks(x); axes[1].set_xticklabels(CONFIGS, fontsize=9)
    axes[1].set_ylim(1, 6.2); axes[1].set_ylabel("mean rating (1-6)")
    axes[1].set_title("(b) Mean rating by position of the named error", fontsize=11)
    axes[1].legend(fontsize=8, loc="lower right")

    fig.suptitle("Category C: does detection depend on whether the named misconception's error "
                 "appears first or later in the work?", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(OUTDIR, "llm_error_position.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    return p


FIGURES = {"signal": fig_signal, "tokens": fig_tokens, "confidence": fig_confidence,
           "response": fig_response, "cost": fig_cost,
           "dprime": fig_dprime_by_misconception, "confusion": fig_confusion,
           "position": fig_error_position}


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rows = load_all_items()
    which = sys.argv[1:] or list(FIGURES)
    for name in which:
        if name not in FIGURES:
            print(f"unknown figure: {name} (have: {list(FIGURES)})"); continue
        p = FIGURES[name](rows)
        print("wrote", p)


if __name__ == "__main__":
    main()
