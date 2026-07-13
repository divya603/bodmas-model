"""Raw JSONL (from run_pilot) -> tidy DataFrame matching the human per-trial schema.

Each model-run is one "subject". We emit the same per-trial fields the human analysis
uses — response (1..6 Likert), statement_correct, category, misconceptions,
probed_misconception, which_target, num_misconceptions, and `correct` (direction match) —
plus factor columns (model, delivery, effort) and LLM extras (thinking_tokens, latency).
So the same accuracy / spread-plot logic runs on human, Bayesian, and LLM data alike.

Rows whose `response` is null (unparseable / errored) are kept but flagged via `parsed`;
filter them out before scoring.

CLI:
  python -m bodmas_llm.parse_results results/raw_*.jsonl --out results/parsed.parquet --summary
"""

from __future__ import annotations

import argparse
import glob
import json
import sys

import polars as pl

FACTOR_COLS = ["model", "delivery", "effort", "practice"]


def load_raw(paths: list[str]) -> list[dict]:
    rows = []
    for pattern in paths:
        for path in glob.glob(pattern):
            with open(path, encoding="utf-8") as f:
                rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def to_dataframe(rows: list[dict]) -> pl.DataFrame:
    recs = []
    for r in rows:
        response = r.get("response")
        parsed = isinstance(response, int)
        recs.append({
            "subject_id": r["subject_id"],
            "id": r["id"],
            "category": r["category"],
            "expression": r["expression"],
            "misconceptions": r["misconceptions"],
            "num_misconceptions": r["num_misconceptions"],
            "probed_misconception": r["probed_misconception"],
            "which_target": r.get("which_target"),
            "statement_correct": r["statement_correct"],
            "correct_answer": r["correct_answer"],           # 'agree' / 'disagree'
            "response": response,                            # 1..6 Likert
            "responded_agree": (response >= 4) if parsed else None,
            "correct": r.get("correct"),                     # direction match (0/1)
            "model": r["model"],
            "delivery": r["delivery"],
            "effort": r["effort"],
            "practice": r["practice"],
            "repeat": r["repeat"],
            "thinking_tokens": r.get("thinking_tokens"),
            "completion_tokens": r.get("completion_tokens"),
            "latency_ms": r.get("latency_ms"),
            "cached": r.get("cached"),
            "parsed": parsed,
            "error": r.get("error"),
            "warning": r.get("warning"),
        })
    return pl.DataFrame(recs)


def summarize(df: pl.DataFrame) -> pl.DataFrame:
    """Accuracy per factor cell x category — the headline comparison."""
    scored = df.filter(pl.col("parsed"))
    return (
        scored.group_by([*FACTOR_COLS, "category"])
        .agg(pl.col("correct").mean().alias("accuracy"), pl.len().alias("n"))
        .sort([*FACTOR_COLS, "category"])
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="raw JSONL path(s) or globs")
    ap.add_argument("--out", default=None, help="output parquet path")
    ap.add_argument("--summary", action="store_true", help="print accuracy-per-cell summary")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    df = to_dataframe(load_raw(args.paths))
    print(f"{df.height} trial rows; {int(df['parsed'].sum())} parsed; "
          f"{df.filter(~pl.col('parsed')).height} unparseable/errored")

    if args.out:
        df.write_parquet(args.out)
        print(f"-> {args.out}")
    if args.summary:
        with pl.Config(tbl_rows=100):
            print(summarize(df))


if __name__ == "__main__":
    main()
