"""Pilot driver: sweep the factorial and write raw per-trial rows to JSONL.

Factors (BODMAS):
  delivery  {independent}          — append is implemented but off by default (numberlink
                                      found independent won everywhere); add "append" to
                                      FACTORS to run the memory contrast.
  effort    {direct, thinking}     — the treatment of interest (biggest effect in pilot).
over `--subjects` seeded synthetic subjects (each a 24-trial mirror of the human draw),
with `--k` repeats per cell. One row per (cell x subject x trial x repeat).

Usage:
  python -m bodmas_llm.run_pilot --model "anthropic/claude-haiku-4.5" --subjects 5 --k 1
  python -m bodmas_llm.run_pilot --model M --only effort=thinking

Caching makes reruns free, so this is safe to re-invoke. Output: results/raw_<tag>.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

from . import RESULTS_DIR, STIMULUS_POOL_PATH
from .client import OpenRouterClient
from .sample_session import load_pool, sample_session
from .session import run_subject

FACTORS = {
    "delivery": ["independent"],          # add "append" to run the memory contrast
    "effort": ["direct", "thinking"],
    "practice": ["none", "examples"],     # 3 worked examples precede the trials, or not
}


def _parse_filter(only: str | None) -> dict[str, set[str]]:
    if not only:
        return {}
    out: dict[str, set[str]] = {}
    for clause in only.split(","):
        k, v = clause.split("=")
        out.setdefault(k.strip(), set()).add(v.strip())
    return out


def iter_cells(filt: dict[str, set[str]]):
    keys = list(FACTORS)
    for combo in itertools.product(*(FACTORS[k] for k in keys)):
        cell = dict(zip(keys, combo))
        if all(cell[k] in vals for k, vals in filt.items()):
            yield cell


def subject_id(model, cell, seed) -> str:
    tag = "all_items" if seed == "all" else f"seed{seed}"
    return f"{model}|{cell['delivery']}|{cell['effort']}|{cell['practice']}|{tag}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="OpenRouter model id, e.g. anthropic/claude-haiku-4.5")
    ap.add_argument("--subjects", type=int, default=5, help="number of synthetic subjects (RNG seeds)")
    ap.add_argument("--k", type=int, default=1, help="repeats per (cell, subject, trial)")
    ap.add_argument("--seed-base", type=int, default=1000, help="first subject seed")
    ap.add_argument("--only", default=None, help='filter cells, e.g. "effort=thinking"')
    ap.add_argument("--all-items", action="store_true",
                    help="run every pool item once (full 240-item coverage) as a single "
                         "'subject', instead of sampled subjects; use with --only to pick one cell")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--max-tokens-direct", type=int, default=8192,
                    help="output token cap for direct cells (reasoning-mandatory models spend "
                         "their minimum reasoning here too, so leave room for the JSON answer)")
    ap.add_argument("--max-tokens-thinking", type=int, default=16000,
                    help="output token cap for thinking cells (must fit CoT + the JSON answer)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0, help="model sampling seed (if honored)")
    ap.add_argument("--out", default=None, help="output JSONL path (default: results/raw_<model>_<ts>.jsonl)")
    args = ap.parse_args()

    pool = load_pool(STIMULUS_POOL_PATH)
    if args.all_items:
        seeds = ["all"]                       # one "subject" = the whole pool, each item once
        sessions = {"all": pool}
    else:
        seeds = [args.seed_base + i for i in range(args.subjects)]
        sessions = {s: sample_session(pool, s) for s in seeds}

    base_params = {"temperature": args.temperature, "top_p": 1.0, "seed": args.seed}
    clients = {
        "direct": OpenRouterClient(model=args.model, params={**base_params, "max_tokens": args.max_tokens_direct}),
        "thinking": OpenRouterClient(model=args.model, params={**base_params, "max_tokens": args.max_tokens_thinking}),
    }

    cells = list(iter_cells(_parse_filter(args.only)))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = args.model.replace("/", "_").replace(":", "_")
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"raw_{safe_model}_{ts}.jsonl"

    n_trials = len(next(iter(sessions.values())))
    total = len(cells) * len(seeds) * args.k
    print(f"{len(cells)} cells x {len(seeds)} subjects x k={args.k} = {total} subject-runs "
          f"({total * n_trials} trial calls before cache)")
    print(f"determinism: temperature={args.temperature}, top_p=1.0, seed={args.seed}  |  "
          f"max_tokens: direct={args.max_tokens_direct}, thinking={args.max_tokens_thinking}")
    print(f"-> {out_path}")

    n_done = 0
    with out_path.open("w", encoding="utf-8") as f:
        for cell in cells:
            for seed in seeds:
                trials = sessions[seed]
                for repeat in range(args.k):
                    sid = subject_id(args.model, cell, seed) + f"|r{repeat}"
                    rows = run_subject(
                        clients[cell["effort"]], trials,
                        delivery=cell["delivery"], effort=cell["effort"],
                        worked_examples=(cell["practice"] == "examples"),
                        repeat=repeat, subject_id=sid, max_workers=args.max_workers,
                    )
                    for row in rows:
                        f.write(json.dumps(row) + "\n")
                    n_done += 1
                    n_err = sum(1 for r in rows if r.get("error"))
                    print(f"  [{n_done}/{total}] {sid}" + (f"  ({n_err} errors)" if n_err else ""))

    print(f"done -> {out_path}")


if __name__ == "__main__":
    main()
