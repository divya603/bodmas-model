# BODMAS LLM experiment

LLM task-analog of the human BODMAS misconception-detection study. A model gets the
**same stimulus** a participant saw (a math expression + a student's step-by-step work +
a belief statement about the student's misconception) and answers the **same 6-point
Likert question** ("How much do you agree that this is what the student believes?"),
scored against the same ground truth (`statement_correct`). Each model-run is one
"subject", so the output drops into the same analysis as the human and Bayesian data for
a three-way comparison.

Adapted from the numberlink LLM pipeline — the OpenRouter client, session layer, seeded
per-subject sampler, and JSONL→schema parser carry over unchanged; the board encoder is
replaced by a text-trace prompt builder (our traces are already text), and the response
is the 1–6 Likert rating instead of yes/no + confidence.

## Design (factorial)

Same pattern as numberlink, minus the `encoding` factor (a wash there, and moot here —
traces are text):

| factor    | levels                 | what it tests                                   |
|-----------|------------------------|-------------------------------------------------|
| effort    | direct / thinking      | reasoning off vs. extended thinking (the arm of interest — biggest effect in the numberlink pilot) |
| delivery  | independent / append   | fresh call per trial vs. one growing conversation (memory) |
| practice  | none / examples        | whether a few worked examples precede the trials |

= **2×2×2 = 8 cells**. Screening scale (per numberlink): **1 synthetic subject** (seed
1000, one balanced 24-trial draw), **k=1**, the **identical 24 trials in identical order
in every cell** (1-to-1 comparisons). Run per model. Signal lives in the factor-level
aggregates, not per-cell. Then scale the chosen format to more subjects + k before any
human-vs-model claims.

Models (via OpenRouter): `anthropic/claude-haiku-4.5`, `openai/gpt-4o`,
`anthropic/claude-opus-4.8`.

## Setup

```bash
cd llm_exp
uv pip install -e .          # or: pip install -e .
cp .env.example .env         # then paste your key into .env:  OPENROUTER_API_KEY=sk-or-...
```

`.env` is gitignored and auto-loaded — never commit it. `results/` (raw JSONL + response
cache) is gitignored too; the cache makes reruns free and lets interrupted runs resume.

## Run

```bash
# smoke test: 1 subject, direct only (24 calls, ~cents) — confirm real calls parse
python -m bodmas_llm.run_pilot --model anthropic/claude-haiku-4.5 --subjects 1 --k 1 --only effort=direct

# full screening sweep for one model (all cells, 1 subject, k=1)
python -m bodmas_llm.run_pilot --model anthropic/claude-haiku-4.5 --subjects 1 --k 1

# parse + accuracy summary
python -m bodmas_llm.parse_results results/raw_*.jsonl --out results/parsed.parquet --summary
```

`--only` filters cells, e.g. `--only effort=thinking,delivery=independent`.
Determinism: `temperature=0, top_p=1, seed=0` (part of the cache key; providers may not
fully honor it, hence k-repeats + majority vote at real scale).

## What the model sees

- **System turn:** the human instructions (verbatim from `InstructionsView.vue`) + the
  1–6 rating format.
- **User turn:** the expression, the student's work (`= step` per line, exactly as
  `TraceJudgmentView.vue` renders it), the belief statement, and the question.

**Answer-leak guard** (`bodmas_prompt.assert_no_leak`, unit-tested over all 240 items):
the prompt may only contain `expression`, `trace`, `belief_statement`, `student_name`;
`statement_correct`, `misconceptions`, `probed_misconception`, `which_target`,
`category`, `num_misconceptions` can never appear.

## Files

```
bodmas_llm/
  bodmas_prompt.py   system prompt + stimulus renderer + leak guard + 1–6 JSON schema
  client.py          OpenRouter client (disk cache, tenacity retries, effort fallback)
  sample_session.py  seeded 24-trial balanced draw (port of the human sample_form)
  session.py         independent / append delivery; per-trial record shape
  run_pilot.py        factorial driver -> results/raw_*.jsonl
  parse_results.py   raw JSONL -> tidy frame in the human per-trial schema
tests/test_prompt.py offline: leak guard + balanced sampling (no network)
data/stimulus_pool.json   the 240-item BODMAS pool (copy of the deployed one)
```
