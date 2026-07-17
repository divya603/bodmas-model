# BODMAS Project — Handoff

A complete, from-scratch orientation for a model picking this up cold. Read this instead of the
conversation history. Repo: `divya603/bodmas-model` (GitHub), branch `main`. **Pushing to `main`
auto-deploys the human experiment** to a live site (see §3).

> **⚠️ STANDING INSTRUCTION TO EVERY AGENT: keep this file current.** As you complete work —
> new scripts, new figures, findings, decisions made, bonuses paid, state changes — update the
> relevant sections of this HANDOFF.md, either incrementally as you go or at the latest before
> your session ends. This file is the single source of truth; the next session must be able to
> pick up cold from it alone.

---

## 0. What this project is (the big picture)

We study **how people (and LLMs, and an ideal observer) detect arithmetic order-of-operations
misconceptions** from a student's step-by-step work. There are **three "observers"** all judging
the **same 240-item stimulus pool**:

1. **Humans** — an online experiment (Smile/Vue, deployed live, run on Prolific).
2. **A Bayesian ideal observer** — infers which misconception(s) generated a trace.
3. **LLMs** — a task-analog run through OpenRouter (haiku-4.5, gpt-4o).

The end goal is a **three-way comparison**: are the misconceptions/conditions that are hard for
people the same ones hard for the LLM and, in principle, for the ideal observer?

**The task (one trial):** a participant/model sees a **math expression**, a **student's
step-by-step work** (which contains 1 or 2 misconceptions), and a **belief statement** claiming
the student holds a particular misconception. They rate, on a **6-point Likert scale** (1=Strongly
Disagree … 6=Strongly Agree), *how well the statement explains the work* — NOT whether the final
answer is right.

### The 6 misconceptions (order-of-operations errors)
| id | meaning |
|---|---|
| `add_before_mul` | does `+` before an adjacent `×` |
| `add_before_div` | does `+` before an adjacent `÷` |
| `sub_before_mul` | does `-` before an adjacent `×` |
| `sub_before_div` | does `-` before an adjacent `÷` |
| `same_priority_rtl` | evaluates equal-priority ops right-to-left instead of left-to-right |
| `outside_bracket_first` | evaluates ops outside a bracket before resolving its contents |

### The 4 stimulus categories (A/B/C/D), 60 items each = 240
- **A** — 1 misconception in trace; statement **names it** → correct answer = **agree**.
- **B** — 1 misconception in trace; statement names a **different/absent** one (foil) → **disagree**.
- **C** — **2** misconceptions in trace; statement names **one of the two** → **agree** (a *partial* explanation).
- **D** — 2 misconceptions; statement names **neither** (foil) → **disagree**.

Scoring collapses the 1–6 rating to binary (**≥4 = agree**), correct if it matches the category's
direction. **Signal-detection framing** (used throughout the LLM analysis): "agree" = "yes";
signal present = A/C; signal absent = B/D; `hit=P(agree|A/C)`, `FA=P(agree|B/D)`,
`d'=z(hit)-z(FA)`, `criterion=-0.5[z(hit)+z(FA)]` (positive criterion = disagree-biased).

---

## 1. Repository map

```
base-task/        The BODMAS model (Python): pool generation, trace sim, Bayesian inference, Streamlit app
src/              The Smile/Vue human experiment (deployed live). User code in src/user/
analysis_human/   Human-data analysis scripts + plots (UNCOMMITTED)
analysis-Bayesian/ Ideal-observer analysis figures (plot_bayes_2misc_heatmap.py; imports base-task/)
Results_combined/ FINAL results doc: results.tex + figs/ (subfolder RENAMED from plots/ on
                  2026-07-15 so tex references figs/<exact-name> and the folder uploads to
                  Overleaf as-is). Results-only, no story; being written STEP BY STEP with the
                  user — so far §1 Bayesian 1-misc (dist_A right-shifted mass, outside()
                  weakest; dist_B three rows + sharp left shift on refuted items, 12/60
                  refutable, absence-of-support point) and §2 LLM 1-misc (thinking ≈ softened
                  ideal observer incl. left shift on refuted; the two direct regimes near-binary
                  and claim-driven in OPPOSITE directions — haiku-direct endorses add/sub &
                  rejects RTL/outside, gpt-4o the reverse; generated sub<÷ item: 2 vs 5 vs 6),
                  and §3 human 1-misc (dist_A at chance / add<× below; dist_B two cells over
                  0.5 = the two failure modes, refuted-row FA drop; closes with
                  human_signal_detection.png — d' spread −1.64..+2.77, two below-chance
                  participants, and the "considering practice trials with feedback before the
                  confirmatory run" note). figs/ now also holds human_signal_detection.png
                  (user copied). Figures pinned with [H] (user wants strict source order);
                  user added soul highlights — preserve them when editing.
                  §4 (2-misc) has a SHARED heatmap explainer up front (C = 6×6 named×partner
                  square, D = 6×15 foil×pair rectangle, green/red diverging, value+n per cell
                  — written once for all three observers) + §4.1 Bayesian (panel (a) all green
                  0.76–1.00, weakest = outside() as target; panel (b) all red ≤0.26, flat →
                  confusions elsewhere are observer properties) + §4.2 LLM heatmap (thinking:
                  green (a)/red (b) except add<× foil row lighting up on add<÷-containing
                  pairs = family confusion; direct: wall of 1s both panels = right on foils
                  for the wrong reason; gpt-4o: RTL/outside() rows green in BOTH panels =
                  claim-driven) + §4.3 human heatmap (noisy-cells caveat; (a) subtraction
                  rows under-endorsed + sub<×/outside() partners mask targets; (b) outside()
                  foil ROW red but outside()-containing pair COLUMNS carry the top false
                  alarms — trace-side confusions, mirror image of gpt-4o's claim-side; shared
                  add-family leak with haiku-thinking; ideal observer flat ⇒ observer
                  properties). **results.tex: ALL PLANNED SECTIONS WRITTEN** (§1 bayes 1misc,
                  §2 LLM 1misc, §3 human 1misc + SDT/practice-trials note, §4 heatmaps ×3).
                  COMPLETE 3x3 figure set as of 2026-07-15: {human,bayes,llm} x {2misc_heatmap,
                  1misc_dist_A, 1misc_dist_B} (9 figures). NOTE: copies are snapshots — re-copy
                  after regenerating any source figure.
llm_exp/          The LLM task-analog experiment (Python, OpenRouter)
llm_exp_buffer/   Reference copy of the teammate's numberlink LLM experiment (delete when done; UNCOMMITTED)
prereg_buffer/    Reference copy of the teammate's numberlink PRE-REGISTRATION (tex + figs)
PreReg/           OUR pre-registration draft: prereg.tex + figs/ (self-contained, user compiles on Overleaf)
scripts/          Smile deploy/data scripts + make_bonus_list.py (bonus tooling; UNCOMMITTED)
data/             Pulled participant data (real-all-main-data.json — gitignored, has demographics)
                  data/private/bonus_paid.csv — the payment LEDGER (gitignored)
public/           consent-form.pdf, debrief.pdf served by the frontend
env/, firebase/   Smile config/secrets (already set up)
```

---

## 2. Component 1 — The BODMAS model (`base-task/`)

Pure-Python model of how a "learner" with misconceptions solves arithmetic.

- **`dag.py`** — FlatDAG representation of an expression (atoms + op nodes, shared references).
- **`parser.py`** — `build_dag(expr)`. NOTE: folds signed-number literals (e.g. `-14`) so re-parsing
  intermediate trace strings matches `_eval`'s representation (a fixed bug).
- **`generator.py`** — `generate_expression(n_ops=4, bracket_prob=…)` random expressions.
- **`pattern_matcher.py`** — classifies 3-node "windows" into Tables 1–5.
- **`learner.py`** — `MISCONCEPTION_FLIPS`: each misconception's bidirectional `to_true`/`to_false`
  validity flips. A **learner = list of 0–2 misconception ids**.
- **`valid_actions.py`**, **`traces.py`** — `generate_traces(dag, misconceptions)` simulates a
  learner and returns all step-by-step traces (AND-across-windows validity). Includes
  `is_zero_divide` guard (never fires ÷0; a fixed "stuck trace" bug).
- **`distance.py`** — `diagnostic_traces()` (traces that behaviorally diverge from expert),
  `tree_edges()`, `correct_answer()`.
- **22 learner profiles** = expert `()` + 6 singles + 15 pairs = `C(6,2)+7`.

### Pool generation — `stimulus_pool.py`
- Builds the **240-item pool** (`stimulus_pool.json`), 60 per category A/B/C/D. Each item has:
  `id, category, expression, trace (list of step strings), misconceptions (present, ground truth),
  probed_misconception (named in statement), statement_correct (bool), which_target (C only:
  'first'/'second'), num_misconceptions, student_name, belief_statement`.
- **Answer-leak-relevant fields** (never shown to a solver): `statement_correct, misconceptions,
  probed_misconception, which_target, category, num_misconceptions`.
- `sample_form(pool, seed)` → one participant's **balanced 24-item draw** (6 per category, distinct
  student names). **This is mirrored in JS (`src/user/utils/sampleForm.js`) — the two MUST stay in
  sync.**

### ⚠️ Category-C chronological rebalance (important recent fix)
Originally `which_target` (first/second) meant *canonical pair order*, which is **invisible to
participants and perfectly confounded with misconception identity** (e.g. `add_before_mul` was
always "first"). Fixed: **`which_target` now means the chronological position of the probed
misconception's error in the trace** (does its error appear first or later than the partner's).
`regenerate_C.py` rebuilt only category C, balanced so **each misconception is the target 5× as the
early error + 5× as the late error** (A/B/D preserved byte-for-byte). Verified: `sample_form` gives
exactly 3 first / 3 second per participant, rotating across participants.

### Bayesian ideal observer
- **`inference.py`** — `posterior_over_profiles(trace)`: posterior over all 22 profiles from a
  trace, via an **epsilon-greedy transition model** (ε=0.05); `marginal_rule_probability()` =
  P(a misconception is in the learner's policy | trace). Known limitation: exact ties between a
  single misconception and pairs containing it when the second never manifests (resolved in practice
  by the marginal, and the professor confirmed "if the marginal is fine, the model is fine").
- **`misconception_difficulty.py`** → `misconception_difficulty.json`: the **ideal-observer
  difficulty baseline** — avg marginal on the true misconception per rule, split alone vs paired.
  **Re-run on the current (rebalanced) pool.** Current ranking hardest→easiest:
  `outside_bracket_first (0.69) < sub_before_div < add_before_div < add_before_mul < sub_before_mul
  < same_priority_rtl (0.82)`. Paired marginals rose after the C fix (e.g. sub_before_mul 0.52→0.74).
- **`test_recovery.py`** — MAP-recovery validation.
- **`app.py`** — Streamlit app, 4 tabs: Expert Learner, Misconception Learner, Infer Learner Type,
  Misconception Difficulty. Run: `cd base-task && streamlit run app.py`.
- **`make_practice_examples.py`** — regenerates the 3 LLM practice examples (writes
  `llm_exp/data/practice_examples.json`).
- **`make_human_practice_items.py`** — regenerates the 3 HUMAN practice trials (writes
  `src/user/data/practice_items.json`). P1 add_before_div named+present (agree), P2
  sub_before_mul present but add_before_mul named (foil, disagree; the foil is actively
  refuted by the trace, IO marginal 0.000), P3 same_priority_rtl + outside_bracket_first
  with outside named (partial match). Each item carries `error_steps` (full-trace indices
  where each misconception fired, found by a state-local check via `_next_dags`: a step an
  expert at that state would never take, attributable to exactly ONE of the item's
  misconceptions) plus a `feedback` statement. Rejection-sampled so every misconception
  fires exactly once, traces are integer-only, expressions are NOT in the 240-item pool,
  and student names (Tara/Sam/Kira) are outside STUDENT_NAMES. P3 additionally requires
  the outside() error as the FIRST step and the RTL error inside the bracket, so the two
  highlights sit in visually separate places.

---

## 3. Component 2 — The human experiment (Smile/Vue, `src/`)

A Smile (codec-lab / gureckislab) Vue-3 experiment, deployed live. **User code in `src/user/`.**
`npm run dev` to run locally; **`git push origin main` auto-deploys** to `www.codec-lab.org` via
GitHub Actions (`gh run list`/`gh run watch` to monitor; a `deploy-error` workflow shows "skipped"
on success — that's normal).

- **`src/user/design.js`** — the timeline (consent → windowsizer → instructions → comprehension quiz
  → practice (3 feedback trials) → experiment → feedback survey → demographics → save → debrief →
  thanks).
- **`src/user/components/trace_judgment/InstructionsView.vue`** — task instructions (button says
  "Next"); announces the 3 practice questions and that they don't count toward the bonus.
- **`src/user/components/trace_judgment/PracticeView.vue`** — 3 practice trials (added 2026-07-16
  because pilot category-A accuracy was at chance). Same trial layout + 3s lock as the main task;
  after the participant submits a rating, the erroneous step(s) are highlighted amber in the trace
  (with a short note per step, e.g. "the student added before dividing here") and a feedback box
  explains what the right answer would be. The feedback NEVER says whether the participant's own
  choice was right or wrong. Items come from `src/user/data/practice_items.json` (generated by
  `base-task/make_human_practice_items.py`, single copy, frontend-only). Responses are recorded
  (view `practice`, same per-trial fields incl. `is_correct`) but do NOT touch the bonus counters.
  Ends with a transition screen ("Practice complete! ... You will judge 24 problems ... These
  count toward your bonus.") with a "Begin task" button before the real task.
- **`src/user/components/trace_judgment/TraceJudgmentView.vue`** — the 24-trial task. Shows expression
  + work (`= step` per line) + belief statement + 6-point Likert. Has:
  - a **3-second answer lock** per trial (options disabled first 3s, with countdown);
  - an **"X of 24" counter** top-right;
  - **bonus scoring**: binary-direction, rescaled `bonus = max(0,(acc−0.5)/0.5)×$2`, recorded per
    trial (`is_correct`) + a `traceJudgmentBonus` summary block in `pageData_exp`;
  - **mouse tracking** (`src/user/utils/useMouseTracking.js`) — sampled cursor path per trial for
    offline bot detection.
- **`src/user/utils/sampleForm.js`** — JS port of `sample_form` (keep in sync with Python!).
- **`src/user/data/stimulus_pool.json`** — the pool copy the frontend reads (must match
  `base-task/stimulus_pool.json`; `regenerate_C.py` writes both).
- **`src/user/components/quizQuestions.js`** — the 1 comprehension question ("What should your rating
  be based on?"), length-matched options + a "handwriting" distractor.
- **`src/builtins/thanks/ThanksView.vue`** — edited in place: upload-progress screen + Prolific
  completion code **`CNIEB9GV`** in BOTH the `prolific` and `web` blocks.
- **`public/consent-form.pdf`** — real NYU IRB form (IRB-FY2026-11440, PI Mark Ho).

### ⚠️ Prolific URL (critical — was the source of a big bug)
Participants MUST enter via a URL routing to `#/welcome/prolific/` with the ID params, or they're
recorded `recruitmentService: "web"` with **no `prolific_id`** (can't be bonused). Working format
(params BEFORE **and** AFTER the hash; trailing slash after codename):
```
https://www.codec-lab.org/e/<codename>/?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}#/welcome/prolific/?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}
```

### Human data + bonusing
- Pull: `npm run getdata` → `data/real-all-main-data.json` (participant records; each has `.data`
  with `seedID`, `recruitmentService`, `pageData_exp.visit_0.data` = 24 trials + bonus block, etc.).
  `npm run getrecruitment` → `data/private/real-main-recruitment.json` (maps `session_id` →
  `prolific_id`). **Join key: data `seedID` == recruitment `session_id`.**
- **`scripts/make_bonus_list.py`** — recomputes each bonus from raw responses vs the answer key
  (tamper-resistant), joins to prolific_id, emits paste-ready `prolific_id,amount` for Prolific's
  bulk-bonus box. `--exclude <id>` to drop your own test runs.
- **Payment ledger (prevents double-paying):** `data/private/bonus_paid.csv` records everyone
  already bonused; plain runs emit ONLY unpaid people. Workflow: run script → pay on Prolific →
  re-run with `--mark-paid` to record. `--include-paid` = audit list. **All 24 pilot participants
  were paid and ledgered on 2026-07-14** ($11.83 total). The 3 batches share one Prolific
  study_id, so the ledger is the only guard against re-paying.
- **DO NOT commit `data/real-all-main-data.json`** (participant demographics; now gitignored).

---

## 4. Component 3 — The LLM experiment (`llm_exp/`)

Task-analog: each model gets the SAME stimulus (expression + trace + belief statement) and gives the
SAME 1–6 Likert rating, scored against `statement_correct`. Adapted from the teammate's numberlink
pipeline (`llm_exp_buffer/`).

- **Setup:** `cd llm_exp && pip install -e .`; key in `llm_exp/.env` as `OPENROUTER_API_KEY=sk-or-…`
  (gitignored, auto-loaded). `results/` (raw JSONL + response cache) is gitignored — **cache makes
  re-runs free**.
- **`bodmas_llm/bodmas_prompt.py`** — system prompt (human instructions verbatim) + stimulus renderer
  + **answer-leak guard** (unit-tested over all 240) + 1–6 JSON schema + 3 worked practice examples
  (`data/practice_examples.json`: EX1 single-match→6, EX2 foil→1, EX3 partial-match→4 with a "can't
  be 6 because incomplete" rationale).
- **`bodmas_llm/client.py`** — OpenRouter client (disk cache keyed by model+params+messages+effort;
  tenacity 5-retry backoff; effort control: thinking=`reasoning{effort:high}`, direct=`reasoning
  {enabled:false}`; empty-`choices` retry-then-error guard). Parser reads `{"rating":1..6}`.
- **`bodmas_llm/sample_session.py`** — port of `sample_form`; `--all-items` runs the full 240 once.
- **`bodmas_llm/run_pilot.py`** — factorial driver: `effort {direct,thinking} × practice
  {none,examples}`, independent delivery only. `--all-items` + `--only effort=…,practice=…` runs one
  cell over all 240. Determinism temp=0/top_p=1/seed=0; caps 8192 direct / 16000 thinking.
- **`bodmas_llm/parse_results.py`** — raw JSONL → tidy frame in the human per-trial schema.

### LLM results (all 240 items, done)
Three regimes (gpt-4o ignores the thinking flag → 0 reasoning tokens, so run once as "direct"):
| regime | accuracy | d' | criterion |
|---|---|---|---|
| **haiku (thinking)** | **0.87** | **2.34** | −0.33 |
| haiku (direct) | 0.68 | 1.27 | +0.80 |
| gpt-4o (direct) | 0.63 | 0.66 | +0.27 |

**Story:** without reasoning both models are **disagree-biased** and near-binary (haiku-direct puts
74% of ratings on "1"); they retain partial competence (d'>0) but a heavy conservative criterion.
**Thinking transforms haiku:** d'→2.34, bias→neutral, **confidence calibrated** (AUC 0.49→0.79, using
Likert magnitude |rating−3.5|), uses the graded scale, and **perceives the partial-match nuance**
(rates C below A: 5.32 vs 5.73). Thinking tokens are RT-like: disagree-direction (732 vs 651) and
errors (1126 vs 672) cost more.

### LLM plots + report
- **`make_llm_plots.py`** → `plots/llm_accuracy_overview.png`, `plots/llm_accuracy_by_misconception.png`.
- **`make_llm_figures.py`** → `plots/llm_signal_detection.png`, `llm_thinking_tokens.png`,
  `llm_confidence.png`, `llm_response_style.png`, + a cost table (stdout). Run: `python3
  make_llm_figures.py [signal|tokens|confidence|response|cost]`.
- **`report/report.tex`** — LaTeX report mirroring the numberlink one (`llm_exp_buffer/report/
  report.tex`). Self-contained: figures in `report/figs/`, referenced as `figs/llm_*.png`. Sections:
  abstract, intro, methods (incl. trial generation), results (all 6 figures + signal-detection &
  cost tables), discussion. Figures are PNG (could switch scripts to also emit PDF for vector).

---

## 5. Component 4 — Human data analysis (`analysis_human/`)

- **`analyze_human.py`** — sanity dashboard + accuracy by misconception (alone/paired) + plots.
  Run from repo root: `python3 analysis_human/analyze_human.py`. Filters to `recruitmentService ==
  'prolific'` done sessions (add `--include-web` for the pre-fix testing sessions).
- **`plot_human_sdt.py`** — per-participant signal detection: sorted d' dot plot + ROC space with
  iso-d' curves (mirrors the LLM fig_signal; same 1/(2N) correction so d' values are comparable).
- **`plot_human_rt.py`** — RT screening: per-trial RTs sorted by d' (3s-lock line) + median-RT-vs-d'.
- **`plot_2misc_heatmap.py`** — 2-misconception (C/D) present×shown heatmaps: C = 6×6 (named
  target × partner), D = 6×15 (named foil × present PAIR; pair-columns containing the foil are
  impossible). Green=agree/red=disagree diverging map. Mirrored by
  `analysis-Bayesian/plot_bayes_2misc_heatmap.py` (marginal P(named rule | trace), 0.5 neutral)
  and `llm_exp/make_llm_2misc_heatmap.py` (3 regimes; reads raw_*.jsonl all_items runs, NOT
  parsed_all.parquet which is only the 24-item factorial pilot). All three figures copied to
  `Results_combined/plots/`. **2-misc analysis across all three observers: DONE.**
  Candidate findings: (1) humans' top false alarms sit on pairs containing outside() (bracket-
  mangled traces make ANY claim believable) while outside() foils themselves are easily rejected;
  (2) false add<× endorsed when add<÷ present — shared by haiku-thinking; (3) ideal observer flat
  (max 0.26) → confusions are observer properties, not stimulus leaks; (4) gpt-4o is claim-driven
  (endorses RTL/outside() statements regardless of trace).
- **1-misconception (A/B) analysis, Bayesian arm DONE:** `analysis-Bayesian/plot_bayes_1misc_marginals.py`
  → `bayes_1misc_marginals.png`: per-item P(named rule | trace) dots (NO averaging — user's
  explicit preference, "not seeing the full picture"), 10 A + 10 B items per misconception.
  Findings: threshold-0.5 accuracy is 100% everywhere (ceiling, uninformative); marginals VARY
  per item — A tight (0.82–1.00; outside() shifted lowest), B structured into "actively refuted"
  (~0.00–0.03: the foil had chances to manifest and visibly didn't) vs "no evidence either way"
  (~0.17–0.26 residual: the foil never had an opportunity). Human + LLM 1-misc counterparts
  still TODO; then copy the set to Results_combined/plots/.
- **1-misc distribution view (Bayesian):** `plot_bayes_1misc_distributions.py` →
  `bayes_1misc_dist_A.png` (6 KDE panels by present misconception; present=named in A) and
  `bayes_1misc_dist_B.png` (same 60 B items, TWO groupings: row 1 by misconception PRESENT,
  row 2 by misconception NAMED/foil). KDEs are boundary-reflected at [0,1] with the 10 raw item
  values as rug ticks. Finding: the PRESENT-grouping is near-uniform (refutation doesn't depend
  on the student's actual bug) but the NAMED-grouping cleanly sorts foils by refutability —
  RTL foils most often actively refuted (largest near-0 mass; equal-priority adjacencies are
  ubiquitous), sub<÷ foils never refuted (pure ~0.2 spike; -/÷ adjacencies rarely arise), others
  in between. I.e. for the ideal observer the action is claim-side, matching the prediction that
  the human arm should instead show trace-side (present-grouping) effects. The B figure has a
  THIRD row: the refuted-only subset (P < 0.15) by named foil — refuted counts per foil:
  add<×=1, add<÷=1, sub<×=3, sub<÷=0 (empty panel), RTL=4, outside()=3; only 12/60 B items are
  refutable at all (80% of foils are merely unsupported, never contradicted). Refuted items pile
  at ~0.00–0.03 except outside(), whose 3 refuted items sit ~0.12 (bracket contradictions are
  soft, mirroring outside()'s weak support in category A).
- **1-misc distributions, human + LLM arms DONE** (same layouts as Bayesian):
  `analysis_human/plot_human_1misc_distributions.py` → `human_1misc_dist_{A,B}.png`;
  `llm_exp/make_llm_1misc_distributions.py` → `llm_1misc_dist_{A,B}.png` (3 regimes, grouped
  bars per panel). ⚠️ Plot style history: KDE was dropped first (user caught oversmoothing +
  boundary double-weighting on discrete ratings; KDE retained ONLY for continuous Bayesian
  marginals), then the HUMAN figures were further collapsed to **BINARY agree/disagree
  probability bars** (rating ≥4 = agree; fixed 0–1 y-axis; user: "the 1–6 scale is getting too
  noisy, no final picture"). Color convention on these bars: GREEN = agree, RED = disagree
  (always response-colored, same as the heatmaps' green/red; suptitle states which answer is
  correct per figure). Binary picture: A hovers at chance everywhere (P(agree) .46–.54)
  except add<× visibly below (.38); in ALL of category B only TWO cells cross 0.5 —
  outside()-PRESENT (.58 FA; bracket-mangled traces) and add<÷-NAMED (.54 FA; add-family leak);
  refutable foils draw fewer false alarms than their named-row counterparts (pooled .29 vs .39,
  add<÷ .25 vs .54). LLM figures keep the full 1–6 scale as **smooth shape-preserving (PCHIP)
  curves through the EXACT per-rating proportions** (user wanted Bayesian-style curves, not
  straight segments): passes through every true value, stays at literal 0 across unused
  ratings (no overshoot/invented bumps — plain KDE was rejected for exactly that), dots mark
  the six exact values, fixed 0–1 y-axis. The refuted-item split comes from
  `analysis-Bayesian/b_item_marginals.json` (written by the Bayesian distributions script) so
  all three observers use the SAME item subset. Findings: (1) human A ratings are polarized
  (modes at the extremes, not middling — the d' split again) but ASYMMETRIC per misconception:
  add<× is majority-disagree (15/24 at ≤3, mode "2"), outside() most-endorsed (13/24 at 5–6,
  zero at "4"); (2) human A difficulty partially
  INVERTS the ideal observer: outside() gets humans' highest A ratings (3.83) despite weakest
  IO evidence, RTL among lowest (3.21) despite strongest; (3) human B false alarms are both
  trace-side (outside()-present bimodal, mean 3.67) and claim-side (add<÷-named 3.67);
  (4) humans rate refutable foils lower (~2.64 vs ~3.0 unrefutable) — weak evidence they use
  refutation; (5) haiku-thinking mirrors IO incl. softer outside() support on A; haiku-direct
  spikes "1" everywhere (even true RTL claims); gpt-4o endorses outside() claims EVEN WHEN THE
  TRACE REFUTES THEM (refuted-row ratings 5–6) — claim-driven, evidence-blind. 1-misc analysis
  now covers all three observers; user copies curated figs to Results_combined/plots/.
- **Synthetic sub<÷ refutable probe (2026-07-15).** The pool has NO B item where the sub<÷ foil
  is refutable, so `analysis-Bayesian/make_synthetic_subdiv_item.py` (seed 20260715) generated
  ONE supplementary item → `analysis-Bayesian/synthetic_items.json` (id SYNB002; ⚠️ NOT in the
  deployed pool, humans never saw it; flagged `synthetic: true`). Expression
  `6 - 8 - 9 ÷ 9 × 11`, present misconception sub<×; at state `-2 - 9 ÷ 9 × 11` the student
  does the DIVISION first at a live sub/÷ decision point → the sub<÷ foil is actively refuted
  (ideal-observer marginal 0.0089). `llm_exp/run_synthetic_item.py` ran it through the 3
  regimes (same determinism params; ratings stored in the same JSON):
  **haiku thinking = 2 (correct, 1107 thinking tokens); haiku direct = 6 (!); gpt-4o = 5.**
  Both no-reasoning models STRONGLY endorse the refuted claim — the sub-family analog of the
  add-family confusion ("subtraction happened early" pattern-matches, never checking which
  operator it beat). Note haiku-direct normally rejects nearly ALL foils (criterion), yet flips
  to 6 here: its low FA rate is criterion, not competence. Rendering: per the user's
  preference both dist-B figures show it as an ORDINARY item (normal orange / regime-bar
  convention, sub<÷ refuted panel titled "n=1"; no special marker or "synthetic" label on
  figures — provenance lives only in synthetic_items.json and here). The item is NOT yet in
  the deployed pool but is slated to be swapped/added when the pool is enlarged (user intends
  to grow the pool later). If/when it goes in: write it into BOTH pool copies
  (base-task + src/user/data), and check sample_form/sampleForm.js balance still holds.
  Single item = illustration, not inference; `make_synthetic_subdiv_item.py` can mass-produce
  a balanced refutable set if this becomes a registered contrast for a future human wave.
- **`analysis-codes/1-Misc/plot_1misc_response_spread.py`** — Likert-spread dot plots for categories
  A (statement matches) and B (foil); x = **misconception present in the trace**, y = 1–6, dot per
  trial per participant, green→red.
- **`analysis-codes/2-Misc/plot_2misc_response_spread.py`** — category-C split by first/second target.
  (Running this is what surfaced the which_target confound that led to the chronological C fix.)

### Key human findings (N=24, complete pilot)
**24 Prolific participants** in 3 batches (Jul 9/10/13 2026): the first 15 saw the OLD category-C
items, the last 9 the rebalanced pool (verified per-session: their C trials match the current pool
6/6). A/B/D items are identical for everyone; C-positional analyses use only the last 9.
- **Accuracy** overall 0.60: A=0.49 (chance!), B=0.61, C=0.64, D=0.66; paired (0.65) > alone (0.55)
  for **every** misconception. The failure mode is *not endorsing correct explanations* (misses).
- **Pooled SDT:** hit=0.57, FA=0.36, **d'=0.51, criterion=+0.09** (weak sensitivity, near-neutral
  bias; below even gpt-4o-direct). The old "disagree-biased" story was A-specific, not global.
- **Huge individual spread** (the headline): per-participant d' from −1.64 to +2.77. Best human
  (0.92 acc, d'=2.77) beats haiku-thinking; ~7 people at chance; 2 reliably BELOW chance.
- **Mean ratings order C > A > B > D** (3.97 vs 3.48 vs ~3.0 vs 2.7): humans rate partial
  explanations ABOVE exact ones, the opposite of haiku-thinking (A 5.73 > C 5.32).
- **No click-through:** zero of 576 trials under 3.5s (3s lock; fastest 4.1s). The d'=−1.64
  participant spent 11 min (median RT 21.9s): systematic reversal, not low effort. One
  always-disagree participant (23/24 disagree, crit +1.73). rho(d', median RT)=0.39.
- **C first/second (new cohort only, n=9):** agree 0.63 (early error) vs 0.56 (late); within noise.
- Stimuli previously **verified correct** (all 240 pass audit; ideal observer ~0.95 on category A);
  the task is genuinely hard for humans, not buggy.

---

## 6. Component 5 — Pre-registration (`PreReg/`)

`PreReg/prereg.tex` + `PreReg/figs/` (3 PNGs copied from `analysis_human/plots/`). Mirrors the
teammate's numberlink prereg (`prereg_buffer/prereg.tex`): same styling, `\decflag` markers, and
an honesty-disclosure box (transparent/informed prereg; pilot N=24 reported in full, hypotheses
locked before a confirmatory cohort). Structure: study info → background (bug/repair-theory
framing, ideal observer as ceiling) → methods (learner model, pool + category tables, worked
example item A044, sampling, procedure, measures incl. SDT defs + bonus formula) → pilot results
built figure-by-figure (accuracy overview → by-misconception → SDT/ROC with individual spread) →
**candidate hypotheses (H1 A-vs-B recognition deficit, H2 C-vs-A partial-explanation rating with
the human/LLM reversal, H3 difficulty ordering vs ideal observer, H4 early/late position in C)**
→ analysis-plan skeleton → decisions box (exclusion rule, N/power, registry, authorship).
**Everything contentious is decision-flagged, nothing locked.** User compiles on **Overleaf**
(copy folder as-is; figures referenced as `figs/<exact-name>`; NO local TeX, don't install one).
Writing style: **no em dashes** (user: "screams AI").

---

## 7. Current state — what's done, what's pending

**DONE:**
- BODMAS model + 240-item pool (with the category-C chronological rebalance).
- Human experiment built + deployed live; consent, quiz, 3s lock, counter, bonus scoring, mouse
  tracking, ThanksView with completion code `CNIEB9GV`, correct Prolific URL.
- **Full pilot: 24 Prolific participants** (15 old-pool C + 9 new-pool C) pulled + analyzed
  (see §5 key findings: A at chance, d' spread −1.64..+2.77, no click-through).
- **All 24 participants bonused and ledgered (2026-07-14)** via `make_bonus_list.py --mark-paid`.
- Bayesian difficulty baseline re-run on the current pool.
- LLM experiment: all 240 items × 3 regimes, 0 errors; all figures + `report/report.tex` written.
- **Pre-registration draft** written (`PreReg/prereg.tex`, §6) — hypotheses/exclusions flagged,
  awaiting discussion.
- **Practice trials built AND DEPLOYED (2026-07-16/17, commit 26af20f):** 3 feedback practice
  trials before the main task (PracticeView.vue + practice_items.json +
  make_human_practice_items.py; instructions updated). This implements the "practice trials
  with feedback before the confirmatory run" note from results.tex §3. Verified: items
  model-checked (IO marginals: P1 named 0.946, P2 foil 0.000, P3 named 0.863), frontend
  compiles, and the live bundle contains the items. Deploy note: the first deploy attempt
  failed with an SSH i/o timeout to the lab server ("create the remote folders" step);
  `gh run rerun <id> --failed` fixed it — transient, not a code issue. User still to eyeball
  the deployed practice flow.

**PENDING / NEXT:**
- **Finalize the prereg decisions** (with the user, discussion-first): hypothesis set +
  directions, participant exclusion rule (candidate: below-chance binomial gate ≤7/24, plus
  no-gate sensitivity), confirmatory N/power analysis, registry + timeline.
- **Run the confirmatory cohort** after locking; C-positional questions especially need n.
- **Three-way comparison** (human × Bayesian × LLM per misconception/category) — the headline.
  All three arms now have data; the framing questions (pooled vs median vs distribution for
  humans, given the individual spread) feed the prereg discussion.
- **BODMAS results report** like the numberlink one; `report/report.tex` is the LLM half.
- **Uncommitted** (decide before relying on them): `analysis_human/`, `scripts/make_bonus_list.py`,
  `PreReg/`, `llm_exp/plots/`, `llm_exp/report/`, `llm_exp/make_llm_plots.py`,
  `llm_exp/make_llm_figures.py`. (The practice-trials change set was committed and deployed,
  commit 26af20f.) (`llm_exp/bodmas_llm/` package IS committed.) `llm_exp_buffer/`
  and `prereg_buffer/` are reference — delete when done.

---

## 8. Commands cheat-sheet

```bash
# Model / pool
cd base-task && python3 stimulus_pool.py            # regenerate full pool (rarely; C-fix uses regenerate_C.py)
cd base-task && python3 regenerate_C.py             # regenerate ONLY category C (preserves A/B/D)
cd base-task && python3 misconception_difficulty.py # re-run Bayesian difficulty baseline
cd base-task && python3 make_human_practice_items.py # regenerate the 3 human practice trials
cd base-task && streamlit run app.py                # the model explorer UI

# Human experiment
npm run dev                                          # local
git push origin main                                 # deploy (auto)
npm run getdata ; npm run getrecruitment             # pull participant + recruitment data
python3 scripts/make_bonus_list.py                   # -> UNPAID prolific_id,amount (ledger-aware)
python3 scripts/make_bonus_list.py --mark-paid       # after paying on Prolific: record in ledger
python3 analysis_human/analyze_human.py              # human accuracy + plots
python3 analysis_human/plot_human_sdt.py             # per-participant d' + ROC/iso-d' figure
python3 analysis_human/plot_human_rt.py              # RT screening figure

# LLM experiment (from llm_exp/)
pip install -e .                                     # once; then put key in .env
python3 -m bodmas_llm.run_pilot --model anthropic/claude-haiku-4.5 --all-items --only effort=thinking,practice=none
python3 -m bodmas_llm.run_pilot --model openai/gpt-4o --all-items --only effort=direct,practice=none
python3 -m bodmas_llm.parse_results results/raw_*.jsonl --out results/parsed.parquet --summary
python3 make_llm_plots.py ; python3 make_llm_figures.py
```

---

## 9. Gotchas / rules

- **`sampleForm.js` (JS) must stay in sync with `sample_form` (Python)** — both sample the 24-item
  form; the live experiment uses the JS one.
- **Pushing to `main` deploys.** Only the frontend (`src/`) affects the deploy; adding Python folders
  is a harmless no-op rebuild.
- **Never commit** `data/real-all-main-data.json` (participant demographics) or `llm_exp/.env`
  (API key). Both are (or should be) gitignored.
- **LLM cache is your friend** — re-running any already-run config costs $0; new spend only for new
  work. Total spent so far ≈ 1M tokens (~$2.5–3.5).
- **User preferences (from working with them):** during design discussions, **finish the discussion
  before writing code** — don't jump to implementation. On a **surprising result, verify our own
  stimuli/task first** before blaming participants. They run things themselves via `! <cmd>` and like
  work pushed (not left local) to avoid loss. **No em dashes in writing** (docs, reports, chat).
  **LaTeX compiles on Overleaf only** — keep tex folders self-contained (`figs/<exact-name>`),
  never install a local TeX toolchain.
- Commit messages end with the current model's co-author line, e.g.
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
