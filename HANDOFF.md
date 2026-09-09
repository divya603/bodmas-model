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
the **same stimulus pool** (240 items for the pilot; extended on 2026-07-17 to the **480-item
refutation design** — see §2):

1. **Humans** — an online experiment (Smile/Vue, deployed live, run on Prolific).
2. **A Bayesian ideal observer** — infers which misconception(s) generated a trace.
3. **LLMs** — a task-analog run through OpenRouter (haiku-4.5, gpt-4o).

The end goal is a **three-way comparison**: are the misconceptions/conditions that are hard for
people the same ones hard for the LLM and, in principle, for the ideal observer?

> **⚠️ A REDESIGN IS IN PROGRESS, see §7b.** As of 2026-09-08 the user has decided to narrow the
> design: **1 misconception per trace only** (categories C and D are dropped), 6 operators, and a
> NEW manipulated factor, **the position of the error (step 1 vs step 3)**. The v3 pool is BUILT
> and verified on branch `pool-v3` but is NOT deployed and NOT yet wired into the frontend, so
> everything described in §0-§6 below is still the CURRENT live design. Read §7b before touching
> the pool, the form sampler, or any figure.

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

### The 4 stimulus categories (A/B/C/D), 120 sampling-eligible items each = 480
- **A** — 1 misconception in trace; statement **names it** → correct answer = **agree**.
- **B** — 1 misconception in trace; statement names a **different/absent** one (foil) → **disagree**.
- **C** — **2** misconceptions in trace; statement names **one of the two** → **agree** (a *partial* explanation).
- **D** — 2 misconceptions; statement names **neither** (foil) → **disagree**.

B and D foils additionally carry a controlled **refutation status** (`foil_status`):
**refuted** = the trace passes a decision point where the foil had a visible chance to manifest
and the student demonstrably didn't take it (ideal-observer marginal < 0.15); **unsupported** =
the foil never had an opportunity (marginal 0.15–0.35). Each participant sees 12 foil trials:
**6 refuted + 6 unsupported, every rule named as a foil exactly twice (once each status)**.

Scoring collapses the 1–6 rating to binary (**≥4 = agree**), correct if it matches the category's
direction. **Signal-detection framing** (used throughout the LLM analysis): "agree" = "yes";
signal present = A/C; signal absent = B/D; `hit=P(agree|A/C)`, `FA=P(agree|B/D)`,
`d'=z(hit)-z(FA)`, `criterion=-0.5[z(hit)+z(FA)]` (positive criterion = disagree-biased).

---

## 1. Repository map

```
base-task/        The BODMAS model (Python): pool generation, trace sim, Bayesian inference, Streamlit app
                  v3 redesign (branch `pool-v3` only, see §7b): generator_v3.py, find_pairs_v3.py,
                  pool_v3.py -> stimulus_pool_v3.json, verify_v3.py, bayes_v3.py -> bayes_per_item_v3.json
src/              The Smile/Vue human experiment (deployed live). User code in src/user/
analysis_human/   Human-data analysis scripts + plots (UNCOMMITTED)
analysis-Bayesian/ Ideal-observer analysis figures (plot_bayes_2misc_heatmap.py; imports base-task/)
Results_combined/ FINAL results doc: results.tex + figs/ (tex references figs/<exact-name>,
                  folder uploads to Overleaf as-is). Results-only, no story. **FULLY
                  RECONCILED 2026-07-28** to the current state of all three arms: §1 Bayes
                  1-misc rewritten for the v2/ε=0 logical oracle (A all exactly 1.000; B
                  refuted exactly 0.000 for ALL six foils incl. outside(), unsupported band
                  0.17–0.29 mean 0.21); §2 LLM rewritten for the terra regime set (thinking
                  meanA 5.63; terra 5.36, evidence-shaped bracket weakness, NOT claim-driven,
                  outside-named foil agree 15%; haiku-direct claim-driven rejection of true
                  RTL/outside; NEW headline: NO regime drops FA on refuted vs unsupported —
                  0.19/0.19, 0.21/0.18, 0.11/0.03 — while humans DO); §3 humans = pooled
                  5-practice cohorts n=40 (A P(agree) .62–.82 overall .73; B no cell over
                  0.5, FA refuted .25 vs unsup .45; SDT d' 0.96 crit −0.09, span −0.67..2.35
                  median 1.18, 35/40 above chance, best d' 2.35 beats both direct regimes but
                  NOT thinking's 2.83); §4.1 bayes 2misc (panel (a) all exactly 1.00, (b)
                  ≤0.29, D-refuted exactly 0); §4.2 (thinking green/red, add-family
                  concentration GONE on v2; terra one notch softer with human-shaped residuals
                  = weak outside() targets + RTL foils on half their pairs (10/20); direct =
                  wall of 1s both panels); §4.3 humans n=40 (all 90 cells occupied median 5;
                  C mean 4.22 ≈ A 4.25, the n=21 C-dip did not persist; D acc .64); §5
                  heatmaps (bayes diag exactly 1.00/off-diag ~0.10 flat; terra pale outside()
                  diagonal cell; human diag means 3.85–4.53, off-diag fully occupied 4–10
                  per cell median 9, mostly red); §6 comparison (humans n=40; human profile
                  flattest: no group below 0.46 or above 0.77 graded, LLMs span the full
                  axis). Figures: human set renamed `human_*_5practice.png` (old
                  *_with_practice REMOVED); observer_scatter_{graded,binary}.png replaced by
                  the pooled n=40 + terra versions; bayes/llm figures are the v2 set.
                  Figures pinned with [H]; user's soul highlights preserved in style (green =
                  key finding, yellow = caveat/contrast) with contents updated to true claims.
                  NOTE: copies are snapshots — re-copy after regenerating any source figure.
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

### Pool generation — `stimulus_pool.py` + `extend_pool.py`
- `stimulus_pool.py` builds the original **240-item pool**, 60 per category. Each item has:
  `id, category, expression, trace (list of step strings), misconceptions (present, ground truth),
  probed_misconception (named in statement), statement_correct (bool), which_target (C only:
  'first'/'second'), num_misconceptions, student_name, belief_statement`.
  ⚠️ Running it standalone now ABORTS unless `--rebuild-240` is passed (it would clobber the
  extended pool with one lacking `foil_status`, which sample_form requires).
- **`extend_pool.py` (2026-07-17, seed 20260717)** extended the pool to the **480 design**
  (was 487 on disk before `drop_ambiguous.py`, see below), preserving all 240 originals
  byte-for-byte (asserted) and adding:
  A +60 (6 rules × 20), B +61 (6 present × 5 foils × {refuted, unsupported} × 2), C +60
  (6 targets × 2 positions × 10, chronological balance kept), D +66 (15 pairs × 4 foils ×
  {refuted, unsupported} × 1). All B/D items (old + new) get **`foil_status`** and
  **`io_foil_marginal`**. Status classification uses TWO signals that must agree:
  *visible refutation* (state-local: the foil forbids the observed step, OR offers an extra
  action never taken — the latter covers outside()'s soft ~0.12 refutations) AND the marginal
  cut (refuted < 0.15 < unsupported ≤ 0.35; observed gap in final pool: 0.148 vs 0.167).
  Items where the signals disagree (e.g. D's combinatorial suppression, where the marginal is
  low for reasons no participant can see) are marked **`ambiguous`**. New expressions deduped
  against pool + the 3 practice items. Writes all THREE pool copies (base-task, src/user/data,
  llm_exp/data).
- **`drop_ambiguous.py` (2026-07-20)** removed the 7 ambiguous items (1 B, 6 D: B057, D003,
  D007, D016, D020, D029, D036) that extend_pool had kept for continuity but sample_form never
  drew. Pool is now a clean **480** (exactly 120 per category; every B (present,foil,status)
  cell = 2, every D (pair,foil,status) cell = 1). Safe: the ambiguous items sat under
  foil_status='ambiguous', a key no sampler requests, so no participant's form changed. Item
  ids left as-is (gaps fine). Writes all THREE pool copies; sampling re-verified 300/500 seeds
  (Py/JS). ⚠️ The LLM raw runs still contain 487 rows (the 7 removed items were run); the
  figure scripts and dashboard filter to current-pool ids, so all analysis is on the 480.
- **Answer-leak-relevant fields** (never shown to a solver): `statement_correct, misconceptions,
  probed_misconception, which_target, category, num_misconceptions, foil_status,
  io_foil_marginal`.
- `sample_form(pool, seed)` → one participant's **balanced 24-item draw** (6 per category, distinct
  student names). NEW balance (requires the extended pool): B = 3 refuted + 3 unsupported foils
  (refuted rules rotate per participant); D = 6 distinct pairs whose foils cover all 6 rules
  exactly once (backtracking assignment) with the refuted set the COMPLEMENT of B's → per form,
  every rule is a foil exactly twice, once refuted once unsupported. Verified balanced over
  500 seeds in BOTH languages; B-vs-D carrier of each rule's refuted slot splits ~50/50 across
  participants. **Mirrored in JS (`src/user/utils/sampleForm.js`) and in
  `llm_exp/bodmas_llm/sample_session.py` — the three MUST stay in sync.**

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
  trace, via a uniform-over-valid transition model; `marginal_rule_probability()` =
  P(a misconception is in the learner's policy | trace). **ε switched 0.05 → 0.0 on 2026-07-20**
  (`DEFAULT_EPSILON`, inference.py). Rationale: every pool trace is generated deterministically by
  one of the 22 profiles (`traces.generate_traces`), so there is no slip process and a nonzero ε is
  a misspecified likelihood. At ε=0 a forbidden step eliminates its profile outright. Verified over
  all 480 items: no item is left with zero surviving profiles, so no `ValueError`, no −inf
  posterior. Keep `epsilon > 0` (app.py's slider) for traces NOT produced by the 22 profiles,
  e.g. real student work. Known limitation, now sharper: exact ties between a single misconception
  and pairs containing it when the second never manifests. At ε=0 those ties are *exact*, so
  `most_likely_profile` breaks them arbitrarily and `test_recovery.py` reports 22/40 MAP — but the
  true profile is **never strictly beaten** (checked n=66: in the argmax set 66/66), and the
  marginal, which every figure uses, is unaffected. Professor: "if the marginal is fine, the model
  is fine."
- **`misconception_difficulty.py`** → `misconception_difficulty.json`: the **ideal-observer
  difficulty baseline** — avg marginal on the true misconception per rule, split alone vs paired.
  **Re-run at ε=0 on the 480 pool (2026-07-20).** Ranking hardest→easiest:
  `outside_bracket_first (0.78) < sub_before_div (0.78) < add_before_mul (0.81) < add_before_div
  (0.82) < sub_before_mul (0.85) < same_priority_rtl (0.85)` — same order as the ε=0.05 run
  (0.68/0.75/0.77/0.78/0.80/0.81) but compressed, and outside() is no longer distinctly hardest
  (0.779 vs 0.784, a 0.005 gap where it used to be 0.068). **All six are exactly 1.000 on the
  `alone` (category A) items**, so the entire remaining spread comes from paired items where the
  second rule never manifests, which is a stimulus-design property rather than an inference one.
  This matters for prereg **H3**: the ideal-observer difficulty gradient is nearly flat at ε=0.
- **`test_recovery.py`** — MAP-recovery validation.

#### ⚠️ `outside_bracket_first` is UNFALSIFIABLE in the current encoding (found 2026-07-21)

This is a model property, not a bug in any script, and it survives the ε=0 switch. It is the
single most important caveat on the Bayesian arm.

**Mechanism.** All six misconceptions flip cells in a 3-node window truth table
(`learner.py: MISCONCEPTION_FLIPS`, keys `(table, op1, op2, role)`). Five of them flip TWO cells
in the SAME Table-2 window: one op promoted (`to_true`), the competing op demoted (`to_false`).
So they *substitute* a move, which forces the learner and makes them testable. Example:

```
12 + 8 × 4 - 3        expert           -> ['12 + 32 - 3']
                      +add_before_mul  -> ['20 × 4 - 3']        <- expert's move is GONE
```

`outside_bracket_first` has `'to_false': set()`, uniquely. Its keys are Table 3
(`a op1 b op2 Y`) and Table 4 (`Y op1 b op2 c`), where the third slot is an UNRESOLVED bracket.
There, the second op was never fireable to begin with (you cannot multiply by an unevaluated
bracket), so there is no competing op cell to demote. The move it actually competes with is
`recurse_Y`, going inside the bracket, which belongs to the bracket's own inner window and is
scanned as a separate level (`traces.inner_valid_actions_for_learner` calls `match_patterns`
on the inner dag with no reference to the enclosing window). `recurse_Y` therefore has no key
that a `to_false` could name. So outside() can only APPEND:

```
12 + 8 × (1 × 4 - 3)  expert                  -> ['12 + 8 × (4 - 3)']
                      +outside_bracket_first  -> ['12 + 8 × (4 - 3)', '20 × (1 × 4 - 3)']
```

**Verified:** adding a rule to the expert removes a legal action in 0 of 1920 pool states for
outside(), versus 249 to 302 states for each of the other five.

**Consequences.**
1. Adding outside() to any profile only enlarges its legal-move set, so no observation can ever
   make that profile impossible. Its marginal can never reach 0. In category B: the other five
   foils hit exactly 0.000 on their refuted items (0 surviving profiles contain the foil);
   outside() always has exactly 1 surviving profile containing it, and averages 0.099 on its
   "refuted" 10 and 0.177 on its unsupported 10.
2. So the trace can only argue against outside() by DILUTION, never by contradiction: a learner
   with outside() who resolves the bracket had 2 legal moves and picked one (p=1/2) where an
   expert had 1 (p=1). Worked example B026 (`11 - 6 ÷ 10 × (11 ÷ 5)`, foil = outside()):
   four surviving profiles at 0.2222 each plus `sub_before_mul + outside_bracket_first` at
   0.1111, i.e. 0.5/(4+0.5). Never 0.
3. **The "refuted" label on the 10 outside()-named category-B items is a misnomer.** They fall
   under the 0.15 cut by dilution, not by logical contradiction like the other 50. Relevant to
   prereg **H5** (FA lower on refuted than unsupported foils): outside() contributes a weaker
   contrast (0.099 vs 0.177) and for a different reason, so either exclude it from H5 or
   pre-register H5 as a per-foil analysis.
4. This is the same asymmetry that pinned the bracket category-C items at 0.68 to 0.70 under
   ε=0.05 (dilution penalty on the present side). ε=0 fixed the present side because hard
   elimination now dominates there; the absent side has nothing to eliminate, so dilution is
   the whole signal and outside() is the one rule where dilution is all you ever get.
5. **Task-validity gap:** the belief statement participants read is a PREFERENCE claim
   ("believes you should calculate outside the brackets before what's inside them"), which
   would force `20 × (1 × 4 - 3)`. The model implements a PERMISSION ("may calculate outside
   first"). Humans and LLMs are being asked about a stronger claim than the one the observer
   scores.

**Model-v2 fix (do NOT do mid-experiment).** `_classify_window` already emits `recurse_Y` as a
first-class candidate reduction for Tables 3 and 4, it just is not keyed. Add a role value
`'recurse_Y'` (keys like `(3, '+', '×', 'recurse_Y')`), put those in outside()'s `to_false`, and
thread the enclosing window's verdict into `inner_valid_actions_for_learner` so the inner scan
can be suppressed. Roughly 20 lines across `learner.py`, `pattern_matcher.py`, `traces.py`. But
it changes which traces each profile generates, so the 480 pool, answer keys, refutation
statuses, every Bayes figure and the whole LLM run would need regenerating, against a live
experiment with n=21 collected. Park it as v2 for a future wave.
- **`app.py`** — Streamlit app, 4 tabs: Expert Learner, Misconception Learner, Infer Learner Type,
  Misconception Difficulty. Run: `cd base-task && streamlit run app.py`.
- **`make_practice_examples.py`** — regenerates the 3 LLM practice examples (writes
  `llm_exp/data/practice_examples.json`).
- **`make_human_practice_items.py`** — regenerates the 5 HUMAN practice trials (writes
  `src/user/data/practice_items.json`). P1 add_before_div named+present (agree), P2
  sub_before_mul present but add_before_mul named (foil, disagree; the foil is actively
  refuted by the trace, IO marginal 0.000), P3 same_priority_rtl + outside_bracket_first
  with outside named (partial match), and 2 MORE C-style partial-match trials added
  2026-07-27 (user request, humans still weak): P4 sub_before_div + add_before_mul with
  sub_before_div named, its error firing chronologically FIRST (`which_target: "first"`);
  P5 sub_before_mul + same_priority_rtl with RTL named, its error firing SECOND
  (`which_target: "second"`). P4/P5 come from `_find_flat_pair` (bracket-free, both rules
  fire exactly once, ≥4 distinct numbers, no standalone 0/±1 anywhere in the trace so no
  trivial ÷1 or +0 steps, final answer differs from expert). With P4/P5 all 6
  misconceptions now appear in practice traces and 5 of 6 as statements (sub_before_mul
  trace-only). Each item carries `error_steps` (full-trace indices
  where each misconception fired, found by a state-local check via `_next_dags`: a step an
  expert at that state would never take, attributable to exactly ONE of the item's
  misconceptions) plus a `feedback` statement. Rejection-sampled so every misconception
  fires exactly once, traces are integer-only, expressions are NOT in the pool,
  and student names (Tara/Sam/Kira/Milo/Anya) are outside STUDENT_NAMES. P3 additionally requires
  the outside() error as the FIRST step and the RTL error inside the bracket, so the two
  highlights sit in visually separate places. P1/P2 are byte-identical to the deployed
  originals; P3 got only a feedback-text tweak (mentions the bold marker).

---

## 3. Component 2 — The human experiment (Smile/Vue, `src/`)

A Smile (codec-lab / gureckislab) Vue-3 experiment, deployed live. **User code in `src/user/`.**
`npm run dev` to run locally; **`git push origin main` auto-deploys** to `www.codec-lab.org` via
GitHub Actions (`gh run list`/`gh run watch` to monitor; a `deploy-error` workflow shows "skipped"
on success — that's normal).

- **`src/user/design.js`** — the timeline (consent → windowsizer → instructions → comprehension quiz
  → practice (3 feedback trials) → experiment → strategy question → feedback survey → demographics →
  save → debrief → thanks).
- **`src/user/components/trace_judgment/StrategyQuestionView.vue`** — standalone free-response screen
  shown right after the task (added 2026-07-17): "Please describe the strategy or strategies you
  used to decide how much you agreed with each statement..." Required (Continue disabled until
  non-empty). Saved as `pageData_strategy` (`{strategy: ...}`). Then Continue leads to the original
  feedback survey (difficulty/enjoyment/general feedback/issues, unchanged).
- **`src/user/components/trace_judgment/InstructionsView.vue`** — task instructions (button says
  "Next"); announces the 3 practice questions and that they don't count toward the bonus.
- **`src/user/components/trace_judgment/PracticeView.vue`** — 5 practice trials (3 added
  2026-07-16 because pilot category-A accuracy was at chance; P4/P5 added 2026-07-27, see §2).
  Same trial layout + 3s lock as the main task;
  after the participant submits a rating, the erroneous step(s) are highlighted amber in the trace
  (with a short note per step, e.g. "the student added before dividing here") and a feedback box
  explains what the right answer would be. On two-misconception items (P3/P4/P5) the highlighted
  step whose misconception the belief statement names additionally gets a bold
  "(the belief statement points to this one)" marker (`isProbedStep`, generic: fires whenever
  `error_steps` ≥ 2 and the step's misconception == `probed_misconception`). The feedback NEVER
  says whether the participant's own
  choice was right or wrong. Items come from `src/user/data/practice_items.json` (generated by
  `base-task/make_human_practice_items.py`, single copy, frontend-only). Responses are recorded
  (view `practice`, same per-trial fields incl. `is_correct`) but do NOT touch the bonus counters.
  Ends with a transition screen ("Practice complete! ... You will judge 24 problems ... These
  count toward your bonus.") with a "Begin task" button before the real task. Counter and
  button labels key off `practiceItems.length`, so the count is not hardcoded in the view;
  InstructionsView.vue says "5 practice questions".
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

### LLM results — CURRENT (gpt-5.6-terra replaces gpt-4o; runs 2026-07-22, figures swapped 2026-07-28)
The user re-ran the suite on 2026-07-22 17:48-17:51: `results/raw_{haiku_thinking,haiku_direct,
gpt56terra_direct}_all480.jsonl` (exactly 480 rows each, clean pool, 0 errors). The old
`raw_*_all487.jsonl` files (including openai/gpt-4o) were DELETED. The figure scripts were
already on the `'gpt-5.6 (direct)'` regime (`openai/gpt-5.6-terra`); all 15 plots regenerated
2026-07-28 and both snapshot sets refreshed (Results_combined/figs LLM figures, report/figs).
| regime | acc | hit | FA | d' | criterion |
|---|---|---|---|---|---|
| haiku (thinking) | 0.892 | 0.975 | 0.192 | 2.83 | -0.54 |
| haiku (direct) | 0.652 | 0.371 | 0.067 | 1.17 | +0.92 |
| **gpt-5.6-terra (direct)** | 0.821 | 0.838 | 0.196 | 1.84 | -0.06 |
terra story: a no-reasoning model with near-neutral criterion and d' 1.84, far above gpt-4o's
0.60 and above haiku-direct; its weak groups (outside()/C, RTL/D, add<x) mirror the humans'
(see the analysis_human_practice_2 scatters), unlike gpt-4o's claim-driven scatter.
⚠️ The haiku numbers differ non-trivially from the 487-era runs (thinking hit 0.92 -> 0.975,
d' 2.37 -> 2.83, criterion -0.25 -> -0.54), so the 07-22 rewrite was NOT a pure cache replay of
the haiku arms; the 487-era responses survive only in the old tables/tex.
⚠️ STILL GPT-4O/487-BASED (prose not yet reconciled): report/report.tex,
Results_combined/results.tex (and its observer_scatter_{binary,graded}.png copies, which are
also still n=21-human + gpt-4o), the historical section below, run_synthetic_item.py
(its stored gpt-4o rating stays as provenance).

### LLM results (HISTORICAL, 487-era runs with gpt-4o, 2026-07-17; raw files deleted 2026-07-22)
Three regimes (gpt-4o ignores the thinking flag → 0 reasoning tokens, so run once as "direct").
Re-run over the full 480 design; 240 originals served from cache ($0), only new items new spend;
0 errors. Raw runs: `results/raw_{haiku_thinking,haiku_direct,gpt4o_direct}_all487.jsonl`
(old 240-item runs archived under `results/archive_240pool/`).
| regime | accuracy | d' | criterion | FA refuted | FA unsupported |
|---|---|---|---|---|---|
| **haiku (thinking)** | **0.87** | **2.38** | −0.25 | **0.09** | 0.26 |
| haiku (direct) | 0.69 | 1.35 | +0.84 | 0.07 | 0.06 |
| gpt-4o (direct) | 0.62 | 0.62 | +0.19 | **0.38** | 0.25 |

Endpoints match the 240-pool pilot (haiku-thinking d' 2.34→2.38 etc.). **NEW refuted-design result
(the H5 contrast):** on the named foil, false-alarm rate by refutation status —
haiku-thinking DROPS on refuted foils (0.26→0.09: it uses the visible contradiction, like the ideal
observer); haiku-direct is flat but floor-bound (rejects almost everything, criterion +0.84);
**gpt-4o goes the WRONG way (0.25→0.38: endorses refuted claims MORE than merely-unsupported ones)**
— evidence-blind / claim-driven, the same story the 2-misc heatmap showed, now quantified on a
balanced refuted vs unsupported split (120 refuted + 120 unsupported foil items pool-wide).

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
  report.tex`). Self-contained: figures in `report/figs/`, referenced as `figs/*.png`. Sections:
  abstract, intro, methods (incl. trial generation), results, discussion. Figures are PNG.
  **FULLY RECONCILED TO THE 480 POOL (2026-07-22).** Both figure scripts (`make_llm_figures.py`,
  `make_llm_plots.py`) now filter `load_all_items()` to the current 480-pool ids and dedup
  (the raw logs hold 487; the 7 dropped ambiguous items were being silently included, and the
  suptitles/prose still said 240). Every number in the prose and both tables was recomputed on
  480 and updated. Key 480 values now in the doc: accuracy 0.617/0.683/0.875 (gpt/haiku-direct/
  haiku-thinking), d′ 0.60/1.33/2.37, criterion +0.19/+0.83/−0.25, agree-rate 43/25/55%,
  haiku-thinking token medians disagree 772 vs agree 629 and error 1076 vs correct 672, quartile
  accuracy 0.94→0.73, ρ(tokens,#misc)=0.11 p=.02 (now weakly significant, was "n.s."), cost
  242k/959k tokens (387k reasoning) for thinking. Two things flipped vs the 240 draft: gpt-4o no
  longer dips below chance on add<× (now 0.52), and the token/#-misc correlation is now
  significant.
  **Two NEW results subsections added (2026-07-22):** (1) "One-misconception structure: present
  vs named, across observers" embedding the three 1-misc present×named heatmaps
  (`figs/bayes_1misc_heatmap.png`, `figs/human_1misc_heatmap_with_practice.png`,
  `figs/llm_1misc_heatmap.png`), including the outside()-unfalsifiability caveat; (2)
  "Three-way comparison on a common accuracy scale" (`\label{sec:comparison}`) embedding
  `figs/observer_scatter_graded.png`, with the P(Acc|trace) definition and the
  Bayes-ceiling/gpt-scatter reading. Discussion's closing paragraph rewritten to point at
  §comparison instead of promising it as future work.

---

## 5. Component 4 — Human data analysis (`analysis_human/`)

- **`analyze_human.py`** — sanity dashboard + accuracy by misconception (alone/paired) + plots.
  Run from repo root: `python3 analysis_human/analyze_human.py`. Filters to `recruitmentService ==
  'prolific'` done sessions (add `--include-web` for the pre-fix testing sessions).
- **`plot_human_sdt.py`** — per-participant signal detection: sorted d' dot plot + ROC space with
  iso-d' curves (mirrors the LLM fig_signal; same 1/(2N) correction so d' values are comparable).
- **`plot_human_rt.py`** — RT screening: per-trial RTs sorted by d' (3s-lock line) + median-RT-vs-d'.
- **1-misconception heatmaps (present × named "confusion matrix"), Bayes arm DONE (2026-07-20):**
  `analysis-Bayesian/plot_bayes_1misc_heatmap.py` → `bayes_1misc_heatmap.png` (copied to
  Results_combined/figs). Rows = misconception PRESENT, cols = NAMED; DIAGONAL = category A
  (agree correct; **at ε=0 all six diagonal cells are exactly 1.00**, previously ~0.9 with
  outside() weakest at 0.89), OFF-DIAGONAL = category B foils
  (disagree correct), split into TWO panels by foil refutation status: (a) refuted → off-diag
  collapses to exactly 0, (b) unsupported → off-diag 0.17–0.25 residual.
  The shared diagonal is the agree reference in both panels. Same CMAP/NORM/cell style as the
  2-misc bayes heatmap. Human counterpart `analysis_human/plot_human_1misc_heatmap.py` (`human_1misc_heatmap_with_practice.png`, n=21, diagonal solid / off-diagonal sparse) and LLM `llm_exp/make_llm_1misc_heatmap.py` (`llm_1misc_heatmap.png`, 3 regimes x 2 panels: thinking green-diagonal + refutation-sensitive, direct all-red off-diagonal, gpt-4o claim-driven RTL/outside columns) DONE 2026-07-20. 1-misc heatmap set complete.
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
- **DOT version of the 2-misc heatmaps, DONE 2026-07-21** (new files, the originals above are
  untouched): `analysis_human/plot_2misc_heatmap_dots.py` →
  `plots/human_2misc_heatmap_dots_practice.png`, and `llm_exp/make_llm_2misc_heatmap_dots.py` →
  `plots/llm_2misc_heatmap_dots.png` (3 regime rows, ~26x26in by design). Same cells and same
  mean colour, but every individual trial/item is drawn inside its cell: horizontal position =
  the 1-6 rating (ticks mark the six positions), dashed centre line = the 3.5 boundary, dots
  stack on repeats. **No Bayes version**: at ε=0 all 30 category-C cells are exactly 1.000 with
  zero spread, so there is nothing to draw.
  Why dots and not mini distributions: cells hold 1-7 values (humans: min 1, median 3, max 7;
  LLM/Bayes: C 3-6, D exactly 2). A density curve through 2 points is theatre, so the raw values
  are shown directly.
  **Finding — the cell mean was systematically hiding bimodality.** Cells whose values span ≥4
  rating points: humans 27 of the 45 cells with n≥3; haiku-direct 31/90; gpt-4o 33/90;
  haiku-thinking only 11/90. And a middling mean is almost never actual middling responses:
  humans `mean 3.5 n=6 → 1,2,2,5,5,6` and `mean 3.4 n=7 → 1,2,2,2,5,6,6`; haiku-direct
  `mean 3.5 n=4 → 1,1,6,6`; gpt-4o `mean 3.2 n=4 → 1,1,5,6`. So a grey "undecided" cell in the
  original heatmaps is usually two confident opposite answers. haiku-thinking is the only
  observer whose cells are genuinely tight and unimodal.
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
  IO evidence, RTL among lowest (3.21) despite strongest. **STALE after the ε=0 switch
  (2026-07-20):** the IO is now exactly 1.00 on every category-A item, so there is no IO gradient
  on A to invert. The human asymmetry is unchanged and still real, but it must be restated as
  "humans vary where the IO is uniformly certain," not as an inversion of IO difficulty; (3) human B false alarms are both
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
  **SUPERSEDED (2026-07-17): the extended pool (extend_pool.py, §2) now contains refutable
  sub<÷ foil items as part of the controlled refutation design — SYNB002 stays in
  synthetic_items.json for provenance only and will not be swapped into the pool.**
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

`PreReg/prereg.tex` + `PreReg/figs/` (7 PNGs, the n=40 human set). **REWRITTEN 2026-08-07** on
the pooled 5-practice cohort; see the 2026-08-07 entry in §7 for the full change list, the new
hypothesis set, and the power numbers. Mirrors the teammate's numberlink prereg
(`prereg_buffer/prereg.tex`): same styling, `\decflag` markers, and an honesty-disclosure box
(transparent/informed prereg; pilot N=40 reported in full, hypotheses locked before a
confirmatory cohort).
**Scope: HUMAN DATA ONLY, FINAL WAVE ONLY** (user's call). No LLM results anywhere; no wave
history; the ideal observer appears only as a methods device (defines foil refutation status +
establishes the task is logically decidable). ⚠️ The v1/v2 pool mixing in the
`outside_bracket_first` cells is deliberately NOT mentioned in the prereg (user instruction);
it lives in §7 here and in `analysis_human/cohorts.py`.
Structure: study info → background (bug/repair-theory framing, ideal observer as logical
oracle) → methods (learner model, pool + category tables, **refuted vs unsupported foils**,
worked example items A044 + refuted-foil B071, sampling incl. the 6+6 refutation balance,
procedure incl. the 5-trial practice block, measures) → pilot results figure-by-figure
(accuracy overview → by-misconception → category-A recognition → foils + refutation contrast →
SDT/ROC → confusion heatmaps) → **locked hypotheses H1 refutation (primary) / H2 recognition /
H3 C-vs-A equivalence / H4 position**, plus a considered-and-dropped note for the old
IO-difficulty hypothesis → analysis plan (mixed models per hypothesis, Holm within family) →
exclusions → power table → decisions box.
User compiles on **Overleaf** (copy folder as-is; figures referenced as `figs/<exact-name>`;
NO local TeX, don't install one). Writing style: **no em dashes** (user: "screams AI").

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
- LLM experiment: all 480 items × 3 regimes, 0 errors; all figures + `report/report.tex` (now
  fully on the 480 pool, 2026-07-22) written.
- **Pre-registration draft** written (`PreReg/prereg.tex`, §6) — hypotheses/exclusions flagged,
  awaiting discussion.
- **Pool extended to the 480 refutation design (2026-07-17):** 487 items on disk (480
  sampling-eligible + 7 preserved ambiguous), foil refutation status controlled in B and D,
  per-form 6 refuted + 6 unsupported foils with every rule in both statuses (see §0/§2).
  All three sampling mirrors updated + verified over 500 seeds each; LLM leak guard extended
  (foil_status, io_foil_marginal) and its tests pass over 487; practice items re-verified
  not in pool; difficulty baseline re-run. This is the pool for the confirmatory wave; the
  pilot ran on the original 240 (all preserved).
- **Practice trials built AND DEPLOYED (2026-07-16/17, commit 26af20f):** 3 feedback practice
  trials before the main task (PracticeView.vue + practice_items.json +
  make_human_practice_items.py; instructions updated). This implements the "practice trials
  with feedback before the confirmatory run" note from results.tex §3. Verified: items
  model-checked (IO marginals: P1 named 0.946, P2 foil 0.000, P3 named 0.863), frontend
  compiles, and the live bundle contains the items. Deploy note: the first deploy attempt
  failed with an SSH i/o timeout to the lab server ("create the remote folders" step);
  `gh run rerun <id> --failed` fixed it — transient, not a code issue. User still to eyeball
  the deployed practice flow.

**DONE (2026-07-21):**
- **Dot versions of the 2-misc heatmaps (human + LLM)**, showing every individual value inside
  its cell instead of only the mean. Full description and the bimodality numbers in §5. Headline:
  the cell mean was hiding two-sided splits in about 60% of human cells with n≥3 and in a third
  of LLM cells, except for haiku-thinking. Relevant to how §4 of results.tex describes those
  heatmaps: "grey cell = undecided" is wrong, it is usually "two confident opposite answers".
- **LLM 1-misc dist figures restyled**: haiku-thinking blue → red, plus per-regime markers
  (thinking star, haiku-direct square, gpt-4o circle) because the three curves coincide exactly
  on many panels. `llm_exp/make_llm_1misc_distributions.py`.
- ⚠️ **Open design question, discussed but NOT built**: a single combined human × LLM × Bayes
  figure for the 1-misc distributions. Blocked on a scale decision. The three arms are not the
  same object today (LLM = 6-point frequency polygon over ratings, Bayes = continuous KDE over
  [0,1], human = ALREADY BINARY because 1-6 was too noisy at n≈21 per cell), the unit differs
  (item for LLM/Bayes, trial for humans), and dist_B row 3 is B-only for LLM/Bayes but pooled
  over B+D for humans. Recommendation on the table: make the common scale **P(agree)** (humans
  and LLMs = proportion rated ≥4, Bayes = proportion with marginal > 0.5), x-axis = the six
  misconceptions, LLM solid / human dashed / Bayes dotted, Wilson intervals on the two sampled
  observers. Note Bayes is a flat line at 1.0 (A) and 0.0 (B) at ε=0.

**⚠️⚠️ MODEL/POOL FORK (v2 staged locally, 2026-07-22) — READ BEFORE ANY CROSS-OBSERVER ANALYSIS:**
The working tree holds an UNCOMMITTED **outside_bracket_first v2**: pattern_matcher.py adds
Table 6 (`a OP Y` / `Y OP a`, op cannot fire, recurse_Y only), traces.py/valid_actions.py mark
Table-6 ops invalid, and `_next_dags` BLOCKS bracket recursion for outside()-learners while any
literal-literal op remains outside. This converts outside() from a PERMISSION (may work outside
first) to a PREFERENCE (must finish outside before going inside), matching the belief-statement
wording, and makes outside() FALSIFIABLE (an inside-first step with outside work available now
eliminates every outside()-containing profile at ε=0). It supersedes-in-v2 the "outside() is
UNFALSIFIABLE" caveat in §2 (the mechanism there described v1 and sketched a different fix).
Because v2 changes which traces outside()-profiles generate, all three stimulus_pool.json
copies were regenerated at 16:57 (one hour before the terra LLM runs); misconception_difficulty
.json and analysis-Bayesian/b_item_marginals.json were also regenerated. User confirmed the
bracket change was known work, but the session that did it never updated HANDOFF.
Versus HEAD the v2 pool has: 8 ids swapped (HEAD-only: B012 B032 D014 D018
D022 D041 D043 D056; working-only: B057 B121 D003 D007 D016 D020 D029 D126 — note this
resurrects 6 of the 7 dropped-ambiguous ids), and of the 472 common ids **437 have different
expressions/traces**, 88 a different probed_misconception, 96 a different foil_status
(statement_correct matches on all common ids). It is a DIFFERENT POOL, not an 8-item patch.
Alignment of the arms as of now:
- **Humans (all cohorts incl. pilot 2) + Bayes** (dashboard/bayes_per_item.json,
  analysis-Bayesian/b_item_marginals.json) = the OLD/HEAD pool. The live site deploys from
  git, and the pool change was never committed, so participants still see the OLD pool.
- **Current LLM raw runs** (raw_*_all480.jsonl, incl. both haiku arms) = the NEW pool. This
  fully explains the "haiku drift" (hit 0.92→0.975 etc.): different stimuli, not model change.
Contaminated outputs (id-joins across the fork): observer_scatter_{binary,graded}_pilot2
(82/456 human trials misgrouped via the new pool's probed_misconception + 4 dropped);
human 1misc heatmap/distributions foil_status panels IF joined via base-task pool (94 B/D
pilot-2 trials would mislabel — scripts should switch to the trial-embedded foil_status).
SAFE (trial-embedded fields only): analyze_human, plot_human_sdt/rt, both human 2misc figures,
the cohort comparison table below (incl. FA refuted/unsupported). LLM-only figures are
internally consistent but describe the NEW pool, so LLM-vs-human/Bayes readings are cross-pool
and provisional. **RESOLVED 2026-07-28:** the fork was INTENTIONAL — the user had told the
07-22 session to replace the pool entirely; that session staged everything locally but never
committed, deployed, or updated HANDOFF, which is why pilot-2 (19 people) ran on v1.
**USER DECISION: v2 is the only pool from now on. v1 is retired** (human records kept solely
to pay the still-unpaid pilot-2 bonuses and re-downloadable from Firebase; no further v1
analysis planned; no v1 LLM re-runs).
**v2 DEPLOYED 2026-07-28**: model files (pattern_matcher/traces/valid_actions) + all 3 pool
copies + misconception_difficulty.json + b_item_marginals.json. Pre-deploy verification:
pool structure exact (120/category, B cells 60 of size 2, D cells 120 of size 1, zero
ambiguous statuses), Python sample_form AND JS sampleForm.js both 500/500 seeds balanced on
v2, practice_items.json regenerated under v2 came out byte-identical (P1-P5 same; P3's
outside() IO marginal is now 1.000 under preference semantics), local bundle grep shows
v2-only content.
⚠️ Cohort bookkeeping: the upcoming confirmatory participants are the V2 COHORT but they also
see 5 practice trials, so cohorts.py 'practice5' alone canNOT separate them from pilot-2
(v1, also 5 practice items). Split on session start after the 2026-07-28 v2 deploy, or
content-match trials against the v2 pool (every 24-item form virtually surely contains
v2-only content). Wire a 'v2' cohort into cohorts.py when the first v2 data arrives.
STILL PENDING after cutover: rebuild dashboard/bayes_per_item.json + dashboard on v2 (its
Bayes data is still v1); regenerate the pilot-2-facing scatters only if ever needed (v1,
low priority now); prereg + results.tex/report.tex reconciliation onto v2 numbers.

**DONE (2026-08-10, corrected a wrong claim about the blue RTL/outside() cell):**
- ⚠️ **results.tex previously claimed "no bracket ever appears in the trace" for the one blue
  cell (RTL present, outside_bracket_first named). This was WRONG and has been corrected.**
  Checked all 4 pool items behind that cell (B028, B058, B110, B111) directly against the
  model (`_next_dags`, same technique as the D073 check earlier): **every one contains a
  bracket.** They split 2/2: B028 forces the same move at every step (claim never testable);
  B058 happens to resolve outside-first, consistent with but not proof of the claim; **B110
  and B111 are actively refuted** — at the first step a legal outside-only move existed
  (e.g. `4+10` in B110, not touching the bracket) and the student went INTO the bracket
  instead, which directly contradicts what an outside()-preference learner would do. Fixed
  paragraph in results.tex §sec:human-1misc-heatmap. Caveat added: can't check whether
  refutation matters WITHIN this cell yet, since B110/B111 each drew only 1 rating in the
  current cohort (n too thin per item, only cell-pooled n=13 is meaningful right now).

**DONE (2026-08-10, NEW 1-misc dots figure + "wrong-side majority" marker, both dot heatmaps):**
- **User request, walking figures one at a time**: the 1-misc present×named heatmap
  (`human_1misc_heatmap_combined_5practice.png`) never had a dots/raw-trial version, unlike
  the 2-misc heatmaps. Built one: **new script `analysis_human/plot_1misc_heatmap_dots.py`**,
  same visual language as `plot_2misc_heatmap_dots.py` (dot strip per cell, amber = split).
  Reuses `IDS/SHORT/CMAP/NORM/DARK_AT` from `plot_human_1misc_heatmap.py` and
  `STRIP_HALF/DOT_DY/.../is_split/rating_x` from `plot_2misc_heatmap_dots.py`; own
  `build_value_grid()` (present×named -> [ratings], combining refuted+unsupported).
- **Result: 34 of 36 cells (incl. all 6 diagonal) contain both agree AND disagree responses**,
  15 substantially split — worse bimodality than either 2-misc panel. This DIRECTLY
  contradicts/corrects a claim already sitting in HANDOFF and results.tex ("errors are spread
  thinly... mercifully boring") — see below, now fixed in results.tex.
- **New second marker, user-designed: "wrong-side majority."** User's ask: box cells where a
  clear majority (they suggested 2/3) landed on the WRONG side of that cell's ground-truth
  correct answer — a different question from "split" (is the mean untrustworthy) — this one
  asks "did most people share a genuine, replicated misconception." Added
  `is_wrong_majority(vals, correct_is_agree)` + `WRONG_EDGE` (blue, `#1959c9`) to
  **`plot_2misc_heatmap_dots.py`** (shared module both dots scripts import from), same
  min-n=3 bar as split, threshold >=2/3 on the wrong side. Validated 2/3 isn't a formal
  significance bar (checked: ~10-19% false-positive rate under pure 50/50 chance at n=10-15
  via binomial) but is a reasonable descriptive "clear lean" flag, which is what was asked
  for. Split and wrong-majority are mutually exclusive by construction (2/3 majority implies
  <=1/3 minority). `draw()` in both scripts now takes/derives `correct_is_agree` per cell
  (1misc: `i==j`; 2misc: constant per panel, True for C, False for D) and draws blue OR amber,
  never both.
- **1-misc result: only 1 of 30 off-diagonal cells is a genuine wrong-majority** —
  `same_priority_rtl` present, `outside_bracket_first` named, 9/13 agree. This SHARPENS and
  partly retracts what was said earlier in this same session about "the RTL/outside()
  cluster": all 5 cells crossing 3.5 do involve RTL/outside(, confirmed still true), but 4 of
  the 5 are near-50/50 splits (6-7 vs 5-9), not shared leans; only the one cell above is a
  real, replicated false belief. **results.tex §sec:human-1misc-heatmap rewritten** to state
  this precisely instead of the old "spread thinly / mercifully boring" line, and the new
  dots figure added right after (`fig:human1hdots`).
- **2-misc result: 0 of 30 category-C cells are wrong-majority (humans never confidently
  reject a true partial explanation), 7 of 60 category-D cells are.** Of those 7, **3 involve
  the add_before_mul/sub_before_mul pair in either direction** (named sub<× with add<×
  present x2, named add<× with sub<× present x1) — this is INDEPENDENT, cell-level
  confirmation of the tentative "×-family" pattern from the earlier `family_followup.py`
  exploratory pass (which found add<× named+sub<× present FA .51/37, sub<× named+add<×
  present FA .48/29, but called it underpowered at n=59). 2 more involve
  `outside_bracket_first` present + false `add_before_div` endorsement. Remaining 2 are
  isolated RTL foils, no shared present-side pattern. **results.tex's existing 2-misc-dots
  paragraph extended** with this (new paragraph after the amber-split paragraph, before the
  figure).
- **Figures copied**: refreshed `human_2misc_heatmap_dots_5practice.png` (now has blue boxes)
  into `PreReg/figs/`, `Results_combined/figs/`, `analysis_human_practice_2/`; NEW
  `human_1misc_heatmap_dots_5practice.png` added to `Results_combined/figs/` and
  `analysis_human_practice_2/` **per explicit user request** ("move the figure into results
  combined") — **NOT added to PreReg** (user didn't ask for that one there; only asked
  results_combined). Ask before adding it to PreReg too if that comes up.
- ⚠️ **NOT YET COMMITTED**: `analysis_human/plot_1misc_heatmap_dots.py` (new),
  `analysis_human/plot_2misc_heatmap_dots.py` (blue-marker addition), `results.tex`, all
  figure copies. All local only.

**DONE (2026-08-10, EVERYTHING REGENERATED AGAIN AT n=62 + SDT plot de-IDed):**
- **Data pulled again.** `data/real-all-main-data.json` now 107 complete prolific sessions.
  `practice5` cohort grew 59 -> **62** (v1 stays 20, v2 grew 39 -> 42, so **3 new v2
  completions**, not the 1 the user expected — flagged and confirmed with the user before
  regenerating, see AskUserQuestion in-session; proceeded at the real n=62).
- ⚠️ **`cohort_count.py`'s per-bucket printout is misleading**: its strict "match==0" rule for
  labeling a session 'v1' almost never fires for the 5-practice v1 wave (those sessions
  incidentally match 1-5 v2-pool items by coincidence), so v1 sessions show up as
  `MIXED(1-5/24)` buckets instead of a clean 'v1' line. The AUTHORITATIVE v1/v2 split is
  prereg_stats.py's rule: match==24/24 -> v2, anything else -> v1. Don't eyeball
  cohort_count.py's raw buckets for the v1 count again; sum the MIXED buckets or just run
  prereg_stats.py.
- **`analysis_human/plot_human_sdt.py` panel (a) no longer labels rows with participant
  seedIDs** (user request — "don't have participant IDs on the side, just say participants").
  Changed `set_yticklabels([s['seed'] ...])` to `set_yticks([])` +
  `set_ylabel("participants (sorted by d')")`. Applies to all cohorts, not just practice5.
- **All 13 human figures + both observer scatters regenerated at n=62**, copied into
  `PreReg/figs/` and `Results_combined/figs/` exactly as the n=59 round. **Both `prereg.tex`
  and `results.tex` fully updated to n=62** (27 + 20 checked replacements respectively,
  including the power table and sample-size decision prose).
- **n=62 headline numbers:** overall acc .677; A .739 B .653 C .712 D .602 (D still weakest).
  SDT pooled hit .726 FA .372 **d' 0.93 crit -0.14**; per-participant d' -0.67..2.77 median
  1.14, **52/62 above 0**. Refutation FA **.288 vs .457**, within-participant dz **0.47**
  (up slightly from 0.45), right direction 41/62.
  **A vs C mean rating is now EXACTLY EQUAL: 4.26 vs 4.26** (dz -0.00) — the gap has been
  monotonically converging to zero across every sample size: +0.02 (N=40) -> -0.06 (N=59) ->
  -0.00 (N=62). This strengthens the H3 equivalence framing considerably; said so explicitly
  in both docs.
  **Bimodality got worse again:** ALL 30/30 category-C cells now contain both agree and
  disagree responses (was 29/30); D substantially-split cells 28 -> 31 of 60.
  H2 weak/strong flipped slightly: weakest still add<× (.65) then RTL (.69); strongest still
  add<÷ (.82). H4 position dz shrank 0.20 -> 0.17 (.74 vs .68).
- **Power table revised** (dz 0.47 for H1 now needs only **N=50**, down from 54 — the
  "N=60 is break-even" framing from the n=59 round no longer applies and was removed).
  H4 dz dropped to 0.17, so its N requirement grew to 366/562. **New candidates: 120 / 200 /
  600** (was 60(not-rec)/120/200/400). At N=120, 5 of 6 H2 misconceptions are now powered
  under Holm (only add<× needs 174); at N=200 all six H2 cells are covered; N~600 needed to
  additionally cover H4.
- Scripts: same `update_prereg_62.py`/`update_results_62.py` pattern as the n=59 round
  (scratchpad, not committed).
- ⚠️ **NOT YET COMMITTED**: `analysis_human/plot_human_sdt.py` (the de-ID change),
  `PreReg/prereg.tex` + `PreReg/figs/*`, `Results_combined/results.tex` +
  `Results_combined/figs/*`, `analysis_human_practice_2/*`. All figure/doc regen is local
  only; nothing pushed (none of it affects the live deploy anyway — only `src/` does).

**DONE (2026-08-10, practice set redesigned + DEPLOYED, commit d73ddb6):**
- **User caught a real design gap**: P1-P5 were structurally A / B / C(first) / C(first) /
  C(second) — THREE category-C items (two of them redundant, both "first") and **zero
  category D**. Fixed per the user's exact spec: P1/P2/P3 untouched; **P4 = the old P5
  content** (sub_before_mul + same_priority_rtl present, probed same_priority_rtl, SECOND);
  **P5 = a new category D item** (add_before_mul + sub_before_div present, probed
  sub_before_mul — absent, a foil, disagree correct).
- `base-task/make_human_practice_items.py`: added `_find_flat_foil(pair, foil,
  required_ops, seedrange, want_refuted=True)`, a sibling of `_find_flat_pair` for the
  D case — searches for a trace where `pair` fires cleanly and PREFERS (with a fallback)
  a foil whose ideal-observer marginal is < 0.15, i.e. actively refuted rather than merely
  unsupported, matching P2's quality and echoing H1 (the refutation contrast is now the
  prereg's primary hypothesis). Found one on the first pass: `sub_before_mul` marginal
  0.000 on `2 × 10 + 12 - 5 ÷ 3`. **Verified the actual mechanism** (didn't just trust the
  marginal): at state `2 × 22 - 5 ÷ 3` a learner with sub_before_mul had `22 - 5` (subtract)
  available right next to `2 × 22` (multiply) and the multiply was taken instead — genuine
  contradiction, not dilution. Feedback text names this specifically.
  Design constraint worth remembering: removing the old P4 would have dropped
  `add_before_mul`/`sub_before_div` from ever appearing as PRESENT in any practice trace, so
  the new D item's pair was deliberately chosen to be exactly those two, preserving the
  "all six misconceptions appear as present somewhere" invariant. Named/probed coverage
  becomes 5 of 6 (only `sub_before_div` never named now, was `sub_before_mul` before) — a
  straight swap, not a regression, and unavoidable with only 5 items naming 6 rules.
  P3's `outside_bracket_first`-fires-first structure was left alone (not touched, matches
  user's "1,2,3 as is").
- Frontend needed ZERO changes: `isProbedStep` in PracticeView.vue already only bolds a step
  when its misconception equals `probed_misconception`, so a D item (where probed is absent
  from both present misconceptions) naturally renders with no bold marker.
  Verified: no duplicate expressions/names across P1-P5, no leak into the pool, `npm run
  build` clean, new content confirmed present in the built JS bundle by grep.
- **Committed (`d73ddb6`) and pushed with the user's explicit go-ahead** ("push, ill look at
  it in the deployed link"). Deploy workflow watched end-to-end, all steps green including
  rsync. Only `base-task/make_human_practice_items.py` +
  `src/user/data/practice_items.json` were committed — the large pile of other uncommitted
  session work (prereg, results.tex, analysis figures) was deliberately left out of this
  push, scope-matched to what was asked.

**DONE (2026-08-07 evening, EVERYTHING REGENERATED AT n=59):**
- **Data pulled.** `data/real-all-main-data.json` now has **104 complete prolific sessions**:
  24 (0 practice, v1) + 21 (3 practice, v1) + 20 (5 practice, v1) + **39 (5 practice, v2)**.
  The `practice5` cohort is **n=59** (19 of the user's 20 new people completed).
- **All 13 figures regenerated** at `--cohort practice5` into `analysis_human_practice_2/`,
  then copied to `PreReg/figs/` (clean names) and `Results_combined/figs/` (`_5practice`
  names). Includes both observer scatters (remember `--scoring binary` is a SEPARATE run;
  the default only writes the graded one).
- **`PreReg/prereg.tex` fully updated to n=59** (27 checked replacements + power table +
  sample-size prose). **`Results_combined/results.tex` human sections also updated** (18
  replacements) because the figure copies would otherwise have contradicted its prose.
- **n=59 headline numbers:** overall acc .678; **A .729 B .667 C .715 D .602**; alone .698
  paired .658; mean ratings A 4.23 C 4.28 B 2.86 D 3.05. SDT pooled hit .722 FA .366
  **d' 0.93 crit -0.12**; per-participant d' -0.67..**2.77** median 1.11, **50/59 above 0**,
  10 at >=2, best 92% acc. **Refutation FA .282 vs .449** (B .28/.38, D .28/.51),
  within-participant .167 SD .368 **dz 0.45**, right direction 39/59.
  Cat-A P(agree) .64-.81 (weakest add<× .64 then **RTL .68**; strongest add<÷ .81).
  Cat-B FA max .47 (outside() by-present). C position .746 vs .684 dz 0.20.
  2-misc: 90/90 cells, median 7 trials. 1-misc diagonal 3.90-4.49; off-diagonal 30 cells
  n 10-15, only **5 above 3.5**.
- **What MOVED vs n=40, and matters:**
  (1) **outside() is no longer a weak category-A cell** (.65 -> .71); **RTL is now the second
  weakest** (.75 -> .68). Any prose naming the weak A claims must say add<× and RTL.
  (2) **D accuracy fell .637 -> .602**, now clearly the weakest category.
  (3) **H1 effect shrank, dz 0.51 -> 0.45**, so required N rose 43 -> **54**. Since the pilot
  is itself 59, a confirmatory N=60 is now BREAK-EVEN. The prereg's recommended candidates
  were changed to **120 / 200 / 400** and it says explicitly that 60 is not recommended.
  (4) H3's A-vs-C gap CHANGED SIGN (+0.02 -> -0.06, C now nominally above A). Still ~zero, so
  the equivalence framing is if anything better justified; the prereg notes the sign flip.
  (5) rho(median RT, d') .06 -> .26, so the "time on task is unrelated to performance" line
  was softened to "only weakly related".
  (6) Bimodality got WORSE with more data: substantially-split D cells 16 -> **28** of 60,
  and 52/60 D cells now contain both agree and disagree responses.
- ⚠️ **Power numbers to reuse:** H1 dz 0.45 -> N=54 (a=.05). H2 per-misconception (a=.05 /
  a=.00625): add<÷ .81 -> 23/37, sub<÷ .78 -> 30/46, sub<× .75 -> 38/60, outside() .71 ->
  56/87, RTL .68 -> 77/120, add<× .64 -> 130/201. H4 dz 0.20 -> 265/407.
- **Family-confusion re-run at n=59 (see the entry below): conclusion UNCHANGED.** Leading
  sibling .287 vs neither .353 (still backwards). Target sibling .387. The ×-only pattern
  firmed up: add<× named with sub<× present **.51 (19/37)**, sub<× named with add<× present
  **.48 (14/29)**, vs ~.35 baseline; the ÷ counterparts stay flat (.32, .31). Still 2 cells,
  still not reportable, but now worth a pre-specified relation-level test if pursued.
- Scratchpad scripts that did the work (not committed): `prereg_stats.py`,
  `family_confusion.py`, `family_followup.py`, `update_prereg.py`, `update_results.py`,
  `gen_examples_tex.py`, `pick_examples.py`, `cohort_count.py`.

**DONE (2026-08-07, FAMILY-CONFUSION ANALYSIS — a NEGATIVE result, plus 2 retired findings):**
- **User hypothesis tested and NOT supported.** The idea: category-D green cells reflect
  operator-family confusion, e.g. sub<× present makes people endorse a sub<÷ claim (share the
  LEADING operator). Tested by holding the named foil FIXED and varying what is present, which
  is the only comparison that controls for each foil's own baseline endorsement rate.
  **All four operator foils go the OPPOSITE way:** leading-sibling present vs neither ->
  add<× .35/.33, add<÷ .33/.39, sub<× .25/.33, sub<÷ .19/.30; pooled **.26 vs .34**. Having the
  leading sibling in the trace REDUCES false alarms.
- **Cause of the apparent effect: a refutation confound.** Leading-sibling trials are 58%
  refuted vs ~45% for the others, and refuted foils draw far fewer FAs (.25 vs .45). Splitting
  on status the family relations wash out (unsupported half: leading .42 / target .42 /
  neither .45).
- **What IS there (weak, 2 cells only):** the TARGET-operator family (both rules attack the
  same operator) is elevated for MULTIPLICATION only: add<× named with sub<× present **.52**
  (13/25) and sub<× named with add<× present **.44** (8/18), vs ~.33 baseline. The division
  counterparts show nothing (.25, .29). Pooled target vs neither is only .36 vs .34, so this
  rests entirely on two small cells. Do not report as a finding yet.
- ⚠️ **TWO PREVIOUSLY RECORDED FINDINGS RETIRED** (both small-n artifacts, do not repeat them):
  (1) "false add<× endorsed when add<÷ present" (§5 candidate finding 2) does NOT replicate at
  n=40: .35 against a .33 baseline. (2) An apparent RTL<->outside() confusion (FA .59 in a
  B-only first pass) evaporates when the foil is held fixed: named RTL with outside() present
  .42 vs .44 absent; named outside() with RTL present .39 vs .34 absent.
- **Bimodality CONFIRMED and quantified at n=40:** 27 of 30 occupied C cells and 41 of 60 D
  cells contain BOTH agree and disagree responses; 11 C and 16 D cells are *substantially*
  split (n>=3 and >=1/3 of trials on the minority side). The user's own example cell
  (pair sub<×+add<×, sub<÷ named, mean 4.00) is literally [2,5,5].
- Scripts (scratchpad, not committed): `family_confusion.py`, `family_followup.py`.
- **If pursuing this**: test at the RELATION level pooled over foils, not cell by cell. At
  n=55 the target-sibling group reaches ~105 trials, roughly 80% power for a .19 difference.
  Cell-level will still be n~5 even at n=60.

**DONE (2026-08-07, prereg round 2 — figures + examples):**
- **`analysis_human/plot_2misc_heatmap_dots.py` now marks split cells** with an amber outline
  (new `is_split()`: n>=3 and minority side >=1/3). Criterion chosen because "both sides of
  3.5" would have outlined 27/30 and 41/60 cells, too many to be informative. Footer text and
  a stdout summary added. Regenerated and refreshed in all three places
  (analysis_human_practice_2/, Results_combined/figs/, PreReg/figs/).
- **Prereg: new subsection "The cell means are hiding two-sided splits"** (§sec:pilot-bimodal)
  embedding the dots figure, with the 27/30 + 41/60 counts, three worked split cells, and the
  methodological consequence (no confirmatory analysis uses a cell mean as its unit).
- **Prereg: the 2-item example box replaced by a FOUR-CATEGORY box** (A039 / B061 / C025 /
  D073), each with erroneous steps highlighted in amber and, for C, the probed step marked
  bold, exactly like the practice trials. Highlighting is model-derived, not hand-marked:
  generated by `scratchpad/gen_examples_tex.py` reusing `_error_steps` from
  `make_human_practice_items.py`, so traces match the pool byte-for-byte. New preamble macro
  `\hlstep` + color `stephl`. ⚠️ Only 2 of 120 D items have exactly one cleanly attributable
  error per misconception (D073 and D102), so the D slot has almost no alternatives.
- ~~data file stale~~ RESOLVED the same day: the user pulled, and everything was regenerated
  at n=59. See the 2026-08-07-evening entry above.

**DONE (2026-08-07, PREREG REWRITTEN ON THE n=40 HUMAN ARM):**
- **`PreReg/prereg.tex` rewritten end to end.** Scope decision (user): the prereg reports
  **human data only** and **only the final wave**, i.e. the pooled 5-practice cohort n=40.
  No LLM results, no wave history, no earlier cohorts. The Bayesian ideal observer STAYS in
  as a methods device only (it defines refuted vs unsupported foil status, and establishes
  the task is logically decidable: exactly 1.0 present, exactly 0 refuted, ~0.2 unsupported).
- ⚠️ **USER INSTRUCTION: the prereg must NOT say that the outside_bracket_first cells mix v1
  and v2 pool semantics.** That caveat is internal to this repo only. It is recorded here and
  in `analysis_human/cohorts.py`; keep it out of the prereg and any external document.
- **Figures**: `PreReg/figs/` now holds the 7-figure n=40 human set (old N=24 ones replaced):
  accuracy_group_category, accuracy_by_misconception (both regenerated at --cohort practice5),
  human_signal_detection, human_1misc_dist_A, human_1misc_dist_B, human_1misc_heatmap,
  human_2misc_heatmap (copied from Results_combined/figs, `_5practice` suffix dropped).
  All 7 \includegraphics refs verified to resolve; labels/refs/braces/math checked.
- **Hypotheses REPLACED.** The old H1-H4 were built on the N=24 no-practice pilot and three of
  them no longer hold at n=40: H1 (A worse than B) REVERSED (A .73 > B .66), H2 (C rated above
  A) GONE (4.22 vs 4.25, dz 0.02), H3 (difficulty tracks the ideal observer) DEAD (no IO
  gradient left at eps=0/v2). New locked set:
  **H1 primary/directional = refutation contrast** (FA .25 refuted vs .45 unsupported;
  within-participant diff .20, SD .39, **dz 0.51**, right direction in 29/40);
  H2 = category-A P(agree) above chance per misconception (.73 overall, .62-.82);
  H3 = C rating equivalent to A (TOST, SESOI flagged);
  H4 = C early vs late (.73 vs .67, dz 0.20, secondary).
  The old IO-difficulty hypothesis is explicitly documented as considered-and-dropped.
- **Multiplicity structure**: H1 is the SINGLE primary test (alpha .05, uncorrected); H2/H3/H4
  are one secondary family of EIGHT tests (H2 contributes 6, one per misconception),
  Holm-corrected, worst case alpha .00625. Fixing this removed an inconsistency in the first
  draft, which said "family of four" in one place and "across the six" in another.
- **Power computed from the pilot** (90%, two-sided = conservative for the directional ones):
  H1 dz 0.51 -> **N=43**; H2 per-misconception at .05/.00625: 22/34 (sub<÷ P=.82), 38/60
  (RTL .75), 113/174 (outside() .65), 179/275 (add<× .62); H4 dz 0.20 -> 265/407.
  N candidates offered: 60 (H1 + strongest H2 cells), 120, ~300 (H1 + all of H2, not H4).
- **Exclusion gate checked against the data:** the candidate below-chance rule (<=7/24,
  one-sided p=.032) excludes **0 of 40** (weakest participant 9/24 = .38), so it is a safety
  net not a filter. Trial RT: min 3.7s, median 18.2s, 4.8% over 60s, 1.5% over 120s, max 554s
  (120s candidate cutoff).
- **Still \decflag in the prereg (user must decide before submitting):** confirmatory N +
  stopping rule, H3 equivalence bounds, performance gate, RT cutoff, registration scope,
  registry/timeline, authorship.
- Stats source of truth for all of the above: recomputed from `data/real-all-main-data.json`
  at `--cohort practice5`, not copied from results.tex.

**DONE (2026-08-07, cohort bookkeeping):**
- **User decision: human figures stay POOLED at n=40** (both 5-practice waves together), not
  split by pool version. Confirmed against the data: 85 complete Prolific sessions total =
  24 (0 practice, v1) + 21 (3 practice, v1) + 20 (5 practice, v1) + 20 (5 practice, v2).
- `analysis_human/cohorts.py` docstring corrected: `practice5` was described as "pilot 2 only"
  but it filters on practice count alone, so it returns all 40. Added the v1/v2 discriminator
  (content-match 24 trials against base-task/stimulus_pool.json: v2 = 24/24, v1 <= 5).
- ⚠️ Consequence to keep reporting internally: pooled `outside_bracket_first` cells mix v1
  permission traces with v2 preference traces (category-A accuracy .55 v1 vs .75 v2).
  Everything else pools cleanly (d' 0.88 v1 / 1.04 v2, FA refuted .26/.24 vs unsup .48/.42).

**DONE (2026-07-28, results.tex fully reconciled):**
- **results.tex rewritten end to end** on the current data (see the Results_combined entry in
  §1 for the section-by-section content): v2/ε=0 Bayes oracle numbers, terra regime set,
  pooled 5-practice humans n=40 with a two-wave/pool-revision sentence in §3's intro, updated
  captions, all figure references pointed at the `_5practice` set. Old `_with_practice`
  figures deleted from figs/. Two claims the rewrite RETIRED because the data no longer
  supports them: (1) "outside() evidence is systematically weakest" (v2 made brackets exactly
  decidable), (2) "best human beats every LLM regime" (best d' 2.35 < thinking's 2.83).
  Two NEW headline claims added: humans are the only non-ideal observer with a refuted vs
  unsupported FA drop (.25 vs .45; all three LLM regimes flat), and the human accuracy
  profile is the flattest across groups while LLMs are spiky.
  ⚠️ report/report.tex (LLM report) is still on gpt-4o/487 prose — NOT yet reconciled.

**DONE (2026-07-28, v2 batch collected + POOLED 5-practice analysis):**
- **20 new participants on the v2 pool** (first post-cutover batch; every session content-matched
  100% against v2 — zero mixed). Also: pilot-2's straggler finished, so the v1 5-practice cohort
  is now 20 (that 20th person is NOT yet bonused; will surface in make_bonus_list together with
  the 20 new people).
- **All 13 figures in `analysis_human_practice_2/` regenerated as the POOLED 5-practice cohort
  (n=40, v1 pilot-2 + v2 batch), same filenames** (user's call: one 5-practice picture; the
  `_pilot2` suffix is historical). Cross-pool joins made safe first: 1misc heatmap/distributions
  now use trial-embedded foil_status (pool join only as fallback); plot_observer_scatter.py
  groups human trials by embedded (probed, category, statement_correct) via new
  `group_scores_direct` (no id join, fixes the 82 misgrouped v1 trials);
  `dashboard/bayes_per_item.json` REBUILT on v2 (Bayes axis + dashboard data now v2; dashboard
  index.html itself still needs assemble/build rerun).
- **Pooled results (n=40):** A .73 B .66 C .70 D .64; individual d' mean 1.10, median 1.18,
  range −0.67..2.35, 35/40 above 0. **Cohort split (v1 n=20 / v2 n=20):** A .72/.74, B .64/.68,
  C .69/.71, D .62/.66, d' 0.88/1.04, crit −0.11/−0.08, FA refuted .26/.24 vs unsupported
  .48/.42 — the practice effect fully replicates on the new pool. The one big cohort
  difference is exactly where v2 changed semantics: **outside()/A accuracy .55 (v1) → .75
  (v2)** (forced outside-first traces make the misconception visibly diagnostic), while
  outside()-named FA barely moved (.38/.33). Keep reporting bracket cells per-cohort; pooled
  bracket cells mix v1 permission traces with v2 preference traces.

**DONE (2026-07-28, pilot 2 collected + analyzed):**
- **19 new Prolific participants on the 5-practice-trial flow** (user launched ~20, one never
  finished; data pulled by the user). Cohort fingerprint: number of recorded practice items
  (3 = Jul-17 cohort n=21, 5 = pilot-2 n=19, 0 = original pilot n=24) via NEW
  `analysis_human/cohorts.py` (`n_practice_items`, `in_cohort`); a `--cohort practice5`
  choice was added to analyze_human.py, plot_human_sdt.py, plot_human_rt.py,
  plot_human_1misc_heatmap.py, plot_human_1misc_distributions.py, plot_2misc_heatmap.py,
  plot_2misc_heatmap_dots.py AND analysis-comparison/plot_observer_scatter.py
  (file suffix `_pilot2`; all fully wired).
- **Figures in NEW `analysis_human_practice_2/`** (user renamed it from analysis_human_pilot2;
  13 files): sanity, accuracy_group_category, accuracy_by_misconception,
  human_signal_detection_pilot2, human_rt_pilot2, human_1misc_heatmap{,_combined}_pilot2,
  human_1misc_dist_A/B, human_2misc_heatmap{,_dots}_pilot2,
  observer_scatter_{binary,graded}_pilot2.
- ⚠️ **LLM raw data REPLACED (discovered 2026-07-28, run by the user):**
  `llm_exp/results/` now holds `raw_{haiku_thinking,haiku_direct,gpt56terra_direct}_all480.jsonl`
  (480 rows each); the old `raw_*_all487.jsonl` set is GONE and **gpt-4o was replaced by
  `openai/gpt-5.6-terra` (direct)**. plot_observer_scatter.py's REGIMES updated accordingly
  (label 'gpt-5.6-terra (direct)'). User confirmed 2026-07-28 they ran the terra model
  themselves. All llm_exp figure scripts were already terra-wired; all 15 plots regenerated and
  snapshot copies refreshed 2026-07-28 (see §4 CURRENT). Tex prose (report.tex, results.tex)
  still describes gpt-4o and awaits reconciliation.
- **Checkpoint result, by cohort (acc by category / pooled SDT):**
  no-practice n=24: A .49 B .61 C .64 D .66, d' 0.51, crit +0.09;
  3-practice n=21: A .61 B .71 C .56 D .72, d' 0.79, crit +0.18, FA refuted .21 vs unsup .36;
  5-practice n=19: A .73 B .64 C .68 D .60, d' 0.84, crit −0.12, FA refuted .26 vs unsup .50.
  Reading: practice moved A far off chance (.49→.73, hit rate .57→.71) and individual d'
  tightened (range −0.67..2.35, 15/19 above 0, median 1.18). BUT the 3→5 step is mostly a
  CRITERION shift toward agreeing (+0.18→−0.12: A/C up, B/D down, d' only 0.79→0.84).
  Likely cause: the practice answer keys are now 4 agree / 1 disagree (P1 agree, P2 disagree,
  P3/P4/P5 agree). If there is a practice v3, balance the keys (e.g. add a D-style disagree
  trial). Refutation contrast (H5 direction) present in both practice cohorts and larger in
  pilot 2 (.26 vs .50).
- **Bonuses PAID + LEDGERED 2026-07-28 for BOTH practice cohorts** (user paid on Prolific,
  `--mark-paid` recorded): Jul-17 3-practice cohort 21 people $13.50 (had never been paid,
  surfaced by the ledger) + pilot-2 19 people $13.33; $26.83 total, 40 ledger rows dated
  2026-07-28 (ledger now 64 entries incl. the 24 pilot people from 07-14). Plain
  make_bonus_list.py re-run emits zero unpaid. Recruitment pull tip: the interactive prompts
  can be skipped with `node scripts/get_recruitment_data.mjs --type real --branch_name main
  --filename data/private/real-main-recruitment.json`.

**DONE (2026-07-27, pushed/deployed 2026-07-28):**
- **Practice set grown 3 → 5 trials** (user request: humans still weak, want more familiarization).
  P4/P5 are C-style partial-match trials (2 misconceptions present, statement names one → agree):
  P4 probed error fires FIRST, P5 probed error fires SECOND (the which_target design, one practice
  trial each). Full item details in §2 (`make_human_practice_items.py`), frontend behavior in §3
  (PracticeView.vue): both error steps highlighted amber, the pointed-to one gets a bold
  "(the belief statement points to this one)" marker (also now applies to P3, whose feedback text
  was tweaked to mention the bold; P1/P2 byte-identical). InstructionsView says 5 practice
  questions. `npm run build` verified clean, new strings confirmed in the bundle. Pushed with
  the user's go-ahead on 2026-07-28; user checks the flow on the DEPLOYED site.

**DONE (2026-07-22):**
- **Three-way observer-comparison scatter** (`analysis-comparison/plot_observer_scatter.py` →
  `observer_scatter_graded.png` + `_binary.png`; copied to `Results_combined/figs/` and
  `llm_exp/report/figs/`). Common scale for all three observers:
  `P(Acc|trace) = P̂(agree)` if the statement is true (present, A/C) else `1 − P̂(agree)`, where
  P̂(agree) = the Bayes marginal directly, or graded `(rating−1)/5` for human/LLM (binary
  `≥4` in the `--scoring binary` variant, same conclusions). **One point = one (named
  misconception × category) group**, because per-item human coverage is a median of 1 rating,
  unusable; per group humans have n=21 (practice cohort), Bayes/LLM ~20 items. Layout: Bayes
  anchored to the y-axis in every panel it appears (panel a Bayes-vs-Human, panel c
  Bayes-vs-LLM per regime), Human on x, LLM the one observer that flips axis between rows b/c
  (a round-robin of 3 observers cannot keep all three axis-fixed). Readings: Bayes is a
  near-ceiling at ε=0 (present 1.00, absent ~0.88) so its panels are flat horizontal bands;
  haiku-thinking sits above the human diagonal, haiku-direct straddles, gpt-4o mostly below and
  widely scattered against Bayes (errors on logically-unambiguous items = claim-driven). The
  outside()/A human point (0.66) is the human→ideal inversion.
- **`report/report.tex` fully reconciled to the 480 pool + two new comparison subsections.**
  See §4 "LLM plots + report" for the full recompute detail and the new-section list.

**DONE (2026-07-20):**
- **ε switched 0.05 → 0.0 and the whole Bayes arm regenerated** (user's call; see §2 Bayesian for
  the rationale and the verification that no item loses all profiles). Regenerated:
  `misconception_difficulty.json`, `analysis-Bayesian/b_item_marginals.json`, all 5 Bayes figures
  (`bayes_1misc_dist_A/B`, `bayes_1misc_heatmap`, `bayes_1misc_heatmap_combined`,
  `bayes_2misc_heatmap`, copied to `Results_combined/figs/`), `dashboard/bayes_per_item.json` →
  `dashboard_data.json` → `index.html`. **Pool untouched** (`extend_pool.py` /
  `make_human_practice_items.py` deliberately NOT re-run, the experiment is live), and the
  refuted/unsupported foil labels are byte-identical at the same 0.15 cut: 0 of 240 B/D items
  relabelled, still 10 refuted per named foil.
  What changed numerically: every present item (A and C, 240 of them) now scores exactly 1.000,
  so the `outside_bracket_first` floor at 0.68–0.70 on the bracket C items is gone. Absent items
  barely moved (B 0.113 → 0.107, D 0.109 → 0.102) but 103 of them are now exactly 0.
  ⚠️ **Consequence to decide on:** the graded structure on the agree side is gone, so
  `bayes_1misc_dist_A.png` is six identical spikes at 1.0 and the 1-misc heatmap diagonal and the
  whole 2-misc panel (a) are uniform 1.00. Those three figures now carry no information and either
  need reframing (report the ideal observer as a logical oracle: 1.0 present vs ≤0.25 absent) or
  dropping. results.tex prose is stale in the same places, see PENDING.
  ⚠️ **What ε=0 does NOT fix:** the outside() column of `bayes_1misc_dist_B` row 3 still never
  spikes at zero, because `outside_bracket_first` is unfalsifiable by construction. Full
  mechanism, evidence and v2 fix in §2 Bayesian, "outside_bracket_first is UNFALSIFIABLE"
  (2026-07-21). Read that before writing any prose about the refuted contrast.
- **results.tex fully on the 480 pool + with-practice human arm**: human sections rewritten for
  the n=21 practice cohort (all four human figures → `_with_practice`, without-practice removed);
  Bayes/LLM sections + title updated to 480 (240 one-misc items, refutation now a controlled
  factor, refuted subset real for every foil). New Bayes 1-misc present×named heatmap added to
  figs (not yet cited in the tex — user writes the doc step by step).
- **Pool finalized to a clean 480** (`drop_ambiguous.py`, §2): the 7 never-sampled ambiguous
  items removed, 120 per category. All figures + the dashboard regenerated on 480.
- **Shareable dashboard published** (Artifact `dashboard/`): stimulus explorer + per-person +
  thinking-errors views over all 480 stimuli × {humans, 2 LLM regimes, Bayes ideal observer},
  with haiku-thinking reasoning inline and an automated failure analysis of its 60 errors
  (two opposite biases: too lenient on foils via precedence-overgeneralization/spurious-rtl,
  too strict on true matches; 14 of the false-agrees are "foil-surface-supported" = the trace
  literally performs the foil op, a WEAK-FOIL stimulus-design signal worth acting on).
  URL is private; user shares from the artifact page. Rebuild+redeploy: assemble_data.py →
  build_dashboard.py → republish same file path. `dashboard/.gitignore` keeps the
  participant-derived data snapshots + built index.html out of git.

**DONE (2026-07-17, extended-pool analysis):**
- **LLM arm re-run on all items × 3 regimes** (0 errors; see the results table above).
  Fixed a `parse_results` polars schema bug (which_target Null-inference on all-items runs →
  `infer_schema_length=None`).
- **All 6 model/observer figures regenerated on the 480 pool and copied to
  `Results_combined/figs/`**: {bayes,llm}×{1misc_dist_A, 1misc_dist_B, 2misc_heatmap}. The
  refuted subset row (dist_B row 3) is now populated for ALL foils including sub<÷ (was empty):
  10/20 refuted per foil (Bayesian) and the LLM version shows gpt-4o endorsing refuted claims.
  Figure scripts de-hardcoded (count labels now dynamic) and the synthetic sub<÷ injection
  disabled (superseded by real refuted items; `INJECT_SYNTHETIC=False`).
  ⚠️ The 4 HUMAN figures in figs/ (human_*.png) are STILL pilot-pool (240) snapshots —
  regenerate once confirmatory human data on the 480 pool is collected.

**PENDING / NEXT:**
- **Update results.tex for ε=0** (numbers below are now wrong in the tex, line refs as of
  2026-07-20): L56 "bracket evidence is systematically the weakest of the six"; L133 "the same
  items where the ideal observer's evidence is weakest"; L216 the difficulty caption; L303–305
  "mean marginal on the named target ranges from 0.68 to 1.00 … weakest cells all have
  outside_bracket_first"; L435 "green diagonal (weakest for outside_bracket_first)". All five rest
  on a graded present-side marginal that no longer exists. Decide the reframing first (logical
  oracle vs keep ε>0 as a secondary robustness analysis), then edit.
- ~~**Restate prereg H3.**~~ DONE 2026-08-07: the IO-difficulty hypothesis was dropped from the
  prereg entirely (documented there as considered-and-not-registered) because at ε=0 on v2 the
  observer is exactly 1.000 on every present item, so there is no gradient to correlate against.
  The "humans invert the ideal observer" finding in §5 is retired for the same reason.
- **Regenerate human figures + results.tex prose** once confirmatory human data arrives on the
  480 pool (the refuted contrast is now a real within-subject factor: 6 refuted + 6 unsupported
  foils per participant).
- **Finalize the 7 remaining prereg `\decflag` decisions** (the hypothesis set, power analysis
  and exclusion gate are now DONE and written up; see the 2026-08-07 entry). Still open and
  needing the user: **confirmatory N + stopping rule** (candidates 60 / 120 / ~300, power table
  in the tex), **H3 equivalence bounds** (candidate dz ±0.3), **confirm the below-chance gate**
  (≤7/24; excludes 0 of the current 40), **trial RT upper cutoff** (candidate 120s),
  **registration scope** (keep LLM/Bayes in a companion doc), **registry + timeline**,
  **authorship**. Note the outside()-unfalsifiability caveat in §2 applies to v1 ONLY; under v2
  refuted foils hit exactly 0 for all six rules, so the H1 refutation wording is safe as
  written.
- **Run the confirmatory cohort** after locking; C-positional questions especially need n.
- **Three-way comparison** (human × Bayesian × LLM per misconception/category) — the headline.
  All three arms now have data; the framing questions (pooled vs median vs distribution for
  humans, given the individual spread) feed the prereg discussion.
- **BODMAS results report** like the numberlink one; `report/report.tex` is the LLM half (now
  on the 480 pool, with the three-way comparison + 1-misc heatmap sections in it).
- **Uncommitted** (decide before relying on them): `analysis_human/`, `scripts/make_bonus_list.py`,
  `PreReg/`, `llm_exp/plots/`, `llm_exp/report/`, `llm_exp/make_llm_plots.py`,
  `llm_exp/make_llm_figures.py`. (The practice-trials change set was committed and deployed,
  commit 26af20f.) (`llm_exp/bodmas_llm/` package IS committed.) `llm_exp_buffer/`
  and `prereg_buffer/` are reference — delete when done.

---

## 7b. THE v3 "POSITION" POOL: decided AND built 2026-09-08 (branch `pool-v3`, not deployed)

> ### START HERE if you are picking up the v3 work cold
>
> **1. You must be on branch `pool-v3`.** The v3 pool does NOT exist on `main`. Either work in
> the existing worktree `/Users/divya/Desktop/NYU/Darpa/bodmas-pool-v3` (already checked out to
> `pool-v3`), or `git checkout pool-v3`. If `ls base-task/stimulus_pool_v3.json` fails, you are
> on the wrong branch and everything below will confuse you.
>
> **2. Confirm the state before changing anything** (about a minute, no side effects):
> ```bash
> cd base-task && python3 verify_v3.py     # must print ALL CHECKS PASSED and exit 0
> cd base-task && python3 bayes_v3.py      # must print 432/432 = 100.0%
> ```
> If either fails, something has been regenerated inconsistently. Rebuild with
> `python3 pool_v3.py` (seed 2026 is fixed, so it reproduces byte-identically) and re-verify.
>
> **3. What is DONE:** the pool is built (432 items), verified, and the ideal observer has been
> run over all of it. Read "The new design", then "Findings the build must respect", then
> "Pool BUILT". The findings section is the important one: it records things that cost real time
> to discover, including one measurement error to avoid repeating (finding 4).
>
> **4. What is NEXT:** the form sampler. `src/user/utils/sampleForm.js` and its Python twin must
> go from a 2-factor rotation to 3-factor (rule x position x foil_status) over 24 trials, then
> the 500-seed balance check gets redone. Everything else is listed under "Downstream work".
>
> **5. What NOT to do yet:**
> - Do NOT copy `stimulus_pool_v3.json` over `llm_exp/data/stimulus_pool.json` or
>   `src/user/data/stimulus_pool.json`. The frontend reads that filename and the sampler cannot
>   handle the new grid, so it would break form assembly.
> - Do NOT push `main`. Pushing `main` deploys the LIVE experiment. Pushing `pool-v3` is fine and
>   deploys only to its own per-branch staging URL.
> - Do NOT assume any figure, `results.tex` number, or the prereg reflects v3. They are all v1/v2.


User decision: narrow the design. The current 4-category / 2-misconception pool is replaced by a
**1-misconception-only design with the position of the error as a new manipulated factor**.
STATUS: the pool is BUILT and VERIFIED on branch `pool-v3` (worktree `../bodmas-pool-v3`), and
the ideal observer has been run over all of it. It is NOT deployed, NOT propagated to the other
two `stimulus_pool.json` copies, and the form sampler cannot sample it yet. This section records
the decisions, the feasibility evidence and the build so none of it has to be re-derived.

### The new design
- **6 misconceptions**, unchanged.
- **1 misconception per trace.** Categories **C and D are dropped entirely**; no 2-misconception
  trials exist any more. Only A (statement names the true misconception -> agree) and B (foil -> disagree).
- **New factor: error position.** The misconception fires at **step 1** or **step 3**.
- **Uniform 6 operators** (so every trace is exactly 6 steps). Chosen over 5 because
  `outside_bracket_first` cannot easily be placed late (see below).
- **Matched expressions**: the SAME expression supplies both the step-1 and the step-3 version,
  so position is manipulated with expression structure held constant. Consequence: **position must
  be between-participants** for a given expression; a participant must never see the same
  expression twice.
- **Refutation (`foil_status` refuted/unsupported) is retained** in B; it is the headline
  three-way result and must survive the redesign.
- **Ideal observer keeps all 22 hypotheses** (expert + 6 singletons + 15 pairs). The user first
  chose 7, then reversed to 22 on 2026-09-08 after finding 7 below. Note the framing that made
  this clear: the 22 are the OBSERVER'S HYPOTHESES, not stimulus types. Every v3 stimulus still
  contains exactly one misconception and no pair hypothesis is ever true of any item; the pair
  hypotheses exist so the observer can represent "the student might ALSO hold rule f", which is
  the only way "this work gives no evidence against that belief" (= unsupported) can differ from
  "the student had a chance to show that belief and demonstrably didn't" (= refuted).
  Consequence: the task is NOT 6-way identification, participants are NOT told there is exactly
  one misconception, and **the deferred `src/user/` instruction-text edit is dropped**; it was
  only ever needed to justify the 7-profile space.

Design grid: A = 6 misconceptions x 2 positions = 12 cells; B = 6 probed x 2 positions x
2 foil_status = 24 cells; 36 cells total (~432 items at 12/cell).

### Feasibility, measured 2026-09-08 (scripts were throwaway; numbers reproduced from
### generator.py + traces.py at n_ops=6, readable + wrong-final-answer required)
Matched pairs where the trace's ONLY expert-illegal move is at step 1 / step 3, per 2000
random expressions: add_before_mul 65, sub_before_mul 112, sub_before_div 46,
same_priority_rtl 45, add_before_div 30, **outside_bracket_first 13** (~1,850 expressions per
12 pairs; ~10s of compute). All six are buildable.

### Findings the build must respect
1. **Step 5+ is unreachable and step 6 is forced.** With n ops every finished trace has exactly
   n steps, and the final step has no choice left. Usable positions at 6 ops are 1-4.
2. **`outside_bracket_first` resists late placement, for a semantic reason.** v2 defines it as a
   *preference* (finish everything outside a bracket before entering it), so violating it is
   structurally an early act. Step-3 yield per 600 expressions: 15 at 5 ops, 45 at 6 ops, 72 at
   7 ops. This is what forced the move to 6 ops.
3. **A misconception often fires more than once.** "Position" is only well defined if the trace
   has **exactly one** expert-illegal move, so that must be an explicit generation constraint,
   not an afterthought.
4. **Do not measure "was this step an error?" with expert trace-edge membership.** Once the
   learner diverges, every later expression is off the expert's trace tree, so edge membership
   marks all subsequent steps as errors. The correct test is whether each move is expert-legal
   *from its own start expression*: `dag_to_str(d) for d in _next_dags(build_dag(prev), [])`.
5. **Arithmetic quality degrades sharply at 6 ops** and the current post-hoc filters do not catch
   it. `_is_clean` rejects only 3+ decimal places. Observed leaks: negative intermediates
   (`-6.75 + (8 x 12) x 2`), negative final answers (36% of candidates at 5 ops), negative
   operands (`5 / 2 + -2 + 7`, 25%), displayed `5 / 0` (2%), runaway magnitudes
   (`11 + 10 x 5 x 7 x 9 - 5 + 12` -> 3144), and degenerate repeated digits
   (`3 + 3 x 3 x 3 x 9 x 9 - 9`, where `9-9=0` annihilates the product).
   **Fix by constraining `generator.py`** (exact-dividing divisors, every intermediate a positive
   integer in a bounded range, no long repeated-digit runs), not by filtering afterwards.
6. **Position is NOT meaningfully entangled with refutation, MEASURED 2026-09-08, concern
   retired.** The worry was that a step-3 item's two leading expert-consistent steps would refute
   more foils. Over 360 foil observations: step-1 mean foil marginal 0.154 with 45% refuted,
   step-3 mean 0.148 with 42% refuted. User decided not to control for it, and the data says
   there is nothing to control for.

7. **⚠️ A 7-profile observer destroys the refutation manipulation AND the Bayes arm.
   MEASURED 2026-09-08.** Over the same 360 foil observations, the 7-profile posterior gave a
   foil marginal of **exactly 0.000 in all 360 cases**, no gradient, so `foil_status` cannot be
   assigned at all. Cause: a foil rule can only carry posterior mass via a profile containing it,
   and with singletons only, the (foil,) profile is eliminated outright by any trace it cannot
   generate. The 22-profile space on the SAME items behaves like the current pool: 43% refuted
   (<0.15), 54% unsupported (0.15-0.35), 2% >0.35, min 0.000 max 0.706 mean 0.151 (v2 for
   comparison: refuted 0.000, unsupported band 0.17-0.29 mean 0.21). An independent structural
   check (is the shown trace still legal for a learner who ALSO holds the foil?) splits
   38% contradicted / 62% never-contradicted, agreeing with the marginal bands.
   Decisive consequence: under 7 profiles the ideal observer scores every A item at exactly
   1.000 and every B item at exactly 0.000, so the Bayes arm becomes a CONSTANT and the
   three-way comparison has nothing left to compare. Under 22 profiles its only informative
   variance is the refutation gradient, which is the headline finding.
   **Recommendation: keep 22 profiles.**

### Built so far on branch `pool-v3` (worktree `../bodmas-pool-v3`), 2026-09-08, UNCOMMITTED
- **`base-task/generator_v3.py`**, constrained generator. NEW FILE; `generator.py` is untouched
  so every existing analysis script keeps working. Draws numbers constructively left-to-right
  with one step of operator lookahead (an earlier repair-pass version was wrong: ordering
  `a - b` re-broke the pair to its left and divisor forcing undid both). Guarantees at the
  literal level, verified 0 violations in 5000 draws: subtraction operands ordered, division
  exact with a PROPER divisor (no `÷ 1`, no `n ÷ n`), × operands <= 6, no run of more than 2
  equal numbers. Also holds `validate_trace()` (non-negative integers only, nothing over 999,
  no zero anywhere, zero is banned because `× 0` collapses the expression and `÷ 0` sits in the
  work as an operation the student visibly never performs) and `error_steps()` (the correct
  expert-legality test from finding 4).
  Literal constraints are necessary but NOT sufficient: `5 - 3 × 6` is fine as literals and goes
  negative once evaluated, and evaluation order is the learner's choice, so `validate_trace()` on
  the displayed trace is the real gate.
- **`base-task/find_pairs_v3.py`**, the matched-pair sampler (`find_matched_pairs`,
  `pairs_for_expression`). Run directly for per-misconception yields.
- Measured yields for 12 matched pairs each (seed 2026): sub_before_mul 20.3% hit rate (59
  draws), add_before_mul 8.4% (143), same_priority_rtl 5.5% (218), sub_before_div 5.0% (242),
  outside_bracket_first 1.1% (1,122), add_before_div 0.9% (1,295). All fast.
- Diagnosticity verified: at epsilon=0, every generated item gives the TRUE rule marginal 1.000
  with every rival at exactly 0.000, at both positions, under either hypothesis space.

### Pool BUILT 2026-09-08 on branch `pool-v3` (worktree `../bodmas-pool-v3`), UNCOMMITTED
- **`base-task/pool_v3.py`** -> writes `base-task/stimulus_pool_v3.json`. Seed 2026.
  **432 items / 216 matched pairs / 216 distinct expressions**, every one of the 36 cells at
  exactly 12 items: A = 6 rules x 2 positions (144 items), B = 6 probed rules x 2 statuses x
  2 positions (288 items). Every trace is 6 steps with exactly ONE expert-illegal move.
  An expression is used by exactly one matched pair (2 items), so no participant can meet the
  same expression twice; that is also why position is between-participants for a given expression.
  A foil is only used when its status is identical at BOTH positions (measured: 93% of foils
  are, so the constraint is nearly free), which keeps a matched pair matched on refutation too.
  Foil marginals came out min 0.000 / max 0.333 / mean 0.126. Numbers shown span 1..960, no
  negatives, no decimals, no zero.
  New item fields vs v2: **`error_position`** (1 or 3) and **`pair_id`** (links the two
  positions of one expression). `category` is now only A or B. `which_target` is retained as
  null for schema compatibility. `foil_status` / `io_foil_marginal` are unchanged in meaning.
- **`base-task/verify_v3.py`**, independent verifier, run it after ANY regeneration
  (`python3 verify_v3.py`, exits non-zero on failure). It re-derives everything from the model
  rather than trusting the builder: re-generates the trace from the expression, re-tests every
  step for expert legality, re-runs the 22-hypothesis observer, and re-checks statement wiring,
  pair matching, expression uniqueness and cell balance. Currently ALL CHECKS PASSED.
- Foil cells are spread across generating rules (never the foil itself), but unevenly: e.g. the
  `sub_before_mul` foil cell draws 18 items from `sub_before_div` traces and only 2 from
  `same_priority_rtl`. Not a confound for the position contrast (which is within-expression) but
  worth evening out if the true-rule identity ever matters to an analysis.
- ⚠️ Pre-existing, NOT a v3 regression: `outside_bracket_first` items read on the surface like
  another rule. `4 + 8 / (4 - 1) -> 12 / (4 - 1)` is formally outside() because the model treats
  "operand is a bracket" as a separate pattern table (3/4) from atom-atom (table 2), so
  add_before_div does NOT license it and the observer separates them cleanly. A human reader may
  not. This is the same outside() weakness already documented for the LLM arm.

### Ideal observer RUN on the full v3 pool, 2026-09-08
- **`base-task/bayes_v3.py`** -> writes **`base-task/bayes_per_item_v3.json`** (the v3
  replacement for `analysis-Bayesian/b_item_marginals.json`). One row per item: probed marginal,
  binary judgment, correctness, MAP hypothesis.
- **Accuracy 432/432 = 100%** at epsilon=0, as expected of a logical oracle.
  A items: probed marginal exactly 1.000 in all 144, at BOTH positions, for all six rules.
  B items: P(agree) 0.000, marginal min 0.000 / mean 0.126 / max 0.333.
- **Refutation separates cleanly and position does not disturb it:**
  refuted pos1 mean 0.012 (max 0.143), refuted pos3 mean 0.001 (max 0.077);
  unsupported pos1 mean 0.246 (min 0.167), unsupported pos3 mean 0.245 (min 0.167).
  Refuted max 0.143 < unsupported min 0.167, so the 0.15 threshold sits in an empty gap and no
  item is near the boundary.
- ⚠️ **The refutation effect is entirely in the GRADED marginal, not the binary judgment.**
  P(agree) is 0.000 for refuted AND unsupported, so a binary-collapsed Bayes arm shows no
  refutation effect at all. Any figure comparing refutation across the three observers must use
  the graded measure on the Bayes side.
- ⚠️ **Do not use `map_profile` as the observer's response; use `probed_marginal`.** On 58/432
  items (13%) the MAP names TWO rules for a one-misconception item, and the partner is
  `outside_bracket_first` in all 58 (all of which contain a bracket). Cause: under v2, outside()
  is the only rule that REMOVES options (it blocks bracket recursion while outside work remains),
  so on a trace that never enters its bracket early, "also holds outside()" makes the observed
  path more likely and the pair strictly outscores the true singleton (e.g. A000: 0.474 vs
  0.105; none of the 58 are ties). Affects no item's correctness and no marginal.
- Minor, in band: outside() as an *unsupported* foil sits a little high (mean 0.281, min 0.250)
  next to e.g. add_before_mul (0.225, min 0.167), for the same reason.

### Bayes FIGURES for v3, BUILT 2026-09-09 (branch `pool-v3`)
The v2 `analysis-Bayesian/plot_bayes_1misc_*.py` scripts all read
`base-task/stimulus_pool.json`, so they render the v2 480-item pool and say nothing about v3.
Ported the three 1-misconception figure scripts; the 2-misconception one is NOT ported and dies
with categories C/D. All new files sit alongside the v2 ones (which are untouched and still
correct for v2) and carry a `_v3` infix in both script and output names.
- **`analysis-Bayesian/bayes_v3_common.py`** shared loader + styling. Reads
  `base-task/bayes_per_item_v3.json` (written by `bayes_v3.py`) rather than recomputing
  posteriors, so a figure can never disagree with the recorded Bayes arm. Exits with a pointed
  message if that file is missing.
- **`plot_bayes_v3_1misc_heatmap.py`** -> `bayes_v3_1misc_heatmap.png` (2 panels, refuted vs
  unsupported, shared category-A diagonal) + `bayes_v3_1misc_heatmap_combined.png`.
  Positions are POOLED here on purpose: splitting the 30 off-diagonal cells by status AND
  position leaves 26 of 120 cells empty and drops the median cell to 3 items, which draws
  sparsity rather than signal. The script prints the position contrast to stdout instead.
  All 30 off-diagonal cells are occupied, but unevenly (2 to 18 items) because foils are spread
  across generating rules unevenly, the imbalance already flagged in "Pool BUILT".
- **`plot_bayes_v3_1misc_marginals.py`** -> `bayes_v3_1misc_marginals.png`. Per-item dots, four
  columns per rule (category x position). Carries the global picture: category A is a flat line
  of dots at exactly 1.000 and category B never rises above 0.333, so the two are separated by
  two thirds of the axis with the decision boundary sitting in empty space.
- **`plot_bayes_v3_1misc_distributions.py`** -> `bayes_v3_1misc_dist_A.png` and `_dist_B.png`.
  Same row structure as v2 (A by present; B by present / by named / refuted-only), with each
  panel split into step-1 and step-3 curves.
- WARNING: **these are NOT KDEs, unlike the v2 distribution script, and that is deliberate.** The
  v3 B marginals take only 11 distinct values in [0, 0.333] and category A takes exactly one, so
  the distributions are discrete; a Gaussian KDE would invent shape between the spikes and smear
  the point mass, which is the oversmoothing the user rejected before. Each panel plots the exact
  observed values as stems whose height is the proportion of that group on that value. The v2 KDE
  stays correct for v2, where the marginals were genuinely continuous.
- WARNING: **the category-B axis is zoomed to [0, 0.4]**, so the 0.5 decision boundary is
  off-scale and the 0.15 refuted cut is drawn instead. On a [0, 1] axis the whole distribution
  sits in the left third and the refuted/unsupported gap, the only structure in the figure, is
  invisible. The marginals figure keeps the full [0, 1] axis, so the global picture is not lost.
- **What the figures show.** (1) Category A is a point mass at 1.000 in all 12 rule x position
  cells, so `dist_A` and the heatmap diagonal carry no information beyond "the observer is a
  logical oracle"; keep them as the reference, do not read structure into them. (2) The
  refuted/unsupported split is clean and bimodal on every foil: a spike at 0 plus a cluster in
  0.167 to 0.333. (3) **Position does nothing to the observer**, refuted 0.012 (pos1) vs 0.001
  (pos3) and unsupported 0.246 vs 0.245, so the two curves overlay in every panel. That is the
  useful null: it makes Bayes the flat reference against which any human or LLM position effect
  reads as an observer property, not a stimulus artifact.
- Verified while building: the stored `foil_status` and the 0.15 cut disagree on 0 of 288 B items,
  and every named foil has exactly 24 refuted items (12 per position).
- **WARNING: the blank cells in the two heatmap panels are a POOL-SAMPLING artifact, not a
  structural fact, MEASURED 2026-09-09.** 13 of the 30 (present x named) cells hold only one
  refutation status in the pool (7 refuted-only, 6 unsupported-only), which is why panel (a) and
  panel (b) each show gaps; the combined heatmap has all 30 cells occupied. It would be easy to
  read those gaps as "this rule pair can never be refuted", and that is WRONG. Probe: drawing 40
  fresh matched pairs per present rule offers BOTH statuses in 29 of the 30 cells, and the one
  holdout (sub<div present, add<div named) splits 172 refuted / 18 unsupported over 200 pairs.
  So every cell can carry either status; the pool just did not sample them.
  Cause: `pool_v3.py` balances (probed foil x status x position) at 12 each and puts NO constraint
  on which generating rule supplies each item, so the present-side breakdown falls where it falls.
  This is the same uneven-spread issue already noted under "Pool BUILT", now measured on the
  status split rather than on counts.
  Consequences: harmless for the position contrast (within-expression) and for the refutation
  contrast pooled over foils (balanced by construction). NOT harmless for any cell-by-cell reading
  of a present x named heatmap, which is exactly how the v2 human heatmaps were analysed (the
  "wrong-side majority" pass). In 13 of 30 cells, present x named identity is perfectly confounded
  with refutation status.
  **Fix if that analysis matters: add a spread constraint over generating rules to the B-item
  selection in `pool_v3.py` and rebuild.** Worth deciding BEFORE the sampler work, since a rebuild
  changes item ids.
- Still MISSING and worth building next: a dedicated **position** figure per observer, and
  **position x refutation**. These five are the v2 set ported, not new v3-specific views.

### Downstream work this triggers (nothing done yet)
- Every 2-misconception figure dies: `analysis_human/plot_2misc_heatmap.py`,
  `plot_2misc_heatmap_dots.py`, `llm_exp/make_llm_2misc_heatmap.py`,
  `make_llm_2misc_heatmap_dots.py`, `analysis-Bayesian/plot_bayes_2misc_heatmap.py`, and
  `Results_combined/results.tex` sections 4.1-4.3.
- ~~The three Bayes 1-misconception figure scripts~~ DONE 2026-09-09, see "Bayes FIGURES
  for v3" above. The human and LLM 1-misc counterparts (`analysis_human/plot_human_1misc_*.py`,
  `llm_exp/make_llm_1misc_*.py`) are still v2-only and need the same port once v3 data exists.
- New figures needed: position effect per observer, and position x refutation. Nothing built
  yet; the v3 Bayes set above ports the v2 views and only reports the position null in passing.
- Form sampling goes from a 2-factor to a 3-factor rotation (rule x position x foil_status) in
  BOTH `src/user/utils/sampleForm.js` and its Python twin; the 500/500 seed-balance verification
  must be redone.
- `inference.py` needs a 7-profile mode; `misconception_difficulty.json` and
  `analysis-Bayesian/b_item_marginals.json` must be regenerated under it.
- All existing human data is on the old design. The 2-misconception results become historical;
  they cannot be replicated under v3.
- The v3 pool is NOT yet propagated to the other two copies (`llm_exp/data/`, `src/user/data/`)
  and must not be until `sampleForm.js` can sample the new grid; the frontend reads
  `stimulus_pool.json`, so dropping v3 in as-is would break form assembly.
- `base-task/make_human_practice_items.py` / `make_practice_examples.py` still assume the old
  categories and must be regenerated for v3.

---

## 7c. THE v4 "POSITION x NAMED" POOL: decided AND built 2026-09-09 (branch `pool-v3`)

**v4 supersedes v3 as the intended design.** v3 is kept on disk and in git for reference; nothing
about it was deleted. Read this section before 7b if you are picking up the current work.

### Why v4 exists
v3 balanced B on (named foil x refutation status x position) and let the PRESENT rule fall where
it may. Consequence, discovered when the v3 Bayes heatmaps were drawn: 13 of the 30 present x named
cells held only ONE refutation status, so the status-split panels had holes. User's call: make the
heatmap the thing that is balanced, and drop refutation as a factor for now.

### The v4 design
- 6 misconceptions, 1 per trace, categories A and B only (C/D stayed dead).
- **Error position kept** (step 1 vs step 3), still via MATCHED PAIRS: one expression supplies both
  the step-1 and the step-3 version, so position moves with expression structure held constant.
- **Refutation dropped as a factor.** `foil_status` and `io_foil_marginal` are still computed and
  stored per item, but nothing is balanced on them.
- **The named foil IS balanced within each present rule**, which is what fills the heatmap columns.

Grid, 240 items in 120 matched pairs:
```
A: present(6) x position(2)            = 12 cells x 10 items = 120
B: present(6) x named(5) x position(2) = 60 cells x  2 items = 120
```
So every rule is the true misconception in exactly 40 items: 10 each of (A,pos1) (A,pos3) (B,pos1)
(B,pos3). The present x named heatmap is FULL, pooled (diagonal 20, off-diagonal 4) and split by
position (diagonal 10, off-diagonal 2).

### ⚠️ 6 operators is FORCED, MEASURED 2026-09-09
Over 2500 bracketed expressions, the count supporting BOTH step 1 and step 3 for
`outside_bracket_first`: **0 at 4 ops, 0 at 5 ops, 32 at 6 ops.** At 4 ops that rule never reaches
step 3 at all (single-error steps were 271 at step 1, 121 at step 2, none at step 3). At 5 ops step 3
is reachable (162 traces) but no single expression does both, so matched pairs are impossible.
Do not "simplify" v4 to shorter expressions; it silently kills the outside() step-3 cell.

### ⚠️ Consequences of dropping refutation, READ BEFORE ANALYSING
1. **Never split the v4 heatmap by `foil_status`.** Status is unbalanced by construction, so a split
   reintroduces exactly the v3 holes that v4 was built to remove. One unsplit heatmap, or split by
   position.
2. **The refutation contrast is no longer a clean test.** Pool-wide it is 58 refuted / 62
   unsupported, which looks fine, but per foil it is lopsided: RTL 16/4 and
   **outside_bracket_first 2 refuted / 18 unsupported**. The three-way refutation result (Bayes
   perfect, humans partial, LLMs not at all) CANNOT be replicated on v4 as a balanced within-subject
   factor. If that finding is wanted in the next wave, refutation has to come back as a factor and
   the pool needs rebuilding.

### Built 2026-09-09 (all on branch `pool-v3`)
- **`base-task/pool_v4.py`** -> `stimulus_pool_v4.json`. Seed 2026, built in 137 pair draws.
  Reuses `generator_v3.py` and `find_pairs_v3.pairs_for_expression` unchanged. A foil is only used
  when its status is the SAME at both positions: status is not a factor any more, but letting it
  flip inside a pair would put a nuisance difference between the two positions being compared, and
  it costs only ~7% of foil options.
- **`base-task/verify_v4.py`** independent verifier, re-derives everything from the model. Asserts
  the exact cell counts that guarantee a full heatmap and asserts 0 empty cells. Currently ALL
  CHECKS PASSED. It deliberately does NOT check refutation balance, only that each stored status
  matches a fresh recomputation and is stable across a pair.
- **`base-task/bayes_v4.py`** -> `bayes_per_item_v4.json`. **240/240 = 100%** at epsilon=0.
  A marginals exactly 1.000 at both positions for all six rules; B P(agree) 0.000, marginal
  min 0.000 / mean 0.130 / max 0.333. 24 of 240 items (10%) have a 2-rule MAP, partner always
  outside_bracket_first, same known cause as v3; use `probed_marginal`, not `map_profile`.
- **`analysis-Bayesian/plot_bayes_v4_1misc_heatmap.py`** -> `bayes_v4_1misc_heatmap.png` (2 panels
  by error position) + `bayes_v4_1misc_heatmap_combined.png`. Both confirmed 0 empty cells.
- **`analysis-Bayesian/bayes_v3_common.py` was RENAMED to `bayes_common.py`** and generalised to
  take a pool path plus an expected item count, since it now serves both v3 and v4 figures. The
  three v3 plot scripts were repointed and re-run unchanged.

### Still to do for v4
- Port the marginals and distributions figures (only the heatmap exists so far). Their v3 versions
  are position-aware already, so it is mostly a loader swap.
- The form sampler still cannot sample v4: `src/user/utils/sampleForm.js` and its Python twin need a
  rule x position rotation over 24 trials, then the 500-seed balance check. Same blocker as v3.
- `stimulus_pool_v4.json` is NOT propagated to `llm_exp/data/` or `src/user/data/`, and must not be
  until the sampler can handle it.
- Practice items (`make_human_practice_items.py`) still assume the old categories.

### Repo strategy (decided 2026-09-08)
Build on a **side branch in a git worktree**, as was done for `outside-bracket-v2`. NOT a new
repo: the deploy secrets, Firebase wiring and Prolific links are bound to this repo, the analysis
scripts are what have to be re-run (not discarded), and `data/real-all-main-data.json` plus
`data/private/bonus_paid.csv` are gitignored and would be silently lost by a copy. Keeping one
history is also the only way to tell which pool generated which figure, the exact failure that
cost 19 participants in the v1/v2 fork.
⚠️ **`deploy.yml` deploys on push to ANY branch** except `feat-*`, `fix-*`, `refactor-*`,
`test-*`, `chore-*`, `style-*`, `docs-*`, `ci-*`, but to a per-branch path
(`/<owner>/<repo>/<branch>/`) with its own codename URL. So a branch named e.g. `pool-v3` gets a
free staging deployment and does NOT touch main's live site.

---

## 8. Commands cheat-sheet

```bash
# v4 position x named pool (CURRENT design, branch `pool-v3` ONLY, see 7c)
cd base-task && python3 pool_v4.py                  # build -> stimulus_pool_v4.json (240 items)
cd base-task && python3 verify_v4.py                # independent checks; RUN AFTER ANY REBUILD
cd base-task && python3 bayes_v4.py                 # ideal observer -> bayes_per_item_v4.json
python3 analysis-Bayesian/plot_bayes_v4_1misc_heatmap.py   # full present x named heatmap

# v3 position pool (SUPERSEDED by v4, kept for reference, see 7b)
cd base-task && python3 find_pairs_v3.py 12         # per-misconception matched-pair yields
cd base-task && python3 pool_v3.py                  # build -> stimulus_pool_v3.json (432 items, ~1 min)
cd base-task && python3 verify_v3.py                # independent checks; RUN AFTER ANY REBUILD, exits nonzero on failure
cd base-task && python3 bayes_v3.py                 # ideal observer over the pool -> bayes_per_item_v3.json
python3 analysis-Bayesian/plot_bayes_v3_1misc_heatmap.py        # present x named, refuted vs unsupported
python3 analysis-Bayesian/plot_bayes_v3_1misc_marginals.py      # per-item dots, category x position
python3 analysis-Bayesian/plot_bayes_v3_1misc_distributions.py  # dist_A (point mass) + dist_B (3 rows)
# the three plot scripts read base-task/bayes_per_item_v3.json, so re-run bayes_v3.py after a rebuild

# Model / pool
cd base-task && python3 stimulus_pool.py            # ABORTS without --rebuild-240 (would clobber the extended pool)
cd base-task && python3 regenerate_C.py             # regenerate ONLY category C (preserves A/B/D)
cd base-task && python3 extend_pool.py              # rebuild the 480 refutation design from a 240 base (writes all 3 copies)
cd base-task && python3 misconception_difficulty.py # re-run Bayesian difficulty baseline
cd base-task && python3 make_human_practice_items.py # regenerate the 3 human practice trials
cd base-task && python3 drop_ambiguous.py            # drop the 7 ambiguous items -> clean 480 (all 3 copies)

# Dashboard (shareable Artifact explorer over all 480 stimuli x 3 observers + humans)
python3 dashboard/assemble_data.py                   # build PII-free dashboard_data.json
python3 dashboard/build_dashboard.py                 # inject data + failure analysis -> dashboard/index.html
# then publish dashboard/index.html as an Artifact (same URL redeploys on rebuild)
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
