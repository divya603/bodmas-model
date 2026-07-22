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
                  — written once for all three observers) + §4.1 Bayesian (at ε=0 panel (a) is
                  uniformly 1.00 in every cell, was 0.76–1.00 with outside() weakest; panel (b)
                  all red ≤0.16, was ≤0.26, flat →
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
                  after regenerating any source figure. bayes/llm figures re-copied on the 480
                  pool 2026-07-17. HUMAN figures now split by cohort (2026-07-20): the pilot
                  (no-practice, 240 pool) plots renamed `human_*_without_practice.png`; the
                  practice cohort (480 pool) plots added as `human_*_with_practice.png` (copied
                  from human_buffer/, currently n=21). results.tex references the
                  `_without_practice` set; the `_with_practice` figures are staged in figs/ but
                  not yet cited (prose written step-by-step with the user).
                  human_signal_detection.png is still the single pilot version (unchanged).
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
  → practice (3 feedback trials) → experiment → strategy question → feedback survey → demographics →
  save → debrief → thanks).
- **`src/user/components/trace_judgment/StrategyQuestionView.vue`** — standalone free-response screen
  shown right after the task (added 2026-07-17): "Please describe the strategy or strategies you
  used to decide how much you agreed with each statement..." Required (Continue disabled until
  non-empty). Saved as `pageData_strategy` (`{strategy: ...}`). Then Continue leads to the original
  feedback survey (difficulty/enjoyment/general feedback/issues, unchanged).
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

### LLM results (all 487 items, re-run on the extended pool 2026-07-17)
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
- **Restate prereg H3.** The ideal-observer difficulty ordering is unchanged in rank but nearly
  flat at ε=0 (0.78 → 0.85 across six rules, all 1.000 on single-misconception items), so H3 as
  written tests a gradient that is now mostly stimulus design. Same for the "humans invert the
  ideal observer" finding in §5.
- **Regenerate human figures + results.tex prose** once confirmatory human data arrives on the
  480 pool (the refuted contrast is now a real within-subject factor: 6 refuted + 6 unsupported
  foils per participant).
- **Finalize the prereg decisions** (with the user, discussion-first): hypothesis set +
  directions — the refutation contrast is now a natural H5 (FA lower on refuted than
  unsupported foils; within-subject 6v6 per participant, pilot hint 0.29 vs 0.39), but see the
  outside()-unfalsifiability caveat in §2 before fixing H5's wording: its "refuted" items are
  diluted, not contradicted —
  participant exclusion rule (candidate: below-chance binomial gate ≤7/24, plus
  no-gate sensitivity), confirmatory N/power analysis, registry + timeline.
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

## 8. Commands cheat-sheet

```bash
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
