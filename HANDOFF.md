# BODMAS pilot (branch `pilot-v4`) — Handoff

A complete, from-scratch orientation for a model picking this up cold. Read this instead of the
conversation history.

> **⚠️ STANDING INSTRUCTION TO EVERY AGENT: keep this file current.** As you complete work (new
> scripts, figures, findings, decisions, deploys), update the relevant sections either as you go or
> at the latest before your session ends. This file is the single source of truth; the next session
> must be able to pick up cold from it alone.

---

## 0. What this branch is, and what it is NOT

`pilot-v4` is a **fresh start for a new pilot**, cut from the older work on 2026-09-09. It carries
ONE stimulus pool, the model that generates it, the Bayesian ideal observer, and the experiment
frontend. Everything else was deliberately deleted.

**Deleted on this branch.** There are **no human results and no LLM results here**, by design. Do
not try to reconstruct them; nothing is lost, but the two halves live in different places:

- **On `main`** (branch `main`, commit `d73ddb6`): the v1/v2 480-item pool and its builders, all
  human data analysis (`analysis_human/`), the whole LLM experiment (`llm_exp/`), the
  pre-registration (`PreReg/`), the results document (`Results_combined/`), the observer-comparison
  scripts, and the dashboard. Check out `main` to get any of it back.
- **Only in THIS branch's own history**, at commit `08b380b` and earlier: the v3 432-item position
  pool (`pool_v3.py`, `stimulus_pool_v3.json`, `verify_v3.py`, `bayes_v3.py`,
  `bayes_per_item_v3.json`) and the v3 Bayes figures. These were never on `main`, and the `pool-v3`
  branch was deleted on 2026-09-09 once `pilot-v4` had superseded it. Recover a file with
  `git show 08b380b:base-task/stimulus_pool_v3.json`. Deleting the branch lost nothing: its tip is
  an ancestor of this branch.

Only two branches exist now: `main` and `pilot-v4`.

So this branch has exactly one arm with data in it: the Bayesian ideal observer, run over the pool.

### The task (one trial)
A participant sees a **math expression**, a **student's step-by-step work** containing exactly one
order-of-operations misconception, and a **belief statement** claiming the student holds a
particular misconception. They rate on a **6-point Likert scale** (1 = Strongly Disagree, 6 =
Strongly Agree) how well the statement explains the work, NOT whether the final answer is right.
Scoring collapses the rating at **>= 4 = agree**.

### The 6 misconceptions
| id | meaning |
|---|---|
| `add_before_mul` | does `+` before an adjacent `×` |
| `add_before_div` | does `+` before an adjacent `÷` |
| `sub_before_mul` | does `-` before an adjacent `×` |
| `sub_before_div` | does `-` before an adjacent `÷` |
| `same_priority_rtl` | evaluates equal-priority ops right-to-left instead of left-to-right |
| `outside_bracket_first` | must finish everything outside a bracket before resolving its contents |

Note `outside_bracket_first` is a **preference**, not a permission: a learner holding it may not
enter a bracket while literal-literal work remains outside. This is the only rule that REMOVES
options rather than adding them, and it has consequences noted throughout.

---

## 1. Repository map

```
base-task/         The model + the pool + the ideal observer (Python). See §2.
analysis-Bayesian/ Ideal-observer figures. See §4.
src/               The Smile/Vue human experiment. User code in src/user/. See §5.
scripts/           Smile deploy/data scripts.
public/            consent-form.pdf, debrief.pdf served by the frontend.
env/, firebase/    Smile config/secrets (already set up).
docs/ tests/ plugins/ analysis/ plans/   Smile framework infrastructure, not ours. Leave alone.
```

---

## 2. The model and the pool (`base-task/`)

### Model core (unchanged from the original project)
- **`dag.py`** FlatDAG representation of an expression (atoms + op nodes, shared references).
- **`parser.py`** `build_dag(expr)`. Folds signed-number literals so re-parsing intermediate trace
  strings matches `_eval`'s representation.
- **`pattern_matcher.py`** classifies 3-node "windows" into Tables 1 to 6.
- **`learner.py`** `MISCONCEPTION_FLIPS`: each misconception's bidirectional `to_true`/`to_false`
  validity flips. A learner is a list of misconception ids.
- **`valid_actions.py`**, **`traces.py`** `generate_traces(dag, misconceptions)` simulates a learner
  and returns ALL step-by-step traces it could produce. Includes the `is_zero_divide` guard.
- **`distance.py`** `correct_answer()`, `tree_edges()`, `diagnostic_traces()`.
- **`generator.py`** the original random expression generator.
- **`inference.py`** `posterior_over_profiles(trace)` and `marginal_rule_probability()`.
  **`DEFAULT_EPSILON = 0.0`**: every pool trace is generated deterministically by one of the
  hypotheses, so there is no slip process and a nonzero epsilon is a misspecified likelihood. At
  epsilon 0 a forbidden step eliminates its hypothesis outright.

### The constrained generator
- **`generator_constrained.py`** (was `generator_v3.py`). Draws numbers constructively left to right
  with one step of operator lookahead. Guarantees, verified 0 violations in 5000 draws: subtraction
  operands ordered, division exact with a proper divisor (no `÷ 1`, no `n ÷ n`), `×` operands <= 6,
  no run of more than 2 equal numbers. Also holds:
  - `validate_trace()`: non-negative integers only, nothing over 999, no zero anywhere. Zero is
    banned because `× 0` collapses the expression and `÷ 0` sits in the work as an operation the
    student visibly never performs.
  - `error_steps(trace)`: **the correct expert-legality test**. It asks, for each step, whether an
    expert could produce that line FROM THE IMMEDIATELY PRECEDING LINE
    (`expert_next` -> `_next_dags(build_dag(prev), [])`).
    ⚠️ Do NOT reimplement this as expert trace-edge membership. Once the learner diverges, every
    later state is off the expert's trace tree, so edge membership marks all subsequent steps as
    errors and no trace ever looks like it has exactly one. This mistake cost real time once.

  Literal constraints are necessary but NOT sufficient: `5 - 3 × 6` is fine as literals and goes
  negative once evaluated, and evaluation order is the learner's choice, so `validate_trace()` on
  the displayed trace is the real gate.
- **`find_pairs.py`** (was `find_pairs_v3.py`). `pairs_for_expression(expr, misconception)` returns
  `{position: trace}` for each requested position the expression supports. A trace qualifies when it
  finishes in `N_OPS` steps, reduces to a number, reaches a DIFFERENT answer than the expert, passes
  `validate_trace`, and has EXACTLY ONE expert-illegal move at the requested position.

### The pool: `pool.py` -> `stimulus_pool.json`
**240 items in 120 matched pairs.** Seed 2026, reproducible byte-for-byte.

Design factors:
- **Misconception present** (6 levels). Each is the true misconception in exactly 40 items.
- **Error position** (step 1 or step 3), manipulated WITHIN expression via matched pairs: both
  members of a `pair_id` share an identical expression and differ only in where the error falls.
- **What the statement names**: category **A** names the present misconception (correct answer
  agree), category **B** names an absent foil (correct answer disagree).

Grid:
```
A: present(6) x position(2)            = 12 cells x 10 items = 120
B: present(6) x named(5) x position(2) = 60 cells x  2 items = 120
```
Each rule's 40 items are 10 each of (A,pos1) (A,pos3) (B,pos1) (B,pos3), and within its 20 B items
the named foil is balanced 5 foils x 2 positions x 2 items. That last constraint is why the
**present x named heatmap has no empty cells**: diagonal 20 and off-diagonal 4 when pooled,
10 and 2 when split by position.

Other guarantees: 120 distinct expressions, each used by exactly one pair, so no participant can
meet the same expression twice. Numbers shown span 1 to 960, no negatives, decimals or zeros.

Item fields: `id, pair_id, category, error_position, expression, n_ops, misconceptions,
num_misconceptions, trace, probed_misconception, statement_correct, which_target (always null,
schema compatibility), student_name, belief_statement`, plus on B items only `foil_status` and
`io_foil_marginal`.

**Answer-leak-relevant fields** (never show a solver): `statement_correct, misconceptions,
probed_misconception, category, num_misconceptions, foil_status, io_foil_marginal`.

### ⚠️ Three things about this pool that will bite you
1. **6 operators is FORCED, not preferred.** Over 2500 bracketed expressions, the number supporting
   BOTH step 1 and step 3 for `outside_bracket_first` is **0 at 4 ops, 0 at 5 ops, 32 at 6 ops**. At
   4 ops that rule never reaches step 3 at all. At 5 ops step 3 is reachable but no single
   expression does both, so matched pairs are impossible. Shortening the expressions silently kills
   the outside() step-3 cell.
2. **`foil_status` is RECORDED but NOT BALANCED.** Refutation (refuted vs unsupported) was dropped
   as a factor. Pool-wide it is 58 refuted / 62 unsupported, which looks fine, but per foil it is
   lopsided: `same_priority_rtl` 16/4 and **`outside_bracket_first` 2 refuted / 18 unsupported**.
   **Never split a figure by `foil_status`** or the cells go lopsided and empty. If a refutation
   contrast is wanted, it has to come back as a design factor and the pool needs rebuilding.
3. **Position is SELECTED, not constructed.** Nothing places the error. A misconception is a
   substitution at one window, so it decides what happens when the learner touches that window, not
   when they touch it. One expression plus one misconception yields many traces (75 for the worked
   example) with the single error landing at steps 1 through 5. The builder filters for the ones at
   step 1 and step 3 and pairs them.

### Verification: `verify.py`
Independent verifier, run after ANY regeneration (`python3 verify.py`, exits non-zero on failure).
Re-derives everything from the model rather than trusting the builder: regenerates the trace from
the expression, re-tests every step for expert legality, re-runs the 22-hypothesis observer, and
re-checks statement wiring, pair matching, expression uniqueness, and the exact cell counts that
guarantee a full heatmap (including asserting 0 empty cells). Currently ALL CHECKS PASSED.

It deliberately does NOT check refutation balance, only that each stored status matches a fresh
recomputation and is stable across a pair.

---

## 3. The Bayesian ideal observer

`bayes.py` -> `bayes_per_item.json`. One row per item: probed marginal, binary judgment,
correctness, MAP hypothesis.

The observer keeps all **22 hypotheses** (expert + 6 singletons + 15 pairs) at epsilon 0. The pair
hypotheses are never true of any item; they exist so the observer can represent "the student might
ALSO hold rule f", which is the only way "no evidence either way" can differ from "had a chance to
show it and demonstrably did not". A 7-hypothesis space collapses every A item to exactly 1.000 and
every B item to exactly 0.000, making the Bayes arm a constant. Keep 22.

**Result: 240/240 = 100%.**
- A items: probed marginal exactly 1.000 in all 120, at both positions, for all six rules.
- B items: P(agree) 0.000, marginal min 0.000 / mean 0.130 / max 0.333, over only 8 distinct values.

⚠️ **Do not use `map_profile` as the observer's response; use `probed_marginal`.** On 24 of 240
items (10%) the MAP names TWO rules for a one-misconception item, and the partner is
`outside_bracket_first` in all of them. Cause: outside() is the only rule that removes options, so
on a trace that never enters its bracket early, "also holds outside()" makes the observed path more
likely and the pair strictly outscores the true singleton. Affects no item's correctness and no
marginal.

---

## 4. Figures (`analysis-Bayesian/`)

- **`bayes_common.py`** shared loader and styling. Reads `base-task/bayes_per_item.json` rather than
  recomputing posteriors, so a figure can never drift from the recorded observer.
- **`plot_bayes_1misc_heatmap.py`** -> `bayes_1misc_heatmap.png` (2 panels split by error position)
  and `bayes_1misc_heatmap_combined.png` (positions pooled). Rows = misconception PRESENT, columns =
  misconception NAMED. Both confirmed 0 empty cells.
- **`plot_bayes_1misc_distributions.py`** -> three figures:
  - `bayes_1misc_dist_A.png` category A. A POINT MASS at 1.000 in every panel. Kept as the
    reference, but it carries no information beyond "the observer is a logical oracle".
  - `bayes_1misc_dist_B.png` category B, two rows, positions overlaid, x zoomed to [0, 0.4] since
    no probed foil marginal exceeds 0.333. **Row 2 (by NAMED rule) is the clean one**: grouping
    variable and plotted marginal are the same rule, so each panel asks one question.
    ⚠️ **Row 1 (by PRESENT rule) is a MIXTURE.** It groups by the rule in the trace but still plots
    the marginal on whichever rule the statement named, so one panel pools five different
    marginals: the panel for `add_before_mul` never plots P(add_before_mul | trace) at all. It is a
    task-level summary of the B trials arising from those traces, not a property of the observer.
    Titles and caption now say this explicitly (they did not on 2026-09-09 and it misled a reader
    immediately). Use the profile figure for the observer-level trace-side question.
  - `bayes_1misc_profile.png` **the one worth looking at.** See below.

### ⚠️ Category A vs B is NOT a difference in the observer
`posterior_over_profiles()` takes ONLY the trace. A and B are properties of the TASK, not of the
inference: the observer computes one posterior per trace, yielding six marginals, and the category
only decides which entry gets read off. The labels exist on the Bayes side purely for
COMPARABILITY, because humans and LLMs do see the statement and are scored per item, so the Bayes
arm has to be scored on the same items to sit on the same axis. Any figure that groups by one rule
while plotting a marginal selected by a different rule is mixing incommensurable quantities; see
the dist_B row 1 warning above.

### ⚠️ The statement is NOT an input to inference
`posterior_over_profiles()` takes ONLY the trace. The named rule enters afterwards purely as an
index: `marginal_rule_probability(post, named)` picks one entry out of a posterior that was already
computed. So category A vs B is not a difference in the observer's computation, only a difference in
WHICH of the six marginals gets read off. Every trace carries all six. This is why `dist_A` is empty
of structure and why the profile figure exists.

### ⚠️ FINDING (2026-09-09): the pool systematically excludes the hardest foils
`bayes_1misc_profile.png` plots all six marginals per trace, and it reveals what the A/B figures
structurally cannot. Over the 1200 (trace, absent rule) combinations in the pool:
- **36 (3.0%) give an ABSENT rule a marginal above 0.35**, and **15 (1.2%) above 0.5**, meaning the
  trace positively FAVOURS a rule the student does not hold. Max observed **0.871** (item B224:
  trace contains add<div, but P(outside() | trace) = 0.871).
- **All 15 of the over-0.5 cases are `outside_bracket_first`**, and 30 of the 36 over-0.35 cases.
  Same root cause as everything else about that rule: it is the only one that REMOVES options, so a
  trace that never enters its bracket early looks like positive evidence FOR it.
- `pool.py: foil_options()` drops any foil whose marginal exceeds `UNSUPPORTED_MAX` (0.35) as "not a
  clean foil". That is a deliberate choice, but its consequence is that **no category-B item ever
  probes a foil the trace actually supports**, so the B items are systematically the easier foils
  and the hardest cases are invisible in every figure except the profile one.
Decide before running participants whether that exclusion is wanted. Keeping it means the disagree
trials never include the genuinely tempting case; removing it means some B items have no defensible
"correct" answer, since the ideal observer itself would agree with the statement.

Notes for any figure added here:
- Category A is a **point mass** at 1.000. Do not draw its "distribution"; there is none.
- Category B marginals take 8 distinct values, so they are discrete. Use exact-value stems, not a
  KDE, which invents shape between the spikes.
- Never split by `foil_status` (see §2).

---

## 5. The human experiment (`src/`)

A Smile (codec-lab / gureckislab) Vue-3 experiment. **User code in `src/user/`.** `npm run dev` to
run locally.

- **`src/user/design.js`** the timeline (consent -> windowsizer -> instructions -> comprehension quiz
  -> practice -> experiment -> strategy question -> feedback survey -> demographics -> save ->
  debrief -> thanks).
- **`src/user/components/trace_judgment/TraceJudgmentView.vue`** the 24-trial task: expression, work
  (`= step` per line), belief statement, 6-point Likert. Has a 3-second answer lock per trial, an
  "X of 24" counter, bonus scoring (binary direction, `bonus = max(0,(acc-0.5)/0.5) x $2`), and
  mouse tracking for offline bot detection.
- **`src/user/components/trace_judgment/PracticeView.vue`** practice trials with feedback: the
  erroneous steps are highlighted amber after the participant answers.
- **`src/user/utils/sampleForm.js`** JS port of the Python `sample_form`.
- **`src/builtins/thanks/ThanksView.vue`** Prolific completion code **`CNIEB9GV`**.
- **`public/consent-form.pdf`** real NYU IRB form (IRB-FY2026-11440, PI Mark Ho).

### ⚠️ The frontend is still on the OLD pool
`src/user/data/stimulus_pool.json` is still the **v1/v2 480-item pool**, and
`src/user/data/practice_items.json` is still the old 5-item practice set with categories C and D in
it. They were kept deliberately so the app still builds and deploys. **They do not match
`base-task/stimulus_pool.json`.** Replacing them is the next work item (§7).

### ⚠️ Prolific URL (was the source of a big bug)
Participants MUST enter via a URL routing to `#/welcome/prolific/` with the ID params, or they are
recorded `recruitmentService: "web"` with no `prolific_id` and cannot be bonused. Working format
(params BEFORE and AFTER the hash, trailing slash after codename):
```
https://www.codec-lab.org/e/<codename>/?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}#/welcome/prolific/?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}
```

### Deploys
`deploy.yml` deploys on push to ANY branch except `feat-* fix-* refactor-* test-* chore-* style-*
docs-* ci-*`, to a per-branch path (`/<owner>/<repo>/<branch>/`) with its own codename URL. So
**pushing `pilot-v4` gets its own staging site and does NOT touch main's live experiment.**
Pushing `main` deploys the live experiment; get explicit go-ahead first.
`gh run list` / `gh run watch` to monitor. A `deploy-error` workflow showing "skipped" on success is
normal.

---

## 6. Commands cheat-sheet

```bash
# Pool and observer (from base-task/)
cd base-task && python3 pool.py       # build -> stimulus_pool.json (240 items, seed 2026)
cd base-task && python3 verify.py     # independent checks; RUN AFTER ANY REBUILD, exits nonzero on failure
cd base-task && python3 bayes.py      # ideal observer -> bayes_per_item.json (expect 240/240)
cd base-task && python3 find_pairs.py 12    # per-misconception matched-pair yields

# Figures (from repo root)
python3 analysis-Bayesian/plot_bayes_1misc_heatmap.py
python3 analysis-Bayesian/plot_bayes_1misc_distributions.py   # dist_A, dist_B, and the profile figure

# Experiment
npm run dev                           # local
git push origin pilot-v4              # deploys to THIS BRANCH's staging URL only
npm run getdata ; npm run getrecruitment   # pull participant + recruitment data
```

---

## 7. What is next

1. **The form sampler.** Nothing can run until this exists. `src/user/utils/sampleForm.js` and a
   Python twin must draw a balanced 24-trial form from the new 240-item pool, rotating over rule x
   position x category. Then verify balance over 500 seeds in BOTH languages; they must stay in sync
   because the live experiment uses the JS one.
2. **Propagate the pool.** Copy `base-task/stimulus_pool.json` to `src/user/data/stimulus_pool.json`
   ONLY after the sampler can handle it. Doing it earlier breaks form assembly.
3. **Practice items.** The generator was deleted with the old categories. A new one is needed that
   produces A and B practice trials for this design. Keep the answer keys balanced: the old set was
   4 agree / 1 disagree, which shifted participants' criterion toward agreeing.
4. **More figures.** The heatmap, the A/B distributions and the six-marginal profile exist. A
   position figure is NOT worth building for the Bayes arm: the observer's answer is identical at
   both positions in 116 of 120 matched pairs (all 60 A pairs, 56 of 60 B pairs), and the 4 that
   differ do not agree on a direction. It becomes worth plotting once a human or LLM arm exists to
   lay against that flat reference.
5. **No human or LLM arm exists on this branch.** If either is wanted, it starts from scratch here.

---

## 8. Gotchas / rules

- **`sampleForm.js` (JS) must stay in sync with `sample_form` (Python).** The live experiment uses
  the JS one.
- **Pushing `main` deploys the live experiment.** Pushing `pilot-v4` deploys only to its own staging
  URL. Ask before pushing experiment-material changes.
- **Never commit** `data/real-all-main-data.json` (participant demographics) or any API key.
- **User preferences:** during design discussions, **finish the discussion before writing code**.
  On a **surprising result, verify our own stimuli and task first** before blaming participants.
  They run things themselves via `! <cmd>` and like work pushed rather than left local.
  **No em dashes in writing** (docs, reports, chat).
  **LaTeX compiles on Overleaf only**: keep tex folders self-contained (`figs/<exact-name>`), never
  install a local TeX toolchain.
- A post-commit hook regenerates `env/.env.git.local` and **fails in this worktree** with
  `Cannot find module '@codenamize/codenamize'` because `node_modules` is not installed here. The
  commit still lands. Run `npm install` if you need the env file.
- Commit messages end with the current model's co-author line.
