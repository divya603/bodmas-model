"""
stimulus_pool.py

Generates a large, balanced pool of stimuli for the misconception-detection
task: an expression, a trace of "student work", and a belief statement about
what the student believes, rated on a 6-point agree/disagree scale.

Every item falls into one of 4 categories:
  A — 1 misconception in the trace, statement names it            (correct)
  B — 1 misconception in the trace, statement names a different one (foil)
  C — 2 misconceptions in the trace, statement names one of the two (correct)
  D — 2 misconceptions in the trace, statement names neither         (foil)

Coverage is deterministic round-robin, not random sampling, so every
misconception and every foil/target assignment appears exactly the same
number of times across the pool — 60 items per category is the smallest
size where every rotation (6 misconceptions, 15 pairs, 4 foils-per-pair)
lands on a whole number of cycles.
"""

import json
import random
import re
from itertools import combinations

from generator import generate_expression
from parser import build_dag
from learner import MISCONCEPTION_FLIPS
from distance import diagnostic_traces, correct_answer, tree_edges
from traces import generate_traces

IDS   = list(MISCONCEPTION_FLIPS.keys())
PAIRS = list(combinations(IDS, 2))

N_OPS         = 4
MAX_ATTEMPTS  = 60
# 24 names == form length, so sample_form can give every item a distinct
# name and no participant ever sees the same student twice. Names stored in
# the pool are placeholders, overwritten at sampling time. Keep in sync with
# STUDENT_NAMES in src/user/utils/sampleForm.js.
STUDENT_NAMES = [
    'Noah', 'Maya', 'Liam', 'Ava', 'Ethan', 'Zoe',
    'Mia', 'Lucas', 'Emma', 'Owen', 'Sofia', 'Caleb',
    'Ruby', 'Jonah', 'Isla', 'Felix', 'Nora', 'Dylan',
    'Priya', 'Marcus', 'Elena', 'Theo', 'Jasmine', 'Omar',
]
NAME_OFFSET   = 2  # decorrelates name rotation from the misconception rotation

STATEMENT_TEMPLATES = {
    'add_before_mul':        "{name} believes addition should be done before multiplication.",
    'add_before_div':        "{name} believes addition should be done before division.",
    'sub_before_mul':        "{name} believes subtraction should be done before multiplication.",
    'sub_before_div':        "{name} believes subtraction should be done before division.",
    'same_priority_rtl':     "{name} believes operations of the same priority should be worked right to left.",
    'outside_bracket_first': "{name} believes you should calculate outside the brackets before what's inside them.",
}


def _make_expression(misconceptions, probed):
    """
    Brackets are mandatory not only when outside_bracket_first is present in
    the trace, but also when the statement merely *names* it (a B/D foil):
    without brackets the statement is trivially wrong, so the item measures
    nothing.
    """
    needs_brackets = ('outside_bracket_first' in misconceptions
                      or probed == 'outside_bracket_first')
    bracket_prob = 1.0 if needs_brackets else 0.4
    return generate_expression(n_ops=N_OPS, bracket_prob=bracket_prob)


def _is_finished(trace):
    """
    True if a trace actually reduces all the way to a single number, rather
    than getting stuck (e.g. the only remaining move is a divide-by-zero,
    which is deliberately never fired — see is_zero_divide). A stuck trace
    is a broken stimulus: it looks like the "work" just stops mid-expression.
    """
    return len(trace[-1].split()) == 1


_LONG_DECIMAL = re.compile(r'\d\.\d{3,}')


def _is_clean(trace):
    """
    True if every number in every step has at most 2 decimal places.
    Divisions like 2 ÷ 7 produce values such as 0.285714 (rounded to 6
    places in valid_actions.py), which look unnatural as hand-written
    student work; rejecting the trace lets the retry loop draw a fresh
    expression instead of showing rounded-and-inconsistent arithmetic.
    """
    return not any(_LONG_DECIMAL.search(step) for step in trace)


def _pick_trace(dag, misconceptions):
    """
    Shortest diagnostic trace (>=1 step no expert trace ever takes) whose
    final answer differs from the expert's — the clearest single piece of
    "work" to show a participant. Falls back to any diagnostic trace if
    none reach a different final answer. Always excludes stuck traces.
    """
    expert_traces = generate_traces(dag, [])
    expert_answer = correct_answer(expert_traces)
    candidates = [t for t in diagnostic_traces(dag, misconceptions, expert_traces)
                  if _is_finished(t) and _is_clean(t)]
    if not candidates:
        return None
    wrong_answer = [t for t in candidates if t[-1] != expert_answer]
    pool = wrong_answer or candidates
    return min(pool, key=lambda t: (len(t), t))


def _pick_trace_for_target(dag, pair, target):
    """
    For a 2-misconception trace, require a step that specifically implicates
    `target` — one that isn't explainable by the expert alone OR by having
    only the OTHER misconception in the pair. Without this, a "partial
    match" item could be answered correctly using evidence that's actually
    about the other misconception, not the one the statement names.
    """
    other = next(m for m in pair if m != target)
    expert_traces = generate_traces(dag, [])
    other_traces  = generate_traces(dag, [other])
    reference     = tree_edges(expert_traces) | tree_edges(other_traces)

    pair_traces = generate_traces(dag, list(pair))

    def _implicates_target(trace):
        return any((trace[i], trace[i + 1]) not in reference
                   for i in range(len(trace) - 1))

    candidates = [t for t in pair_traces
                  if _implicates_target(t) and _is_finished(t) and _is_clean(t)]
    if not candidates:
        return None

    expert_answer = correct_answer(expert_traces)
    wrong_answer  = [t for t in candidates if t[-1] != expert_answer]
    pool = wrong_answer or candidates
    return min(pool, key=lambda t: (len(t), t))


def _error_order(dag, trace, target, other):
    """
    Chronological position of `target`'s error relative to its pair partner
    `other` in the trace: 'first' if the target's deviation appears before the
    partner's, 'second' if after, 'ambiguous' if a single step is explainable
    by both, None if no attributable deviation is found. This is what
    which_target means in the rebuilt category C — early vs late error, the
    only "1st/2nd" a participant can actually perceive (canonical pair order is
    invisible to them and confounded with misconception identity).
    """
    expert = tree_edges(generate_traces(dag, []))
    tdev   = tree_edges(generate_traces(dag, [target])) - expert
    odev   = tree_edges(generate_traces(dag, [other]))  - expert
    for i in range(len(trace) - 1):
        e = (trace[i], trace[i + 1])
        if e in expert:
            continue
        in_t, in_o = e in tdev, e in odev
        if in_t and not in_o:
            return 'first'
        if in_o and not in_t:
            return 'second'
        if in_t and in_o:
            return 'ambiguous'
    return None


def _pick_trace_ordered(dag, target, other, position):
    """
    Like _pick_trace_for_target, but only accepts traces where the target's
    error is chronologically `position` ('first'/'second') relative to the
    partner's. Returns None (triggering an expression retry) when this
    expression can't produce the requested ordering — that rejection sampling
    is what lets us balance each misconception across early- and late-error.
    """
    pair          = tuple(sorted((target, other), key=IDS.index))
    expert_traces = generate_traces(dag, [])
    other_traces  = generate_traces(dag, [other])
    reference     = tree_edges(expert_traces) | tree_edges(other_traces)
    pair_traces   = generate_traces(dag, list(pair))

    def _implicates_target(trace):
        return any((trace[i], trace[i + 1]) not in reference
                   for i in range(len(trace) - 1))

    candidates = [t for t in pair_traces
                  if _implicates_target(t) and _is_finished(t) and _is_clean(t)
                  and _error_order(dag, t, target, other) == position]
    if not candidates:
        return None

    expert_answer = correct_answer(expert_traces)
    wrong_answer  = [t for t in candidates if t[-1] != expert_answer]
    pool = wrong_answer or candidates
    return min(pool, key=lambda t: (len(t), t))


def _build_category_C(per_target_position=5):
    """
    Rebuilt category C, balanced by the chronological position of the target's
    error. Each of the 6 misconceptions is the target `per_target_position`
    times as the EARLY error (which_target='first') and the same number as the
    LATE error ('second'), rotating through its 5 partners and rejection-
    sampling expressions until the requested ordering holds. Result: 60 items,
    each misconception 5 'first' + 5 'second' — no confound with identity.
    """
    items = []
    n_names = len(STUDENT_NAMES)
    counter = 0
    for target in IDS:
        partners = [m for m in IDS if m != target]
        for position in ('first', 'second'):
            made, pi, guard = 0, 0, 0
            while made < per_target_position and guard < 8 * len(partners):
                other = partners[pi % len(partners)]
                pi += 1
                guard += 1
                pair   = tuple(sorted((target, other), key=IDS.index))
                name   = STUDENT_NAMES[counter % n_names]
                picker = (lambda dag, t=target, o=other, p=position:
                          _pick_trace_ordered(dag, t, o, p))
                item = _build_item(f'C{counter:03d}', 'C', list(pair), target, True,
                                   which_target=position, trace_picker=picker,
                                   student_name=name)
                if item:
                    items.append(item)
                    made += 1
                    counter += 1
            if made < per_target_position:
                print(f"WARN: only made {made}/{per_target_position} C items "
                      f"for target={target} position={position}")
    return items


def _build_item(item_id, category, misconceptions, probed, statement_correct,
                 which_target=None, trace_picker=None, student_name=None):
    trace_picker = trace_picker or (lambda dag: _pick_trace(dag, misconceptions))
    student_name = student_name or STUDENT_NAMES[0]
    for _ in range(MAX_ATTEMPTS):
        expr  = _make_expression(misconceptions, probed)
        dag   = build_dag(expr)
        trace = trace_picker(dag)
        if trace is not None:
            break
    else:
        return None

    return dict(
        id=item_id,
        category=category,
        expression=expr,
        n_ops=N_OPS,
        misconceptions=list(misconceptions),
        num_misconceptions=len(misconceptions),
        trace=trace,
        probed_misconception=probed,
        statement_correct=statement_correct,
        which_target=which_target,
        student_name=student_name,
        belief_statement=STATEMENT_TEMPLATES[probed].format(name=student_name),
    )


def build_pool(n_per_category=60):
    items = []
    n_names = len(STUDENT_NAMES)

    # A — single, statement matches
    # name index folds in the repetition count (i // 6), not just i itself, so
    # a given misconception cycles through *different* names across its 10
    # occurrences rather than being permanently paired with just one
    for i in range(n_per_category):
        mid  = IDS[i % len(IDS)]
        name = STUDENT_NAMES[(i % len(IDS) + i // len(IDS)) % n_names]
        item = _build_item(f'A{i:03d}', 'A', [mid], mid, True, student_name=name)
        if item:
            items.append(item)

    # B — single, statement is a foil (cycles all 5 non-present misconceptions per mid)
    for i in range(n_per_category):
        mid    = IDS[i % len(IDS)]
        others = [m for m in IDS if m != mid]
        foil   = others[(i // len(IDS)) % len(others)]
        name   = STUDENT_NAMES[(i % len(IDS) + i // len(IDS) + 3) % n_names]
        item   = _build_item(f'B{i:03d}', 'B', [mid], foil, False, student_name=name)
        if item:
            items.append(item)

    # C — pair, statement names one present misconception, balanced by the
    # chronological position of the target's error (which_target='first'/'second')
    items.extend(_build_category_C(per_target_position=n_per_category // len(IDS) // 2))

    # D — pair, statement names neither (cycles all 4 non-member misconceptions per pair)
    for i in range(n_per_category):
        pair   = PAIRS[i % len(PAIRS)]
        others = [m for m in IDS if m not in pair]
        foil   = others[(i // len(PAIRS)) % len(others)]
        name   = STUDENT_NAMES[(i % len(PAIRS) + i // len(PAIRS) + 2) % n_names]
        item   = _build_item(f'D{i:03d}', 'D', list(pair), foil, False, None, student_name=name)
        if item:
            items.append(item)

    return items


def _index_pool(pool):
    """Lookup indices used by sample_form to fetch an item matching an exact
    (present, probed[, target/status]) combination, rather than sampling blind.
    B/D keys include foil_status ('refuted'/'unsupported'; written by
    extend_pool.py), so sample_form REQUIRES the extended pool. Items with
    foil_status 'ambiguous' land under unreachable keys and are never drawn.
    'D_any' is a status-agnostic D index used only by the emergency fallback."""
    idx = {'A': {}, 'B': {}, 'C': {}, 'D': {}, 'D_any': {}}
    for it in pool:
        cat = it['category']
        if cat == 'A':
            key = it['misconceptions'][0]
        elif cat == 'B':
            key = (it['misconceptions'][0], it['probed_misconception'],
                   it.get('foil_status'))
        elif cat == 'C':
            key = (it['probed_misconception'], it['which_target'])   # (target, early/late)
        else:  # D
            key = (tuple(it['misconceptions']), it['probed_misconception'],
                   it.get('foil_status'))
            idx['D_any'].setdefault(key[:2], []).append(it)
        idx[cat].setdefault(key, []).append(it)
    return idx


def _match_foils(rules, pairs, ok):
    """Backtracking perfect matching: assign each rule a distinct pair with
    ok(rule, pair) true. Returns {rule: pair} in `rules` order, or None."""
    assignment, used = {}, set()

    def bt(i):
        if i == len(rules):
            return True
        for p in pairs:
            if p in used or not ok(rules[i], p):
                continue
            used.add(p)
            assignment[rules[i]] = p
            if bt(i + 1):
                return True
            used.discard(p)
            del assignment[rules[i]]
        return False

    return assignment if bt(0) else None


def sample_form(pool, seed=None, n_per_category=6):
    """
    Build one participant's form with exact within-form balance, not just
    balance-in-aggregate-across-participants:

      A — each of the 6 misconceptions appears exactly once.
      B — each of the 6 misconceptions appears exactly once as the present
          misconception, AND (via a fixed nonzero cyclic shift, chosen per
          participant) exactly once as the foil — no misconception is
          over/under-used as foil within this form. Of the 6 foils, exactly 3
          are ACTIVELY REFUTED by the trace (foil_status='refuted') and 3 are
          merely unsupported; which rules are refuted rotates per participant.
      C — each of the 6 misconceptions probed once as target; exactly half
          shown as the EARLY error (which_target='first') and half as the LATE
          error ('second'), with which misconceptions are 'first' rotated per
          participant so coverage is even across the sample. Pairs kept distinct
          within a form.
      D — 6 distinct pairs whose foils cover all 6 misconceptions exactly once
          (backtracking assignment; foil never a member of its pair), with the
          refuted set the COMPLEMENT of B's. Across B+D every rule is therefore
          named as a foil exactly twice per form: once refuted, once
          unsupported — 6 refuted + 6 unsupported foil trials per participant.

    Requires the EXTENDED pool (extend_pool.py: foil_status on all B/D items).

    Different participants get a different shift/offset/pair-sample (all
    derived from `seed`), so coverage varies across participants while every
    individual form stays exactly balanced on its own.

    Every item also gets a distinct student name (24 names, 24 items), so
    no participant ever sees the same student twice.
    """
    if n_per_category != 6:
        raise ValueError("sample_form's exact-balance guarantees require "
                          "n_per_category == 6 (== number of misconceptions)")

    rng = random.Random(seed)
    idx = _index_pool(pool)
    form = []

    # A — one of each misconception
    for mid in IDS:
        form.append(rng.choice(idx['A'][mid]))

    # B — one of each misconception as present; foil = fixed nonzero shift,
    # so foil also covers all 6 exactly once (a cyclic shift is a bijection).
    # 3 foils refuted + 3 unsupported; which rules are refuted rotates via r,
    # and category D below uses the complementary set.
    k = rng.choice([1, 2, 3, 4, 5])
    r = rng.randrange(len(IDS))
    refuted_B = {IDS[(r + j) % len(IDS)] for j in range(len(IDS) // 2)}
    for i, mid in enumerate(IDS):
        foil = IDS[(i + k) % len(IDS)]
        status = 'refuted' if foil in refuted_B else 'unsupported'
        form.append(rng.choice(idx['B'][(mid, foil, status)]))

    # C — each of the 6 misconceptions probed once as target; 3 shown as the
    # early error ('first'), 3 as the late error ('second'); which 3 are 'first'
    # rotates per participant via `shift`. Retry until the 6 underlying pairs
    # are distinct (partners can otherwise collide); fall back to allowing a
    # repeat rather than failing.
    shift = rng.randrange(len(IDS))
    positions = ['first' if (i + shift) % len(IDS) < n_per_category // 2 else 'second'
                 for i in range(len(IDS))]

    def _pick_C(require_distinct):
        chosen, used = [], set()
        for mid, position in zip(IDS, positions):
            cands = idx['C'].get((mid, position), [])
            fresh = [it for it in cands
                     if tuple(sorted(it['misconceptions'])) not in used]
            if require_distinct and not fresh:
                return None
            it = rng.choice(fresh or cands)
            used.add(tuple(sorted(it['misconceptions'])))
            chosen.append(it)
        return chosen

    c_items = None
    for _ in range(25):
        c_items = _pick_C(require_distinct=True)
        if c_items is not None:
            break
    form.extend(c_items if c_items is not None else _pick_C(require_distinct=False))

    # D — 6 distinct pairs whose foils cover all 6 rules exactly once (foil
    # never a member of its pair), refuted set = complement of B's, so each
    # rule is refuted exactly once per form across B+D combined.
    refuted_D = set(IDS) - refuted_B

    def _d_status(rule):
        return 'refuted' if rule in refuted_D else 'unsupported'

    d_items = None
    for _ in range(50):
        pairs = rng.sample(PAIRS, n_per_category)
        order = IDS[:]
        rng.shuffle(order)
        ok = lambda rule, pair: (rule not in pair
                                 and idx['D'].get((pair, rule, _d_status(rule))))
        assignment = _match_foils(order, pairs, ok)
        if assignment:
            d_items = [rng.choice(idx['D'][(pair, rule, _d_status(rule))])
                       for rule, pair in assignment.items()]
            break
    if d_items is None:
        # emergency fallback (should not happen with the full 120-cell pool):
        # old rotation logic, status ignored — a mildly unbalanced form beats
        # a crashed session
        d_items = []
        offset = rng.randrange(4)
        for j, pair in enumerate(rng.sample(PAIRS, n_per_category)):
            others = [m for m in IDS if m not in pair]
            foil = others[(offset + j) % len(others)]
            d_items.append(rng.choice(idx['D_any'][(pair, foil)]))
    form.extend(d_items)

    rng.shuffle(form)

    # Assign each item a distinct student name (copies, so the shared pool
    # dicts are never mutated), rewriting the belief statement to match.
    names = STUDENT_NAMES[:]
    rng.shuffle(names)
    form = [dict(it,
                 student_name=name,
                 belief_statement=it['belief_statement'].replace(it['student_name'], name, 1))
            for it, name in zip(form, names)]
    return form


if __name__ == '__main__':
    import sys
    import time
    from collections import Counter

    if '--rebuild-240' not in sys.argv:
        sys.exit("stimulus_pool.json is now the EXTENDED 480-design pool "
                 "(extend_pool.py). Running this script would overwrite it "
                 "with a fresh 240-item pool WITHOUT foil_status, which "
                 "sample_form requires. If you really want that, pass "
                 "--rebuild-240 and then re-run extend_pool.py afterwards.")

    t0   = time.time()
    pool = build_pool()
    dt   = time.time() - t0

    print(f"Generated {len(pool)} items in {dt:.1f}s")
    print("By category:", dict(Counter(it['category'] for it in pool)))

    for cat in 'ABCD':
        cat_items = [it for it in pool if it['category'] == cat]
        present_counts = Counter(m for it in cat_items for m in it['misconceptions'])
        probed_counts  = Counter(it['probed_misconception'] for it in cat_items)
        print(f"\nCategory {cat} (n={len(cat_items)}):")
        print(f"  present as ground truth : {dict(sorted(present_counts.items()))}")
        print(f"  probed / named in stmt  : {dict(sorted(probed_counts.items()))}")

    with open('stimulus_pool.json', 'w') as f:
        json.dump(pool, f, indent=2)
    print(f"\nWrote stimulus_pool.json ({len(pool)} items)")

    form = sample_form(pool, seed=1)
    form_counts = Counter(it['category'] for it in form)
    right_wrong = Counter(it['statement_correct'] for it in form)
    print(f"\nExample participant form: {len(form)} items")
    print("  by category:", dict(form_counts))
    print("  statement_correct counts:", dict(right_wrong))
