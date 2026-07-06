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
STUDENT_NAMES = ['Noah', 'Maya', 'Liam', 'Ava', 'Ethan', 'Zoe']
NAME_OFFSET   = 2  # decorrelates name rotation from the misconception rotation

STATEMENT_TEMPLATES = {
    'add_before_mul':        "{name} believes addition should be done before multiplication.",
    'add_before_div':        "{name} believes addition should be done before division.",
    'sub_before_mul':        "{name} believes subtraction should be done before multiplication.",
    'sub_before_div':        "{name} believes subtraction should be done before division.",
    'same_priority_rtl':     "{name} believes operations of the same priority should be worked right to left.",
    'outside_bracket_first': "{name} believes you should calculate outside the brackets before what's inside them.",
}


def _make_expression(misconceptions):
    bracket_prob = 1.0 if 'outside_bracket_first' in misconceptions else 0.4
    return generate_expression(n_ops=N_OPS, bracket_prob=bracket_prob)


def _pick_trace(dag, misconceptions):
    """
    Shortest diagnostic trace (>=1 step no expert trace ever takes) whose
    final answer differs from the expert's — the clearest single piece of
    "work" to show a participant. Falls back to any diagnostic trace if
    none reach a different final answer.
    """
    expert_traces = generate_traces(dag, [])
    expert_answer = correct_answer(expert_traces)
    candidates = diagnostic_traces(dag, misconceptions, expert_traces)
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

    candidates = [t for t in pair_traces if _implicates_target(t)]
    if not candidates:
        return None

    expert_answer = correct_answer(expert_traces)
    wrong_answer  = [t for t in candidates if t[-1] != expert_answer]
    pool = wrong_answer or candidates
    return min(pool, key=lambda t: (len(t), t))


def _build_item(item_id, category, misconceptions, probed, statement_correct,
                 which_target=None, trace_picker=None, student_name=None):
    trace_picker = trace_picker or (lambda dag: _pick_trace(dag, misconceptions))
    student_name = student_name or STUDENT_NAMES[0]
    for _ in range(MAX_ATTEMPTS):
        expr  = _make_expression(misconceptions)
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

    # C — pair, statement names one of the two (alternates first/second per pair)
    for i in range(n_per_category):
        pair   = PAIRS[i % len(PAIRS)]
        target = pair[0] if (i // len(PAIRS)) % 2 == 0 else pair[1]
        which  = 'first' if target == pair[0] else 'second'
        name   = STUDENT_NAMES[(i % len(PAIRS) + i // len(PAIRS)) % n_names]
        picker = lambda dag, pair=pair, target=target: _pick_trace_for_target(dag, pair, target)
        item   = _build_item(f'C{i:03d}', 'C', list(pair), target, True, which,
                              trace_picker=picker, student_name=name)
        if item:
            items.append(item)

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
    (present, probed[, target]) combination, rather than sampling blind."""
    idx = {'A': {}, 'B': {}, 'C': {}, 'D': {}}
    for it in pool:
        cat = it['category']
        if cat == 'A':
            key = it['misconceptions'][0]
        elif cat == 'B':
            key = (it['misconceptions'][0], it['probed_misconception'])
        else:  # C, D
            key = (tuple(it['misconceptions']), it['probed_misconception'])
        idx[cat].setdefault(key, []).append(it)
    return idx


def sample_form(pool, seed=None, n_per_category=6):
    """
    Build one participant's form with exact within-form balance, not just
    balance-in-aggregate-across-participants:

      A — each of the 6 misconceptions appears exactly once.
      B — each of the 6 misconceptions appears exactly once as the present
          misconception, AND (via a fixed nonzero cyclic shift, chosen per
          participant) exactly once as the foil — no misconception is
          over/under-used as foil within this form.
      C — n_per_category distinct pairs, split exactly in half between
          which_target='first' and 'second' (requires n_per_category even).
      D — n_per_category distinct pairs, foil rotated across each pair's 4
          non-member misconceptions via a per-participant offset.

    Different participants get a different shift/offset/pair-sample (all
    derived from `seed`), so coverage varies across participants while every
    individual form stays exactly balanced on its own.
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
    # so foil also covers all 6 exactly once (a cyclic shift is a bijection)
    k = rng.choice([1, 2, 3, 4, 5])
    for i, mid in enumerate(IDS):
        foil = IDS[(i + k) % len(IDS)]
        form.append(rng.choice(idx['B'][(mid, foil)]))

    # C — 6 distinct pairs, alternating target by position -> exact 3/3 split
    for j, pair in enumerate(rng.sample(PAIRS, n_per_category)):
        target = pair[0] if j % 2 == 0 else pair[1]
        form.append(rng.choice(idx['C'][(pair, target)]))

    # D — 6 distinct pairs, foil rotated across each pair's 4 non-members
    offset = rng.randrange(4)
    for j, pair in enumerate(rng.sample(PAIRS, n_per_category)):
        others = [m for m in IDS if m not in pair]
        foil   = others[(offset + j) % len(others)]
        form.append(rng.choice(idx['D'][(pair, foil)]))

    rng.shuffle(form)
    return form


if __name__ == '__main__':
    import time
    from collections import Counter

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
