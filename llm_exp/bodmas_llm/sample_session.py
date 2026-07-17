"""Python port of the human task's per-participant form sampler
(base-task/stimulus_pool.py `sample_form` / `_index_pool`, same logic as the live
src/user/utils/sampleForm.js).

Each call to `sample_session(pool, seed)` reproduces ONE participant's 24-trial draw:
  A — each of the 6 misconceptions once (statement matches, agree correct).
  B — each of the 6 once as present; foil = fixed cyclic shift (disagree correct);
      3 foils actively refuted + 3 unsupported, the refuted set rotating per participant.
  C — each of the 6 once as target; 3 shown as the early error ('first') and 3 as the
      late error ('second'), rotated per participant; underlying pairs kept distinct.
  D — 6 distinct pairs whose foils cover all 6 rules exactly once (disagree correct);
      refuted set = complement of B's, so across B+D each rule is named as a foil
      twice per form: once refuted, once unsupported. Requires the extended pool.
Then globally shuffled, and every item gets a distinct student name.

Seeded with `random.Random(seed)` so each synthetic "subject" is reproducible, and N
seeds give N distinct subjects (mirroring N human participants). Grouping key downstream
is `subject_id`; the item identity is `id`.
"""

from __future__ import annotations

import json
import random
from itertools import combinations
from pathlib import Path

IDS = [
    "add_before_mul", "add_before_div", "sub_before_mul",
    "sub_before_div", "same_priority_rtl", "outside_bracket_first",
]
PAIRS = list(combinations(IDS, 2))

# Keep in sync with STUDENT_NAMES in base-task/stimulus_pool.py (24 names).
STUDENT_NAMES = [
    "Noah", "Maya", "Liam", "Ava", "Ethan", "Zoe",
    "Mia", "Lucas", "Emma", "Owen", "Sofia", "Caleb",
    "Ruby", "Jonah", "Isla", "Felix", "Nora", "Dylan",
    "Priya", "Marcus", "Elena", "Theo", "Jasmine", "Omar",
]


def load_pool(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _index_pool(pool: list[dict]) -> dict:
    """B/D keys include foil_status ('refuted'/'unsupported'; written by
    base-task/extend_pool.py), so sample_session REQUIRES the extended pool.
    Items with foil_status 'ambiguous' are never drawn. 'D_any' is the
    status-agnostic D index used only by the emergency fallback."""
    idx = {"A": {}, "B": {}, "C": {}, "D": {}, "D_any": {}}
    for it in pool:
        cat = it["category"]
        if cat == "A":
            key = it["misconceptions"][0]
        elif cat == "B":
            key = (it["misconceptions"][0], it["probed_misconception"],
                   it.get("foil_status"))
        elif cat == "C":
            key = (it["probed_misconception"], it["which_target"])
        else:  # D
            key = (tuple(it["misconceptions"]), it["probed_misconception"],
                   it.get("foil_status"))
            idx["D_any"].setdefault(key[:2], []).append(it)
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


def sample_session(pool: list[dict], seed: int, n_per_category: int = 6) -> list[dict]:
    """Reproduce one participant's balanced 24-trial form. Deterministic for (pool, seed)."""
    if n_per_category != 6:
        raise ValueError("n_per_category must be 6 (== number of misconceptions)")

    rng = random.Random(seed)
    idx = _index_pool(pool)
    form: list[dict] = []

    # A — one of each misconception
    for mid in IDS:
        form.append(rng.choice(idx["A"][mid]))

    # B — one of each as present; foil = fixed nonzero cyclic shift (a bijection).
    # 3 foils refuted + 3 unsupported (refuted set rotates via r; D uses the complement)
    k = rng.choice([1, 2, 3, 4, 5])
    r = rng.randrange(len(IDS))
    refuted_B = {IDS[(r + j) % len(IDS)] for j in range(len(IDS) // 2)}
    for i, mid in enumerate(IDS):
        foil = IDS[(i + k) % len(IDS)]
        status = "refuted" if foil in refuted_B else "unsupported"
        form.append(rng.choice(idx["B"][(mid, foil, status)]))

    # C — each misconception once as target; 3 'first' / 3 'second' (rotated); distinct pairs
    shift = rng.randrange(len(IDS))
    positions = ["first" if (i + shift) % len(IDS) < n_per_category // 2 else "second"
                 for i in range(len(IDS))]

    def _pick_C(require_distinct):
        chosen, used = [], set()
        for mid, position in zip(IDS, positions):
            cands = idx["C"].get((mid, position), [])
            fresh = [it for it in cands if tuple(sorted(it["misconceptions"])) not in used]
            if require_distinct and not fresh:
                return None
            it = rng.choice(fresh or cands)
            used.add(tuple(sorted(it["misconceptions"])))
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
        return "refuted" if rule in refuted_D else "unsupported"

    d_items = None
    for _ in range(50):
        pairs = rng.sample(PAIRS, n_per_category)
        order = IDS[:]
        rng.shuffle(order)
        ok = lambda rule, pair: (rule not in pair
                                 and idx["D"].get((pair, rule, _d_status(rule))))
        assignment = _match_foils(order, pairs, ok)
        if assignment:
            d_items = [rng.choice(idx["D"][(pair, rule, _d_status(rule))])
                       for rule, pair in assignment.items()]
            break
    if d_items is None:
        # emergency fallback: old rotation logic, status ignored
        d_items = []
        offset = rng.randrange(4)
        for j, pair in enumerate(rng.sample(PAIRS, n_per_category)):
            others = [m for m in IDS if m not in pair]
            foil = others[(offset + j) % len(others)]
            d_items.append(rng.choice(idx["D_any"][(pair, foil)]))
    form.extend(d_items)

    rng.shuffle(form)

    # distinct student name per item (copies, so the shared pool dicts are never mutated)
    names = STUDENT_NAMES[:]
    rng.shuffle(names)
    form = [
        dict(it,
             student_name=name,
             belief_statement=it["belief_statement"].replace(it["student_name"], name, 1))
        for it, name in zip(form, names)
    ]
    return form
