"""Python port of the human task's per-participant form sampler
(base-task/stimulus_pool.py `sample_form` / `_index_pool`, same logic as the live
src/user/utils/sampleForm.js).

Each call to `sample_session(pool, seed)` reproduces ONE participant's 24-trial draw:
  A — each of the 6 misconceptions once (statement matches, agree correct).
  B — each of the 6 once as present; foil = fixed cyclic shift (disagree correct).
  C — each of the 6 once as target; 3 shown as the early error ('first') and 3 as the
      late error ('second'), rotated per participant; underlying pairs kept distinct.
  D — 6 distinct pairs, foil rotated across each pair's non-members (disagree correct).
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
    idx = {"A": {}, "B": {}, "C": {}, "D": {}}
    for it in pool:
        cat = it["category"]
        if cat == "A":
            key = it["misconceptions"][0]
        elif cat == "B":
            key = (it["misconceptions"][0], it["probed_misconception"])
        elif cat == "C":
            key = (it["probed_misconception"], it["which_target"])
        else:  # D
            key = (tuple(it["misconceptions"]), it["probed_misconception"])
        idx[cat].setdefault(key, []).append(it)
    return idx


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

    # B — one of each as present; foil = fixed nonzero cyclic shift (a bijection)
    k = rng.choice([1, 2, 3, 4, 5])
    for i, mid in enumerate(IDS):
        foil = IDS[(i + k) % len(IDS)]
        form.append(rng.choice(idx["B"][(mid, foil)]))

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

    # D — 6 distinct pairs, foil rotated across each pair's non-members
    offset = rng.randrange(4)
    for j, pair in enumerate(rng.sample(PAIRS, n_per_category)):
        others = [m for m in IDS if m not in pair]
        foil = others[(offset + j) % len(others)]
        form.append(rng.choice(idx["D"][(pair, foil)]))

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
