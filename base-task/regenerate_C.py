#!/usr/bin/env python3
"""
regenerate_C.py

Regenerate ONLY the category-C items of stimulus_pool.json, preserving the
existing A/B/D items exactly (so already-run participants' non-C data stays
comparable). The new C items are balanced by the chronological position of the
target's error — each misconception is the target 5x as the early error
(which_target='first') and 5x as the late error ('second') — replacing the old
canonical-order C items, which were ~all 'first' and confounded with identity.

Run from base-task/:
    python3 regenerate_C.py
"""

import json
import random
import shutil
import time
from collections import Counter

from stimulus_pool import _build_category_C, sample_form

POOL = 'stimulus_pool.json'
FRONTEND_COPY = '../src/user/data/stimulus_pool.json'


def main():
    random.seed(0)  # reproducible regeneration
    pool = json.load(open(POOL))

    kept = {c: [it for it in pool if it['category'] == c] for c in 'ABD'}
    print("preserved:", {c: len(kept[c]) for c in 'ABD'})

    t0 = time.time()
    new_C = _build_category_C(per_target_position=5)
    print(f"generated {len(new_C)} category-C items in {time.time() - t0:.1f}s")

    # reassemble in A, B, C, D order
    new_pool = kept['A'] + kept['B'] + new_C + kept['D']

    # ── verify balance ──────────────────────────────────────────────
    first = Counter(it['probed_misconception'] for it in new_C if it['which_target'] == 'first')
    second = Counter(it['probed_misconception'] for it in new_C if it['which_target'] == 'second')
    print("\nC target x which_target balance (want 5 / 5 each):")
    for m in sorted(set(first) | set(second)):
        print(f"  {m:24s} first={first[m]:2d}  second={second[m]:2d}")

    # ── verify a sampled form is 3 first / 3 second ─────────────────
    form = sample_form(new_pool, seed=1)
    cform = [it for it in form if it['category'] == 'C']
    wt = Counter(it['which_target'] for it in cform)
    print(f"\nexample form category-C which_target counts: {dict(wt)}  (want first=3, second=3)")

    with open(POOL, 'w') as f:
        json.dump(new_pool, f, indent=2)
    shutil.copy(POOL, FRONTEND_COPY)
    print(f"\nwrote {len(new_pool)} items to {POOL} and copied to {FRONTEND_COPY}")


if __name__ == '__main__':
    main()
