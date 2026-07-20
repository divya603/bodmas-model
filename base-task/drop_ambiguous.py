#!/usr/bin/env python3
"""
drop_ambiguous.py

Remove the 7 'ambiguous' foil items (1 in B, 6 in D) that extend_pool.py kept
in the file for continuity but that sample_form never draws. Dropping them lands
the pool on exactly 480 sampling-eligible items (120 per category), with every
B (present,foil,status) cell at 2 and every D (pair,foil,status) cell at 1.

Safe by construction: the ambiguous items sit under foil_status='ambiguous',
which no sampler key ever requests, so removing them cannot change any
participant's form. Item ids are left as-is (gaps are fine; ids are labels).

Writes all three pool copies. Run from base-task/:
    python3 drop_ambiguous.py
"""
import json
import shutil
from collections import Counter

POOL = 'stimulus_pool.json'
FRONTEND = '../src/user/data/stimulus_pool.json'
LLM = '../llm_exp/data/stimulus_pool.json'


def main():
    pool = json.load(open(POOL))
    amb = [it['id'] for it in pool if it.get('foil_status') == 'ambiguous']
    keep = [it for it in pool if it.get('foil_status') != 'ambiguous']
    print(f"dropping {len(amb)} ambiguous: {amb}")
    cats = Counter(it['category'] for it in keep)
    print(f"remaining {len(keep)} items, by category: {dict(sorted(cats.items()))}")
    assert len(keep) == 480 and all(cats[c] == 120 for c in 'ABCD'), "expected clean 480"

    with open(POOL, 'w') as f:
        json.dump(keep, f, indent=2)
    shutil.copy(POOL, FRONTEND)
    shutil.copy(POOL, LLM)
    print(f"wrote {POOL}, {FRONTEND}, {LLM}")


if __name__ == '__main__':
    main()
