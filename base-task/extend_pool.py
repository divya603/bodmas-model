#!/usr/bin/env python3
"""
extend_pool.py

Extend the 240-item pool to ~480, preserving every existing item's stimulus
fields exactly (regenerate_C precedent), and make the foil's REFUTATION STATUS
a controlled first-class factor in categories B and D.

Target design (per category):
  A - 6 rules x 20 items                                   (60 kept + 60 new)
  B - 6 present x 5 foils x {refuted, unsupported} x 2     (60 kept, classified + complement)
  C - 6 targets x {first, second} x 10                     (60 kept + 60 new)
  D - 15 pairs x 4 foils x {refuted, unsupported} x 1      (60 kept, classified + complement)

Refutation status of the named foil, given the trace. Two signals:
  visible  - the trace passes a decision point where the foil had a concrete
             opportunity to manifest and the student demonstrably didn't take
             it. Computed state-locally: either the observed step is INVALID
             under the (present + foil) learner (the foil forbids what the
             student did), or the foil OFFERS an extra action at some state
             that the student never took (covers outside_bracket_first's soft
             refutations, where the conventional action stays valid).
  marginal - ideal-observer P(foil in policy | trace), cut at 0.15 (the cut
             already registered in analysis-Bayesian/b_item_marginals.json).

  refuted      visible AND marginal < 0.15
  unsupported  not visible AND 0.15 <= marginal <= 0.35
  ambiguous    the two signals disagree (e.g. D's combinatorial suppression:
               marginal is low only because foil-profiles fail to explain one
               of the two errors - nothing a participant can see). Excluded
               from the sampling cells; old ambiguous items stay in the pool
               file for continuity but are never drawn.
  supported    marginal > 0.35 (a confounded foil) - rejected outright for new
               items, loudly flagged if an old item lands there.

Every B/D item (old and new) gets: foil_status (str) and io_foil_marginal
(float). A/C items are untouched. Old items' existing fields are asserted
unchanged. New expressions are deduped against the whole pool, the 3 human
practice items, and each other.

Run from base-task/ (several minutes):
    python3 extend_pool.py
"""

import json
import random
import shutil
import time
from collections import Counter

from parser import build_dag
from misconceptions import dag_to_str
from traces import _next_dags
from stimulus_pool import (IDS, PAIRS, STUDENT_NAMES, _build_item,
                           _pick_trace_ordered)
from inference import posterior_over_profiles, marginal_rule_probability

POOL = 'stimulus_pool.json'
FRONTEND_COPY = '../src/user/data/stimulus_pool.json'
LLM_COPY = '../llm_exp/data/stimulus_pool.json'
PRACTICE = '../src/user/data/practice_items.json'

REFUTED_CUT = 0.15
SUPPORTED_CUT = 0.35
SEED = 20260717
MAX_TRIES = 4000   # per needed item; hardest observed cells yield ~1-7%


# ── refutation classification ───────────────────────────────────────

def _next_strs(state, misconceptions):
    dag = build_dag(state)
    return {dag_to_str(d) for d in _next_dags(dag, list(misconceptions))}


def _visibly_refuted(trace, present, foil):
    """The foil had a concrete, participant-visible chance to manifest and the
    student didn't take it: at some state, adding the foil to the learner's
    policy either forbids the observed step or offers an extra action that was
    never taken."""
    for t in range(len(trace) - 1):
        base = _next_strs(trace[t], present)
        withf = _next_strs(trace[t], list(present) + [foil])
        if trace[t + 1] not in withf:
            return True          # foil forbids what the student actually did
        if withf - base:
            return True          # foil offered a distinct action, never taken
    return False


def classify(item):
    """-> (status, marginal). Status in refuted/unsupported/ambiguous/supported."""
    marg = marginal_rule_probability(
        posterior_over_profiles(item['trace']), item['probed_misconception'])
    visible = _visibly_refuted(item['trace'], item['misconceptions'],
                               item['probed_misconception'])
    if marg > SUPPORTED_CUT:
        status = 'supported'
    elif visible and marg < REFUTED_CUT:
        status = 'refuted'
    elif not visible and marg >= REFUTED_CUT:
        status = 'unsupported'
    else:
        status = 'ambiguous'
    return status, round(marg, 4)


# ── generation helpers ──────────────────────────────────────────────

def _gen_foil_item(item_id, category, present, foil, want_status, name,
                   used_exprs):
    """Rejection-sample a B/D item whose foil classifies as want_status."""
    for _ in range(MAX_TRIES):
        item = _build_item(item_id, category, list(present), foil, False,
                           student_name=name)
        if item is None or item['expression'] in used_exprs:
            continue
        status, marg = classify(item)
        if status != want_status:
            continue
        item['foil_status'] = status
        item['io_foil_marginal'] = marg
        used_exprs.add(item['expression'])
        return item
    return None


def _gen_A(item_id, mid, name, used_exprs):
    for _ in range(MAX_TRIES):
        item = _build_item(item_id, 'A', [mid], mid, True, student_name=name)
        if item is not None and item['expression'] not in used_exprs:
            used_exprs.add(item['expression'])
            return item
    return None


def _gen_C(item_id, target, other, position, name, used_exprs):
    pair = tuple(sorted((target, other), key=IDS.index))
    picker = (lambda dag, t=target, o=other, p=position:
              _pick_trace_ordered(dag, t, o, p))
    for _ in range(8):   # _build_item already retries expressions internally
        item = _build_item(item_id, 'C', list(pair), target, True,
                           which_target=position, trace_picker=picker,
                           student_name=name)
        if item is not None and item['expression'] not in used_exprs:
            used_exprs.add(item['expression'])
            return item
    return None


def main():
    random.seed(SEED)
    t0 = time.time()

    pool = json.load(open(POOL))
    originals = {it['id']: json.loads(json.dumps(it)) for it in pool}
    by_cat = {c: [it for it in pool if it['category'] == c] for c in 'ABCD'}
    practice = json.load(open(PRACTICE))
    used_exprs = ({it['expression'] for it in pool}
                  | {p['expression'] for p in practice})

    # ── 1. classify existing B and D items ──────────────────────────
    print("classifying existing B/D items ...", flush=True)
    status_counts = {'B': Counter(), 'D': Counter()}
    for cat in 'BD':
        for it in by_cat[cat]:
            status, marg = classify(it)
            it['foil_status'] = status
            it['io_foil_marginal'] = marg
            status_counts[cat][status] += 1
    print(f"  existing B: {dict(status_counts['B'])}")
    print(f"  existing D: {dict(status_counts['D'])}")
    for cat in 'BD':
        if status_counts[cat]['supported']:
            print(f"  WARNING: {status_counts[cat]['supported']} existing {cat} "
                  f"items are foil-SUPPORTED (marginal > {SUPPORTED_CUT})")

    # ── 2. extend A: +60 (same rotation formulas as build_pool) ─────
    print("\nextending category A ...", flush=True)
    new_A = []
    for i in range(60, 120):
        mid = IDS[i % len(IDS)]
        name = STUDENT_NAMES[(i % len(IDS) + i // len(IDS)) % len(STUDENT_NAMES)]
        item = _gen_A(f'A{i:03d}', mid, name, used_exprs)
        if item:
            new_A.append(item)
        else:
            print(f"  WARN: could not build A item for {mid}")
    print(f"  built {len(new_A)} new A items")

    # ── 3. top up B to 2 refuted + 2 unsupported per (present, foil) ─
    print("\ntopping up category B ...", flush=True)
    new_B, b_id = [], 60
    b_fail = []
    for mi, mid in enumerate(IDS):
        for foil in IDS:
            if foil == mid:
                continue
            have = Counter(it['foil_status'] for it in by_cat['B']
                           if it['misconceptions'][0] == mid
                           and it['probed_misconception'] == foil)
            for status in ('refuted', 'unsupported'):
                for _ in range(max(0, 2 - have[status])):
                    name = STUDENT_NAMES[(mi + b_id) % len(STUDENT_NAMES)]
                    item = _gen_foil_item(f'B{b_id:03d}', 'B', [mid], foil,
                                          status, name, used_exprs)
                    if item:
                        new_B.append(item)
                        b_id += 1
                    else:
                        b_fail.append((mid, foil, status))
            print(f"  {mid:22s} foil={foil:22s} done", flush=True)
    print(f"  built {len(new_B)} new B items; failures: {b_fail or 'none'}")

    # ── 4. extend C: +60, chronologically balanced (5 more per cell) ─
    print("\nextending category C ...", flush=True)
    new_C, c_id = [], 60
    for target in IDS:
        partners = [m for m in IDS if m != target]
        for position in ('first', 'second'):
            made, pi, guard = 0, 0, 0
            while made < 5 and guard < 8 * len(partners):
                other = partners[pi % len(partners)]
                pi += 1
                guard += 1
                name = STUDENT_NAMES[c_id % len(STUDENT_NAMES)]
                item = _gen_C(f'C{c_id:03d}', target, other, position, name,
                              used_exprs)
                if item:
                    new_C.append(item)
                    made += 1
                    c_id += 1
            if made < 5:
                print(f"  WARN: only {made}/5 new C items for "
                      f"target={target} position={position}")
    print(f"  built {len(new_C)} new C items")

    # ── 5. top up D to 1 refuted + 1 unsupported per (pair, foil) ────
    print("\ntopping up category D ...", flush=True)
    new_D, d_id = [], 60
    d_fail = []
    for pi_, pair in enumerate(PAIRS):
        for foil in [m for m in IDS if m not in pair]:
            have = Counter(it['foil_status'] for it in by_cat['D']
                           if tuple(it['misconceptions']) == pair
                           and it['probed_misconception'] == foil)
            for status in ('refuted', 'unsupported'):
                for _ in range(max(0, 1 - have[status])):
                    name = STUDENT_NAMES[(pi_ + d_id) % len(STUDENT_NAMES)]
                    item = _gen_foil_item(f'D{d_id:03d}', 'D', pair, foil,
                                          status, name, used_exprs)
                    if item:
                        new_D.append(item)
                        d_id += 1
                    else:
                        d_fail.append((pair, foil, status))
        print(f"  pair {pair} done", flush=True)
    print(f"  built {len(new_D)} new D items; failures: {d_fail or 'none'}")

    # ── assemble, verify, write ──────────────────────────────────────
    new_pool = (by_cat['A'] + new_A + by_cat['B'] + new_B
                + by_cat['C'] + new_C + by_cat['D'] + new_D)

    # every original item is preserved: all its original fields unchanged
    for it in new_pool:
        if it['id'] in originals:
            orig = originals[it['id']]
            assert all(it[k] == orig[k] for k in orig), f"mutated: {it['id']}"
    assert len([it for it in new_pool if it['id'] in originals]) == 240

    # cell balance over the SAMPLING-ELIGIBLE items
    print("\n=== balance ===")
    cats = Counter(it['category'] for it in new_pool)
    print("category counts:", dict(cats))
    a_cells = Counter(it['misconceptions'][0] for it in new_pool
                      if it['category'] == 'A')
    print("A per rule (want 20):", dict(a_cells))
    b_cells = Counter((it['misconceptions'][0], it['probed_misconception'],
                       it['foil_status']) for it in new_pool
                      if it['category'] == 'B')
    bad_b = {k: v for k, v in b_cells.items()
             if k[2] in ('refuted', 'unsupported') and v < 2}
    print(f"B cells at full 2/2 strength: "
          f"{sum(1 for k, v in b_cells.items() if k[2] in ('refuted', 'unsupported') and v >= 2)}/60"
          f"{'  SHORT: ' + str(bad_b) if bad_b else ''}")
    c_cells = Counter((it['probed_misconception'], it['which_target'])
                      for it in new_pool if it['category'] == 'C')
    print("C per (target, position) (want 10):",
          sorted(set(c_cells.values())))
    d_cells = Counter((tuple(it['misconceptions']), it['probed_misconception'],
                       it['foil_status']) for it in new_pool
                      if it['category'] == 'D')
    n_full_d = sum(1 for (p, f, s), v in d_cells.items()
                   if s in ('refuted', 'unsupported') and v >= 1)
    print(f"D (pair, foil, status) cells covered: {n_full_d}/120")
    amb = Counter(it['category'] for it in new_pool
                  if it.get('foil_status') == 'ambiguous')
    print(f"ambiguous (kept, never sampled): {dict(amb) or 0}")

    margs = [it['io_foil_marginal'] for it in new_pool
             if it.get('foil_status') == 'refuted']
    umargs = [it['io_foil_marginal'] for it in new_pool
              if it.get('foil_status') == 'unsupported']
    print(f"refuted marginals:     n={len(margs)}  max={max(margs):.3f}")
    print(f"unsupported marginals: n={len(umargs)}  min={min(umargs):.3f}")

    with open(POOL, 'w') as f:
        json.dump(new_pool, f, indent=2)
    shutil.copy(POOL, FRONTEND_COPY)
    shutil.copy(POOL, LLM_COPY)
    print(f"\nwrote {len(new_pool)} items to {POOL} and copied to "
          f"{FRONTEND_COPY} and {LLM_COPY}  ({time.time() - t0:.0f}s)")


if __name__ == '__main__':
    main()
