"""
pool.py

Builds the v4 "position x named" stimulus pool.

v4 vs v3
--------
v3 balanced B on (named foil x refutation status x position) and let the
PRESENT rule fall where it may, which left 13 of the 30 present x named
heatmap cells holding only one refutation status, so the status-split panels
had holes. v4 makes the heatmap the thing that is balanced:

  * refutation is NO LONGER A FACTOR. foil_status is still computed and stored
    per item, but nothing is balanced on it. Consequence to respect: only the
    COMBINED present x named heatmap is guaranteed full. Splitting it by
    refutation status brings the v3 holes straight back.
  * the named foil IS balanced within each present rule, which is what fills
    the heatmap's columns.

Design
------
Two categories, one misconception per trace, as in v3:
  A - the statement NAMES the misconception in the trace   -> agree
  B - the statement names a FOIL                           -> disagree

Grid, 240 items in 120 matched pairs:
  A: present(6) x position(2)            = 12 cells x 10 items = 120
  B: present(6) x named(5) x position(2) = 60 cells x  2 items = 120

So every rule is the true misconception in exactly 40 items (20 per position,
half of them named and half foiled), and the present x named heatmap has 20 on
each diagonal cell and 4 in each of the 30 off-diagonal cells. No empty boxes.

Matched pairs, as in v3: one expression supplies BOTH the step-1 and the step-3
version of an item, so position is manipulated with expression structure held
constant. This is why N_OPS is 6 and not 5 or 4. Measured over 2500 bracketed
expressions, the number that support both step 1 and step 3 for
outside_bracket_first is 0 at 4 ops, 0 at 5 ops, and 32 at 6 ops: at 4 ops that
rule never reaches step 3 at all.

A foil is only used when its refutation status is the SAME at both positions.
Status is not a factor any more, but letting it flip inside a pair would put a
nuisance difference between the two positions we are trying to compare. It
costs about 7% of foil options, which is cheap insurance.
"""

import json
import random
from collections import Counter, defaultdict
from itertools import combinations

from learner import MISCONCEPTION_FLIPS
from inference import posterior_over_profiles, marginal_rule_probability
from generator_constrained import generate_expression
from find_pairs import pairs_for_expression, N_OPS, POSITIONS

IDS        = list(MISCONCEPTION_FLIPS.keys())
HYPOTHESES = [()] + [(m,) for m in IDS] + list(combinations(IDS, 2))

A_PAIRS_PER_RULE = 10   # -> 20 A items per rule, 10 per position
B_PAIRS_PER_CELL = 2    # per (present, named) -> 4 items, 2 per position

REFUTED_MAX     = 0.15
UNSUPPORTED_MAX = 0.35

STATEMENT_TEMPLATES = {
    'add_before_mul':        "{name} believes addition should be done before multiplication.",
    'add_before_div':        "{name} believes addition should be done before division.",
    'sub_before_mul':        "{name} believes subtraction should be done before multiplication.",
    'sub_before_div':        "{name} believes subtraction should be done before division.",
    'same_priority_rtl':     "{name} believes operations of the same priority should be worked right to left.",
    'outside_bracket_first': "{name} believes you should calculate outside the brackets before what's inside them.",
}

STUDENT_NAMES = [
    'Noah', 'Maya', 'Liam', 'Ava', 'Ethan', 'Zoe',
    'Mia', 'Lucas', 'Emma', 'Owen', 'Sofia', 'Caleb',
    'Ruby', 'Jonah', 'Isla', 'Felix', 'Nora', 'Dylan',
    'Priya', 'Marcus', 'Elena', 'Theo', 'Jasmine', 'Omar',
]


def _status(marginal):
    if marginal < REFUTED_MAX:
        return 'refuted'
    if marginal <= UNSUPPORTED_MAX:
        return 'unsupported'
    return 'high'


def foil_options(pair):
    """
    {foil_rule: (status, {position: marginal})} for every rule other than the
    true one whose status is identical at both positions and is not 'high'.
    'high' means the trace positively supports the rule, so it is not a clean
    foil; a status that flips between positions would unmatch the pair.
    """
    true_m = pair['misconception']
    post = {p: posterior_over_profiles(pair['traces'][p], profiles=HYPOTHESES)
            for p in POSITIONS}
    out = {}
    for f in IDS:
        if f == true_m:
            continue
        marg = {p: marginal_rule_probability(post[p], f) for p in POSITIONS}
        st = {p: _status(marg[p]) for p in POSITIONS}
        if st[POSITIONS[0]] == st[POSITIONS[1]] and st[POSITIONS[0]] != 'high':
            out[f] = (st[POSITIONS[0]], marg)
    return out


def draw_pair(misconception, rng, bracket_prob, max_draws=40_000):
    """One matched pair for `misconception`, or None if the budget runs out."""
    for _ in range(max_draws):
        expr = generate_expression(n_ops=N_OPS, bracket_prob=bracket_prob, rng=rng)
        if expr is None:
            continue
        got = pairs_for_expression(expr, misconception)
        if all(p in got for p in POSITIONS):
            return {'expression': expr, 'misconception': misconception,
                    'traces': {p: got[p] for p in POSITIONS}}
    return None


def build(seed=2026, verbose=True):
    rng = random.Random(seed)

    a_need = {m: A_PAIRS_PER_RULE for m in IDS}
    b_need = {(p, f): B_PAIRS_PER_CELL for p in IDS for f in IDS if p != f}
    a_pairs = defaultdict(list)
    b_pairs = defaultdict(list)
    seen_expressions = set()
    draws = Counter()
    stalls = 0

    def still_needed():
        return any(a_need.values()) or any(b_need.values())

    while still_needed():
        progressed = False
        for m in IDS:
            if not still_needed():
                break
            # This rule is done once its A quota and every B cell it can feed
            # are full. B cells are keyed by PRESENT rule, so only rule m can
            # fill the m-row of the heatmap.
            row_need = sum(v for (p, _f), v in b_need.items() if p == m)
            if a_need[m] == 0 and row_need == 0:
                continue

            bp = 1.0 if m == 'outside_bracket_first' else 0.6
            pair = draw_pair(m, rng, bp)
            draws[m] += 1
            if pair is None or pair['expression'] in seen_expressions:
                continue

            # An expression is used by exactly ONE pair, so a participant can
            # never meet the same expression twice.
            if a_need[m] > 0:
                seen_expressions.add(pair['expression'])
                a_pairs[m].append(pair)
                a_need[m] -= 1
                progressed = True
                continue

            opts = foil_options(pair)
            wanted = [f for f in opts if b_need.get((m, f), 0) > 0]
            if not wanted:
                continue
            f = max(wanted, key=lambda k: b_need[(m, k)])   # scarcest cell first
            seen_expressions.add(pair['expression'])
            b_pairs[(m, f)].append((pair, opts[f][0], opts[f][1]))
            b_need[(m, f)] -= 1
            progressed = True

        if verbose:
            print(f"  remaining pairs: A={sum(a_need.values()):3d}  "
                  f"B={sum(b_need.values()):3d}", end='\r')
        if not progressed:
            stalls += 1
            if stalls > 50:
                missing = {k: v for k, v in b_need.items() if v} | \
                          {k: v for k, v in a_need.items() if v}
                raise SystemExit(f"\nstalled with cells unfilled: {missing}")
        else:
            stalls = 0

    if verbose:
        print(f"  generated {len(seen_expressions)} expressions "
              f"({sum(draws.values())} pair draws)              ")

    # emit items
    items, pair_no = [], 0

    def emit(pair, probed, correct, foil_status=None, marginals=None):
        nonlocal pair_no
        pid = f"P{pair_no:03d}"
        pair_no += 1
        for pos in POSITIONS:
            name = STUDENT_NAMES[len(items) % len(STUDENT_NAMES)]
            it = {
                'id':                   f"{'A' if correct else 'B'}{len(items):03d}",
                'pair_id':              pid,
                'category':             'A' if correct else 'B',
                'error_position':       pos,
                'expression':           pair['expression'],
                'n_ops':                N_OPS,
                'misconceptions':       [pair['misconception']],
                'num_misconceptions':   1,
                'trace':                pair['traces'][pos],
                'probed_misconception': probed,
                'statement_correct':    correct,
                'which_target':         None,   # schema compatibility; C/D are gone
                'student_name':         name,
                'belief_statement':     STATEMENT_TEMPLATES[probed].format(name=name),
            }
            if foil_status:
                # recorded, NOT balanced: see the module docstring
                it['foil_status']      = foil_status
                it['io_foil_marginal'] = round(marginals[pos], 4)
            items.append(it)

    for m in IDS:
        for pair in a_pairs[m]:
            emit(pair, m, True)
    for (present, f), lst in b_pairs.items():
        for pair, st, marg in lst:
            emit(pair, f, False, st, marg)

    return items


def summarise(items):
    print(f"\n{len(items)} items, {len({i['pair_id'] for i in items})} matched pairs, "
          f"{len({i['expression'] for i in items})} distinct expressions")
    print("  category x position:",
          dict(Counter((i['category'], i['error_position']) for i in items)))
    print("  trace lengths:", dict(Counter(len(i['trace']) for i in items)))

    print(f"\n  items per PRESENT rule (want 40: 20 per position, 20 A / 20 B):")
    for m in IDS:
        sub = [i for i in items if i['misconceptions'][0] == m]
        c = Counter((i['category'], i['error_position']) for i in sub)
        print(f"    {m:24s} n={len(sub):3d}  " +
              "  ".join(f"{cat}/pos{p}={c[(cat, p)]:2d}"
                        for cat in 'AB' for p in POSITIONS))

    print(f"\n  present x named heatmap occupancy (diagonal = A, want 20; "
          f"off-diagonal = B, want 4):")
    grid = Counter((i['misconceptions'][0], i['probed_misconception']) for i in items)
    empty = [(p, n) for p in IDS for n in IDS if grid[(p, n)] == 0]
    header = "".join(f"{n[:9]:>11s}" for n in IDS)
    print(f"    {'present \\ named':>24s}{header}")
    for p in IDS:
        print(f"    {p:>24s}" + "".join(f"{grid[(p, n)]:>11d}" for n in IDS))
    print(f"    EMPTY CELLS: {len(empty)}" + (f"  {empty}" if empty else "  (heatmap is full)"))

    print("\n  refutation status, RECORDED not balanced (expect uneven):")
    cs = Counter(i['foil_status'] for i in items if i['category'] == 'B')
    print(f"    {dict(cs)}")
    per = Counter((i['probed_misconception'], i['foil_status'])
                  for i in items if i['category'] == 'B')
    for f in IDS:
        print(f"    {f:24s} refuted={per[(f,'refuted')]:2d}  "
              f"unsupported={per[(f,'unsupported')]:2d}")


if __name__ == '__main__':
    print("Building v4 pool...")
    items = build()
    summarise(items)
    with open('stimulus_pool.json', 'w', encoding='utf-8') as fh:
        json.dump(items, fh, ensure_ascii=False, indent=1)
    print("\nwrote stimulus_pool.json")
