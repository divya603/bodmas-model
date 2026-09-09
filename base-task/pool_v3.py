"""
pool_v3.py

Builds the v3 "position" stimulus pool (HANDOFF 7b).

Design
------
Two categories, no 2-misconception items:
  A - trace contains 1 misconception, the statement NAMES it        -> agree
  B - trace contains 1 misconception, the statement names a FOIL    -> disagree

Two new things relative to v1/v2:
  * error_position - the misconception fires at step 1 or step 3, and it is the
    trace's ONLY expert-illegal move, so the position is unambiguous.
  * matched pairs - the step-1 and step-3 versions of an item come from the SAME
    expression, so position is manipulated with expression structure held
    constant. Both members of a pair carry the same pair_id.

B items keep the refutation manipulation (foil_status refuted/unsupported),
computed from the marginal of the NAMED rule under the observer's 22
hypotheses (expert + 6 singletons + 15 pairs). The pair hypotheses are never
true of any item; they are what lets the observer express "no evidence either
way about that rule", which is what separates unsupported from refuted. A foil
is only used when its status is the SAME at both positions (93% of the time),
so a matched pair is matched on refutation too.

Grid: A = 6 rules x 2 positions; B = 6 probed rules x 2 positions x 2 statuses.
36 cells at TARGET_PER_CELL items each.
"""

import json
import random
from collections import Counter, defaultdict
from itertools import combinations

from learner import MISCONCEPTION_FLIPS
from inference import posterior_over_profiles, marginal_rule_probability
from generator_v3 import generate_expression
from find_pairs_v3 import pairs_for_expression, N_OPS, POSITIONS

IDS         = list(MISCONCEPTION_FLIPS.keys())
HYPOTHESES  = [()] + [(m,) for m in IDS] + list(combinations(IDS, 2))

TARGET_PER_CELL = 12
REFUTED_MAX     = 0.15    # marginal < this  -> the trace argues against the rule
UNSUPPORTED_MAX = 0.35    # in between       -> the rule never had an opportunity

STATEMENT_TEMPLATES = {
    'add_before_mul':        "{name} believes addition should be done before multiplication.",
    'add_before_div':        "{name} believes addition should be done before division.",
    'sub_before_mul':        "{name} believes subtraction should be done before multiplication.",
    'sub_before_div':        "{name} believes subtraction should be done before division.",
    'same_priority_rtl':     "{name} believes operations of the same priority should be worked right to left.",
    'outside_bracket_first': "{name} believes you should calculate outside the brackets before what's inside them.",
}

# 24 names == form length, so no participant sees a student twice. Placeholders
# here; sample_form overwrites them. Keep in sync with src/user/utils/sampleForm.js.
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
    true one whose status is identical at both positions. Rules whose status
    flips between positions are dropped, so a matched pair stays matched on
    refutation. 'high' (marginal > UNSUPPORTED_MAX) is also dropped: the trace
    positively supports the rule, so it is not a clean foil.
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


def draw_pair(misconception, rng, bracket_prob, max_draws=20_000):
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


def build(target=TARGET_PER_CELL, seed=2026, verbose=True):
    rng = random.Random(seed)

    a_need = {m: target for m in IDS}
    b_need = {(f, s): target for f in IDS for s in ('refuted', 'unsupported')}
    a_pairs = defaultdict(list)
    b_pairs = defaultdict(list)
    seen_expressions = set()
    tries = Counter()

    def still_needed():
        return any(a_need.values()) or any(b_need.values())

    while still_needed():
        # Round-robin the TRUE misconception so no foil cell ends up dominated by
        # one generating rule. outside_bracket_first needs a bracket to appear at
        # all; other rules get a bracket often too, because a bracket is what lets
        # outside_bracket_first be REFUTED as a foil (its scarcest cell).
        for m in IDS:
            if not still_needed():
                break
            bp = 1.0 if m == 'outside_bracket_first' else 0.6
            pair = draw_pair(m, rng, bp)
            tries[m] += 1
            if pair is None or pair['expression'] in seen_expressions:
                continue

            # An expression is used by exactly ONE item pair, so a participant can
            # never meet the same expression twice.
            if a_need[m] > 0:
                seen_expressions.add(pair['expression'])
                a_pairs[m].append(pair)
                a_need[m] -= 1
                continue

            opts = foil_options(pair)
            wanted = [(f, st) for f, (st, _) in opts.items() if b_need.get((f, st), 0) > 0]
            if not wanted:
                continue
            # Fill the scarcest cell first.
            f, st = max(wanted, key=lambda k: b_need[k])
            seen_expressions.add(pair['expression'])
            b_pairs[(f, st)].append((pair, opts[f][1]))
            b_need[(f, st)] -= 1

        if verbose:
            print(f"  remaining: A={sum(a_need.values()):3d}  B={sum(b_need.values()):3d}", end='\r')

    if verbose:
        print(f"  generated {len(seen_expressions)} expressions "
              f"({sum(tries.values())} pair draws)          ")

    # ── emit items ──────────────────────────────────────────────────
    items, pair_id = [], 0

    def emit(pair, probed, correct, foil_status=None, marginals=None):
        nonlocal pair_id
        pid = f"P{pair_id:03d}"
        pair_id += 1
        for pos in POSITIONS:
            name = STUDENT_NAMES[len(items) % len(STUDENT_NAMES)]
            it = {
                'id':                 f"{'A' if correct else 'B'}{len(items):03d}",
                'pair_id':            pid,
                'category':           'A' if correct else 'B',
                'error_position':     pos,
                'expression':         pair['expression'],
                'n_ops':              N_OPS,
                'misconceptions':     [pair['misconception']],
                'num_misconceptions': 1,
                'trace':              pair['traces'][pos],
                'probed_misconception': probed,
                'statement_correct':  correct,
                'which_target':       None,     # retained for schema compatibility; C/D are gone
                'student_name':       name,
                'belief_statement':   STATEMENT_TEMPLATES[probed].format(name=name),
            }
            if foil_status:
                it['foil_status']      = foil_status
                it['io_foil_marginal'] = round(marginals[pos], 4)
            items.append(it)

    for m in IDS:
        for pair in a_pairs[m]:
            emit(pair, m, True)
    for (f, st), lst in b_pairs.items():
        for pair, marg in lst:
            emit(pair, f, False, st, marg)

    return items


def summarise(items):
    print(f"\n{len(items)} items, {len({i['pair_id'] for i in items})} matched pairs, "
          f"{len({i['expression'] for i in items})} distinct expressions")
    print("  category x position:",
          dict(Counter((i['category'], i['error_position']) for i in items)))
    print("  trace lengths:", dict(Counter(len(i['trace']) for i in items)))
    print("  misconceptions per item:", dict(Counter(i['num_misconceptions'] for i in items)))
    print("\n  A cells (probed rule x position), want %d each:" % TARGET_PER_CELL)
    ca = Counter((i['probed_misconception'], i['error_position'])
                 for i in items if i['category'] == 'A')
    for m in IDS:
        print(f"    {m:24s} " + "  ".join(f"pos{p}={ca[(m,p)]:2d}" for p in POSITIONS))
    print("\n  B cells (probed rule x status x position), want %d each:" % TARGET_PER_CELL)
    cb = Counter((i['probed_misconception'], i['foil_status'], i['error_position'])
                 for i in items if i['category'] == 'B')
    for m in IDS:
        row = "  ".join(f"{s[:5]}/pos{p}={cb[(m,s,p)]:2d}"
                        for s in ('refuted', 'unsupported') for p in POSITIONS)
        print(f"    {m:24s} {row}")
    print("\n  true rule behind each foil cell (should be spread, never the foil itself):")
    for f in IDS:
        c = Counter(i['misconceptions'][0] for i in items
                    if i['category'] == 'B' and i['probed_misconception'] == f)
        assert f not in c, f"foil {f} probed on a trace generated by itself"
        print(f"    {f:24s} {dict(c)}")


if __name__ == '__main__':
    print("Building v3 pool...")
    items = build()
    summarise(items)
    with open('stimulus_pool_v3.json', 'w', encoding='utf-8') as fh:
        json.dump(items, fh, ensure_ascii=False, indent=1)
    print(f"\nwrote stimulus_pool_v3.json")
