"""
generator_v3.py

Constrained expression generation for the v3 "position" pool (HANDOFF 7b).

Why this exists instead of generator.py
---------------------------------------
generator.py draws every number uniformly from 1..12 and leaves quality to
post-hoc filters (stimulus_pool._is_clean rejects only 3+ decimal places).
That holds up at n_ops=4. At n_ops=6 it does not: two extra operations
compound both magnitude and non-integer division, so most candidate traces
fail on arithmetic rather than on design, and the ones that survive are a
narrow, misconception-dependent slice of expression space.

Measured on 5-op traces that passed the old filters: 36% ended on a negative
answer, 25% displayed a negative operand ("5 / 2 + -2 + 7"), 2% displayed
"5 / 0". At 6 ops we also saw runaway magnitudes (-> 3144) and degenerate
repeated-digit runs ("3 + 3 x 3 x 3 x 9 x 9 - 9", where 9-9=0 annihilates
the product).

So this module shapes the draw so clean arithmetic is the common case, and
does the rejecting explicitly in validate_trace() rather than implicitly.

Note on generator artifacts: the v3 design takes both the step-1 and the
step-3 version of an item from the SAME expression, so any regularity this
sampler introduces is held constant across the position manipulation and
cannot masquerade as a position effect.
"""

import random
import re

OPS_ALL = ['+', '-', '×', '÷']
_MUL = ('×', '÷')

# Bounds. MAX_VALUE keeps every displayed number to at most 3 digits so the
# arithmetic stays checkable by eye across a 6-step trace.
MAX_VALUE      = 999
MUL_OPERAND_MAX = 6     # operands adjacent to × are kept small to bound growth
NUM_MIN, NUM_MAX = 1, 12
MAX_RUN        = 2      # at most this many equal numbers in a row


# ── number shaping ─────────────────────────────────────────────────
#
# Constructive left-to-right draw rather than a set of repair passes. Repair
# passes fight each other: ordering "a - b" so a >= b can re-break the pair to
# its left, and forcing a divisor can undo both. Here each number is drawn from
# the candidates its incoming operator allows, with one step of lookahead at the
# outgoing operator, so no later choice can invalidate an earlier one.

def _proper_divisors(v):
    """Divisors of v strictly between 1 and v, so no '÷ 1' and no 'n ÷ n'."""
    return [d for d in range(2, v) if v % d == 0]


def _candidates(prev, op, next_op):
    """Values allowed at this slot, given the operator to its left and right."""
    if op is None:
        cand = list(range(NUM_MIN, NUM_MAX + 1))
    elif op == '+':
        cand = list(range(NUM_MIN, NUM_MAX + 1))
    elif op == '-':
        cand = list(range(NUM_MIN, prev))          # keep the difference positive
    elif op == '×':
        cand = list(range(2, MUL_OPERAND_MAX + 1))  # bound magnitude growth
    elif op == '÷':
        cand = _proper_divisors(prev)               # exact, non-trivial division
    else:
        cand = []

    # Lookahead: if this value is about to be divided, it must be divisible.
    if next_op == '÷':
        divisible = [v for v in cand if _proper_divisors(v)]
        if divisible:
            cand = divisible

    return cand


def _draw_numbers(operators, rng):
    """
    Draw n_ops+1 numbers satisfying every operator's local constraint.
    Returns None if a slot runs out of candidates (caller redraws operators).
    """
    n_ops = len(operators)
    nums  = []
    for i in range(n_ops + 1):
        op      = operators[i - 1] if i > 0 else None
        next_op = operators[i] if i < n_ops else None
        prev    = nums[-1] if nums else None
        cand    = _candidates(prev, op, next_op)
        if not cand:
            return None
        nums.append(rng.choice(cand))
    return nums


def _has_long_run(numbers):
    """True if more than MAX_RUN equal numbers appear consecutively."""
    run = 1
    for a, b in zip(numbers, numbers[1:]):
        run = run + 1 if a == b else 1
        if run > MAX_RUN:
            return True
    return False


# ── expression generation ──────────────────────────────────────────

def generate_expression(n_ops=6, bracket_prob=0.4, ops=None, rng=None, max_attempts=200):
    """
    A random expression with shaped numbers.

    n_ops        : number of operators (the trace will have exactly n_ops steps)
    bracket_prob : probability of wrapping one contiguous sub-expression;
                   pass 1.0 for outside_bracket_first items, which need a bracket
    rng          : random.Random instance, for reproducible pools

    Returns the expression string, or None if max_attempts draws all failed.
    """
    rng = rng or random
    ops = ops or OPS_ALL

    for _ in range(max_attempts):
        operators = [rng.choice(ops) for _ in range(n_ops)]
        if not any(o in _MUL for o in operators):
            continue                      # precedence must actually matter

        numbers = _draw_numbers(operators, rng)
        if numbers is None or _has_long_run(numbers):
            continue

        parts = []
        for i in range(n_ops):
            parts += [str(numbers[i]), operators[i]]
        parts.append(str(numbers[n_ops]))

        if rng.random() < bracket_prob and n_ops >= 2:
            size    = rng.randint(1, min(2, n_ops - 1))
            start   = rng.randint(0, n_ops - size)
            p_start = 2 * start
            p_end   = 2 * (start + size) + 1
            parts   = parts[:p_start] + [f"({' '.join(parts[p_start:p_end])})"] + parts[p_end:]

        return ' '.join(parts)

    return None


# ── trace validation ───────────────────────────────────────────────

_TOKEN = re.compile(r'[()]')


def _values(step):
    """Every numeric token in one trace line, as strings."""
    return [t for t in _TOKEN.sub(' ', step).split() if t not in OPS_ALL]


def validate_trace(trace, max_value=MAX_VALUE, allow_zero=False):
    """
    True if every line of the trace is something we are willing to show a
    participant: non-negative integers only, nothing over max_value, and
    (by default) no zero anywhere.

    Zero is excluded because it makes degenerate items: a single "x 0" collapses
    the rest of the expression, and "/ 0" cannot be fired at all, so it sits in
    the displayed work as an operation the student visibly never does.
    """
    for step in trace:
        for tok in _values(step):
            if not tok.isdigit():          # catches '-4', '0.25', '79.6'
                return False
            v = int(tok)
            if v > max_value:
                return False
            if v == 0 and not allow_zero:
                return False
    return True


# ── error position ─────────────────────────────────────────────────
#
# NOTE (HANDOFF 7b, finding 4): do NOT identify erroneous steps by asking
# whether an edge appears in the expert's trace tree. Once the learner
# diverges, every later expression is off that tree entirely, so edge
# membership reports all subsequent steps as errors. The only correct test is
# whether each move was expert-legal FROM ITS OWN start expression.

_expert_next_cache = {}


def expert_next(expr):
    """The set of expressions an expert could legally produce in one move."""
    if expr not in _expert_next_cache:
        from parser import build_dag
        from traces import _next_dags
        from misconceptions import dag_to_str
        try:
            _expert_next_cache[expr] = {dag_to_str(d) for d in _next_dags(build_dag(expr), [])}
        except Exception:
            _expert_next_cache[expr] = set()
    return _expert_next_cache[expr]


def error_steps(trace):
    """1-based indices of the steps that were not expert-legal when taken."""
    return [i for i in range(1, len(trace))
            if trace[i] not in expert_next(trace[i - 1])]
