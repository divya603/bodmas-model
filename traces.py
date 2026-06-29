"""
traces.py

Generate all valid expert-learner traces for an arithmetic expression.

A trace is a list of expression strings from the initial expression down to
the final reduced number.  At each step the next state is obtained by firing
one valid BODMAS action — either a top-level op whose truth-table row is ⊤,
or a valid op inside a bracket atom.

Both kinds of move are always considered together, so the output contains
every possible ordering an expert learner could legally choose.
"""

from misconceptions import dag_to_str
from pattern_matcher import match_patterns
from valid_actions import (
    compute_valid_actions,
    fire_operator,
    inner_valid_actions,
    fire_inner_op,
)


def _is_done(dag):
    return len(dag.ops) == 0 and len(dag.atoms) == 1 and dag.atoms[0].is_number()


def _next_dags(dag):
    """All DAGs reachable in one valid expert move."""
    matches = match_patterns(dag)
    actions = compute_valid_actions(dag, matches)

    nexts = []
    for a in actions:
        if a['valid']:
            nexts.append(fire_operator(dag, a['op_index']))

    for ia in inner_valid_actions(dag):
        nexts.append(fire_inner_op(dag, ia['atom_index'], ia['inner_op_index']))

    return nexts


def generate_traces(dag):
    """
    Return all valid expert-learner traces reachable from dag.

    Each trace is a list of expression strings:
        ['3 + 4 × 5', '3 + 20', '23']

    An expression with k independent valid moves at the first step produces
    k sub-trees, each explored recursively.
    """
    current = dag_to_str(dag)

    if _is_done(dag):
        return [[current]]

    nexts = _next_dags(dag)
    if not nexts:
        return [[current]]   # stuck — partial trace (shouldn't happen normally)

    traces = []
    for next_dag in nexts:
        for sub in generate_traces(next_dag):
            traces.append([current] + sub)
    return traces


# ── quick test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from parser import build_dag

    cases = [
        '3 × 4',                   # single op → 1 trace
        '3 + 4 + 5',               # two + ops, associative → 2 traces
        '3 + 4 × 5',               # × must go first → 1 trace
        '3 + 4 + 5 × 2',           # × or first + valid first → 3 traces
        '3 × 4 × 5',               # both × valid → 2 traces
        '5 + 2 × (4 + 4) - 8',    # bracket must resolve before × → 1 trace
        '5 × 3 + (4 + 2)',         # × valid before bracket resolves → 2 traces
    ]

    for expr in cases:
        dag    = build_dag(expr)
        traces = generate_traces(dag)
        print(f"\n{expr}  →  {len(traces)} trace(s)")
        for i, t in enumerate(traces):
            print(f"  [{i+1}]  " + "  →  ".join(t))
