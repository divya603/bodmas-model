"""
traces.py

Generate all valid traces for any learner (expert or misconception learner).

A learner is a list of up to 2 misconception IDs (empty list = expert).
A trace is a list of expression strings from the initial expression down
to the final reduced number.

At each step, every op whose truth-table row ANDs to True (using the
learner's overridden truth table) is a valid next move.  Both top-level
and inner-bracket ops are considered simultaneously, giving every possible
ordering the learner could legally choose.
"""

from misconceptions import dag_to_str
from pattern_matcher import match_patterns
from valid_actions import _is_correct_in_window, fire_operator, fire_inner_op, is_zero_divide
from learner import is_correct_for_learner


# ── learner-aware valid-actions ────────────────────────────────────

def compute_valid_actions_for_learner(dag, matches, misconceptions):
    """
    Same as compute_valid_actions but uses the learner's truth table.
    Misconceptions flip specific F → T entries; everything else is default.
    """
    top_ids = {op.node_id: i for i, op in enumerate(dag.ops)}

    actions = []
    for m in matches:
        if m['op_id'] not in top_ids:
            continue
        idx = top_ids[m['op_id']]

        if m['table'] == 1:          # standalone — always valid for everyone
            actions.append(dict(op_id=m['op_id'], op_label=m['op_label'],
                                op_index=idx, truth=[], valid=True))
            continue

        truth = []
        for w in m['windows']:
            correct = is_correct_for_learner(w, misconceptions)
            w_idx   = idx if w['role'] == 'left' else idx - 1
            truth.append((w_idx, w['pattern'], correct))

        actions.append(dict(
            op_id=m['op_id'], op_label=m['op_label'], op_index=idx,
            truth=truth, valid=all(c for _, _, c in truth),
        ))

    return sorted(actions, key=lambda a: a['op_index'])


def inner_valid_actions_for_learner(dag, misconceptions):
    """Valid ops inside bracket atoms, evaluated with the learner's truth table."""
    result = []
    for atom_index, atom in enumerate(dag.atoms):
        if not (atom.is_bracket() and atom.inner_dag):
            continue
        inner        = atom.inner_dag
        inner_acts   = compute_valid_actions_for_learner(
                           inner, match_patterns(inner), misconceptions)
        for ia in inner_acts:
            if not ia['valid'] or is_zero_divide(inner, ia['op_index']):
                continue
            left  = inner.atoms[ia['op_index']]
            right = inner.atoms[ia['op_index'] + 1]
            result.append(dict(
                atom_index     = atom_index,
                inner_op_index = ia['op_index'],
                op_id          = ia['op_id'],
                op_label       = ia['op_label'],
                left_label     = left.label  if left.is_number()  else f'({dag_to_str(left.inner_dag)})',
                right_label    = right.label if right.is_number() else f'({dag_to_str(right.inner_dag)})',
            ))
    return result


# ── trace generation ───────────────────────────────────────────────

def _is_done(dag):
    return len(dag.ops) == 0 and len(dag.atoms) == 1 and dag.atoms[0].is_number()


def _next_dags(dag, misconceptions):
    """All DAGs reachable in one valid move for this learner."""
    matches = match_patterns(dag)
    actions = compute_valid_actions_for_learner(dag, matches, misconceptions)

    nexts = []
    for a in actions:
        if a['valid'] and not is_zero_divide(dag, a['op_index']):
            nexts.append(fire_operator(dag, a['op_index']))

    for ia in inner_valid_actions_for_learner(dag, misconceptions):
        nexts.append(fire_inner_op(dag, ia['atom_index'], ia['inner_op_index']))

    return nexts


def generate_traces(dag, misconceptions=()):
    """
    Return all valid traces for a learner defined by misconceptions.

    misconceptions : tuple/list of misconception IDs, max 2.
                     Empty = expert learner.

    Each trace is a list of expression strings:
        ['3 + 4 × 5', '3 + 20', '23']
    """
    current = dag_to_str(dag)

    if _is_done(dag):
        return [[current]]

    nexts = _next_dags(dag, misconceptions)
    if not nexts:
        return [[current]]   # stuck — partial trace

    traces = []
    for next_dag in nexts:
        for sub in generate_traces(next_dag, misconceptions):
            traces.append([current] + sub)
    return traces


# ── quick test ────────────────────────────────────────────────────

if __name__ == '__main__':
    from parser import build_dag

    expr = '3 + 4 × 5 - 2'
    dag  = build_dag(expr)

    print(f"Expression: {expr}\n")

    expert = generate_traces(dag, [])
    print(f"Expert  ({len(expert)} trace(s)):")
    for t in expert:
        print("  " + "  →  ".join(t))

    print()
    m1 = generate_traces(dag, ['add_before_mul'])
    print(f"add_before_mul  ({len(m1)} trace(s)):")
    for t in m1:
        print("  " + "  →  ".join(t))

    print()
    m2 = generate_traces(dag, ['same_priority_rtl'])
    print(f"same_priority_rtl  ({len(m2)} trace(s)):")
    for t in m2:
        print("  " + "  →  ".join(t))

    print()
    m3 = generate_traces(dag, ['add_before_mul', 'same_priority_rtl'])
    print(f"add_before_mul + same_priority_rtl  ({len(m3)} trace(s)):")
    for t in m3:
        print("  " + "  →  ".join(t))
