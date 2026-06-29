"""
valid_actions.py

Computes which top-level operators are valid BODMAS moves via the
truth-table AND: op_i is valid iff every 3-node window it participates
in marks firing op_i as a correct move.

ValidActions(G) = { op_i | ∧_{P ∋ op_i} P(op_i) = T }
"""

from dag import Node, FlatDAG

_ADD = {'+', '-'}
_MUL = {'×', '÷'}


# ── correctness rules ──────────────────────────────────────────────

def _op2_correct(op1, op2):
    """
    Is firing op2 the correct BODMAS move when op1 sits to its left?

    True when:
      - op2 is strictly stronger than op1  (e.g. a+b×c → fire ×)
      - op1 == op2 and it's an associative operator  (a+b+c or a×b×c → either order ok)

    False for all mixed same-group cases (a+b-c, a-b+c, a×b÷c, a÷b×c, …)
    because LTR gives the correct result and RTL may not.
    """
    if op1 in _ADD and op2 in _MUL:        return True   # op2 strictly stronger
    if op1 == op2 and op2 in {'+', '×'}:  return True   # purely associative
    return False


def _is_correct_in_window(window):
    """
    Returns True iff firing op_i (in its role as op1 or op2)
    is a correct BODMAS move in this 3-node window.
    """
    table = window['table']
    role  = window['role']
    parts = window['pattern'].split()
    op1, op2 = parts[1], parts[3]

    if table == 1: return True
    if table == 5: return False   # Y in middle — must recurse first, neither op can fire

    if table == 3:                # a op1 b op2 Y
        if role == 'left':        # op_i = op1, fires (a op1 b), both literals
            return not (op1 in _ADD and op2 in _MUL)
        return False              # op_i = op2, right neighbour is Y — can't fire yet

    if table == 4:                # Y op1 b op2 c
        if role == 'right':       # op_i = op2, fires (b op2 c), both literals
            return _op2_correct(op1, op2)
        return False              # op_i = op1, left neighbour is Y — can't fire yet

    # table == 2: a op1 b op2 c — all literals
    if role == 'left':
        return not (op1 in _ADD and op2 in _MUL)
    return _op2_correct(op1, op2)


# ── main API ───────────────────────────────────────────────────────

def compute_valid_actions(dag, matches):
    """
    For every top-level operator in dag, compute its truth-table row
    and overall validity.

    Returns a list of dicts (sorted by op_index):
        op_id     : str
        op_label  : str
        op_index  : int
        truth     : list of (window_index: int, pattern: str, correct: bool)
        valid     : bool   — AND over all truth values
    """
    top_ids = {op.node_id: i for i, op in enumerate(dag.ops)}

    actions = []
    for m in matches:
        if m['op_id'] not in top_ids:
            continue
        idx = top_ids[m['op_id']]

        if m['table'] == 1:          # standalone — always valid
            actions.append(dict(op_id=m['op_id'], op_label=m['op_label'],
                                op_index=idx, truth=[], valid=True))
            continue

        truth = []
        for w in m['windows']:
            correct = _is_correct_in_window(w)
            w_idx   = idx if w['role'] == 'left' else idx - 1
            truth.append((w_idx, w['pattern'], correct))

        actions.append(dict(
            op_id=m['op_id'], op_label=m['op_label'], op_index=idx,
            truth=truth, valid=all(c for _, _, c in truth),
        ))

    return sorted(actions, key=lambda a: a['op_index'])


# ── firing ─────────────────────────────────────────────────────────

def _eval(a, op, b):
    va, vb = float(a.label), float(b.label)
    r = {'+': va + vb, '-': va - vb, '×': va * vb, '÷': va / vb}[op.label]
    return Node(str(int(r)) if r == int(r) else str(round(r, 6)))


def fire_operator(dag, op_index):
    """Evaluate dag.ops[op_index] and return a new FlatDAG."""
    i         = op_index
    new_atom  = _eval(dag.atoms[i], dag.ops[i], dag.atoms[i + 1])
    new_atoms = dag.atoms[:i] + [new_atom] + dag.atoms[i + 2:]
    new_ops   = dag.ops[:i] + dag.ops[i + 1:]
    if i > 0:             new_ops[i - 1].children[1] = new_atom
    if i < len(new_ops):  new_ops[i].children[0]     = new_atom
    return FlatDAG(new_atoms, new_ops, dag.counter)


# ── bracket helpers ────────────────────────────────────────────────

def get_bracket_atoms(dag):
    """Return [{atom_index, inner_expr}] for all top-level bracket atoms."""
    from misconceptions import dag_to_str
    return [
        dict(atom_index=i, inner_expr=dag_to_str(atom.inner_dag))
        for i, atom in enumerate(dag.atoms)
        if atom.is_bracket() and atom.inner_dag
    ]


def fire_inner_op(dag, atom_index, inner_op_index):
    """
    Fire inner_op_index inside the bracket at dag.atoms[atom_index].
    Unwraps the bracket if it reduces to a single literal.
    Returns a new top-level FlatDAG.
    """
    atom      = dag.atoms[atom_index]
    new_inner = fire_operator(atom.inner_dag, inner_op_index)

    # Unwrap fully-reduced bracket to a bare literal
    if (len(new_inner.ops) == 0 and len(new_inner.atoms) == 1
            and new_inner.atoms[0].is_number()):
        new_atom = new_inner.atoms[0]
    else:
        new_atom = Node('()', node_id=atom.node_id)
        new_atom.inner_dag = new_inner

    new_atoms = list(dag.atoms)
    new_atoms[atom_index] = new_atom
    if atom_index > 0:             dag.ops[atom_index - 1].children[1] = new_atom
    if atom_index < len(dag.ops):  dag.ops[atom_index].children[0]     = new_atom
    return FlatDAG(new_atoms, dag.ops, dag.counter)


def inner_valid_actions(dag):
    """
    For each bracket atom, return the valid inner ops as flat action records:
        atom_index  : int
        inner_op_index : int
        op_label    : str
        left_label  : str
        right_label : str
    """
    from pattern_matcher import match_patterns

    result = []
    for atom_index, atom in enumerate(dag.atoms):
        if not (atom.is_bracket() and atom.inner_dag):
            continue
        inner        = atom.inner_dag
        inner_acts   = compute_valid_actions(inner, match_patterns(inner))
        for ia in inner_acts:
            if not ia['valid']:
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
