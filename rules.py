from dag import Node, FlatDAG

OPERATORS = {'+', '-', '×', '÷'}


# ── helpers ───────────────────────────────────────────────────────

def _assign_ids(node, counter):
    """Walk a Node tree and give IDs to any operator/bracket/unary without one."""
    if node is None:
        return
    if (node.is_operator() or node.is_bracket() or node.is_unary()) and node.node_id is None:
        node.node_id = f'op{counter[0]}'
        counter[0] += 1
    for child in node.children:
        _assign_ids(child, counter)


def _stitch(new_ops, i, new_atom):
    """
    After removing ops[i], update the neighbours to point to new_atom.
    new_ops is already ops[:i] + ops[i+1:]
    """
    if i > 0:
        new_ops[i-1].children[1] = new_atom
    if i < len(new_ops):
        new_ops[i].children[0]   = new_atom


# ── Eval ──────────────────────────────────────────────────────────

def can_eval(dag, op_id):
    i, op = dag.find_op(op_id)
    if op is None:
        return False
    return dag.atoms[i].is_number() and dag.atoms[i+1].is_number()


def apply_eval(dag, op_id):
    """Eval[⊕]: a ⊕ b  →  ⟦a ⊕ b⟧"""
    i, op = dag.find_op(op_id)
    a = float(dag.atoms[i].label)
    b = float(dag.atoms[i+1].label)
    if   op.label == '+': result = a + b
    elif op.label == '-': result = a - b
    elif op.label == '×': result = a * b
    elif op.label == '÷': result = a / b
    label    = str(int(result)) if result == int(result) else str(result)
    new_atom = Node(label)

    new_atoms = dag.atoms[:i] + [new_atom] + dag.atoms[i+2:]
    new_ops   = dag.ops[:i]   + dag.ops[i+1:]
    _stitch(new_ops, i, new_atom)
    return FlatDAG(new_atoms, new_ops, dag.counter)


# ── DistribRight ──────────────────────────────────────────────────

def can_distrib_right(dag, op_id):
    i, op = dag.find_op(op_id)
    if op is None: return False
    right = dag.atoms[i+1]
    if not right.is_bracket(): return False
    inner = right.child()
    return inner is not None and inner.is_operator()


def apply_distrib_right(dag, op_id):
    """DistribRight[⊗,⊕]: X ⊗ (Y ⊕ Z)  →  (X ⊗ Y) ⊕ (X ⊗ Z)"""
    i, op    = dag.find_op(op_id)
    X        = dag.atoms[i]
    bracket  = dag.atoms[i+1]
    inner_op = bracket.child()
    Y        = inner_op.left()
    Z        = inner_op.right()
    outer    = op.label
    inner    = inner_op.label

    lb          = Node(outer, children=[X, Y])
    rb          = Node(outer, children=[X, Z])
    new_inner   = Node(inner, children=[lb, rb])
    new_bracket = Node('()', children=[new_inner])
    _assign_ids(new_bracket, dag.counter)

    new_atoms = dag.atoms[:i] + [new_bracket] + dag.atoms[i+2:]
    new_ops   = dag.ops[:i]   + dag.ops[i+1:]
    _stitch(new_ops, i, new_bracket)
    return FlatDAG(new_atoms, new_ops, dag.counter)


# ── DistribLeft ───────────────────────────────────────────────────

def can_distrib_left(dag, op_id):
    i, op = dag.find_op(op_id)
    if op is None: return False
    left = dag.atoms[i]
    if not left.is_bracket(): return False
    inner = left.child()
    return inner is not None and inner.is_operator()


def apply_distrib_left(dag, op_id):
    """DistribLeft[⊗,⊕]: (Y ⊕ Z) ⊗ X  →  (Y ⊗ X) ⊕ (Z ⊗ X)"""
    i, op    = dag.find_op(op_id)
    bracket  = dag.atoms[i]
    X        = dag.atoms[i+1]
    inner_op = bracket.child()
    Y        = inner_op.left()
    Z        = inner_op.right()
    outer    = op.label
    inner    = inner_op.label

    lb          = Node(outer, children=[Y, X])
    rb          = Node(outer, children=[Z, X])
    new_inner   = Node(inner, children=[lb, rb])
    new_bracket = Node('()', children=[new_inner])
    _assign_ids(new_bracket, dag.counter)

    new_atoms = dag.atoms[:i] + [new_bracket] + dag.atoms[i+2:]
    new_ops   = dag.ops[:i]   + dag.ops[i+1:]
    _stitch(new_ops, i, new_bracket)
    return FlatDAG(new_atoms, new_ops, dag.counter)


# ── PartialDistrib (misconception) ────────────────────────────────

def can_partial_distrib(dag, op_id):
    return can_distrib_right(dag, op_id)


def apply_partial_distrib(dag, op_id):
    """PartialDistrib[⊗,⊕]: X ⊗ (Y ⊕ Z)  →  (X ⊗ Y) ⊕ Z  [MISCONCEPTION]"""
    i, op    = dag.find_op(op_id)
    X        = dag.atoms[i]
    bracket  = dag.atoms[i+1]
    inner_op = bracket.child()
    Y        = inner_op.left()
    Z        = inner_op.right()
    outer    = op.label
    inner    = inner_op.label

    lb          = Node(outer, children=[X, Y])
    new_inner   = Node(inner, children=[lb, Z])
    new_bracket = Node('()', children=[new_inner])
    _assign_ids(new_bracket, dag.counter)

    new_atoms = dag.atoms[:i] + [new_bracket] + dag.atoms[i+2:]
    new_ops   = dag.ops[:i]   + dag.ops[i+1:]
    _stitch(new_ops, i, new_bracket)
    return FlatDAG(new_atoms, new_ops, dag.counter)


# ── Commute ───────────────────────────────────────────────────────

def can_commute(dag, op_id):
    _, op = dag.find_op(op_id)
    return op is not None


def apply_commute(dag, op_id):
    """Commute[⊕]: X ⊕ Y  →  Y ⊕ X"""
    i, op = dag.find_op(op_id)
    L = dag.atoms[i]
    R = dag.atoms[i+1]

    new_atoms      = list(dag.atoms)
    new_atoms[i]   = R
    new_atoms[i+1] = L

    op.children = [R, L]
    if i > 0:
        dag.ops[i-1].children[1] = R
    if i+1 < len(dag.ops):
        dag.ops[i+1].children[0] = L

    return FlatDAG(new_atoms, dag.ops, dag.counter)


# ── Assoc ─────────────────────────────────────────────────────────

def can_assoc(dag, op_id):
    i, op = dag.find_op(op_id)
    if op is None:
        return False
    left = dag.atoms[i]
    if not left.is_bracket():
        return False
    inner_op = left.child()
    return inner_op is not None and inner_op.is_operator() and inner_op.label == op.label


def apply_assoc(dag, op_id):
    """Assoc[⊕]: (X ⊕ Y) ⊕ Z  →  X ⊕ (Y ⊕ Z)"""
    i, op    = dag.find_op(op_id)
    bracket  = dag.atoms[i]
    inner_op = bracket.child()
    X = inner_op.left()
    Y = inner_op.right()
    Z = dag.atoms[i+1]

    new_inner   = Node(op.label, children=[Y, Z])
    new_bracket = Node('()', children=[new_inner])
    _assign_ids(new_bracket, dag.counter)

    new_atoms      = list(dag.atoms)
    new_atoms[i]   = X
    new_atoms[i+1] = new_bracket

    op.children = [X, new_bracket]
    if i > 0:
        dag.ops[i-1].children[1] = X
    if i+1 < len(dag.ops):
        dag.ops[i+1].children[0] = new_bracket

    return FlatDAG(new_atoms, dag.ops, dag.counter)


# ── ParenElim ─────────────────────────────────────────────────────

def can_paren_elim(dag, bracket_id):
    """Fires on a bracket atom whose inner child is a single number (no inner operator)."""
    for atom in dag.atoms:
        if atom.is_bracket() and atom.node_id == bracket_id:
            inner = atom.child()
            return inner is not None and inner.is_number()
    return False


def apply_paren_elim(dag, bracket_id):
    """ParenElim: (X) → X"""
    idx = next(j for j, a in enumerate(dag.atoms)
               if a.is_bracket() and a.node_id == bracket_id)
    bracket = dag.atoms[idx]
    X = bracket.child()

    new_atoms      = list(dag.atoms)
    new_atoms[idx] = X

    if idx > 0:
        dag.ops[idx-1].children[1] = X
    if idx < len(dag.ops):
        dag.ops[idx].children[0]   = X

    return FlatDAG(new_atoms, dag.ops, dag.counter)


# ── DropDouble / CancelDouble ─────────────────────────────────────
#
# ⊕⊕X → ⊕X  (DropDouble)   /   ⊕⊕X → X  (CancelDouble)
#
# Fires on a unary atom whose operator is applied twice in a row.
# Two structural forms are recognised:
#   direct : u⊕( u⊕(X) )               e.g.  --3
#   bracket: u⊕( bracket( u⊕(X) ) )    e.g.  -(−3)

def _double_inner(atom):
    """
    If atom is u⊕ wrapping another u⊕ (directly or via a bracket),
    return (True, X, op_label).  Otherwise (False, None, None).
    """
    if not atom.is_unary():
        return False, None, None
    op    = atom.label
    child = atom.child()
    if child is None:
        return False, None, None
    if child.is_unary() and child.label == op:          # direct: --X
        return True, child.child(), op
    if child.is_bracket():                              # bracket: -(−X)
        inner = child.child()
        if inner is not None and inner.is_unary() and inner.label == op:
            return True, inner.child(), op
    return False, None, None


def can_drop_double(dag, atom_id):
    _, atom = dag.find_atom(atom_id)
    if atom is None:
        return False
    ok, _, _ = _double_inner(atom)
    return ok


def apply_drop_double(dag, atom_id):
    """DropDouble[⊕]: ⊕⊕X → ⊕X  — keep one unary application"""
    idx, atom = dag.find_atom(atom_id)
    _, X, op  = _double_inner(atom)

    new_node = Node(op, children=[X], node_id=f'op{dag.counter[0]}')
    dag.counter[0] += 1

    new_atoms      = list(dag.atoms)
    new_atoms[idx] = new_node
    if idx > 0:
        dag.ops[idx-1].children[1] = new_node
    if idx < len(dag.ops):
        dag.ops[idx].children[0]   = new_node

    return FlatDAG(new_atoms, dag.ops, dag.counter)


def can_cancel_double(dag, atom_id):
    return can_drop_double(dag, atom_id)


def apply_cancel_double(dag, atom_id):
    """CancelDouble[⊕]: ⊕⊕X → X  — two applications cancel"""
    idx, atom = dag.find_atom(atom_id)
    _, X, _   = _double_inner(atom)

    new_atoms      = list(dag.atoms)
    new_atoms[idx] = X
    if idx > 0:
        dag.ops[idx-1].children[1] = X
    if idx < len(dag.ops):
        dag.ops[idx].children[0]   = X

    return FlatDAG(new_atoms, dag.ops, dag.counter)


# ── OpConfusion ───────────────────────────────────────────────────

def can_op_confusion(dag, op_id):
    _, op = dag.find_op(op_id)
    return op is not None


def apply_op_confusion(dag, op_id, target_label):
    """OpConfusion[⊕, ⊗]: X ⊕ Y → X ⊗ Y  — replace the operator in place"""
    _, op = dag.find_op(op_id)
    op.label = target_label
    return FlatDAG(dag.atoms, dag.ops, dag.counter)


# ── available actions ─────────────────────────────────────────────

def available_actions(dag):
    """
    Return all (rule_name, op_id[, extra]) tuples where a rule can fire
    at the top level of the flat DAG.
    OpConfusion entries carry a third element: the target operator label.
    ParenElim entries use the bracket's node_id rather than an op_id.
    """
    actions = []
    for i, op in enumerate(dag.ops):
        left_atom  = dag.atoms[i]
        right_atom = dag.atoms[i+1]

        if left_atom.is_number() and right_atom.is_number():
            actions.append(('Eval', op.node_id))

        if right_atom.is_bracket():
            inner = right_atom.child()
            if inner is not None and inner.is_operator():
                actions.append(('DistribRight',   op.node_id))
                actions.append(('PartialDistrib', op.node_id))

        if left_atom.is_bracket():
            inner = left_atom.child()
            if inner is not None and inner.is_operator():
                actions.append(('DistribLeft', op.node_id))
                if inner.label == op.label:
                    actions.append(('Assoc', op.node_id))

        actions.append(('Commute', op.node_id))

        for target in sorted(OPERATORS - {op.label}):
            actions.append(('OpConfusion', op.node_id, target))

    for atom in dag.atoms:
        if atom.is_bracket() and atom.node_id:
            inner = atom.child()
            if inner is not None and inner.is_number():
                actions.append(('ParenElim', atom.node_id))
        if atom.is_unary() and atom.node_id:
            ok, _, _ = _double_inner(atom)
            if ok:
                actions.append(('DropDouble',   atom.node_id))
                actions.append(('CancelDouble', atom.node_id))

    return actions
