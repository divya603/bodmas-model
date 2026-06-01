from dag import Node, FlatDAG

OPERATORS = {'+', '-', '×', '÷'}


# ── helpers ───────────────────────────────────────────────────────

def _new_id(counter):
    op_id = f'op{counter[0]}'
    counter[0] += 1
    return op_id


def _make_bracket(atoms, op_label, counter):
    """
    Wrap two atoms into a bracket: (atoms[0] op_label atoms[1]).
    Returns the bracket Node with its inner_dag set.
    """
    op_node = Node(op_label, children=list(atoms), node_id=_new_id(counter))
    bracket = Node('()', node_id=_new_id(counter))
    bracket.inner_dag = FlatDAG(list(atoms), [op_node], counter)
    return bracket


def _wrap_remainder(atoms, ops, counter):
    """
    If there is only one atom, return it directly.
    If there are multiple, wrap them in a new bracket with the given ops.
    """
    if len(atoms) == 1:
        return atoms[0]
    bracket = Node('()', node_id=_new_id(counter))
    bracket.inner_dag = FlatDAG(list(atoms), list(ops), counter)
    return bracket


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
    return right.inner_dag is not None and len(right.inner_dag.ops) >= 1


def apply_distrib_right(dag, op_id):
    """DistribRight[⊗,⊕]: X ⊗ (Y ⊕ …)  →  (X ⊗ Y) ⊕ (X ⊗ rest)"""
    i, op      = dag.find_op(op_id)
    X          = dag.atoms[i]
    bracket    = dag.atoms[i+1]
    inner      = bracket.inner_dag
    inner_op   = inner.ops[0]
    Y          = inner.atoms[0]
    Z          = _wrap_remainder(inner.atoms[1:], inner.ops[1:], dag.counter)
    outer      = op.label
    inner_lbl  = inner_op.label

    lb          = _make_bracket([X, Y], outer, dag.counter)
    rb          = _make_bracket([X, Z], outer, dag.counter)
    new_bracket = _make_bracket([lb, rb], inner_lbl, dag.counter)

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
    return left.inner_dag is not None and len(left.inner_dag.ops) >= 1


def apply_distrib_left(dag, op_id):
    """DistribLeft[⊗,⊕]: (Y ⊕ …) ⊗ X  →  (Y ⊗ X) ⊕ (rest ⊗ X)"""
    i, op      = dag.find_op(op_id)
    bracket    = dag.atoms[i]
    X          = dag.atoms[i+1]
    inner      = bracket.inner_dag
    inner_op   = inner.ops[0]
    Y          = inner.atoms[0]
    Z          = _wrap_remainder(inner.atoms[1:], inner.ops[1:], dag.counter)
    outer      = op.label
    inner_lbl  = inner_op.label

    lb          = _make_bracket([Y, X], outer, dag.counter)
    rb          = _make_bracket([Z, X], outer, dag.counter)
    new_bracket = _make_bracket([lb, rb], inner_lbl, dag.counter)

    new_atoms = dag.atoms[:i] + [new_bracket] + dag.atoms[i+2:]
    new_ops   = dag.ops[:i]   + dag.ops[i+1:]
    _stitch(new_ops, i, new_bracket)
    return FlatDAG(new_atoms, new_ops, dag.counter)


# ── PartialDistrib (misconception) ────────────────────────────────

def can_partial_distrib(dag, op_id):
    return can_distrib_right(dag, op_id)


def apply_partial_distrib(dag, op_id):
    """PartialDistrib[⊗,⊕]: X ⊗ (Y ⊕ …)  →  (X ⊗ Y) ⊕ rest  [MISCONCEPTION]"""
    i, op      = dag.find_op(op_id)
    X          = dag.atoms[i]
    bracket    = dag.atoms[i+1]
    inner      = bracket.inner_dag
    inner_op   = inner.ops[0]
    Y          = inner.atoms[0]
    Z          = _wrap_remainder(inner.atoms[1:], inner.ops[1:], dag.counter)
    outer      = op.label
    inner_lbl  = inner_op.label

    lb          = _make_bracket([X, Y], outer, dag.counter)
    new_bracket = _make_bracket([lb, Z], inner_lbl, dag.counter)

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
    if op is None: return False
    left = dag.atoms[i]
    if not left.is_bracket(): return False
    inner = left.inner_dag
    if inner is None or len(inner.ops) != 1: return False
    return inner.ops[0].label == op.label


def apply_assoc(dag, op_id):
    """Assoc[⊕]: (X ⊕ Y) ⊕ Z  →  X ⊕ (Y ⊕ Z)"""
    i, op   = dag.find_op(op_id)
    bracket = dag.atoms[i]
    inner   = bracket.inner_dag
    X       = inner.atoms[0]
    Y       = inner.atoms[1]
    Z       = dag.atoms[i+1]

    new_bracket = _make_bracket([Y, Z], op.label, dag.counter)

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
    """Fires when a bracket's inner dag has no ops (single atom inside)."""
    _, atom = dag.find_atom(bracket_id)
    if atom is None or not atom.is_bracket(): return False
    return atom.inner_dag is not None and not atom.inner_dag.ops


def apply_paren_elim(dag, bracket_id):
    """ParenElim: (X) → X"""
    idx, bracket = dag.find_atom(bracket_id)
    X = bracket.inner_dag.atoms[0]

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
#   direct : u⊕( u⊕(X) )                         e.g.  --3
#   bracket: u⊕( bracket with single u⊕(X) )      e.g.  -(−3)

def _double_inner(atom):
    """
    If atom is u⊕ wrapping another u⊕ (directly or via a single-atom bracket),
    return (True, X, op_label).  Otherwise (False, None, None).
    """
    if not atom.is_unary():
        return False, None, None
    op    = atom.label
    child = atom.child()
    if child is None:
        return False, None, None
    # direct: u⊕(u⊕(X))
    if child.is_unary() and child.label == op:
        return True, child.child(), op
    # bracket: u⊕( (u⊕(X)) )
    if child.is_bracket() and child.inner_dag:
        inner = child.inner_dag
        if not inner.ops and len(inner.atoms) == 1:
            inner_atom = inner.atoms[0]
            if inner_atom.is_unary() and inner_atom.label == op:
                return True, inner_atom.child(), op
    return False, None, None


def can_drop_double(dag, atom_id):
    _, atom = dag.find_atom(atom_id)
    if atom is None: return False
    ok, _, _ = _double_inner(atom)
    return ok


def apply_drop_double(dag, atom_id):
    """DropDouble[⊕]: ⊕⊕X → ⊕X  — keep one unary application"""
    idx, atom = dag.find_atom(atom_id)
    _, X, op  = _double_inner(atom)

    new_node = Node(op, children=[X], node_id=_new_id(dag.counter))

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


# ── Recursive scan ────────────────────────────────────────────────

def _scan_atom(atom, actions):
    """Recurse into a bracket or unary atom to find available actions inside."""
    if atom.is_bracket() and atom.inner_dag:
        if not atom.inner_dag.ops:
            actions.append(('ParenElim', atom.node_id))
        _scan_flat_dag(atom.inner_dag, actions)

    elif atom.is_unary() and atom.node_id:
        ok, _, _ = _double_inner(atom)
        if ok:
            actions.append(('DropDouble',   atom.node_id))
            actions.append(('CancelDouble', atom.node_id))
        if atom.child():
            _scan_atom(atom.child(), actions)


def _scan_flat_dag(dag, actions):
    """
    Collect every available action in a FlatDAG, recursing into
    bracket and unary atoms so the whole expression tree is covered.
    """
    # ── ops ───────────────────────────────────────────────────────
    for i, op in enumerate(dag.ops):
        L = dag.atoms[i]
        R = dag.atoms[i+1]

        if L.is_number() and R.is_number():
            actions.append(('Eval', op.node_id))

        if R.is_bracket() and R.inner_dag and len(R.inner_dag.ops) >= 1:
            actions.append(('DistribRight',   op.node_id))
            actions.append(('PartialDistrib', op.node_id))

        if L.is_bracket() and L.inner_dag and len(L.inner_dag.ops) >= 1:
            actions.append(('DistribLeft', op.node_id))
            if len(L.inner_dag.ops) == 1 and L.inner_dag.ops[0].label == op.label:
                actions.append(('Assoc', op.node_id))

        actions.append(('Commute', op.node_id))

        for target in sorted(OPERATORS - {op.label}):
            actions.append(('OpConfusion', op.node_id, target))

    # ── atoms — recurse deeper ────────────────────────────────────
    for atom in dag.atoms:
        _scan_atom(atom, actions)


# ── apply dispatch ────────────────────────────────────────────────

def _apply_in_dag(dag, node_id, fn):
    """
    Find node_id anywhere in the dag (top level or inside brackets),
    call fn(level_dag, node_id) at the level where it lives, then
    propagate the change back up. Returns the updated top-level FlatDAG,
    or None if node_id is not found anywhere.
    """
    # found at this level (op or atom)?
    i, _ = dag.find_op(node_id)
    if i is not None:
        return fn(dag, node_id)
    j, _ = dag.find_atom(node_id)
    if j is not None:
        return fn(dag, node_id)

    # recurse into bracket atoms
    for k, atom in enumerate(dag.atoms):
        inner_dag = None
        if atom.is_bracket():
            inner_dag = atom.inner_dag
        elif atom.is_unary() and atom.child() and atom.child().is_bracket():
            inner_dag = atom.child().inner_dag

        if inner_dag is None:
            continue

        result_inner = _apply_in_dag(inner_dag, node_id, fn)
        if result_inner is None:
            continue

        # rebuild the atom with the updated inner dag
        if atom.is_bracket():
            new_atom = Node('()', node_id=atom.node_id)
            new_atom.inner_dag = result_inner
        else:
            new_bracket = Node('()', node_id=atom.child().node_id)
            new_bracket.inner_dag = result_inner
            new_atom = Node(atom.label, children=[new_bracket], node_id=atom.node_id)

        new_atoms = list(dag.atoms)
        new_atoms[k] = new_atom
        if k > 0:
            dag.ops[k-1].children[1] = new_atom
        if k < len(dag.ops):
            dag.ops[k].children[0] = new_atom
        return FlatDAG(new_atoms, dag.ops, dag.counter)

    return None


def apply_action(dag, action):
    """Apply a single action tuple (as returned by available_actions) to dag."""
    rule    = action[0]
    node_id = action[1]

    DISPATCH = {
        'Eval':           apply_eval,
        'DistribRight':   apply_distrib_right,
        'DistribLeft':    apply_distrib_left,
        'PartialDistrib': apply_partial_distrib,
        'Commute':        apply_commute,
        'Assoc':          apply_assoc,
        'ParenElim':      apply_paren_elim,
        'DropDouble':     apply_drop_double,
        'CancelDouble':   apply_cancel_double,
    }

    if rule == 'OpConfusion':
        target = action[2]
        fn = lambda d, nid: apply_op_confusion(d, nid, target)
    elif rule in DISPATCH:
        fn = DISPATCH[rule]
    else:
        raise ValueError(f"Unknown rule: {rule}")

    result = _apply_in_dag(dag, node_id, fn)
    if result is None:
        raise ValueError(f"Node '{node_id}' not found in dag")
    return result


# ── available actions ─────────────────────────────────────────────

def available_actions(dag):
    """
    Return all (rule_name, node_id[, extra]) tuples where a rule can fire
    anywhere in the expression — recursively through all bracket levels.
    """
    actions = []
    _scan_flat_dag(dag, actions)
    return actions
