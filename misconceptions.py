"""
misconceptions.py

Defines 5 Table-2 misconceptions and applies them to a FlatDAG.

Cross-precedence (ids 1-4): fire the WRONG (lower-precedence) op in a
3-node all-literal window:
  a wrong b strong c  ->  fire wrong (left)   e.g. a + b x c -> [a+b] x c
  a strong b wrong c  ->  fire wrong (right)  e.g. a x b + c -> a x [b+c]

Same-priority RTL (id 5): fire the RIGHT op in a same-precedence window,
simulating right-to-left evaluation. Excluded: a+b+c and a*b*c (associative).
"""

from dag import Node, FlatDAG


# ── misconception registry ────────────────────────────────────────

MISCONCEPTIONS = [
    {'id': 'add_before_mul', 'name': 'Addition before ×', 'wrong': '+', 'strong': '×',
     'patterns': {'a + b × c', 'a × b + c'}},
    {'id': 'add_before_div', 'name': 'Addition before ÷', 'wrong': '+', 'strong': '÷',
     'patterns': {'a + b ÷ c', 'a ÷ b + c'}},
    {'id': 'sub_before_mul', 'name': 'Subtraction before ×', 'wrong': '-', 'strong': '×',
     'patterns': {'a - b × c', 'a × b - c'}},
    {'id': 'sub_before_div', 'name': 'Subtraction before ÷', 'wrong': '-', 'strong': '÷',
     'patterns': {'a - b ÷ c', 'a ÷ b - c'}},
    {'id': 'same_priority_rtl', 'name': 'Same priority right to left', 'wrong': None, 'strong': None,
     'patterns': {'a + b - c', 'a - b + c', 'a - b - c',
                  'a × b ÷ c', 'a ÷ b × c', 'a ÷ b ÷ c'}},
    # Table 3 misconception
    {'id': 'outside_bracket_first', 'name': 'Evaluate outside bracket first', 'wrong': None, 'strong': None,
     'patterns': {'a + b × Y', 'a + b ÷ Y', 'a - b × Y', 'a - b ÷ Y',   # Table 3
                  'Y × b + c', 'Y × b - c', 'Y ÷ b + c', 'Y ÷ b - c'}}, # Table 4
]

_ADDITIVE       = {'+', '-'}
_MULTIPLICATIVE = {'×', '÷'}


def applicable_misconceptions(matches):
    """
    Given the pattern-match table for an expression, return the subset of
    MISCONCEPTIONS whose trigger patterns appear in the table.
    """
    seen_patterns = set()
    for m in matches:
        if m['table'] == 1:
            continue
        for w in m['windows']:
            seen_patterns.add(w['pattern'])
    return [m for m in MISCONCEPTIONS if seen_patterns & m['patterns']]


# ── helpers ───────────────────────────────────────────────────────

def _is_lit(atom):
    return atom.is_number()


def _compute(a, op, b):
    va, vb = float(a.label), float(b.label)
    o = op.label
    if   o == '+': r = va + vb
    elif o == '-': r = va - vb
    elif o == '×': r = va * vb
    elif o == '÷': r = va / vb
    return Node(str(int(r)) if r == int(r) else str(round(r, 6)))


def _fire(dag, i):
    """Evaluate dag.ops[i] in place, return new FlatDAG."""
    new_atom  = _compute(dag.atoms[i], dag.ops[i], dag.atoms[i + 1])
    new_atoms = dag.atoms[:i] + [new_atom] + dag.atoms[i + 2:]
    new_ops   = dag.ops[:i]   + dag.ops[i + 1:]
    if i > 0:             new_ops[i - 1].children[1] = new_atom
    if i < len(new_ops):  new_ops[i].children[0]     = new_atom
    return FlatDAG(new_atoms, new_ops, dag.counter)


# ── core scan ─────────────────────────────────────────────────────

def _scan(dag, wrong, strong):
    """
    Find first position where the misconception pattern matches and apply it.
    Returns new FlatDAG or None.
    """
    n = len(dag.ops)

    for i in range(n - 1):
        if not (_is_lit(dag.atoms[i]) and
                _is_lit(dag.atoms[i + 1]) and
                _is_lit(dag.atoms[i + 2])):
            continue
        op1, op2 = dag.ops[i].label, dag.ops[i + 1].label

        if op1 == wrong and op2 == strong:    # a wrong b strong c → fire wrong (left)
            return _fire(dag, i)
        if op1 == strong and op2 == wrong:    # a strong b wrong c → fire wrong (right)
            return _fire(dag, i + 1)

    # recurse into brackets
    for k, atom in enumerate(dag.atoms):
        if not (atom.is_bracket() and atom.inner_dag):
            continue
        new_inner = _scan(atom.inner_dag, wrong, strong)
        if new_inner is None:
            continue
        new_atom = Node('()', node_id=atom.node_id)
        new_atom.inner_dag = new_inner
        new_atoms = list(dag.atoms)
        new_atoms[k] = new_atom
        if k > 0:             dag.ops[k - 1].children[1] = new_atom
        if k < len(dag.ops):  dag.ops[k].children[0]     = new_atom
        return FlatDAG(new_atoms, dag.ops, dag.counter)

    return None


# ── same-priority RTL scan ────────────────────────────────────────

def _scan_same_priority_rtl(dag):
    """
    Find the first 3-node all-literal window where both ops are same-precedence
    but NOT both identical-associative (i.e. not a+b+c or a*b*c), then fire
    op2 (right op) — simulating right-to-left evaluation.
    """
    n = len(dag.ops)
    for i in range(n - 1):
        if not (_is_lit(dag.atoms[i]) and
                _is_lit(dag.atoms[i + 1]) and
                _is_lit(dag.atoms[i + 2])):
            continue
        op1, op2 = dag.ops[i].label, dag.ops[i + 1].label
        same_add = op1 in _ADDITIVE and op2 in _ADDITIVE
        same_mul = op1 in _MULTIPLICATIVE and op2 in _MULTIPLICATIVE
        if not (same_add or same_mul):
            continue
        if op1 == '+' and op2 == '+':   # a+b+c — associative, skip
            continue
        if op1 == '×' and op2 == '×':  # a*b*c — associative, skip
            continue
        return _fire(dag, i + 1)        # fire right op (the RTL misconception)

    # recurse into brackets
    for k, atom in enumerate(dag.atoms):
        if not (atom.is_bracket() and atom.inner_dag):
            continue
        new_inner = _scan_same_priority_rtl(atom.inner_dag)
        if new_inner is None:
            continue
        new_atom = Node('()', node_id=atom.node_id)
        new_atom.inner_dag = new_inner
        new_atoms = list(dag.atoms)
        new_atoms[k] = new_atom
        if k > 0:             dag.ops[k - 1].children[1] = new_atom
        if k < len(dag.ops):  dag.ops[k].children[0]     = new_atom
        return FlatDAG(new_atoms, dag.ops, dag.counter)

    return None


# ── outside-bracket-first scan (Table 3) ─────────────────────────

def _scan_outside_bracket_first(dag):
    """
    Table 3: a + b × Y  — op1 additive, op2 multiplicative, right atom is Y → fire op1
    Table 4: Y × b + c  — op1 multiplicative, op2 additive, left atom is Y  → fire op2
    """
    n = len(dag.ops)
    for i in range(n - 1):
        op1, op2 = dag.ops[i].label, dag.ops[i + 1].label

        # Table 3: a + b × Y
        if (_is_lit(dag.atoms[i]) and _is_lit(dag.atoms[i + 1]) and
                not _is_lit(dag.atoms[i + 2]) and
                op1 in _ADDITIVE and op2 in _MULTIPLICATIVE):
            return _fire(dag, i)

        # Table 4: Y × b + c
        if (not _is_lit(dag.atoms[i]) and _is_lit(dag.atoms[i + 1]) and
                _is_lit(dag.atoms[i + 2]) and
                op1 in _MULTIPLICATIVE and op2 in _ADDITIVE):
            return _fire(dag, i + 1)

    # recurse into brackets
    for k, atom in enumerate(dag.atoms):
        if not (atom.is_bracket() and atom.inner_dag):
            continue
        new_inner = _scan_outside_bracket_first(atom.inner_dag)
        if new_inner is None:
            continue
        new_atom = Node('()', node_id=atom.node_id)
        new_atom.inner_dag = new_inner
        new_atoms = list(dag.atoms)
        new_atoms[k] = new_atom
        if k > 0:             dag.ops[k - 1].children[1] = new_atom
        if k < len(dag.ops):  dag.ops[k].children[0]     = new_atom
        return FlatDAG(new_atoms, dag.ops, dag.counter)

    return None


# ── public API ────────────────────────────────────────────────────

def apply_misconception(dag, misconception_id):
    """
    Apply the named misconception to the first matching position in dag.
    Returns a new FlatDAG, or None if the pattern is not present.
    """
    m = next((x for x in MISCONCEPTIONS if x['id'] == misconception_id), None)
    if m is None:
        return None
    if misconception_id == 'same_priority_rtl':
        return _scan_same_priority_rtl(dag)
    if misconception_id == 'outside_bracket_first':
        return _scan_outside_bracket_first(dag)
    return _scan(dag, m['wrong'], m['strong'])


# ── dag → string ──────────────────────────────────────────────────

def _atom_str(atom):
    if atom.is_number():
        return atom.label
    if atom.is_unary():
        return atom.label[1:] + _atom_str(atom.child())
    if atom.is_bracket():
        return '(' + dag_to_str(atom.inner_dag) + ')'
    return str(atom.label)


def dag_to_str(dag):
    if not dag.ops:
        return _atom_str(dag.atoms[0]) if dag.atoms else ''
    parts = []
    for i, atom in enumerate(dag.atoms):
        parts.append(_atom_str(atom))
        if i < len(dag.ops):
            parts.append(dag.ops[i].label)
    return ' '.join(parts)
