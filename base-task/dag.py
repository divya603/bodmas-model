class Node:
    def __init__(self, label, children=None, node_id=None):
        self.label     = label
        self.children  = children if children is not None else []
        self.node_id   = node_id
        self.inner_dag = None   # FlatDAG; set for bracket nodes only

    def is_number(self):
        try:
            float(self.label)
            return True
        except ValueError:
            return False

    def is_operator(self):
        return self.label in {'+', '-', '×', '÷'}

    def is_bracket(self):
        return self.label == '()'

    def is_unary(self):
        return self.label in {'u-', 'u+'}

    def left(self):
        return self.children[0] if len(self.children) > 0 else None

    def right(self):
        return self.children[1] if len(self.children) > 1 else None

    def child(self):
        """For unary nodes: returns the single wrapped child."""
        return self.children[0] if len(self.children) == 1 else None


class FlatDAG:
    """
    Flat DAG representation of an arithmetic expression at any level.

    atoms : [a0, a1, ..., aN]     — numbers, bracket nodes, or unary nodes
    ops   : [o0, o1, ..., o(N-1)] — operator nodes

    Invariant:
        ops[i].left()  is atoms[i]      (same Python object)
        ops[i].right() is atoms[i+1]    (same Python object)

    Bracket nodes carry their own inner FlatDAG in node.inner_dag,
    so the same flat structure applies recursively at every level.

    Example  5+(2+3-5)*4:
        Top-level  atoms = [5,  bracket,  4]
                   ops   = [+,  ×]
        bracket.inner_dag:
                   atoms = [2,  3,  5]
                   ops   = [+,  -]          ← 3 shared between both ops
    """
    def __init__(self, atoms, ops, counter):
        self.atoms   = atoms
        self.ops     = ops
        self.counter = counter

    def is_done(self):
        return len(self.ops) == 0

    def find_op(self, op_id):
        """Return (index, op_node) for op_id, or (None, None)."""
        for i, op in enumerate(self.ops):
            if op.node_id == op_id:
                return i, op
        return None, None

    def find_atom(self, atom_id):
        """Return (index, atom_node) for atom_id, or (None, None)."""
        for j, a in enumerate(self.atoms):
            if a.node_id == atom_id:
                return j, a
        return None, None


# ── Printing ──────────────────────────────────────────────────────

def _atom_label(atom):
    if atom.is_number():
        return atom.label
    elif atom.is_bracket():
        return f"{atom.node_id or '?'}(())"
    elif atom.is_unary():
        sym = atom.label[1:]
        return f"{atom.node_id or '?'}({sym}{_atom_label(atom.child())})"
    return str(atom.label)


def print_dag(dag, indent=0):
    pad = "  " * indent

    if dag.is_done():
        print(pad + f"Result: {_atom_label(dag.atoms[0])}")
        _print_bracket_internals(dag.atoms, indent)
        return

    parts = []
    for i, atom in enumerate(dag.atoms):
        parts.append(_atom_label(atom))
        if i < len(dag.ops):
            op = dag.ops[i]
            parts.append(f"[{op.node_id}:{op.label}]")
    print(pad + "Sequence:  " + "  ".join(parts))
    _print_bracket_internals(dag.atoms, indent)


def _print_bracket_internals(atoms, indent, seen=None):
    if seen is None:
        seen = set()
    pad = "  " * indent
    for atom in atoms:
        if atom.is_bracket() and id(atom) not in seen:
            seen.add(id(atom))
            print(f"\n{pad}  {atom.node_id}(()) contains:")
            print_dag(atom.inner_dag, indent + 2)
        elif atom.is_unary() and atom.child() and atom.child().is_bracket():
            bracket = atom.child()
            if id(bracket) not in seen:
                seen.add(id(bracket))
                print(f"\n{pad}  {bracket.node_id}(()) contains:")
                print_dag(bracket.inner_dag, indent + 2)
