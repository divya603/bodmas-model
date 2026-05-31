class Node:
    def __init__(self, label, children=None, node_id=None):
        self.label = label
        self.children = children if children is not None else []
        self.node_id = node_id

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
        return self.children[0] if len(self.children) == 1 else None


class FlatDAG:
    """
    Canonical flat DAG representation of an arithmetic expression.

    atoms : [a0, a1, ..., aN]      — numbers or bracket nodes, in order
    ops   : [o0, o1, ..., o(N-1)]  — operator nodes, in order

    Invariant:
        ops[i].left()  == atoms[i]
        ops[i].right() == atoms[i+1]

    Adjacent atoms are the SAME Python object in both neighbour ops,
    which is how sharing is represented.

    Example  5+(2-3)*4:
        atoms = [Node(5),  Node(()),  Node(4)]
        ops   = [Node(+),  Node(×)]
        Node(()) is the identical object in ops[0].right and ops[1].left
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


# ──────────────────────────────────────────────────────────────────
# Printing
# ──────────────────────────────────────────────────────────────────

def _atom_label(atom):
    if atom.is_number():
        return atom.label
    elif atom.is_bracket():
        return f"{atom.node_id or '?'}(())"
    elif atom.is_unary():
        sym = atom.label[1:]   # 'u-' → '-', 'u+' → '+'
        return f"{atom.node_id or '?'}({sym}{_atom_label(atom.child())})"
    return str(atom.label)


def _print_inner(node, prefix="", is_last=True):
    """Pretty-print the Node tree inside a bracket."""
    if node is None:
        return
    connector = "└── " if is_last else "├── "
    sym       = node.label[1:] if node.is_unary() else node.label
    display   = f"{node.node_id}({sym})" if node.node_id else sym
    print(prefix + connector + display)
    child_prefix = prefix + ("    " if is_last else "│   ")
    for i, child in enumerate(node.children):
        _print_inner(child, child_prefix, i == len(node.children) - 1)


def print_dag(dag):
    """
    Print the flat DAG.
    Shows the top-level atom/op sequence, then the interior of each bracket.
    """
    if dag.is_done():
        atom = dag.atoms[0]
        print(f"Result: {_atom_label(atom)}")
        return

    # top-level sequence
    parts = []
    for i, atom in enumerate(dag.atoms):
        parts.append(_atom_label(atom))
        if i < len(dag.ops):
            op = dag.ops[i]
            parts.append(f"[{op.node_id}:{op.label}]")
    print("Top level:  " + "  ".join(parts))

    # bracket internals
    seen = set()
    for atom in dag.atoms:
        if atom.is_bracket() and id(atom) not in seen:
            seen.add(id(atom))
            print(f"\n  {atom.node_id}(()) contains:")
            _print_inner(atom.child(), "    ", True)