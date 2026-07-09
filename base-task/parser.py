from dag import Node, FlatDAG

OPERATORS = {'+', '-', '×', '÷'}


def tokenize(expression):
    tokens = []
    expect_unary = True   # True at start, after '(', or after a binary operator
    i = 0
    while i < len(expression):
        c = expression[i]
        if c == ' ':
            i += 1
        elif c.isdigit() or c == '.':
            j = i
            while j < len(expression) and (expression[j].isdigit() or expression[j] == '.'):
                j += 1
            tokens.append(expression[i:j])
            i = j
            expect_unary = False
        elif c in '+-−':          # '−' is U+2212 unicode minus
            base = '-' if c == '−' else c
            tokens.append('u' + base if expect_unary else base)
            expect_unary = True
            i += 1
        elif c in '*×':
            tokens.append('×')
            expect_unary = True
            i += 1
        elif c in '/÷':
            tokens.append('÷')
            expect_unary = True
            i += 1
        elif c == '(':
            tokens.append('(')
            expect_unary = True
            i += 1
        elif c == ')':
            tokens.append(')')
            expect_unary = False
            i += 1
        else:
            raise ValueError(f"Unknown character: '{c}'")
    return tokens


# ── Core parser ───────────────────────────────────────────────────

def _parse_flat(tokens, pos, counter):
    """
    Parse a flat atom/op sequence up to ')' or end of tokens.
    Returns (FlatDAG, new_pos).

    This is the single parsing routine used at every level — top level
    and inside brackets — so the flat-DAG structure is uniform throughout.
    """
    atoms     = []
    op_labels = []

    while pos < len(tokens) and tokens[pos] != ')':
        atom, pos = _parse_atom(tokens, pos, counter)
        atoms.append(atom)
        if pos < len(tokens) and tokens[pos] in OPERATORS:
            op_labels.append(tokens[pos])
            pos += 1

    # Build op nodes; adjacent atoms share the same Python object.
    op_nodes = []
    for i, label in enumerate(op_labels):
        op_id = f'op{counter[0]}'
        counter[0] += 1
        op_nodes.append(Node(label, children=[atoms[i], atoms[i+1]], node_id=op_id))

    return FlatDAG(atoms, op_nodes, counter), pos


def _parse_atom(tokens, pos, counter):
    """Parse one atom: number, bracket (→ inner FlatDAG), or unary-prefix node."""
    tok = tokens[pos]

    if tok in ('u-', 'u+'):
        inner, pos = _parse_atom(tokens, pos + 1, counter)
        if inner.is_number():
            # fold the sign directly into a signed-number atom (e.g. "-14")
            # instead of a unary wrapper node, so it matches how _eval
            # represents negative results — is_number() must be True for a
            # bare negative literal, or downstream code (pattern_matcher's
            # literal checks, inference's action enumeration) treats it as
            # an unresolved sub-expression and refuses to touch it
            value = float(inner.label)
            if tok == 'u-':
                value = -value
            label = str(int(value)) if value == int(value) else str(value)
            return Node(label), pos
        op_id = f'op{counter[0]}'
        counter[0] += 1
        return Node(tok, children=[inner], node_id=op_id), pos

    if tok == '(':
        op_id = f'op{counter[0]}'
        counter[0] += 1
        pos += 1
        inner_dag, pos = _parse_flat(tokens, pos, counter)
        if pos >= len(tokens) or tokens[pos] != ')':
            raise ValueError("Expected ')'")
        pos += 1
        bracket = Node('()', node_id=op_id)
        bracket.inner_dag = inner_dag
        return bracket, pos

    # plain number (or unrecognised token treated as atom)
    return Node(tok), pos + 1


def build_dag(expression):
    tokens    = tokenize(expression)
    counter   = [1]
    dag, _    = _parse_flat(tokens, 0, counter)
    return dag
