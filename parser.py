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


# ── inside brackets: right-leaning tree ──────────────────────────

def _parse_inner_expr(tokens, pos, counter):
    left, pos = _parse_inner_atom(tokens, pos, counter)
    if pos < len(tokens) and tokens[pos] in OPERATORS:
        op_label = tokens[pos]
        op_id    = f'op{counter[0]}'
        counter[0] += 1
        pos += 1
        right, pos = _parse_inner_expr(tokens, pos, counter)
        return Node(op_label, children=[left, right], node_id=op_id), pos
    return left, pos


def _parse_inner_atom(tokens, pos, counter):
    tok = tokens[pos]
    if tok in ('u-', 'u+'):
        op_id = f'op{counter[0]}'
        counter[0] += 1
        inner, pos = _parse_inner_atom(tokens, pos + 1, counter)
        return Node(tok, children=[inner], node_id=op_id), pos
    if tok == '(':
        op_id = f'op{counter[0]}'
        counter[0] += 1
        pos += 1
        inner, pos = _parse_inner_expr(tokens, pos, counter)
        if pos >= len(tokens) or tokens[pos] != ')':
            raise ValueError("Expected ')'")
        pos += 1
        return Node('()', children=[inner], node_id=op_id), pos
    return Node(tok), pos + 1


# ── top level: flat atom/operator sequence ────────────────────────

def _parse_top_atom(tokens, pos, counter):
    """Parse one atom at the top level: number, bracket, or unary-prefix node."""
    tok = tokens[pos]
    if tok in ('u-', 'u+'):
        op_id = f'op{counter[0]}'
        counter[0] += 1
        inner, pos = _parse_top_atom(tokens, pos + 1, counter)
        return Node(tok, children=[inner], node_id=op_id), pos
    if tok == '(':
        op_id = f'op{counter[0]}'
        counter[0] += 1
        pos += 1
        inner, pos = _parse_inner_expr(tokens, pos, counter)
        if pos >= len(tokens) or tokens[pos] != ')':
            raise ValueError("Expected ')'")
        pos += 1
        return Node('()', children=[inner], node_id=op_id), pos
    return Node(tok), pos + 1


def build_dag(expression):
    """
    Parse expression into a canonical FlatDAG.

    Atoms between adjacent operators are the SAME Python object in both
    operator nodes — this is the shared-node (true DAG) representation.
    """
    tokens    = tokenize(expression)
    counter   = [1]
    atoms     = []
    op_labels = []

    pos = 0
    while pos < len(tokens):
        atom, pos = _parse_top_atom(tokens, pos, counter)
        atoms.append(atom)
        if pos < len(tokens) and tokens[pos] in OPERATORS:
            op_labels.append(tokens[pos])
            pos += 1

    # atoms[i] is literally the same object in ops[i-1].right and ops[i].left
    op_nodes = []
    for i, label in enumerate(op_labels):
        op_id   = f'op{counter[0]}'
        counter[0] += 1
        op_node = Node(label, children=[atoms[i], atoms[i+1]], node_id=op_id)
        op_nodes.append(op_node)

    return FlatDAG(atoms, op_nodes, counter)