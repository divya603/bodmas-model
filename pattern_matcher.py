"""
pattern_matcher.py

For every operator in a FlatDAG (recursing into brackets), find all 3-node
windows it participates in and classify each window against Tables 1–5.

Table 1 — standalone:  a OP b       (single op, no neighbours)
Table 2 — all literals: a OP1 b OP2 c
Table 3 — Y on right:   a OP1 b OP2 Y
Table 4 — Y on left:    Y OP1 a OP2 b
Table 5 — Y in middle:  a OP1 Y OP2 b

where a, b, c are numeric literals and Y is any bracket/unary (unreduced) node.
"""

from dag import Node, FlatDAG


# ── helpers ───────────────────────────────────────────────────────────────────

def _is_lit(atom):
    return atom.is_number()

def _sym(atom):
    return atom.label if atom.is_number() else 'Y'


# ── window classification ─────────────────────────────────────────────────────

def _classify_window(a0, op1, a1, op2, a2):
    """
    Classify the 3-node window  a0 op1 a1 op2 a2  into one of Tables 2–5.

    Returns a dict:
        table      : 2 | 3 | 4 | 5 | None  (None = multiple Ys, not in tables)
        pattern    : human-readable pattern string
        reductions : subset of {'op1', 'op2', 'recurse_Y'}
            'op1'       — left op can fire  (evaluate a0 op1 a1)
            'op2'       — right op can fire (evaluate a1 op2 a2)
            'recurse_Y' — must reduce the Y bracket before either op fires
    """
    lit0, lit1, lit2 = _is_lit(a0), _is_lit(a1), _is_lit(a2)

    if lit0 and lit1 and lit2:                    # Table 2
        return dict(
            table=2,
            pattern=f'a {op1.label} b {op2.label} c',
            reductions=['op1', 'op2'],
        )

    if lit0 and (not lit1) and lit2:              # Table 5: Y in middle
        return dict(
            table=5,
            pattern=f'a {op1.label} Y {op2.label} c',
            reductions=['recurse_Y'],
        )

    if lit0 and lit1 and (not lit2):              # Table 3: Y on right
        return dict(
            table=3,
            pattern=f'a {op1.label} b {op2.label} Y',
            reductions=['op1', 'recurse_Y'],
        )

    if (not lit0) and lit1 and lit2:              # Table 4: Y on left
        return dict(
            table=4,
            pattern=f'Y {op1.label} b {op2.label} c',
            reductions=['op2', 'recurse_Y'],
        )

    # multiple Ys — not covered by tables 2–5
    s0 = 'a' if lit0 else 'Y'
    s1 = 'b' if lit1 else 'Y'
    s2 = 'c' if lit2 else 'Y'
    return dict(
        table=None,
        pattern=f'{s0} {op1.label} {s1} {op2.label} {s2}',
        reductions=['recurse_Y'],
    )


# ── per-level scan ────────────────────────────────────────────────────────────

def _scan_level(dag, results, inside_bracket):
    n = len(dag.ops)
    if n == 0:
        return

    for i, op in enumerate(dag.ops):

        if n == 1:
            # Table 1 — standalone: only matches when both atoms are literals
            a0, a1 = dag.atoms[0], dag.atoms[1]
            if not (_is_lit(a0) and _is_lit(a1)):
                continue                          # no pattern matches — skip
            pat = f'(a {op.label} b)' if inside_bracket else f'a {op.label} b'
            results.append(dict(
                op_id=op.node_id, op_label=op.label,
                table=1, pattern=pat, windows=[],
            ))
            continue

        windows = []

        # ops[i] as the RIGHT op: window (atoms[i-1], ops[i-1], atoms[i], ops[i], atoms[i+1])
        if i > 0:
            w = _classify_window(
                dag.atoms[i - 1], dag.ops[i - 1],
                dag.atoms[i],     op,
                dag.atoms[i + 1],
            )
            if w['table'] is not None:            # skip multi-Y windows (no table match)
                w['role'] = 'right'
                windows.append(w)

        # ops[i] as the LEFT op: window (atoms[i], ops[i], atoms[i+1], ops[i+1], atoms[i+2])
        if i < n - 1:
            w = _classify_window(
                dag.atoms[i],     op,
                dag.atoms[i + 1], dag.ops[i + 1],
                dag.atoms[i + 2],
            )
            if w['table'] is not None:            # skip multi-Y windows (no table match)
                w['role'] = 'left'
                windows.append(w)

        if windows:                               # only add if at least one window matched
            results.append(dict(
                op_id=op.node_id, op_label=op.label,
                table=None, windows=windows,
            ))


def _scan(dag, results, inside_bracket=False):
    _scan_level(dag, results, inside_bracket)
    for atom in dag.atoms:
        if atom.is_bracket() and atom.inner_dag:
            _scan(atom.inner_dag, results, inside_bracket=True)
        elif atom.is_unary() and atom.child() and atom.child().is_bracket():
            _scan(atom.child().inner_dag, results, inside_bracket=True)


# ── public API ────────────────────────────────────────────────────────────────

def match_patterns(dag):
    """
    Return a list of match records for every operator in the expression,
    recursing through all bracket levels.

    Each record:
        op_id     : str
        op_label  : str
        table     : 1 | None   — 1 means standalone (Table 1); None means
                                  table is per-window (see 'windows')
        pattern   : str        — set only when table == 1
        windows   : list of window dicts, each containing:
            role       : 'left' | 'right'   ops[i]'s role in this window
            table      : 2 | 3 | 4 | 5 | None
            pattern    : str
            reductions : list — subset of 'op1', 'op2', 'recurse_Y'
    """
    results = []
    _scan(dag, results)
    return results


# ── pretty printer ────────────────────────────────────────────────────────────

def print_matches(matches):
    for m in matches:
        if m['table'] == 1:
            print(f"  {m['op_id']} ({m['op_label']})  [standalone]")
            print(f"      Table 1: {m['pattern']}")
        else:
            nw = len(m['windows'])
            print(f"  {m['op_id']} ({m['op_label']})  [{nw} window{'s' if nw != 1 else ''}]")
            for w in m['windows']:
                reds = ', '.join(w['reductions'])
                tbl  = f"Table {w['table']}" if w['table'] else 'multi-Y'
                print(f"      as {w['role']:5s} op  →  {tbl}: {w['pattern']}   [{reds}]")


# ── quick test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from parser import build_dag

    tests = [
        '5 + 3',
        '4 + 5 + 6 × 2 + 3',
        '5 + 4 × (6 - 9)',
        '2 × (3 + 4) - 1',
        '(2 + 3) × (4 - 1)',
    ]
    for expr in tests:
        dag     = build_dag(expr)
        matches = match_patterns(dag)
        print(f"\n{'─' * 55}")
        print(f"  {expr}")
        print_matches(matches)
