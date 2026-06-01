import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from parser import build_dag
from rules import available_actions, apply_action

st.set_page_config(page_title="BODMAS DAG", layout="wide")
st.title("BODMAS DAG")

expr = st.text_input("Expression", key="expr_val")
if "expr_val" not in st.session_state:
    st.session_state.expr_val = "5+(2-3)*4"

# reset history whenever expression changes
if st.session_state.get("last_expr") != expr:
    st.session_state.last_expr = expr
    st.session_state.history   = []


# ── DOT generation ────────────────────────────────────────────────

def _add_node_decl(node, lines, seen):
    nid = f"n{id(node)}"
    if id(node) in seen:
        return nid
    seen.add(id(node))
    if node.is_number():
        lines.append(f'  {nid} [label="{node.label}", shape=rectangle, style="rounded,filled", fillcolor="#AED6F1", color="#2471A3"];')
    elif node.is_operator():
        lbl = node.label.replace('"', '\\"')
        nid_label = node.node_id or ""
        lines.append(f'  {nid} [label="{lbl}\\n{nid_label}", shape=circle, style=filled, fillcolor="#FAD7A0", color="#D68910", fontsize=9, width=0.6, height=0.6, fixedsize=true];')
    elif node.is_bracket():
        lines.append(f'  {nid} [label="( )", shape=ellipse, style=filled, fillcolor="#A9DFBF", color="#1E8449"];')
    elif node.is_unary():
        sym = node.label[1:]
        lines.append(f'  {nid} [label="{sym}", shape=diamond, style=filled, fillcolor="#F9E79F", color="#B7950B"];')
    return nid


def _render_flat_dag(dag, lines, seen, rendered):
    if id(dag) in rendered:
        return
    rendered.add(id(dag))
    for op in dag.ops:
        _add_node_decl(op, lines, seen)
    for atom in dag.atoms:
        _add_node_decl(atom, lines, seen)
    for i, op in enumerate(dag.ops):
        op_nid    = f"n{id(op)}"
        left_nid  = f"n{id(dag.atoms[i])}"
        right_nid = f"n{id(dag.atoms[i+1])}"
        lines.append(f"  {op_nid} -> {left_nid};")
        lines.append(f"  {op_nid} -> {right_nid};")
    if len(dag.ops) > 1:
        same = "; ".join(f"n{id(op)}" for op in dag.ops)
        lines.append(f"  {{ rank=same; {same}; }}")
    for atom in dag.atoms:
        if atom.is_bracket() and atom.inner_dag:
            bnid = f"n{id(atom)}"
            _render_flat_dag(atom.inner_dag, lines, seen, rendered)
            if atom.inner_dag.ops:
                for inner_op in atom.inner_dag.ops:
                    lines.append(f"  {bnid} -> n{id(inner_op)};")
            elif atom.inner_dag.atoms:
                lines.append(f"  {bnid} -> n{id(atom.inner_dag.atoms[0])};")
        elif atom.is_unary() and atom.child():
            unid = f"n{id(atom)}"
            child = atom.child()
            _add_node_decl(child, lines, seen)
            lines.append(f"  {unid} -> n{id(child)};")
            if child.is_bracket() and child.inner_dag:
                _render_flat_dag(child.inner_dag, lines, seen, rendered)
                cnid = f"n{id(child)}"
                if child.inner_dag.ops:
                    for inner_op in child.inner_dag.ops:
                        lines.append(f"  {cnid} -> n{id(inner_op)};")
                elif child.inner_dag.atoms:
                    lines.append(f"  {cnid} -> n{id(child.inner_dag.atoms[0])};")


def dag_to_dot(dag):
    lines = [
        "digraph {",
        '  graph [rankdir=TB, nodesep=0.3, ranksep=0.5, size="4,4"];',
        '  node [fontname="Helvetica", fontsize=11];',
        '  edge [color="#888888", arrowsize=0.8];',
    ]
    seen = set(); rendered = set()
    _render_flat_dag(dag, lines, seen, rendered)
    lines.append("}")
    return "\n".join(lines)


# ── dag → expression string ───────────────────────────────────────

def _atom_str(atom):
    if atom.is_number():
        return atom.label
    elif atom.is_unary():
        sym = atom.label[1:]
        return f"{sym}{_atom_str(atom.child())}"
    elif atom.is_bracket():
        return f"({_dag_str(atom.inner_dag)})"
    return str(atom.label)

def _dag_str(dag):
    if dag.is_done():
        return _atom_str(dag.atoms[0])
    parts = []
    for i, atom in enumerate(dag.atoms):
        parts.append(_atom_str(atom))
        if i < len(dag.ops):
            parts.append(dag.ops[i].label)
    return " ".join(parts)


# ── action label ──────────────────────────────────────────────────

def action_label(a):
    if len(a) == 2:
        return f"{a[0]}  @  {a[1]}"
    return f"{a[0]}  @  {a[1]}  →  {a[2]}"


# ── main ──────────────────────────────────────────────────────────

if expr and expr.strip():
    try:
        dag = build_dag(expr.strip())

        def show_dag_with_actions(current_dag, key_prefix):
            dag_col, actions_col = st.columns([2, 1])
            with dag_col:
                st.graphviz_chart(dag_to_dot(current_dag), use_container_width=False)
            with actions_col:
                st.markdown("**Available actions**")
                for i, a in enumerate(available_actions(current_dag)):
                    if st.button(action_label(a), key=f"{key_prefix}_{i}"):
                        st.session_state.history.append((a, apply_action(current_dag, a)))

        # initial DAG — clicking clears history and starts fresh
        init_col, init_actions_col = st.columns([2, 1])
        with init_col:
            st.graphviz_chart(dag_to_dot(dag), use_container_width=False)
        with init_actions_col:
            st.markdown("**Available actions**")
            for i, a in enumerate(available_actions(dag)):
                if st.button(action_label(a), key=f"init_{i}"):
                    st.session_state.history = [(a, apply_action(dag, a))]

        # chain of steps
        for step, (action, step_dag) in enumerate(st.session_state.get("history", [])):
            st.divider()
            st.markdown(f"**After:  {action_label(action)}**")
            st.code(_dag_str(step_dag), language=None)
            show_dag_with_actions(step_dag, key_prefix=f"step{step}")

    except Exception as e:
        st.error(f"Parse error: {e}")
