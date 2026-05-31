import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from parser import build_dag

st.set_page_config(page_title="BODMAS DAG", layout="centered")

st.title("BODMAS DAG")

expr = st.text_input("Expression", value="5+(2-3)*4")


def _add_subtree(node, lines, seen):
    nid = f"n{id(node)}"
    if id(node) in seen:
        return nid
    seen.add(id(node))

    if node.is_number():
        lines.append(f'  {nid} [label="{node.label}", shape=rectangle, style="rounded,filled", fillcolor="#AED6F1", color="#2471A3"];')
    elif node.is_operator():
        lbl = node.label.replace('"', '\\"')
        lines.append(f'  {nid} [label="{lbl}", shape=circle, style=filled, fillcolor="#FAD7A0", color="#D68910", width=0.55, height=0.55, fixedsize=true];')
        for child in node.children:
            cid = _add_subtree(child, lines, seen)
            lines.append(f"  {nid} -> {cid};")
    elif node.is_bracket():
        lines.append(f'  {nid} [label="( )", shape=ellipse, style=filled, fillcolor="#A9DFBF", color="#1E8449"];')
        if node.child():
            cid = _add_subtree(node.child(), lines, seen)
            lines.append(f"  {nid} -> {cid};")
    elif node.is_unary():
        sym = node.label[1:]
        lines.append(f'  {nid} [label="{sym}", shape=diamond, style=filled, fillcolor="#F9E79F", color="#B7950B"];')
        if node.child():
            cid = _add_subtree(node.child(), lines, seen)
            lines.append(f"  {nid} -> {cid};")

    return nid


def dag_to_dot(dag):
    lines = [
        "digraph {",
        '  graph [rankdir=TB, nodesep=0.3, ranksep=0.5, size="4,4"];',
        '  node [fontname="Helvetica", fontsize=11];',
        '  edge [color="#888888", arrowsize=0.8];',
    ]
    seen = set()

    if dag.is_done():
        _add_subtree(dag.atoms[0], lines, seen)
    else:
        for op in dag.ops:
            _add_subtree(op, lines, seen)
        if len(dag.ops) > 1:
            same = "; ".join(f"n{id(op)}" for op in dag.ops)
            lines.append(f"  {{ rank=same; {same}; }}")

    lines.append("}")
    return "\n".join(lines)


if expr.strip():
    try:
        dag = build_dag(expr.strip())
        _, col, _ = st.columns([1, 2, 1])
        with col:
            st.graphviz_chart(dag_to_dot(dag), use_container_width=True)
    except Exception as e:
        st.error(f"Parse error: {e}")
