import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from parser import build_dag
from pattern_matcher import match_patterns
from misconceptions import dag_to_str
from valid_actions import compute_valid_actions, fire_operator, fire_inner_op, inner_valid_actions, is_zero_divide
from generator import generate_expression
from traces import generate_traces
from learner import MISCONCEPTION_FLIPS, flipped_cells as get_flipped_cells
from distance import compare as compare_traces

# ── operator colours ──────────────────────────────────────────────

PALETTE   = ['#e74c3c','#2980b9','#27ae60','#e67e22','#8e44ad',
             '#f1c40f','#16a085','#c0392b','#1abc9c','#d35400']
_SYMBOLS  = ['+', '-', '×', '÷', '*', '/']
OP_COLORS = {sym: PALETTE[i % len(PALETTE)] for i, sym in enumerate(_SYMBOLS)}

MISCONCEPTION_LABELS = {
    'add_before_mul':        'Addition before ×',
    'add_before_div':        'Addition before ÷',
    'sub_before_mul':        'Subtraction before ×',
    'sub_before_div':        'Subtraction before ÷',
    'same_priority_rtl':     'Same priority right-to-left',
    'outside_bracket_first': 'Outside bracket first',
}
LABEL_TO_ID = {v: k for k, v in MISCONCEPTION_LABELS.items()}

def colorize(text):
    out = ''
    for ch in text:
        if ch in OP_COLORS:
            out += f'<b style="color:{OP_COLORS[ch]}">{ch}</b>'
        else:
            out += ch
    return out

def atom_label(atom):
    if atom.is_number():  return atom.label
    if atom.is_bracket(): return f'({dag_to_str(atom.inner_dag)})'
    return '?'

def render_truth_table(actions, n_ops, flipped_up=None, flipped_down=None):
    """
    flipped_up   : set of (op_index, w_idx) flipped F→T (orange T*)
    flipped_down : set of (op_index, w_idx) flipped T→F (purple F*)
    """
    n_windows = max(0, n_ops - 1)
    if not actions or n_ops == 0:
        return ''
    if flipped_up   is None: flipped_up   = set()
    if flipped_down is None: flipped_down = set()

    win_labels = [None] * n_windows
    for a in actions:
        for w_idx, pattern, _ in a['truth']:
            if 0 <= w_idx < n_windows and win_labels[w_idx] is None:
                win_labels[w_idx] = pattern
    win_labels = [lbl or f'W{i+1}' for i, lbl in enumerate(win_labels)]

    H  = 'style="padding:5px 14px;text-align:center;color:#888;font-weight:normal;border-bottom:1px solid #ddd;font-size:0.82rem"'
    L  = 'style="padding:5px 14px;font-family:monospace;border-bottom:1px solid #f0f0f0"'
    Tc = 'style="padding:5px 14px;text-align:center;color:#27ae60;font-weight:bold"'
    Fc = 'style="padding:5px 14px;text-align:center;color:#e74c3c;font-weight:bold"'
    Fu = 'style="padding:5px 14px;text-align:center;color:#e67e22;font-weight:bold"'  # F→T flip
    Fd = 'style="padding:5px 14px;text-align:center;color:#8e44ad;font-weight:bold"'  # T→F flip
    Dc = 'style="padding:5px 14px;text-align:center;color:#ccc"'

    header = f'<th {H}>op</th>'
    for lbl in win_labels:
        header += f'<th {H}>{colorize(lbl)}</th>'
    header += f'<th {H}>valid</th>'

    rows = ''
    for a in actions:
        row_map = {w_idx: c for w_idx, _, c in a['truth']}
        rows += f'<tr><td {L}>{colorize(a["op_label"])}</td>'
        for w_idx in range(n_windows):
            cell = (a['op_index'], w_idx)
            if w_idx in row_map:
                if cell in flipped_up:
                    rows += f'<td {Fu}>T*</td>'
                elif cell in flipped_down:
                    rows += f'<td {Fd}>F*</td>'
                elif row_map[w_idx]:
                    rows += f'<td {Tc}>T</td>'
                else:
                    rows += f'<td {Fc}>F</td>'
            else:
                rows += f'<td {Dc}>—</td>'
        icon  = '✓' if a['valid'] else '✗'
        color = '#27ae60' if a['valid'] else '#e74c3c'
        rows += f'<td style="padding:5px 14px;text-align:center;font-weight:bold;color:{color}">{icon}</td></tr>'

    return f'<table style="border-collapse:collapse;font-size:0.95rem;margin-top:0.4rem"><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table>'

def render_traces(traces):
    for i, t in enumerate(traces):
        parts = [colorize(s) for s in t]
        st.markdown(
            f'<p style="font-family:monospace;font-size:0.88rem;line-height:1.8">'
            f'<b style="color:#888">[{i+1}]</b>&nbsp;&nbsp;'
            + '&nbsp;&nbsp;<b>→</b>&nbsp;&nbsp;'.join(parts)
            + '</p>',
            unsafe_allow_html=True,
        )

# ── page ──────────────────────────────────────────────────────────

st.set_page_config(page_title="BODMAS Step-by-Step", layout="centered")
st.title("BODMAS Step-by-Step")

# ── expression input + generate (shared across tabs) ──────────────

gcol1, gcol2, gcol3 = st.columns([2, 2, 4])
with gcol1:
    n_ops_gen = st.number_input("Operators", min_value=2, max_value=8, value=4, step=1)
with gcol2:
    st.markdown("<div style='margin-top:1.75rem'>", unsafe_allow_html=True)
    if st.button("Generate", use_container_width=True):
        st.session_state['_expr_input'] = generate_expression(n_ops=int(n_ops_gen))
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

expr_input = st.text_input(
    "Expression",
    placeholder="e.g. 4 + 5 + 6 × 2 + 3",
    label_visibility="collapsed",
    key='_expr_input',
)

if not (expr_input and expr_input.strip()):
    st.stop()

expr = expr_input.strip()

if st.session_state.get('_expr') != expr:
    try:
        dag0 = build_dag(expr)
    except Exception as e:
        st.error(f"Parse error: {e}")
        st.stop()
    st.session_state['_expr']    = expr
    st.session_state['_dag']     = dag0
    st.session_state['_history'] = [dag_to_str(dag0)]

dag          = st.session_state['_dag']
initial_dag  = build_dag(expr)

# ── tabs ──────────────────────────────────────────────────────────

tab_expert, tab_learner = st.tabs(["Expert Learner", "Misconception Learner"])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — Expert Learner
# ══════════════════════════════════════════════════════════════════

with tab_expert:

    current = dag_to_str(dag)
    st.markdown(
        f'<p style="font-family:monospace;font-size:1.3rem;margin-bottom:1rem">'
        f'{colorize(current)}</p>',
        unsafe_allow_html=True,
    )

    try:
        matches = match_patterns(dag)
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

    # truth table — top level
    st.markdown("**Truth Table**")
    actions = compute_valid_actions(dag, matches)
    n_ops   = len(dag.ops)
    tt_html = render_truth_table(actions, n_ops)
    if tt_html:
        st.markdown(tt_html, unsafe_allow_html=True)

    # truth table — brackets
    for atom_index, atom in enumerate(dag.atoms):
        if not (atom.is_bracket() and atom.inner_dag):
            continue
        inner         = atom.inner_dag
        inner_matches = match_patterns(inner)
        inner_actions = compute_valid_actions(inner, inner_matches)
        inner_html    = render_truth_table(inner_actions, len(inner.ops))
        if inner_html:
            label = dag_to_str(inner)
            st.markdown(
                f'<p style="font-family:monospace;color:#888;font-size:0.85rem;margin-top:1rem">'
                f'Inside ({colorize(label)})</p>',
                unsafe_allow_html=True,
            )
            st.markdown(inner_html, unsafe_allow_html=True)

    # valid actions
    st.divider()
    st.markdown("**Valid Actions**")
    valid_ops = [a for a in actions if a['valid'] and not is_zero_divide(dag, a['op_index'])]
    inner_ops = inner_valid_actions(dag)

    if n_ops == 0:
        st.success(f"Done! Result = **{current}**")
    elif valid_ops:
        cols = st.columns(len(valid_ops))
        for col, a in zip(cols, valid_ops):
            left  = atom_label(dag.atoms[a['op_index']])
            right = atom_label(dag.atoms[a['op_index'] + 1])
            btn   = f"{left} {a['op_label']} {right}"
            if col.button(btn, key=f"fire_{a['op_id']}", use_container_width=True):
                new_dag = fire_operator(dag, a['op_index'])
                st.session_state['_dag'] = new_dag
                st.session_state['_history'].append(dag_to_str(new_dag))
                st.rerun()
    elif inner_ops:
        st.info("Resolve bracket(s) first.")
        cols = st.columns(len(inner_ops))
        for col, ia in zip(cols, inner_ops):
            btn = f"{ia['left_label']} {ia['op_label']} {ia['right_label']}"
            if col.button(btn, key=f"inner_{ia['atom_index']}_{ia['op_id']}", use_container_width=True):
                new_dag = fire_inner_op(dag, ia['atom_index'], ia['inner_op_index'])
                st.session_state['_dag'] = new_dag
                st.session_state['_history'].append(dag_to_str(new_dag))
                st.rerun()
    else:
        st.warning("No valid actions found.")

    # step history
    history = st.session_state.get('_history', [])
    if len(history) > 1:
        st.divider()
        st.markdown("**Steps**")
        parts = [colorize(s) for s in history]
        st.markdown(
            '<p style="font-family:monospace;font-size:0.95rem;line-height:2">'
            + '&nbsp;&nbsp;→&nbsp;&nbsp;'.join(parts)
            + '</p>',
            unsafe_allow_html=True,
        )
        if st.button("Reset", key="reset"):
            st.session_state['_dag']     = build_dag(expr)
            st.session_state['_history'] = [dag_to_str(st.session_state['_dag'])]
            st.rerun()

    # all expert traces
    st.divider()
    expert_traces = generate_traces(initial_dag, [])
    with st.expander(f"Expert traces — {len(expert_traces)} total"):
        render_traces(expert_traces)


# ══════════════════════════════════════════════════════════════════
# TAB 2 — Misconception Learner
# ══════════════════════════════════════════════════════════════════

with tab_learner:
    from traces import compute_valid_actions_for_learner

    st.markdown("Select up to **2 misconceptions** to define this learner:")
    selected_labels = st.multiselect(
        "Misconceptions",
        options=list(MISCONCEPTION_LABELS.values()),
        max_selections=2,
        label_visibility="collapsed",
    )
    selected_ids = [LABEL_TO_ID[lbl] for lbl in selected_labels]

    if not selected_ids:
        st.info("No misconceptions selected — this learner is identical to the expert.")

    # ── truth table with flipped cells highlighted ─────────────────

    st.markdown("**Truth Table**")

    l_matches       = match_patterns(initial_dag)
    learner_actions = compute_valid_actions_for_learner(initial_dag, l_matches, selected_ids)
    expert_actions0 = compute_valid_actions(initial_dag, l_matches)
    up, down        = get_flipped_cells(expert_actions0, learner_actions)

    tt_html = render_truth_table(learner_actions, len(initial_dag.ops),
                                 flipped_up=up, flipped_down=down)
    if tt_html:
        st.markdown(tt_html, unsafe_allow_html=True)
        if up or down:
            legend = []
            if up:   legend.append('<span style="color:#e67e22">T*</span> = flipped F→T (learner fires wrong op)')
            if down: legend.append('<span style="color:#8e44ad">F*</span> = flipped T→F (learner blocks correct op)')
            st.markdown(
                '<p style="font-size:0.8rem;margin-top:0.3rem">' + '&nbsp;&nbsp;|&nbsp;&nbsp;'.join(legend) + '</p>',
                unsafe_allow_html=True,
            )

    # bracket truth tables
    for atom_index, atom in enumerate(initial_dag.atoms):
        if not (atom.is_bracket() and atom.inner_dag):
            continue
        inner        = atom.inner_dag
        inner_m      = match_patterns(inner)
        inner_l_acts = compute_valid_actions_for_learner(inner, inner_m, selected_ids)
        inner_e_acts = compute_valid_actions(inner, inner_m)
        i_up, i_down = get_flipped_cells(inner_e_acts, inner_l_acts)
        inner_html   = render_truth_table(inner_l_acts, len(inner.ops),
                                          flipped_up=i_up, flipped_down=i_down)
        if inner_html:
            label = dag_to_str(inner)
            st.markdown(
                f'<p style="font-family:monospace;color:#888;font-size:0.85rem;margin-top:1rem">'
                f'Inside ({colorize(label)})</p>',
                unsafe_allow_html=True,
            )
            st.markdown(inner_html, unsafe_allow_html=True)

    # ── score vs expert ────────────────────────────────────────────

    st.divider()
    learner_traces = generate_traces(initial_dag, selected_ids)
    score          = compare_traces(expert_traces, learner_traces)

    s1, s2, s3 = st.columns(3)
    s1.metric("Edge Jaccard", f"{score['edge_jaccard']:.2f}",
              help="Similarity of transition sets (1 = identical to expert, 0 = completely different)")
    s2.metric("Correct rate", f"{score['correct_rate']:.0%}",
              help="Fraction of this learner's traces that reach the right answer")
    s3.metric("Shared transitions", f"{score['shared_edges']}",
              help=f"Expert-only: {score['expert_only']}  |  Learner-only: {score['learner_only']}")

    # ── traces ─────────────────────────────────────────────────────

    st.markdown(f"**{len(learner_traces)} trace(s)** for this learner")
    render_traces(learner_traces)
