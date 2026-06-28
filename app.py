import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from parser import build_dag
from pattern_matcher import match_patterns
from misconceptions import MISCONCEPTIONS, applicable_misconceptions, apply_misconception, dag_to_str

# ── operator colors ───────────────────────────────────────────────

PALETTE = [
    '#e74c3c',   # red
    '#2980b9',   # blue
    '#27ae60',   # green
    '#e67e22',   # orange
    '#8e44ad',   # purple
    '#f1c40f',   # yellow
    '#16a085',   # teal
    '#c0392b',   # dark red
    '#1abc9c',   # mint
    '#d35400',   # burnt orange
]

_SYMBOLS = ['+', '-', '×', '÷', '*', '/']
OP_COLORS = {sym: PALETTE[i % len(PALETTE)] for i, sym in enumerate(_SYMBOLS)}

def colorize(text):
    """Wrap every operator character in a colored bold span."""
    out = ''
    for ch in text:
        if ch in OP_COLORS:
            out += f'<b style="color:{OP_COLORS[ch]}">{ch}</b>'
        else:
            out += ch
    return out


# ── page ──────────────────────────────────────────────────────────

st.set_page_config(page_title="BODMAS Pattern Matcher", layout="centered")
st.title("BODMAS Pattern Matcher")

expr = st.text_input("Expression", placeholder="e.g. 4 + 5 + 6 × 2 + 3", label_visibility="collapsed")

if not (expr and expr.strip()):
    st.stop()

try:
    dag     = build_dag(expr.strip())
    matches = match_patterns(dag)
except Exception as e:
    st.error(f"Parse error: {e}")
    st.stop()

# colored expression header
st.markdown(
    f'<p style="font-family:monospace;font-size:1.25rem;margin-bottom:1.2rem">'
    f'{colorize(expr.strip())}</p>',
    unsafe_allow_html=True,
)

# ── HTML table ────────────────────────────────────────────────────

th = 'style="padding:6px 16px;text-align:left;color:#888;font-weight:normal;border-bottom:1px solid #ddd"'
td_op  = 'style="padding:6px 16px;color:#aaa;font-family:monospace"'
td_lbl = 'style="padding:6px 16px;font-family:monospace;font-size:1.05rem"'
td_pat = 'style="padding:6px 16px;font-family:monospace"'
tr_sep = 'style="border-bottom:1px solid #f0f0f0"'

rows_html = ''
for m in matches:
    if m['table'] == 1:
        patterns_html = colorize(m['pattern'])
    else:
        parts = [colorize(w['pattern']) for w in m['windows']]
        patterns_html = '&nbsp;&nbsp;&nbsp;·&nbsp;&nbsp;&nbsp;'.join(parts)

    rows_html += (
        f'<tr {tr_sep}>'
        f'<td {td_op}>{m["op_id"]}</td>'
        f'<td {td_lbl}>{colorize(m["op_label"])}</td>'
        f'<td {td_pat}>{patterns_html}</td>'
        f'</tr>'
    )

table_html = f'''
<table style="width:100%;border-collapse:collapse;font-size:0.95rem">
  <thead>
    <tr>
      <th {th}>op</th>
      <th {th}>label</th>
      <th {th}>matched patterns</th>
    </tr>
  </thead>
  <tbody>
    {rows_html}
  </tbody>
</table>
'''

st.markdown(table_html, unsafe_allow_html=True)

# ── misconceptions ────────────────────────────────────────────────

st.divider()
st.markdown("**Misconceptions**")

# reset active misconception when expression changes
if st.session_state.get('_last_expr') != expr:
    st.session_state['_last_expr']   = expr
    st.session_state['_active_misc'] = None

applicable = applicable_misconceptions(matches)

if not applicable:
    st.caption("No misconception patterns found in this expression.")
else:
    cols = st.columns(len(applicable))
    for col, m in zip(cols, applicable):
        if col.button(m['name'], key=m['id'], use_container_width=True):
            st.session_state['_active_misc'] = m['id']

active = st.session_state.get('_active_misc')
if active and active not in {m['id'] for m in applicable}:
    active = None
if active:
    m       = next(x for x in MISCONCEPTIONS if x['id'] == active)
    result  = apply_misconception(dag, active)
    orig    = dag_to_str(dag)
    if result is None:
        st.info(f"No '{m['name']}' pattern in this expression.")
    else:
        after = dag_to_str(result)
        st.markdown(
            f'<p style="font-family:monospace;font-size:1rem;margin-top:0.8rem">'
            f'<b>{m["name"]}:</b>&nbsp;&nbsp;'
            f'{colorize(orig)}&nbsp;&nbsp;→&nbsp;&nbsp;{colorize(after)}'
            f'</p>',
            unsafe_allow_html=True,
        )
