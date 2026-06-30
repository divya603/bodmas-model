"""
learner.py  —  Misconception flip rules (bidirectional)

Each misconception swaps validity in specific windows:
  to_true  : entries that flip F → T  (learner thinks this wrong move is valid)
  to_false : entries that flip T → F  (learner thinks this correct move is invalid)

A learner = a list of up to 2 misconception IDs.
The expert learner = [] — no flips, pure default truth table.
"""

_ADD = {'+', '-'}
_MUL = {'×', '÷'}

MISCONCEPTION_FLIPS = {

    # ── cross-precedence: swap validity between the two ops ────────
    # In a + b × c: learner fires + (F→T) and considers × invalid (T→F)
    # In a × b + c: learner fires + (F→T) and considers × invalid (T→F)

    'add_before_mul': {
        'to_true':  {(2, '+', '×', 'left'),  (2, '×', '+', 'right')},
        'to_false': {(2, '+', '×', 'right'), (2, '×', '+', 'left')},
    },

    'add_before_div': {
        'to_true':  {(2, '+', '÷', 'left'),  (2, '÷', '+', 'right')},
        'to_false': {(2, '+', '÷', 'right'), (2, '÷', '+', 'left')},
    },

    'sub_before_mul': {
        'to_true':  {(2, '-', '×', 'left'),  (2, '×', '-', 'right')},
        'to_false': {(2, '-', '×', 'right'), (2, '×', '-', 'left')},
    },

    'sub_before_div': {
        'to_true':  {(2, '-', '÷', 'left'),  (2, '÷', '-', 'right')},
        'to_false': {(2, '-', '÷', 'right'), (2, '÷', '-', 'left')},
    },

    # ── same-priority RTL: swap left (T→F) and right (F→T) ────────
    # Excludes purely associative (a+b+c, a×b×c) — both ops stay valid there.

    'same_priority_rtl': {
        'to_true': {
            (2, '+', '-', 'right'),
            (2, '-', '+', 'right'),
            (2, '-', '-', 'right'),
            (2, '×', '÷', 'right'),
            (2, '÷', '×', 'right'),
            (2, '÷', '÷', 'right'),
        },
        'to_false': {
            (2, '+', '-', 'left'),
            (2, '-', '+', 'left'),
            (2, '-', '-', 'left'),
            (2, '×', '÷', 'left'),
            (2, '÷', '×', 'left'),
            (2, '÷', '÷', 'left'),
        },
    },

    # ── outside bracket first: only F→T, no correct op to invalidate ─
    # The right op in Table 3 and left op in Table 4 are always F anyway
    # (Y not resolved), so there is nothing correct to flip down.

    'outside_bracket_first': {
        'to_true': {
            (3, '+', '×', 'left'), (3, '+', '÷', 'left'),
            (3, '-', '×', 'left'), (3, '-', '÷', 'left'),
            (4, '×', '+', 'right'), (4, '×', '-', 'right'),
            (4, '÷', '+', 'right'), (4, '÷', '-', 'right'),
        },
        'to_false': set(),
    },
}


# ── lookup ────────────────────────────────────────────────────────

def _key(window):
    parts = window['pattern'].split()
    return (window['table'], parts[1], parts[3], window['role'])


def is_correct_for_learner(window, misconceptions):
    """
    Validity of firing this op for a learner with the given misconceptions.

    Priority:
      1. F→T flip from any misconception  →  True
      2. T→F flip from any misconception  →  False
      3. Default truth table              →  _is_correct_in_window(window)
    """
    from valid_actions import _is_correct_in_window
    k = _key(window)

    for mid in misconceptions:
        if k in MISCONCEPTION_FLIPS.get(mid, {}).get('to_true', set()):
            return True

    for mid in misconceptions:
        if k in MISCONCEPTION_FLIPS.get(mid, {}).get('to_false', set()):
            return False

    return _is_correct_in_window(window)


def applicable_misconceptions(dag):
    """
    Return the subset of misconception IDs that are relevant to this dag —
    i.e., at least one of their flip keys appears in the expression's windows.
    """
    from pattern_matcher import match_patterns
    window_keys = set()
    for m in match_patterns(dag):
        for w in m.get('windows', []):
            window_keys.add(_key(w))

    result = []
    for mid, flips in MISCONCEPTION_FLIPS.items():
        all_keys = flips.get('to_true', set()) | flips.get('to_false', set())
        if all_keys & window_keys:
            result.append(mid)
    return result


def flipped_cells(expert_actions, learner_actions):
    """
    Compare expert and learner truth tables.
    Returns:
      up   : set of (op_index, w_idx) flipped F→T  (wrong, shown orange)
      down : set of (op_index, w_idx) flipped T→F  (suppressed, shown purple)
    """
    expert_map = {
        (a['op_index'], w_idx): correct
        for a in expert_actions
        for w_idx, _, correct in a['truth']
    }
    up, down = set(), set()
    for a in learner_actions:
        for w_idx, _, correct in a['truth']:
            cell = (a['op_index'], w_idx)
            was = expert_map.get(cell)
            if was is None:
                continue
            if correct and not was:
                up.add(cell)
            elif not correct and was:
                down.add(cell)
    return up, down
