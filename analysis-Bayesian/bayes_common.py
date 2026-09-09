#!/usr/bin/env python3
"""
bayes_common.py

Shared loading + styling for the versioned Bayesian ideal-observer figures
(v3 = the position pool, v4 = the position x named pool). Pass the pool you
want to load_rows(); the v3 default is kept so existing callers are unchanged.

The v3 figures read `base-task/bayes_per_item_v3.json` (written by
base-task/bayes_v3.py) rather than recomputing posteriors from the pool. That
file IS the recorded Bayes arm of the v3 experiment, so reading it guarantees
the figures and the recorded observer can never drift apart. Regenerate it with
`cd base-task && python3 bayes_v3.py` after any pool or model change.

v3 differences the figures have to respect:
  - Only categories A and B exist. There are no 2-misconception items.
  - Every item carries `error_position` (1 or 3), the new manipulated factor,
    and `pair_id` linking the two positions of one expression.
  - Category A is a POINT MASS: all 144 items score exactly 1.000. Anything
    that tries to show its shape is drawing noise that is not there.
  - Category B marginals take 11 distinct values in [0, 0.333]. They are
    effectively discrete, so these figures use exact-value stems rather than
    the KDEs the v2 scripts used on continuous marginals.
"""

import json
import os

from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_TASK = os.path.join(os.path.dirname(HERE), 'base-task')
ROWS_PATH    = os.path.join(BASE_TASK, 'bayes_per_item_v3.json')
ROWS_PATH_V4 = os.path.join(BASE_TASK, 'bayes_per_item_v4.json')

IDS = ['add_before_mul', 'add_before_div', 'sub_before_mul', 'sub_before_div',
       'same_priority_rtl', 'outside_bracket_first']
SHORT = {
    'add_before_mul': 'add<×', 'add_before_div': 'add<÷',
    'sub_before_mul': 'sub<×', 'sub_before_div': 'sub<÷',
    'same_priority_rtl': 'RTL', 'outside_bracket_first': 'outside()',
}
POSITIONS = [1, 3]

BLUE, ORANGE = '#2a78d6', '#eb6834'
# position is the new v3 factor; one hue per position, used in every figure
POS_COLOR = {1: '#2a78d6', 3: '#eb6834'}
POS_MARKER = {1: 'o', 3: '^'}
POS_LABEL = {1: 'error at step 1', 3: 'error at step 3'}

CMAP = LinearSegmentedColormap.from_list('divmarg',
                                         ['#e34948', '#f0efec', '#008300'])
NORM = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
DARK_AT = 0.28

REFUTED_CUT = 0.15


def load_rows(path=ROWS_PATH, expect=432):
    """Load recorded observer responses, with a pointed error if the file has
    not been generated yet. `expect` is the item count for the pool being read
    (432 for v3, 240 for v4); it only drives a warning."""
    if not os.path.exists(path):
        raise SystemExit(
            f"missing {path}\n"
            "Generate it first:  cd base-task && python3 bayes_v3.py")
    rows = json.load(open(path, encoding='utf-8'))
    if len(rows) != expect:
        print(f"warning: expected {expect} items in {os.path.basename(path)}, "
              f"found {len(rows)}")
    return rows


def split_ab(rows):
    """(category A rows, category B rows)."""
    return ([r for r in rows if r['category'] == 'A'],
            [r for r in rows if r['category'] == 'B'])
