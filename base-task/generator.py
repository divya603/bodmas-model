"""
generator.py

Randomly generates arithmetic order-of-operations expressions.
No exponents. Guarantees at least one × or ÷ so precedence matters.
"""

import random

_ALL_OPS = ['+', '-', '×', '÷']


def generate_expression(
    n_ops=4,
    num_range=(1, 12),
    ops=None,
    bracket_prob=0.4,
    seed=None,
):
    """
    Generate a random arithmetic expression string.

    Parameters
    ----------
    n_ops       : number of operators (expression has n_ops+1 numbers)
    num_range   : (low, high) inclusive range for random integers
    ops         : list of allowed operators; defaults to all four
    bracket_prob: probability that one sub-expression is wrapped in brackets
    seed        : optional random seed for reproducibility
    """
    if ops is None:
        ops = _ALL_OPS
    if seed is not None:
        random.seed(seed)

    mul_ops = [op for op in ops if op in ('×', '÷')]

    while True:
        numbers   = [random.randint(*num_range) for _ in range(n_ops + 1)]
        operators = [random.choice(ops) for _ in range(n_ops)]

        # avoid division by zero
        for i, op in enumerate(operators):
            if op == '÷' and numbers[i + 1] == 0:
                numbers[i + 1] = random.randint(1, num_range[1])

        # retry until at least one × or ÷ appears (so precedence is exercised)
        if mul_ops and not any(op in ('×', '÷') for op in operators):
            continue

        break

    # build alternating parts list: [n0, op0, n1, op1, ..., nN]
    parts = []
    for i in range(n_ops):
        parts.append(str(numbers[i]))
        parts.append(operators[i])
    parts.append(str(numbers[n_ops]))

    # optionally wrap a contiguous sub-expression in brackets
    if random.random() < bracket_prob and n_ops >= 2:
        bracket_size = random.randint(1, min(2, n_ops - 1))
        op_start     = random.randint(0, n_ops - bracket_size)
        p_start      = 2 * op_start
        p_end        = 2 * (op_start + bracket_size) + 1   # inclusive last index
        inner        = ' '.join(parts[p_start:p_end])
        parts        = parts[:p_start] + [f'({inner})'] + parts[p_end:]

    return ' '.join(parts)
