"""
distance.py

Compares two learners' trace trees by treating each tree as a set of
directed edges (from_expr, to_expr) and computing Jaccard similarity.

The number of traces per learner doesn't matter — all paths are flattened
into a single edge set before comparison.
"""

from itertools import combinations


def tree_edges(traces):
    """
    Flatten a list of traces into a set of (from, to) transition pairs.
    Each trace is a list of expression strings.
    """
    edges = set()
    for trace in traces:
        for i in range(len(trace) - 1):
            edges.add((trace[i], trace[i + 1]))
    return edges


def jaccard(set_a, set_b):
    """Jaccard similarity between two sets.  Returns 1.0 if both are empty."""
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def correct_answer(expert_traces):
    """Extract the correct final answer from the expert trace set."""
    return expert_traces[0][-1] if expert_traces else None


def correct_rate(learner_traces, answer):
    """Fraction of learner traces that reach the correct answer."""
    if not learner_traces:
        return 0.0
    return sum(1 for t in learner_traces if t[-1] == answer) / len(learner_traces)


def compare(expert_traces, learner_traces):
    """
    Compare an expert trace tree with a learner trace tree.

    Returns a dict:
        edge_jaccard   : float  — similarity of transition sets (0=disjoint, 1=identical)
        correct_rate   : float  — fraction of learner traces that reach the right answer
        shared_edges   : int    — number of transitions both learners agree on
        expert_only    : int    — transitions only the expert makes
        learner_only   : int    — transitions only the learner makes
    """
    e_edges = tree_edges(expert_traces)
    l_edges = tree_edges(learner_traces)
    answer  = correct_answer(expert_traces)

    return dict(
        edge_jaccard  = jaccard(e_edges, l_edges),
        correct_rate  = correct_rate(learner_traces, answer),
        shared_edges  = len(e_edges & l_edges),
        expert_only   = len(e_edges - l_edges),
        learner_only  = len(l_edges - e_edges),
    )


# ── quick test ────────────────────────────────────────────────────

if __name__ == '__main__':
    from parser import build_dag
    from traces import generate_traces
    from learner import MISCONCEPTION_FLIPS

    expressions = [
        '3 + 4 × 5',
        '3 + 4 + 5 × 2',
        '10 × 10 × 9 × 5 - 3',
        '5 + 2 × (4 + 4) - 8',
    ]

    misconceptions = list(MISCONCEPTION_FLIPS.keys())

    for expr in expressions:
        dag           = build_dag(expr)
        expert_traces = generate_traces(dag, [])
        print(f"\n{expr}")
        print(f"  Expert: {len(expert_traces)} trace(s), answer = {correct_answer(expert_traces)}")

        for mid in misconceptions:
            l_traces = generate_traces(dag, [mid])
            result   = compare(expert_traces, l_traces)
            if result['edge_jaccard'] < 1.0:   # only show if learner differs
                print(f"  {mid:30s}  "
                      f"jaccard={result['edge_jaccard']:.2f}  "
                      f"correct={result['correct_rate']:.0%}  "
                      f"shared={result['shared_edges']}  "
                      f"expert_only={result['expert_only']}  "
                      f"learner_only={result['learner_only']}")
