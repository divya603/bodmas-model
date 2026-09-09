"""
hidden.py

Inference when one intermediate state of the trace is HIDDEN.

The observed likelihood factorises one term per step:

    P(s1..sT | L, s0) = prod_t pi_L(s_{t+1} | s_t)

Hiding state s_k means the participant (and the observer) see
s0..s_{k-1}, s_{k+1}..sT, with the two factors that touch s_k collapsed into a
marginal over every value s_k could have taken:

    P(observed | L, s0) = [prod_{t != k-1, k} pi_L(s_{t+1} | s_t)]
                          * sum_{s_k} pi_L(s_k | s_{k-1}) pi_L(s_{k+1} | s_k)

At epsilon 0 the inner sum runs over the states a learner with policy L can
legally reach from s_{k-1}, keeping only those from which s_{k+1} is also
legally reachable. A profile is eliminated only if NO bridging state exists,
which is strictly weaker than the fully observed case, so hiding a step can
only ever flatten the posterior, never sharpen it.
"""

import math

from parser import build_dag
from misconceptions import dag_to_str
from traces import _next_dags
from inference import transition_prob, DEFAULT_EPSILON, ALL_PROFILES


def two_step_prob(prev, nxt, misconceptions, epsilon=DEFAULT_EPSILON):
    """sum over the hidden state of pi(hidden|prev) * pi(next|hidden)."""
    total = 0.0
    for d in _next_dags(build_dag(prev), misconceptions):
        mid = dag_to_str(d)
        p1 = transition_prob(build_dag(prev), mid, misconceptions, epsilon)
        if p1 <= 0:
            continue
        p2 = transition_prob(build_dag(mid), nxt, misconceptions, epsilon)
        if p2 > 0:
            total += p1 * p2
    return total


def hidden_log_likelihood(trace, hide_index, misconceptions, epsilon=DEFAULT_EPSILON):
    """log P(observed | L, s0) with trace[hide_index] hidden."""
    T = len(trace) - 1
    if not 1 <= hide_index <= T - 1:
        raise ValueError(f"cannot hide index {hide_index} of a {len(trace)}-line trace; "
                         f"s0 and the final answer stay visible")
    log_p = 0.0
    for t in range(T):
        if t == hide_index - 1:                      # the collapsed pair
            p = two_step_prob(trace[t], trace[t + 2], misconceptions, epsilon)
        elif t == hide_index:                        # already consumed above
            continue
        else:
            p = transition_prob(build_dag(trace[t]), trace[t + 1], misconceptions, epsilon)
        if p <= 0:
            return float('-inf')
        log_p += math.log(p)
    return log_p


def hidden_posterior(trace, hide_index, epsilon=DEFAULT_EPSILON, profiles=None, priors=None):
    """P(L | observed trace with trace[hide_index] hidden)."""
    profiles = profiles if profiles is not None else ALL_PROFILES
    if priors is None:
        priors = {L: 1.0 / len(profiles) for L in profiles}

    log_post = {}
    for L in profiles:
        ll = hidden_log_likelihood(trace, hide_index, L, epsilon)
        log_post[L] = (float('-inf') if ll == float('-inf') or priors[L] <= 0
                       else math.log(priors[L]) + ll)

    finite = [v for v in log_post.values() if v != float('-inf')]
    if not finite:
        raise ValueError('No profile assigns nonzero mass to this observation')
    mx = max(finite)
    unnorm = {L: (math.exp(v - mx) if v != float('-inf') else 0.0) for L, v in log_post.items()}
    tot = sum(unnorm.values())
    return {L: v / tot for L, v in unnorm.items()}


def visible_trace(trace, hide_index):
    """The lines a participant would actually see."""
    return [s for i, s in enumerate(trace) if i != hide_index]


# ── general case: any set of hidden lines ──────────────────────────

def gap_prob(start, end, n_steps, misconceptions, epsilon=DEFAULT_EPSILON):
    """P(reach `end` from `start` in exactly n_steps) under pi_L, summing over
    every intermediate path. n_steps == 1 is the ordinary one-step case."""
    dist = {start: 1.0}
    for _ in range(n_steps - 1):
        nxt = {}
        for s, w in dist.items():
            for d in _next_dags(build_dag(s), misconceptions):
                m = dag_to_str(d)
                p = transition_prob(build_dag(s), m, misconceptions, epsilon)
                if p > 0:
                    nxt[m] = nxt.get(m, 0.0) + w * p
        dist = nxt
        if not dist:
            return 0.0
    return sum(w * transition_prob(build_dag(s), end, misconceptions, epsilon)
               for s, w in dist.items())


def multi_hidden_log_likelihood(trace, hide_set, misconceptions, epsilon=DEFAULT_EPSILON):
    """log P(observed | L, s0) with every line in `hide_set` hidden."""
    T = len(trace) - 1
    hide = set(hide_set)
    if not hide <= set(range(1, T)):
        raise ValueError("s0 and the final answer must stay visible")
    visible = [i for i in range(len(trace)) if i not in hide]
    log_p = 0.0
    for a, b in zip(visible, visible[1:]):
        p = gap_prob(trace[a], trace[b], b - a, misconceptions, epsilon)
        if p <= 0:
            return float('-inf')
        log_p += math.log(p)
    return log_p


def multi_hidden_posterior(trace, hide_set, epsilon=DEFAULT_EPSILON, profiles=None, priors=None):
    profiles = profiles if profiles is not None else ALL_PROFILES
    if priors is None:
        priors = {L: 1.0 / len(profiles) for L in profiles}
    log_post = {}
    for L in profiles:
        ll = multi_hidden_log_likelihood(trace, hide_set, L, epsilon)
        log_post[L] = (float('-inf') if ll == float('-inf') or priors[L] <= 0
                       else math.log(priors[L]) + ll)
    finite = [v for v in log_post.values() if v != float('-inf')]
    if not finite:
        raise ValueError('No profile assigns nonzero mass to this observation')
    mx = max(finite)
    unnorm = {L: (math.exp(v - mx) if v != float('-inf') else 0.0) for L, v in log_post.items()}
    tot = sum(unnorm.values())
    return {L: v / tot for L, v in unnorm.items()}
