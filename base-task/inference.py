"""
inference.py

Bayesian inference over learner type (one of the 22 misconception profiles:
expert + 6 singles + 15 pairs) given an observed trace, using an
epsilon-greedy transition model:

    π_L(s'|s) = (1-ε) · 1/|ValidActions_L(s)|   if s' ∈ ValidActions_L(s)
              + ε      · 1/|AllActions(s)|        if s' ∈ AllActions(s)

ValidActions_L(s) is exactly traces._next_dags(dag, L) — the states a
learner with policy L can legally reach in one step.

AllActions(s) is every single-step reduction that is arithmetically
well-defined right now (top-level or inside a bracket), regardless of any
policy's notion of correctness — the noise/slip space. Defining it this way
(rather than as the union of the 22 policies) guarantees no observed
transition is ever assigned exactly zero probability for any L, unless it's
a physically impossible move.

DEFAULT_EPSILON is 0.0: every trace in the stimulus pool is generated
deterministically by one of the 22 profiles (traces.generate_traces), so
there is no slip process to model and a nonzero epsilon is a misspecified
likelihood. At epsilon = 0 the transition model is uniform-over-valid and a
step that a profile forbids eliminates that profile outright, which is the
correct inference for noiseless traces. Verified over all 480 pool items:
no item is left with zero surviving profiles. Pass epsilon > 0 explicitly
(as app.py's slider does) to soften this for traces that were NOT produced
by one of the 22 profiles, e.g. real student work.

Likelihood of a trace s0..sT under L (Markov, one factor per step):
    P(s1..sT | L, s0) = ∏ₜ π_L(s_{t+1} | s_t)

Posterior over learner type (uniform prior by default):
    P(L | s0..sT) ∝ P(L) · P(s1..sT | L, s0)

Marginal probability of a single rule R being part of the learner's policy:
    P(R ∈ L | trace) = Σ_L P(L | trace) · [R ∈ L]
"""

import math
from itertools import combinations

from parser import build_dag
from misconceptions import dag_to_str
from valid_actions import fire_operator, fire_inner_op, is_zero_divide
from traces import _next_dags
from learner import MISCONCEPTION_FLIPS

IDS = list(MISCONCEPTION_FLIPS.keys())
ALL_PROFILES = [()] + [(m,) for m in IDS] + list(combinations(IDS, 2))  # 22 profiles

DEFAULT_EPSILON = 0.0


# ── AllActions(s): policy-agnostic noise space ─────────────────────

def _bracket_actions(dag, atom_index):
    inner = dag.atoms[atom_index].inner_dag
    out = []
    for i in range(len(inner.ops)):
        if (inner.atoms[i].is_number() and inner.atoms[i + 1].is_number()
                and not is_zero_divide(inner, i)):
            out.append(fire_inner_op(dag, atom_index, i))
    return out


def all_actions(dag):
    """
    Every single-step reduction that is arithmetically well-defined right
    now — top-level or inside a bracket — regardless of any policy. This is
    the epsilon/noise action space: deliberately policy-agnostic so no
    observed move is ever assigned exactly zero probability.
    """
    nexts = []
    for i in range(len(dag.ops)):
        if (dag.atoms[i].is_number() and dag.atoms[i + 1].is_number()
                and not is_zero_divide(dag, i)):
            nexts.append(fire_operator(dag, i))
    for atom_index, atom in enumerate(dag.atoms):
        if atom.is_bracket() and atom.inner_dag:
            nexts.extend(_bracket_actions(dag, atom_index))
    return nexts


# ── transition model ────────────────────────────────────────────────

def transition_prob(dag, next_str, misconceptions, epsilon=DEFAULT_EPSILON):
    """π_L(s'|s) for a single step; s' given as its canonical expression string."""
    valid_strs = [dag_to_str(d) for d in _next_dags(dag, misconceptions)]
    every_strs = [dag_to_str(d) for d in all_actions(dag)]

    p = 0.0
    if valid_strs and next_str in valid_strs:
        p += (1 - epsilon) / len(valid_strs)
    if every_strs and next_str in every_strs:
        p += epsilon / len(every_strs)
    return p


def trace_log_likelihood(trace, misconceptions, epsilon=DEFAULT_EPSILON):
    """
    Sum of log π_L(s_{t+1}|s_t) over an observed trace (list of expression
    strings, s0..sT). Log-space avoids underflow over many steps; returns
    -inf if any single transition has probability 0 under this L.
    """
    log_p = 0.0
    for t in range(len(trace) - 1):
        dag = build_dag(trace[t])
        p = transition_prob(dag, trace[t + 1], misconceptions, epsilon)
        if p <= 0:
            return float('-inf')
        log_p += math.log(p)
    return log_p


# ── posterior over learner type ─────────────────────────────────────

def posterior_over_profiles(trace, epsilon=DEFAULT_EPSILON, profiles=None, priors=None):
    """
    P(L | trace) for every learner profile, normalized to sum to 1.
    `priors` defaults to uniform over `profiles` (all 22 by default).
    Returns a dict {profile_tuple: posterior_probability}.
    """
    profiles = profiles if profiles is not None else ALL_PROFILES
    if priors is None:
        priors = {L: 1.0 / len(profiles) for L in profiles}

    log_post = {}
    for L in profiles:
        ll = trace_log_likelihood(trace, L, epsilon)
        if ll == float('-inf') or priors[L] <= 0:
            log_post[L] = float('-inf')
        else:
            log_post[L] = math.log(priors[L]) + ll

    finite = [v for v in log_post.values() if v != float('-inf')]
    if not finite:
        raise ValueError('No profile assigns nonzero posterior mass to this trace')
    max_lp = max(finite)

    unnorm = {L: (math.exp(v - max_lp) if v != float('-inf') else 0.0) for L, v in log_post.items()}
    total = sum(unnorm.values())
    return {L: v / total for L, v in unnorm.items()}


def marginal_rule_probability(posterior, rule_id):
    """P(rule_id is part of the learner's policy | trace)."""
    return sum(p for L, p in posterior.items() if rule_id in L)


def most_likely_profile(posterior):
    """(profile, probability) for the MAP estimate."""
    return max(posterior.items(), key=lambda kv: kv[1])


def format_profile(profile):
    return 'expert' if not profile else ' + '.join(profile)


# ── quick test ────────────────────────────────────────────────────

if __name__ == '__main__':
    from generator import generate_expression

    for mid in [(), ('add_before_mul',), ('add_before_mul', 'same_priority_rtl')]:
        expr = generate_expression(n_ops=4, bracket_prob=0.3)
        dag = build_dag(expr)
        from traces import generate_traces
        candidates = generate_traces(dag, list(mid))
        trace = min(candidates, key=lambda t: (len(t), t))

        print(f"\nTrue profile: {format_profile(mid)}")
        print(f"Trace: {'  ->  '.join(trace)}")

        post = posterior_over_profiles(trace)
        ranked = sorted(post.items(), key=lambda kv: -kv[1])[:5]
        for L, p in ranked:
            print(f"  {format_profile(L):40s} {p:.4f}")
