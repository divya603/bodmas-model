"""
Learner System — built on formal policy constraints.

A Learner is defined purely by a conjunction of PolicyConstraints.
No precedence map — the policies themselves constrain P directly.

    Learner = φ₁(P) ∧ φ₂(P) ∧ ... ∧ φₙ(P)

The learner picks action(s) with highest P value.
Multiple actions returned → tie → tree branches.
"""

from typing import List, Dict
from formal_policy import (
    Action, PolicyConstraint, Constraint,
    Resolver, get_policy, POLICY_REGISTRY
)


# =============================================================================
# LEARNER
# =============================================================================

class Learner:
    """
    A learner defined by a conjunction of formal policy constraints.

    Policies constrain P: A → ℕ
    Learner picks action(s) with highest P.
    Ties → multiple actions returned (tree branches).
    """

    def __init__(self, name: str,
                 policy_names: List[str],
                 description: str = ""):
        self.name = name
        self.description = description
        self.policies: List[PolicyConstraint] = [get_policy(n) for n in policy_names]
        self.resolver = Resolver()

    def get_constraints(self, actions: List[Action]) -> List[Constraint]:
        """
        Collect all constraints from all policies (conjunction).
        This is the full set of ordering constraints on P.
        """
        all_constraints = []
        for policy in self.policies:
            all_constraints.extend(policy.generate_constraints(actions))
        return all_constraints

    def choose(self, actions: List[Action]) -> List[Action]:
        """
        Choose the action(s) with highest priority under this learner's policies.

        Returns:
            Single action in list  → deterministic, tree continues on one path
            Multiple actions       → tie, tree branches into all of them
            Empty list             → no actions available, expression fully evaluated
        """
        if not actions:
            return []

        constraints = self.get_constraints(actions)
        return self.resolver.choose(actions, constraints)

    def priority_map(self, actions: List[Action]) -> Dict[Action, int]:
        """
        Return the full P: A → ℕ assignment for inspection/debugging.
        Higher value = chosen first.
        """
        constraints = self.get_constraints(actions)
        return self.resolver.resolve(actions, constraints)

    def explain(self, actions: List[Action]) -> str:
        """
        Print a readable explanation of how P is assigned for these actions.
        Shows which policies generated which constraints.
        """
        lines = [f"\nLearner: {self.name}"]
        lines.append(f"Description: {self.description}")
        lines.append(f"\nPolicies:")
        for p in self.policies:
            lines.append(f"  [{p.name}]: {p.description}")

        lines.append(f"\nGenerated constraints on P:")
        for policy in self.policies:
            cs = policy.generate_constraints(actions)
            if cs:
                lines.append(f"  From [{policy.name}]:")
                for c in cs:
                    lines.append(f"    P({c.higher}) > P({c.lower})")
            else:
                lines.append(f"  From [{policy.name}]: (no constraints)")

        priority = self.priority_map(actions)
        lines.append(f"\nResolved P assignment:")
        for action, p in sorted(priority.items(), key=lambda x: -x[1]):
            lines.append(f"  P({action}) = {p}")

        chosen = self.choose(actions)
        lines.append(f"\nChosen action(s): {chosen}")
        if len(chosen) > 1:
            lines.append("  → TIE: tree branches into all of the above")
        else:
            lines.append("  → Deterministic choice")

        return "\n".join(lines)

    def __repr__(self):
        return f"Learner({self.name!r}, policies={[p.name for p in self.policies]})"


# =============================================================================
# PRESET LEARNER PROFILES
# =============================================================================

LEARNER_PROFILES: Dict[str, Dict] = {

    "expert": {
        "policies": [
            "brackets_first",
            "exponentiation_first",
            "multiplication_before_addition",
            "leftmost_first",       # tiebreaker among same-operator actions
        ],
        "description": "Correct BODMAS: brackets → ^ → */ → +- → left to right"
    },

    "addition_first": {
        "policies": [
            "addition_before_multiplication",
            "leftmost_first",       # tiebreaker among same-operator actions
        ],
        "description": "Believes addition comes before multiplication (wrong)"
    },

    "multiplication_first": {
        "policies": [
            "multiplication_before_addition",
            "leftmost_first",
        ],
        "description": "Knows * before + but ignores brackets and ^"
    },

    "left_to_right": {
        "policies": [
            "left_to_right_strict", # pure positional — ignores operator type
        ],
        "description": "Evaluates strictly left to right, ignores operator type"
    },

    "right_to_left": {
        "policies": [
            "right_to_left_strict", # pure positional — ignores operator type
        ],
        "description": "Evaluates strictly right to left (wrong direction)"
    },

    "bracket_ignorer": {
        "policies": [
            "brackets_ignored",
            "leftmost_among_non_brackets",
        ],
        "description": "Ignores brackets, goes left to right among non-bracket actions"
    },

    "novice": {
        "policies": [
            "all_equal",
        ],
        "description": "No knowledge — all actions equally valid, full branching"
    },
}


def create_learner(profile_name: str) -> Learner:
    """Create a learner from a preset profile."""
    if profile_name not in LEARNER_PROFILES:
        raise ValueError(
            f"Unknown profile: {profile_name!r}. "
            f"Available: {list(LEARNER_PROFILES)}"
        )
    profile = LEARNER_PROFILES[profile_name]
    return Learner(
        name=profile_name,
        policy_names=profile["policies"],
        description=profile["description"]
    )


def create_custom_learner(name: str,
                           policy_names: List[str],
                           description: str = "") -> Learner:
    """Create a custom learner with any combination of policies."""
    return Learner(name=name, policy_names=policy_names, description=description)
