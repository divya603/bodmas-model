"""
Formal Policy Constraint System for BODMAS Learner Modelling

Core idea:
    P: A → ℕ  (priority function over actions)

A Policy is a logical constraint on P, expressed as:
    ∀a₁, a₂ ∈ A . condition(a₁, a₂) → P(a₁) > P(a₂)

A Learner is a conjunction of policies:
    φ₁(P) ∧ φ₂(P) ∧ ... ∧ φₙ(P)

The learner picks the action(s) with highest P value.
Ties → multiple valid actions (tree branches).
"""

from dataclasses import dataclass
from typing import List, Tuple, Set, Dict, Optional
from abc import ABC, abstractmethod
from collections import defaultdict


# =============================================================================
# ACTION — what a learner can do at each step
# =============================================================================

@dataclass(frozen=True)
class Action:
    """
    An action the learner can take on the expression.

    operator:   the operation type ('+', '-', '*', '/', '^', 'brackets')
    position:   index in the expression (left = lower index)
    depth:      nesting depth (0 = outermost)
    """
    operator: str
    position: int
    depth: int = 0

    def __repr__(self):
        return f"Action({self.operator!r}, pos={self.position}, depth={self.depth})"


# Predicate helpers — these are the is_add, is_mult etc. from the formal language
def is_add(a: Action) -> bool:
    return a.operator == '+'

def is_sub(a: Action) -> bool:
    return a.operator == '-'

def is_mult(a: Action) -> bool:
    return a.operator == '*'

def is_div(a: Action) -> bool:
    return a.operator == '/'

def is_exp(a: Action) -> bool:
    return a.operator == '^'

def is_bracket(a: Action) -> bool:
    return a.operator == 'brackets'

def is_additive(a: Action) -> bool:
    return a.operator in ('+', '-')

def is_multiplicative(a: Action) -> bool:
    return a.operator in ('*', '/')


# =============================================================================
# CONSTRAINT — a single ordering constraint between two actions
# =============================================================================

@dataclass(frozen=True)
class Constraint:
    """
    Represents: P(higher) > P(lower)
    i.e. 'higher' should be chosen before 'lower'
    """
    higher: Action   # P(higher) > P(lower)
    lower: Action

    def __repr__(self):
        return f"P({self.higher}) > P({self.lower})"


# =============================================================================
# POLICY CONSTRAINT — abstract base class
# =============================================================================

class PolicyConstraint(ABC):
    """
    A policy is a logical rule that generates ordering constraints.

    Formally:
        ∀a₁, a₂ ∈ A . condition(a₁, a₂) → P(a₁) > P(a₂)

    Given a list of available actions, generate_constraints returns
    all concrete (a1, a2) pairs where this rule applies.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def generate_constraints(self, actions: List[Action]) -> List[Constraint]:
        """
        Generate all concrete ordering constraints for this policy
        given the available actions.

        Returns list of Constraint(higher, lower) meaning P(higher) > P(lower)
        """
        pass

    def __repr__(self):
        return f"Policy({self.name})"


# =============================================================================
# CONCRETE POLICIES
# =============================================================================

class AdditionBeforeMultiplication(PolicyConstraint):
    """
    Formal rule:
        ∀a₁, a₂ . is_add(a₁) ∧ is_mult(a₂) → P(a₁) > P(a₂)
    """

    @property
    def name(self) -> str:
        return "addition_before_multiplication"

    @property
    def description(self) -> str:
        return "∀a₁,a₂. is_add(a₁) ∧ is_mult(a₂) → P(a₁) > P(a₂)"

    def generate_constraints(self, actions: List[Action]) -> List[Constraint]:
        constraints = []
        for a1 in actions:
            for a2 in actions:
                if is_additive(a1) and is_multiplicative(a2):
                    constraints.append(Constraint(higher=a1, lower=a2))
        return constraints


class MultiplicationBeforeAddition(PolicyConstraint):
    """
    Formal rule:
        ∀a₁, a₂ . is_mult(a₁) ∧ is_add(a₂) → P(a₁) > P(a₂)

    This is the correct BODMAS rule.
    """

    @property
    def name(self) -> str:
        return "multiplication_before_addition"

    @property
    def description(self) -> str:
        return "∀a₁,a₂. is_mult(a₁) ∧ is_add(a₂) → P(a₁) > P(a₂)"

    def generate_constraints(self, actions: List[Action]) -> List[Constraint]:
        constraints = []
        for a1 in actions:
            for a2 in actions:
                if is_multiplicative(a1) and is_additive(a2):
                    constraints.append(Constraint(higher=a1, lower=a2))
        return constraints


class ExponentiationFirst(PolicyConstraint):
    """
    Formal rule:
        ∀a₁, a₂ . is_exp(a₁) ∧ ¬is_exp(a₂) ∧ ¬is_bracket(a₂) → P(a₁) > P(a₂)
    """

    @property
    def name(self) -> str:
        return "exponentiation_first"

    @property
    def description(self) -> str:
        return "∀a₁,a₂. is_exp(a₁) ∧ ¬is_exp(a₂) ∧ ¬is_bracket(a₂) → P(a₁) > P(a₂)"

    def generate_constraints(self, actions: List[Action]) -> List[Constraint]:
        constraints = []
        for a1 in actions:
            for a2 in actions:
                if is_exp(a1) and not is_exp(a2) and not is_bracket(a2):
                    constraints.append(Constraint(higher=a1, lower=a2))
        return constraints


class LeftmostFirst(PolicyConstraint):
    """
    Tiebreaker: among same-operator actions, prefer the leftmost.

    Formal rule:
        ∀a₁, a₂ . op(a₁) = op(a₂) ∧ pos(a₁) < pos(a₂) → P(a₁) > P(a₂)

    This is a TIEBREAKER — only orders actions of the same operator type.
    Does not conflict with precedence rules (mult vs add).
    Use LeftToRightStrict for pure positional ordering ignoring operator type.
    """

    @property
    def name(self) -> str:
        return "leftmost_first"

    @property
    def description(self) -> str:
        return "∀a₁,a₂. op(a₁)=op(a₂) ∧ pos(a₁) < pos(a₂) → P(a₁) > P(a₂)"

    def generate_constraints(self, actions: List[Action]) -> List[Constraint]:
        constraints = []
        for a1 in actions:
            for a2 in actions:
                # Only tiebreak between same operator type
                if a1.operator == a2.operator and a1.position < a2.position:
                    constraints.append(Constraint(higher=a1, lower=a2))
        return constraints


class RightmostFirst(PolicyConstraint):
    """
    Tiebreaker: among same-operator actions, prefer the rightmost.

    Formal rule:
        ∀a₁, a₂ . op(a₁) = op(a₂) ∧ pos(a₁) > pos(a₂) → P(a₁) > P(a₂)
    """

    @property
    def name(self) -> str:
        return "rightmost_first"

    @property
    def description(self) -> str:
        return "∀a₁,a₂. op(a₁)=op(a₂) ∧ pos(a₁) > pos(a₂) → P(a₁) > P(a₂)"

    def generate_constraints(self, actions: List[Action]) -> List[Constraint]:
        constraints = []
        for a1 in actions:
            for a2 in actions:
                if a1.operator == a2.operator and a1.position > a2.position:
                    constraints.append(Constraint(higher=a1, lower=a2))
        return constraints


class LeftToRightStrict(PolicyConstraint):
    """
    Pure positional ordering — ignores operator type entirely.

    Formal rule:
        ∀a₁, a₂ . pos(a₁) < pos(a₂) → P(a₁) > P(a₂)

    This is the LEFT-TO-RIGHT ONLY learner policy.
    Will conflict with precedence rules if combined with them.
    Use leftmost_first (tiebreaker) if you want position as secondary sort.
    """

    @property
    def name(self) -> str:
        return "left_to_right_strict"

    @property
    def description(self) -> str:
        return "∀a₁,a₂. pos(a₁) < pos(a₂) → P(a₁) > P(a₂)  [all operators]"

    def generate_constraints(self, actions: List[Action]) -> List[Constraint]:
        constraints = []
        for a1 in actions:
            for a2 in actions:
                if a1.position < a2.position:
                    constraints.append(Constraint(higher=a1, lower=a2))
        return constraints


class RightToLeftStrict(PolicyConstraint):
    """
    Pure positional ordering right to left — ignores operator type.

    Formal rule:
        ∀a₁, a₂ . pos(a₁) > pos(a₂) → P(a₁) > P(a₂)
    """

    @property
    def name(self) -> str:
        return "right_to_left_strict"

    @property
    def description(self) -> str:
        return "∀a₁,a₂. pos(a₁) > pos(a₂) → P(a₁) > P(a₂)  [all operators]"

    def generate_constraints(self, actions: List[Action]) -> List[Constraint]:
        constraints = []
        for a1 in actions:
            for a2 in actions:
                if a1.position > a2.position:
                    constraints.append(Constraint(higher=a1, lower=a2))
        return constraints


class BracketsFirst(PolicyConstraint):
    """
    Formal rule:
        ∀a₁, a₂ . is_bracket(a₁) ∧ ¬is_bracket(a₂) → P(a₁) > P(a₂)

    Always evaluate inside brackets before anything else.
    """

    @property
    def name(self) -> str:
        return "brackets_first"

    @property
    def description(self) -> str:
        return "∀a₁,a₂. is_bracket(a₁) ∧ ¬is_bracket(a₂) → P(a₁) > P(a₂)"

    def generate_constraints(self, actions: List[Action]) -> List[Constraint]:
        constraints = []
        for a1 in actions:
            for a2 in actions:
                if is_bracket(a1) and not is_bracket(a2):
                    constraints.append(Constraint(higher=a1, lower=a2))
        return constraints


class BracketsIgnored(PolicyConstraint):
    """
    Formal rule:
        ∀a₁, a₂ . ¬is_bracket(a₁) ∧ is_bracket(a₂) → P(a₁) > P(a₂)

    Always prefer non-bracket actions over bracket actions.
    (Brackets get lowest priority — effectively ignored)
    """

    @property
    def name(self) -> str:
        return "brackets_ignored"

    @property
    def description(self) -> str:
        return "∀a₁,a₂. ¬is_bracket(a₁) ∧ is_bracket(a₂) → P(a₁) > P(a₂)"

    def generate_constraints(self, actions: List[Action]) -> List[Constraint]:
        constraints = []
        for a1 in actions:
            for a2 in actions:
                if not is_bracket(a1) and is_bracket(a2):
                    constraints.append(Constraint(higher=a1, lower=a2))
        return constraints


class LeftmostAmongNonBrackets(PolicyConstraint):
    """
    Left to right ordering but only among non-bracket actions.

    Formal rule:
        ∀a₁, a₂ . ¬is_bracket(a₁) ∧ ¬is_bracket(a₂) ∧ pos(a₁) < pos(a₂)
                   → P(a₁) > P(a₂)

    Used by bracket_ignorer — goes left to right but only among
    non-bracket actions. Avoids conflict with brackets_ignored.
    """

    @property
    def name(self) -> str:
        return "leftmost_among_non_brackets"

    @property
    def description(self) -> str:
        return "∀a₁,a₂. ¬bracket(a₁) ∧ ¬bracket(a₂) ∧ pos(a₁)<pos(a₂) → P(a₁)>P(a₂)"

    def generate_constraints(self, actions: List[Action]) -> List[Constraint]:
        constraints = []
        for a1 in actions:
            for a2 in actions:
                if (not is_bracket(a1) and not is_bracket(a2)
                        and a1.position < a2.position):
                    constraints.append(Constraint(higher=a1, lower=a2))
        return constraints


class AllEqual(PolicyConstraint):
    """
    Formal rule:
        ∀a₁, a₂ . True   (no constraints — all actions equally valid)

    This is the novice policy — no ordering imposed at all.
    Generates no constraints → all actions tie → full branching.
    """

    @property
    def name(self) -> str:
        return "all_equal"

    @property
    def description(self) -> str:
        return "No constraints — all actions have equal priority (novice)"

    def generate_constraints(self, actions: List[Action]) -> List[Constraint]:
        return []
    """
    Formal rule:
        ∀a₁, a₂ . True   (no constraints — all actions equally valid)

    This is the novice policy — no ordering imposed at all.
    Generates no constraints → all actions tie → full branching.
    """

    @property
    def name(self) -> str:
        return "all_equal"

    @property
    def description(self) -> str:
        return "No constraints — all actions have equal priority (novice)"

    def generate_constraints(self, actions: List[Action]) -> List[Constraint]:
        return []   # No ordering constraints → everything ties


# =============================================================================
# POLICY REGISTRY
# =============================================================================

POLICY_REGISTRY: Dict[str, PolicyConstraint] = {
    "addition_before_multiplication": AdditionBeforeMultiplication(),
    "multiplication_before_addition": MultiplicationBeforeAddition(),
    "exponentiation_first":           ExponentiationFirst(),
    "leftmost_first":                 LeftmostFirst(),
    "rightmost_first":                RightmostFirst(),
    "left_to_right_strict":           LeftToRightStrict(),
    "right_to_left_strict":           RightToLeftStrict(),
    "brackets_first":                 BracketsFirst(),
    "brackets_ignored":               BracketsIgnored(),
    "leftmost_among_non_brackets":    LeftmostAmongNonBrackets(),
    "all_equal":                      AllEqual(),
}


def get_policy(name: str) -> PolicyConstraint:
    if name not in POLICY_REGISTRY:
        raise ValueError(f"Unknown policy: {name!r}. Available: {list(POLICY_REGISTRY)}")
    return POLICY_REGISTRY[name]


# =============================================================================
# RESOLVER — finds P assignment from constraints
# =============================================================================

class Resolver:
    """
    Given a set of constraints on P, find a valid assignment P: A → ℕ.

    Approach: build a directed graph where edge a1 → a2 means P(a1) > P(a2),
    then use topological sort to assign priority levels.

    Nodes at the same topological level → tied priority → tree branches.

    If there is a cycle → contradiction (conflicting constraints).
    """

    def resolve(self, actions: List[Action],
                constraints: List[Constraint]) -> Dict[Action, int]:
        """
        Resolve constraints into a priority assignment P: Action → int.

        Higher int = higher priority = chosen first.
        Tied int = tied priority = tree branches into all tied actions.

        Returns dict mapping each action to its priority level.
        """
        if not actions:
            return {}

        # Build adjacency:
        # must_beat[a] = set of actions that a must have higher priority than
        # beaten_by[a] = set of actions that must have higher priority than a
        must_beat: Dict[Action, Set[Action]] = defaultdict(set)
        beaten_by: Dict[Action, Set[Action]] = defaultdict(set)

        for action in actions:
            must_beat[action]   # ensure all actions are in the graph
            beaten_by[action]

        for constraint in constraints:
            if constraint.higher in must_beat and constraint.lower in must_beat:
                # P(higher) > P(lower)
                must_beat[constraint.higher].add(constraint.lower)
                beaten_by[constraint.lower].add(constraint.higher)

        # Kahn's topological sort — processing order:
        # Actions with nobody above them (beaten_by empty) go FIRST
        # These are the HIGHEST priority actions
        # We assign priority levels descending from top

        in_degree = {a: len(beaten_by[a]) for a in actions}

        # Level starts high — actions processed first get the highest level
        # We'll collect groups, then assign levels in reverse
        groups = []   # groups[0] = highest priority group
        remaining = set(actions)

        while remaining:
            # Find actions with no one above them among remaining
            current_group = [a for a in remaining if in_degree[a] == 0]

            if not current_group:
                # Cycle detected
                unresolved = list(remaining)
                raise ValueError(
                    f"Conflicting constraints — cycle detected among: {unresolved}\n"
                    f"This learner has contradictory policies."
                )

            groups.append(current_group)

            # Remove this group — update in_degrees
            for action in current_group:
                remaining.remove(action)
                for lower in must_beat[action]:
                    if lower in remaining:
                        in_degree[lower] -= 1

        # Assign priority: groups[0] = highest priority = len(groups)-1
        # groups[-1] = lowest priority = 0
        priority = {}
        max_level = len(groups) - 1
        for level_idx, group in enumerate(groups):
            assigned_priority = max_level - level_idx
            for action in group:
                priority[action] = assigned_priority

        return priority

    def choose(self, actions: List[Action],
               constraints: List[Constraint]) -> List[Action]:
        """
        Given actions and constraints, return the action(s) with highest priority.

        Returns a list:
        - Single item  → deterministic choice
        - Multiple items → tie → tree branches into all of them
        """
        if not actions:
            return []

        priority = self.resolve(actions, constraints)
        max_priority = max(priority.values())
        return [a for a in actions if priority[a] == max_priority]
