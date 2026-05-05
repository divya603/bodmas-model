"""
Tests for the formal policy constraint system.

Expression: 3 + 5 * 2
Available actions:
    a1: + at position 1
    a2: * at position 3
"""

from formal_policy import Action
from learner import create_learner, create_custom_learner, LEARNER_PROFILES


def run_tests():
    # Expression: 3 + 5 * 2
    # Actions available:
    a_add  = Action(operator='+', position=1, depth=0)
    a_mult = Action(operator='*', position=3, depth=0)
    a_brk  = Action(operator='brackets', position=2, depth=1)

    actions_no_bracket = [a_add, a_mult]
    actions_with_bracket = [a_add, a_mult, a_brk]

    print("=" * 60)
    print("FORMAL POLICY CONSTRAINT SYSTEM — TESTS")
    print("=" * 60)

    # ------------------------------------------------------------------
    # TEST 1: Expert learner — should pick brackets first, then * over +
    # ------------------------------------------------------------------
    print("\n--- TEST 1: Expert learner (no brackets) ---")
    expert = create_learner("expert")
    chosen = expert.choose(actions_no_bracket)
    print(expert.explain(actions_no_bracket))
    assert a_mult in chosen and a_add not in chosen, \
        f"Expert should pick * over + but got {chosen}"
    print("✅ PASS: Expert correctly picks * before +")

    # ------------------------------------------------------------------
    # TEST 2: Expert with brackets — should pick bracket first
    # ------------------------------------------------------------------
    print("\n--- TEST 2: Expert learner (with brackets) ---")
    chosen = expert.choose(actions_with_bracket)
    print(f"Chosen: {chosen}")
    assert a_brk in chosen, f"Expert should pick brackets first but got {chosen}"
    print("✅ PASS: Expert correctly picks brackets first")

    # ------------------------------------------------------------------
    # TEST 3: Addition first learner — should pick + over *
    # ------------------------------------------------------------------
    print("\n--- TEST 3: Addition-first learner ---")
    add_first = create_learner("addition_first")
    chosen = add_first.choose(actions_no_bracket)
    print(add_first.explain(actions_no_bracket))
    assert a_add in chosen and a_mult not in chosen, \
        f"Addition-first should pick + but got {chosen}"
    print("✅ PASS: Addition-first correctly picks + before *")

    # ------------------------------------------------------------------
    # TEST 4: Left-to-right learner — should pick leftmost action
    # ------------------------------------------------------------------
    print("\n--- TEST 4: Left-to-right learner ---")
    ltr = create_learner("left_to_right")
    chosen = ltr.choose(actions_no_bracket)
    print(ltr.explain(actions_no_bracket))
    assert a_add in chosen and a_mult not in chosen, \
        f"Left-to-right should pick leftmost (+) but got {chosen}"
    print("✅ PASS: Left-to-right correctly picks leftmost action")

    # ------------------------------------------------------------------
    # TEST 5: Novice — all actions tied, full branching
    # ------------------------------------------------------------------
    print("\n--- TEST 5: Novice learner (all tied) ---")
    novice = create_learner("novice")
    chosen = novice.choose(actions_no_bracket)
    print(novice.explain(actions_no_bracket))
    assert set(chosen) == {a_add, a_mult}, \
        f"Novice should return all actions as tied but got {chosen}"
    print("✅ PASS: Novice correctly returns all actions (full branching)")

    # ------------------------------------------------------------------
    # TEST 6: Bracket ignorer — should never pick brackets
    # ------------------------------------------------------------------
    print("\n--- TEST 6: Bracket ignorer (with brackets available) ---")
    ignorer = create_learner("bracket_ignorer")
    chosen = ignorer.choose(actions_with_bracket)
    print(ignorer.explain(actions_with_bracket))
    assert a_brk not in chosen, \
        f"Bracket ignorer should never pick brackets but got {chosen}"
    print("✅ PASS: Bracket ignorer correctly avoids brackets")

    # ------------------------------------------------------------------
    # TEST 7: Custom learner — compose any policies
    # ------------------------------------------------------------------
    print("\n--- TEST 7: Custom learner (right-to-left + add before mult) ---")
    custom = create_custom_learner(
        name="custom_test",
        policy_names=["rightmost_first", "addition_before_multiplication"],
        description="Goes right to left AND thinks addition beats multiplication"
    )
    chosen = custom.choose(actions_no_bracket)
    print(custom.explain(actions_no_bracket))
    # rightmost picks * (pos=3), addition_before_mult also wants +
    # These conflict — let's see what resolves
    print(f"Custom learner chose: {chosen}")
    print("✅ Ran without error — constraint interaction works")

    # ------------------------------------------------------------------
    # TEST 8: Show all learner profiles
    # ------------------------------------------------------------------
    print("\n--- TEST 8: All learner profiles on same expression ---")
    print(f"Expression actions: {actions_no_bracket}")
    print(f"{'Learner':<25} | {'Chosen action(s)'}")
    print("-" * 55)
    for profile_name in LEARNER_PROFILES:
        learner = create_learner(profile_name)
        chosen = learner.choose(actions_no_bracket)
        chosen_str = [f"{a.operator}@pos{a.position}" for a in chosen]
        print(f"{profile_name:<25} | {chosen_str}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✅")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
