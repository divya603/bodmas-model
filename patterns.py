# ══════════════════════════════════════════════════════════════════════════════
# RULE TABLE
#
# Each entry:
#   id       — unique slug
#   pattern  — structural template (human-readable description of the shape)
#   validity — "RULE" or "ANTI-RULE"
#   ops      — dict mapping placeholder names to the set of operators that
#              make this entry valid/invalid.
#              Placeholder conventions:
#                "⊕"      the single operator in 1-op patterns
#                "⊕","⊗"  left and right ops in 2-op patterns
#                "⊗","⊕"  outer and inner ops in bracket patterns
#                "outer","inner"  for nested unary patterns
#                "from","to"  for op-confusion entries
#   note     — why it is valid or invalid
#
# Anti-rules are structurally plausible rewrites that are mathematically
# wrong for those specific operators — they model student misconceptions.
# ══════════════════════════════════════════════════════════════════════════════

RULE_TABLE = [

    # ══════════════════════════════════════════════════════════════════
    # 1. EVAL BINARY
    #    Shape:  a ⊕ b → [||a ⊕ b||]     (a, b are number leaves)
    #    Always a rule; precedence is a separate concern (section 2).
    # ══════════════════════════════════════════════════════════════════
    dict(id="Eval",
         pattern="a ⊕ b → [||a ⊕ b||]",
         validity="RULE",
         ops={"⊕": {"+", "-", "×", "÷"}},
         note="evaluate two numbers; divisor ≠ 0 for ÷"),

    # ══════════════════════════════════════════════════════════════════
    # 2. PRECEDENCE
    #    Shape:  X ⊕ Y ⊗ Z   (three atoms, two ops at the same level)
    #    ⊕ = left op,  ⊗ = right op.
    #    "eval ⊕ first" means collapse (atoms[0], op⊕, atoms[1]) first.
    #    "eval ⊗ first" means collapse (atoms[1], op⊗, atoms[2]) first.
    # ══════════════════════════════════════════════════════════════════

    # right op is higher-prec → eval right (⊗) first
    dict(id="Prec.right-hi",
         pattern="X ⊕ Y ⊗ Z: eval ⊗ first",
         validity="RULE",
         ops={"⊕": {"+", "-"}, "⊗": {"×", "÷"}},
         note="×/÷ on the right binds tighter than +/- on the left"),

    # left op is higher-prec → eval left (⊕) first
    dict(id="Prec.left-hi",
         pattern="X ⊕ Y ⊗ Z: eval ⊕ first",
         validity="RULE",
         ops={"⊕": {"×", "÷"}, "⊗": {"+", "-"}},
         note="×/÷ on the left must be evaluated before +/- on the right"),

    # same precedence → eval left (⊕) first  [left-to-right convention]
    dict(id="Prec.same-left",
         pattern="X ⊕ Y ⊗ Z: eval ⊕ first",
         validity="RULE",
         ops={"⊕": {"+", "-", "×", "÷"}, "⊗": {"+", "-", "×", "÷"}},
         note="same-precedence pair: left-to-right is the standard convention"),

    # same precedence, right first — only safe when left op is fully associative
    dict(id="Prec.same-right.ok",
         pattern="X ⊕ Y ⊗ Z: eval ⊗ first",
         validity="RULE",
         ops={"⊕": {"+", "×"}, "⊗": {"+", "-", "×", "÷"}},
         note="right-first gives same result when left op is + or × (fully associative)"),

    # ANTI-RULES
    # classic left-to-right error: left is +/-, right is ×/÷ → student evals left first
    dict(id="Prec.lo-before-hi",
         pattern="X ⊕ Y ⊗ Z: eval ⊕ first",
         validity="ANTI-RULE",
         ops={"⊕": {"+", "-"}, "⊗": {"×", "÷"}},
         note="left-to-right misconception: eval +/- before ×/÷"),

    # skip high-prec left: left is ×/÷, right is +/- → student evals right first
    dict(id="Prec.skip-left-hi",
         pattern="X ⊕ Y ⊗ Z: eval ⊗ first",
         validity="ANTI-RULE",
         ops={"⊕": {"×", "÷"}, "⊗": {"+", "-"}},
         note="student skips left ×/÷ and evals right +/- first"),

    # same-prec, right first, but left op is - or ÷ → result differs
    dict(id="Prec.same-right.nonassoc",
         pattern="X ⊕ Y ⊗ Z: eval ⊗ first",
         validity="ANTI-RULE",
         ops={"⊕": {"-", "÷"}, "⊗": {"+", "-", "×", "÷"}},
         note="right-first changes result when left op is - or ÷ (not fully associative)"),

    # ══════════════════════════════════════════════════════════════════
    # 3. COMMUTE
    #    Shape:  X ⊕ Y → Y ⊕ X
    # ══════════════════════════════════════════════════════════════════
    dict(id="Commute",
         pattern="X ⊕ Y → Y ⊕ X",
         validity="RULE",
         ops={"⊕": {"+", "×"}},
         note="commutativity holds for + and ×"),

    dict(id="Commute.AR",
         pattern="X ⊕ Y → Y ⊕ X",
         validity="ANTI-RULE",
         ops={"⊕": {"-", "÷"}},
         note="- and ÷ are not commutative"),

    # ══════════════════════════════════════════════════════════════════
    # 4. ASSOC SAME-OP
    #    Shape:  (X ⊕ Y) ⊕ Z  ↔  X ⊕ (Y ⊕ Z)
    # ══════════════════════════════════════════════════════════════════
    dict(id="Assoc.same",
         pattern="(X ⊕ Y) ⊕ Z → X ⊕ (Y ⊕ Z)",
         validity="RULE",
         ops={"⊕": {"+", "×"}},
         note="+ and × are associative"),

    dict(id="Assoc.same.AR",
         pattern="(X ⊕ Y) ⊕ Z → X ⊕ (Y ⊕ Z)",
         validity="ANTI-RULE",
         ops={"⊕": {"-", "÷"}},
         note="- and ÷ are not associative; rebracket changes the result"),

    # ══════════════════════════════════════════════════════════════════
    # 5. ASSOC MIXED-OP
    #    Shape:  (X ⊕ Y) ⊗ Z  →  X ⊕ (Y ⊗ Z)
    #    ⊕ = op inside the bracket, ⊗ = op outside
    # ══════════════════════════════════════════════════════════════════

    # valid cases: result is unchanged after rebracket
    dict(id="AssocMix.+-",
         pattern="(X ⊕ Y) ⊗ Z → X ⊕ (Y ⊗ Z)",
         validity="RULE",
         ops={"⊕": {"+"}, "⊗": {"-"}},
         note="X+Y-Z = X+(Y-Z); result is identical"),

    dict(id="AssocMix.×÷",
         pattern="(X ⊕ Y) ⊗ Z → X ⊕ (Y ⊗ Z)",
         validity="RULE",
         ops={"⊕": {"×"}, "⊗": {"÷"}},
         note="X×Y÷Z = X×(Y÷Z); result is identical"),

    # invalid cases: rebracket changes the result
    dict(id="AssocMix.-.AR",
         pattern="(X ⊕ Y) ⊗ Z → X ⊕ (Y ⊗ Z)",
         validity="ANTI-RULE",
         ops={"⊕": {"-"}, "⊗": {"+", "-"}},
         note="(X-Y)⊗Z ≠ X-(Y⊗Z) for ⊗ ∈ {+,-}"),

    dict(id="AssocMix.÷.AR",
         pattern="(X ⊕ Y) ⊗ Z → X ⊕ (Y ⊗ Z)",
         validity="ANTI-RULE",
         ops={"⊕": {"÷"}, "⊗": {"×", "÷"}},
         note="(X÷Y)⊗Z ≠ X÷(Y⊗Z) for ⊗ ∈ {×,÷}"),

    # ══════════════════════════════════════════════════════════════════
    # 6. PAREN ELIM  (bracket flattening)
    #    Shape:  A ⊗ (X ⊕ Y) → A ⊗ X ⊕ Y   (naive drop, no adjustment)
    #    ⊗ = outer op,  ⊕ = inner op
    # ══════════════════════════════════════════════════════════════════

    # bracket is genuinely redundant: inner is ×/÷ (higher prec than outer +/-)
    dict(id="PE.redundant",
         pattern="A ⊗ (X ⊕ Y) → A ⊗ X ⊕ Y",
         validity="RULE",
         ops={"⊗": {"+", "-"}, "⊕": {"×", "÷"}},
         note="inner ×/÷ still binds tighter after bracket removed — result unchanged"),

    # outer + with additive inner: + is associative
    dict(id="PE.+.additive",
         pattern="A ⊗ (X ⊕ Y) → A ⊗ X ⊕ Y",
         validity="RULE",
         ops={"⊗": {"+"}, "⊕": {"+", "-"}},
         note="A+(X±Y) flattens correctly; + associativity preserves result"),

    # outer × with multiplicative inner: × is associative / ×÷ left-to-right is same
    dict(id="PE.×.multiplicative",
         pattern="A ⊗ (X ⊕ Y) → A ⊗ X ⊕ Y",
         validity="RULE",
         ops={"⊗": {"×"}, "⊕": {"×", "÷"}},
         note="A×(X×Y) and A×(X÷Y) flatten correctly"),

    # outer - with additive inner: valid only with sign distribution (not a naive drop)
    dict(id="PE.-.sign-dist",
         pattern="A-(X ⊕ Y) → A ∓ X ∓ Y  (signs distributed)",
         validity="RULE",
         ops={"⊗": {"-"}, "⊕": {"+", "-"}},
         note="A-(X+Y)→A-X-Y; A-(X-Y)→A-X+Y — must distribute the minus, not just drop bracket"),

    # ANTI-RULES
    dict(id="PE.-.naive",
         pattern="A ⊗ (X ⊕ Y) → A ⊗ X ⊕ Y  (naive)",
         validity="ANTI-RULE",
         ops={"⊗": {"-"}, "⊕": {"+", "-"}},
         note="naive drop without distributing the minus sign"),

    dict(id="PE.×.additive",
         pattern="A ⊗ (X ⊕ Y) → A ⊗ X ⊕ Y  (naive)",
         validity="ANTI-RULE",
         ops={"⊗": {"×"}, "⊕": {"+", "-"}},
         note="must distribute × over each term; naive drop gives wrong result"),

    dict(id="PE.÷.any",
         pattern="A ⊗ (X ⊕ Y) → A ⊗ X ⊕ Y  (naive)",
         validity="ANTI-RULE",
         ops={"⊗": {"÷"}, "⊕": {"+", "-", "×", "÷"}},
         note="÷ outer: naive bracket drop always changes the result"),

    # ══════════════════════════════════════════════════════════════════
    # 7. DISTRIB RIGHT
    #    Shape:  X ⊗ (Y ⊕ Z) → X⊗Y ⊕ X⊗Z
    #    ⊗ = outer op,  ⊕ = inner op
    # ══════════════════════════════════════════════════════════════════
    dict(id="DistR",
         pattern="X ⊗ (Y ⊕ Z) → X⊗Y ⊕ X⊗Z",
         validity="RULE",
         ops={"⊗": {"×"}, "⊕": {"+", "-"}},
         note="× distributes over + and -"),

    dict(id="DistR.AR",
         pattern="X ⊗ (Y ⊕ Z) → X⊗Y ⊕ X⊗Z",
         validity="ANTI-RULE",
         ops={"⊗": {"+", "-", "÷"}, "⊕": {"+", "-", "×", "÷"}},
         note="only × distributes; applying this shape with any other outer op is wrong"),

    dict(id="DistR.partial",
         pattern="X ⊗ (Y ⊕ Z) → X⊗Y ⊕ Z",
         validity="ANTI-RULE",
         ops={"⊗": {"×"}, "⊕": {"+", "-"}},
         note="partial distribution: X distributed into Y but Z left unscaled"),

    # ══════════════════════════════════════════════════════════════════
    # 8. DISTRIB LEFT
    #    Shape:  (Y ⊕ Z) ⊗ X → Y⊗X ⊕ Z⊗X
    #    ⊗ = outer op,  ⊕ = inner op
    # ══════════════════════════════════════════════════════════════════
    dict(id="DistL",
         pattern="(Y ⊕ Z) ⊗ X → Y⊗X ⊕ Z⊗X",
         validity="RULE",
         ops={"⊗": {"×", "÷"}, "⊕": {"+", "-"}},
         note="× and ÷ both distribute from the right over + and -"),

    dict(id="DistL.AR",
         pattern="(Y ⊕ Z) ⊗ X → Y⊗X ⊕ Z⊗X",
         validity="ANTI-RULE",
         ops={"⊗": {"+", "-"}, "⊕": {"+", "-", "×", "÷"}},
         note="+ and - do not distribute"),

    dict(id="DistL.partial",
         pattern="(Y ⊕ Z) ⊗ X → Y⊗X ⊕ Z",
         validity="ANTI-RULE",
         ops={"⊗": {"×", "÷"}, "⊕": {"+", "-"}},
         note="partial distribution: Y scaled by X but Z left unscaled"),

    # ══════════════════════════════════════════════════════════════════
    # 9. DROP DOUBLE  (nested unary elimination)
    #    Shape:  outer(inner(X)) → result
    # ══════════════════════════════════════════════════════════════════
    dict(id="DropDouble.--",
         pattern="-(-X) → X",
         validity="RULE",
         ops={"outer": {"u-"}, "inner": {"u-"}},
         note="double negation cancels"),

    dict(id="DropDouble.++",
         pattern="+(+X) → X",
         validity="RULE",
         ops={"outer": {"u+"}, "inner": {"u+"}},
         note="double positive is identity"),

    dict(id="DropDouble.-+",
         pattern="-(+X) → -X",
         validity="RULE",
         ops={"outer": {"u-"}, "inner": {"u+"}},
         note="negating a positive is negative"),

    dict(id="DropDouble.+-",
         pattern="+(-X) → -X",
         validity="RULE",
         ops={"outer": {"u+"}, "inner": {"u-"}},
         note="positive of a negative is negative"),

    dict(id="DropSingle",
         pattern="-X → X",
         validity="ANTI-RULE",
         ops={"outer": {"u-"}},
         note="single negation incorrectly dropped"),

    dict(id="DropDouble.partial",
         pattern="-(-X) → -X",
         validity="ANTI-RULE",
         ops={"outer": {"u-"}, "inner": {"u-"}},
         note="only one of the two negations removed"),

    # ══════════════════════════════════════════════════════════════════
    # 10. CANCEL DOUBLE  (binary op with unary-negated right operand)
    #     Shape:  X ⊗ (-Y)
    #     ⊗ = binary op,  right operand is unary-negated
    # ══════════════════════════════════════════════════════════════════
    dict(id="Cancel.sub-neg",
         pattern="X - (-Y) → X + Y",
         validity="RULE",
         ops={"⊗": {"-"}, "unary": {"u-"}},
         note="subtracting a negative is the same as adding"),

    dict(id="Cancel.add-neg",
         pattern="X + (-Y) → X - Y",
         validity="RULE",
         ops={"⊗": {"+"}, "unary": {"u-"}},
         note="adding a negative is the same as subtracting"),

    dict(id="Cancel.sub-neg.AR",
         pattern="X - (-Y) → X - Y",
         validity="ANTI-RULE",
         ops={"⊗": {"-"}, "unary": {"u-"}},
         note="forgot to flip sign; kept minus instead of flipping to plus"),

    dict(id="Cancel.add-neg.AR",
         pattern="X + (-Y) → X + Y",
         validity="ANTI-RULE",
         ops={"⊗": {"+"}, "unary": {"u-"}},
         note="ignored the negative sign of Y; treated it as positive"),

    # ══════════════════════════════════════════════════════════════════
    # 11. NEGATE DISTRIBUTE
    #     Shape:  -(X ⊕ Y)  where the bracket contains a binary op
    # ══════════════════════════════════════════════════════════════════
    dict(id="NegDist.+",
         pattern="-(X + Y) → -X - Y",
         validity="RULE",
         ops={"⊕": {"+"}},
         note="distribute minus: both terms get negated"),

    dict(id="NegDist.-",
         pattern="-(X - Y) → -X + Y",
         validity="RULE",
         ops={"⊕": {"-"}},
         note="distribute minus: Y's sign flips"),

    dict(id="NegDist.+.partial",
         pattern="-(X + Y) → -X + Y",
         validity="ANTI-RULE",
         ops={"⊕": {"+"}},
         note="only negated the first term; forgot to negate Y"),

    dict(id="NegDist.-.partial",
         pattern="-(X - Y) → -X - Y",
         validity="ANTI-RULE",
         ops={"⊕": {"-"}},
         note="failed to flip Y's sign when distributing minus"),

    dict(id="NegDist.×.both",
         pattern="-(X × Y) → (-X) × (-Y)",
         validity="ANTI-RULE",
         ops={"⊕": {"×"}},
         note="incorrectly negated both factors; correct is -(X×Y) = (-X)×Y or X×(-Y)"),

    # ══════════════════════════════════════════════════════════════════
    # 12. OP CONFUSION
    #     Shape:  X ⊕ Y  but student applies  X ⊗ Y  with the wrong op
    # ══════════════════════════════════════════════════════════════════
    dict(id="OpConf.×→+",  pattern="X × Y  read as  X + Y",  validity="ANTI-RULE", ops={"from": {"×"}, "to": {"+"}},  note="multiply confused with add"),
    dict(id="OpConf.+→×",  pattern="X + Y  read as  X × Y",  validity="ANTI-RULE", ops={"from": {"+"}, "to": {"×"}},  note="add confused with multiply"),
    dict(id="OpConf.÷→-",  pattern="X ÷ Y  read as  X - Y",  validity="ANTI-RULE", ops={"from": {"÷"}, "to": {"-"}},  note="divide confused with subtract"),
    dict(id="OpConf.-→÷",  pattern="X - Y  read as  X ÷ Y",  validity="ANTI-RULE", ops={"from": {"-"}, "to": {"÷"}},  note="subtract confused with divide"),
    dict(id="OpConf.-→+",  pattern="X - Y  read as  X + Y",  validity="ANTI-RULE", ops={"from": {"-"}, "to": {"+"}},  note="subtract confused with add"),
    dict(id="OpConf.+→-",  pattern="X + Y  read as  X - Y",  validity="ANTI-RULE", ops={"from": {"+"}, "to": {"-"}},  note="add confused with subtract"),
    dict(id="OpConf.÷→×",  pattern="X ÷ Y  read as  X × Y",  validity="ANTI-RULE", ops={"from": {"÷"}, "to": {"×"}},  note="divide confused with multiply"),
    dict(id="OpConf.×→÷",  pattern="X × Y  read as  X ÷ Y",  validity="ANTI-RULE", ops={"from": {"×"}, "to": {"÷"}},  note="multiply confused with divide"),
]


# ── Convenience views ─────────────────────────────────────────────────────────

RULES      = [e for e in RULE_TABLE if e["validity"] == "RULE"]
ANTI_RULES = [e for e in RULE_TABLE if e["validity"] == "ANTI-RULE"]

def rules_by_pattern(prefix):
    return [e for e in RULE_TABLE if e["id"].startswith(prefix)]


if __name__ == "__main__":
    print(f"Total: {len(RULE_TABLE)}  ({len(RULES)} RULE, {len(ANTI_RULES)} ANTI-RULE)\n")
    sections = ["Eval", "Prec", "Commute", "Assoc", "PE",
                "DistR", "DistL", "DropDouble", "DropSingle",
                "Cancel", "NegDist", "OpConf"]
    for s in sections:
        g  = rules_by_pattern(s)
        r  = sum(1 for e in g if e["validity"] == "RULE")
        ar = sum(1 for e in g if e["validity"] == "ANTI-RULE")
        print(f"  {s:<12}  {r:>2} rules   {ar:>2} anti-rules")
