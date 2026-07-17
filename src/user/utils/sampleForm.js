// JS port of base-task/stimulus_pool.py's sample_form/_index_pool.
// Keep in sync with that file if the pool schema or misconception set changes.

const IDS = [
  'add_before_mul',
  'add_before_div',
  'sub_before_mul',
  'sub_before_div',
  'same_priority_rtl',
  'outside_bracket_first',
]

const PAIRS = []
for (let i = 0; i < IDS.length; i++) {
  for (let j = i + 1; j < IDS.length; j++) {
    PAIRS.push([IDS[i], IDS[j]])
  }
}

// One name per form item (24), so no participant ever sees the same student
// twice. Names baked into the pool JSON are placeholders; they are replaced
// here at sampling time. Keep in sync with STUDENT_NAMES in stimulus_pool.py.
const STUDENT_NAMES = [
  'Noah', 'Maya', 'Liam', 'Ava', 'Ethan', 'Zoe',
  'Mia', 'Lucas', 'Emma', 'Owen', 'Sofia', 'Caleb',
  'Ruby', 'Jonah', 'Isla', 'Felix', 'Nora', 'Dylan',
  'Priya', 'Marcus', 'Elena', 'Theo', 'Jasmine', 'Omar',
]

// Small seeded PRNG (mulberry32) so a given seed always reproduces the same form.
function makeRng(seed) {
  let s = seed >>> 0
  function next() {
    s = (s + 0x6d2b79f5) | 0
    let t = Math.imul(s ^ (s >>> 15), 1 | s)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
  return {
    choice(arr) {
      return arr[Math.floor(next() * arr.length)]
    },
    randrange(n) {
      return Math.floor(next() * n)
    },
    sample(arr, k) {
      const remaining = arr.slice()
      const out = []
      for (let i = 0; i < k; i++) {
        const idx = Math.floor(next() * remaining.length)
        out.push(remaining[idx])
        remaining.splice(idx, 1)
      }
      return out
    },
    shuffle(arr) {
      for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(next() * (i + 1))
        ;[arr[i], arr[j]] = [arr[j], arr[i]]
      }
      return arr
    },
  }
}

// B/D keys include foil_status ('refuted'/'unsupported'; written by
// base-task/extend_pool.py), so sampleForm REQUIRES the extended pool. Items
// with foil_status 'ambiguous' land under unreachable keys and are never
// drawn. DAny is a status-agnostic D index used only by the emergency fallback.
function indexPool(pool) {
  const idx = { A: {}, B: {}, C: {}, D: {}, DAny: {} }
  for (const it of pool) {
    let key
    if (it.category === 'A') {
      key = it.misconceptions[0]
    } else if (it.category === 'B') {
      key = `${it.misconceptions[0]}|${it.probed_misconception}|${it.foil_status}`
    } else if (it.category === 'C') {
      key = `${it.probed_misconception}|${it.which_target}` // (target, early/late error)
    } else {
      key = `${it.misconceptions.join(',')}|${it.probed_misconception}|${it.foil_status}`
      ;(idx.DAny[`${it.misconceptions.join(',')}|${it.probed_misconception}`] ??= []).push(it)
    }
    ;(idx[it.category][key] ??= []).push(it)
  }
  return idx
}

// Backtracking perfect matching: assign each rule a distinct pair (by key)
// with ok(rule, pairKey) true. Returns Map(rule -> pairKey) or null.
function matchFoils(rules, pairKeys, ok) {
  const assignment = new Map()
  const used = new Set()
  function bt(i) {
    if (i === rules.length) return true
    for (const p of pairKeys) {
      if (used.has(p) || !ok(rules[i], p)) continue
      used.add(p)
      assignment.set(rules[i], p)
      if (bt(i + 1)) return true
      used.delete(p)
      assignment.delete(rules[i])
    }
    return false
  }
  return bt(0) ? assignment : null
}

/**
 * Build one participant's form with exact within-form balance:
 *   A — each of the 6 misconceptions appears exactly once.
 *   B — each of the 6 misconceptions appears exactly once as present, AND
 *       (via a fixed nonzero cyclic shift) exactly once as foil; 3 foils
 *       actively refuted by the trace + 3 merely unsupported, the refuted set
 *       rotating per participant.
 *   C — each of the 6 misconceptions probed once as target; exactly 3 shown as
 *       the early error ('first') and 3 as the late error ('second'), rotating
 *       per participant; underlying pairs kept distinct.
 *   D — 6 distinct pairs whose foils cover all 6 rules exactly once (foil
 *       never a member of its pair), refuted set = the COMPLEMENT of B's, so
 *       across B+D every rule is named as a foil exactly twice per form:
 *       once refuted, once unsupported.
 * Different seeds vary the shift/offset/pair-sample, so coverage differs
 * across participants while every individual form stays exactly balanced.
 * Every item also gets a distinct student name (24 names, 24 items), so no
 * participant ever sees the same student twice.
 */
export function sampleForm(pool, seed) {
  const rng = makeRng(seed)
  const idx = indexPool(pool)
  const form = []

  // A — one of each misconception
  for (const mid of IDS) {
    form.push(rng.choice(idx.A[mid]))
  }

  // B — one of each misconception as present; foil = fixed nonzero shift.
  // 3 foils refuted + 3 unsupported; which rules are refuted rotates via r,
  // and category D below uses the complementary set.
  const k = rng.choice([1, 2, 3, 4, 5])
  const r = rng.randrange(IDS.length)
  const refutedB = new Set(
    Array.from({ length: IDS.length / 2 }, (_, j) => IDS[(r + j) % IDS.length])
  )
  IDS.forEach((mid, i) => {
    const foil = IDS[(i + k) % IDS.length]
    const status = refutedB.has(foil) ? 'refuted' : 'unsupported'
    form.push(rng.choice(idx.B[`${mid}|${foil}|${status}`]))
  })

  // C — each of the 6 misconceptions probed once as target; 3 shown as the
  // early error ('first'), 3 as the late error ('second'); which 3 are 'first'
  // rotates per participant via `cShift`. Retry until the 6 underlying pairs
  // are distinct (partners can otherwise collide); fall back to allowing a
  // repeat rather than failing.
  const cShift = rng.randrange(IDS.length)
  const positions = IDS.map((_, i) => ((i + cShift) % IDS.length < 3 ? 'first' : 'second'))
  const pickC = (requireDistinct) => {
    const chosen = []
    const used = new Set()
    for (let i = 0; i < IDS.length; i++) {
      const cands = idx.C[`${IDS[i]}|${positions[i]}`] || []
      const fresh = cands.filter((it) => !used.has([...it.misconceptions].sort().join(',')))
      if (requireDistinct && fresh.length === 0) return null
      const it = rng.choice(fresh.length ? fresh : cands)
      used.add([...it.misconceptions].sort().join(','))
      chosen.push(it)
    }
    return chosen
  }
  let cItems = null
  for (let a = 0; a < 25 && cItems === null; a++) cItems = pickC(true)
  form.push(...(cItems || pickC(false)))

  // D — 6 distinct pairs whose foils cover all 6 rules exactly once (foil
  // never a member of its pair), refuted set = complement of B's, so each
  // rule is refuted exactly once per form across B+D combined.
  const refutedD = new Set(IDS.filter((m) => !refutedB.has(m)))
  const dStatus = (rule) => (refutedD.has(rule) ? 'refuted' : 'unsupported')
  let dItems = null
  for (let a = 0; a < 50 && dItems === null; a++) {
    const pairKeys = rng.sample(PAIRS, 6).map((p) => p.join(','))
    const order = rng.shuffle(IDS.slice())
    const ok = (rule, pairKey) =>
      !pairKey.split(',').includes(rule) && (idx.D[`${pairKey}|${rule}|${dStatus(rule)}`] || []).length > 0
    const assignment = matchFoils(order, pairKeys, ok)
    if (assignment) {
      dItems = order.map((rule) => rng.choice(idx.D[`${assignment.get(rule)}|${rule}|${dStatus(rule)}`]))
    }
  }
  if (dItems === null) {
    // emergency fallback (should not happen with the full 120-cell pool):
    // old rotation logic, status ignored — a mildly unbalanced form beats
    // a crashed session
    dItems = []
    const offset = rng.randrange(4)
    rng.sample(PAIRS, 6).forEach((pair, j) => {
      const others = IDS.filter((m) => !pair.includes(m))
      const foil = others[(offset + j) % others.length]
      dItems.push(rng.choice(idx.DAny[`${pair.join(',')}|${foil}`]))
    })
  }
  form.push(...dItems)

  rng.shuffle(form)

  // Assign each item a distinct student name (copies, so the shared pool
  // objects are never mutated), rewriting the belief statement to match.
  const names = rng.shuffle(STUDENT_NAMES.slice())
  return form.map((it, i) => ({
    ...it,
    student_name: names[i],
    belief_statement: it.belief_statement.replace(it.student_name, names[i]),
  }))
}

export function randomSeed() {
  return Math.floor(Math.random() * 2 ** 31)
}
