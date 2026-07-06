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

function indexPool(pool) {
  const idx = { A: {}, B: {}, C: {}, D: {} }
  for (const it of pool) {
    let key
    if (it.category === 'A') {
      key = it.misconceptions[0]
    } else if (it.category === 'B') {
      key = `${it.misconceptions[0]}|${it.probed_misconception}`
    } else {
      key = `${it.misconceptions.join(',')}|${it.probed_misconception}`
    }
    ;(idx[it.category][key] ??= []).push(it)
  }
  return idx
}

/**
 * Build one participant's form with exact within-form balance:
 *   A — each of the 6 misconceptions appears exactly once.
 *   B — each of the 6 misconceptions appears exactly once as present, AND
 *       (via a fixed nonzero cyclic shift) exactly once as foil.
 *   C — 6 distinct pairs, split exactly 3/3 between which_target 'first'/'second'.
 *   D — 6 distinct pairs, foil rotated across each pair's 4 non-members.
 * Different seeds vary the shift/offset/pair-sample, so coverage differs
 * across participants while every individual form stays exactly balanced.
 */
export function sampleForm(pool, seed) {
  const rng = makeRng(seed)
  const idx = indexPool(pool)
  const form = []

  // A — one of each misconception
  for (const mid of IDS) {
    form.push(rng.choice(idx.A[mid]))
  }

  // B — one of each misconception as present; foil = fixed nonzero shift
  const k = rng.choice([1, 2, 3, 4, 5])
  IDS.forEach((mid, i) => {
    const foil = IDS[(i + k) % IDS.length]
    form.push(rng.choice(idx.B[`${mid}|${foil}`]))
  })

  // C — 6 distinct pairs, alternating target by position -> exact 3/3 split
  rng.sample(PAIRS, 6).forEach((pair, j) => {
    const target = j % 2 === 0 ? pair[0] : pair[1]
    form.push(rng.choice(idx.C[`${pair.join(',')}|${target}`]))
  })

  // D — 6 distinct pairs, foil rotated across each pair's 4 non-members
  const offset = rng.randrange(4)
  rng.sample(PAIRS, 6).forEach((pair, j) => {
    const others = IDS.filter((m) => !pair.includes(m))
    const foil = others[(offset + j) % others.length]
    form.push(rng.choice(idx.D[`${pair.join(',')}|${foil}`]))
  })

  return rng.shuffle(form)
}

export function randomSeed() {
  return Math.floor(Math.random() * 2 ** 31)
}
