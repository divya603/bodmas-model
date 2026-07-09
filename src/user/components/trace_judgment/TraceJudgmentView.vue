<script setup>
import { ref, watch, onBeforeUnmount } from 'vue'
import useViewAPI from '@/core/composables/useViewAPI'
import { Button } from '@/uikit/components/ui/button'
import { ConstrainedTaskWindow } from '@/uikit/layouts'
import pool from '@/user/data/stimulus_pool.json'
import { sampleForm, randomSeed } from '@/user/utils/sampleForm'
import { useMouseTracking } from '@/user/utils/useMouseTracking'

const api = useViewAPI()

// records a sampled mouse path per trial for offline bot detection (bots tend
// to not move the cursor or move in perfectly straight lines)
const mouse = useMouseTracking()

const LIKERT = [
  { value: 1, label: 'Strongly Disagree' },
  { value: 2, label: 'Disagree' },
  { value: 3, label: 'Somewhat Disagree' },
  { value: 4, label: 'Somewhat Agree' },
  { value: 5, label: 'Agree' },
  { value: 6, label: 'Strongly Agree' },
]

// draw this participant's 24-item form once, then persist the seed so a
// page reload mid-experiment doesn't resample a different form
if (!api.persist.isDefined('formSeed')) {
  api.persist.formSeed = randomSeed()
}
const form = sampleForm(pool, api.persist.formSeed)

const trials = api.steps.append(form.map((item) => ({ ...item })))
trials.append([{ id: 'summary' }])

// ── bonus scoring ───────────────────────────────────────────────────
// Collapse the 6-point Likert to a binary agree/disagree judgment (>= 4 =
// agree) and score each trial against the item's ground-truth direction
// (statement_correct). The performance bonus is rescaled so chance (50%)
// earns $0 and perfect earns MAX_BONUS. We score only the direction, never
// confidence magnitude, so the bonus can't push participants toward the
// extremes of the scale and distort the Likert data we care about.
const MAX_BONUS = 2.0
const CHANCE = 0.5

if (!api.persist.isDefined('nScored')) api.persist.nScored = 0
if (!api.persist.isDefined('nCorrect')) api.persist.nCorrect = 0

function scoreResponse(value) {
  const respondedAgree = value >= 4
  const correctAgree = api.stepData.statement_correct === true
  const isCorrect = respondedAgree === correctAgree
  api.stepData.responded_agree = respondedAgree
  api.stepData.correct_agree = correctAgree
  api.stepData.is_correct = isCorrect
  api.persist.nScored = api.persist.nScored + 1
  if (isCorrect) api.persist.nCorrect = api.persist.nCorrect + 1
}

function computeBonus() {
  const n = api.persist.nScored || 0
  const accuracy = n ? api.persist.nCorrect / n : 0
  const bonus = Math.round(Math.max(0, (accuracy - CHANCE) / (1 - CHANCE)) * MAX_BONUS * 100) / 100
  return { n, accuracy, bonus }
}

const selected = ref(null)

// ── read-before-answer lock ─────────────────────────────────────────
// Disable the response options for the first UNLOCK_DELAY_MS of each trial so
// participants actually read the expression, trace, and belief statement
// before rating. Re-arms on every new trial.
const UNLOCK_DELAY_MS = 3000
const locked = ref(false)
const remaining = ref(0)
let lockTimer = null
let lockInterval = null

function clearLockTimers() {
  if (lockTimer) {
    clearTimeout(lockTimer)
    lockTimer = null
  }
  if (lockInterval) {
    clearInterval(lockInterval)
    lockInterval = null
  }
}

function startLock() {
  clearLockTimers()
  locked.value = true
  remaining.value = Math.ceil(UNLOCK_DELAY_MS / 1000)
  lockInterval = setInterval(() => {
    remaining.value = Math.max(0, remaining.value - 1)
    if (remaining.value <= 0) {
      clearInterval(lockInterval)
      lockInterval = null
    }
  }, 1000)
  lockTimer = setTimeout(() => {
    locked.value = false
    remaining.value = 0
  }, UNLOCK_DELAY_MS)
}

// re-arm the lock whenever a new trial is shown (not on the summary step)
watch(
  () => api.stepIndex,
  () => {
    if (api.path[0] !== 'summary') startLock()
    else clearLockTimers()
  },
  { immediate: true }
)

onBeforeUnmount(clearLockTimers)

if (!api.isTimerStarted()) {
  api.startTimer()
}
mouse.start() // begin tracking for the first trial

function selectResponse(value) {
  if (locked.value) return
  selected.value = value
}

function submit() {
  if (selected.value === null) return
  api.stepData.response = selected.value
  scoreResponse(selected.value)
  api.stepData.rt = api.elapsedTime()
  api.stepData.mouse = mouse.getPoints()
  api.recordStep()
  selected.value = null
  api.startTimer()
  mouse.reset() // fresh path for the next trial
  api.goNextStep()
}

function autofill() {
  while (api.stepIndex < api.nSteps) {
    if (api.path[0] !== 'summary') {
      const value = api.faker.rchoice([1, 2, 3, 4, 5, 6])
      api.stepData.response = value
      scoreResponse(value)
      api.stepData.rt = api.faker.rnorm(4000, 800)
    }
    api.recordStep()
    api.goNextStep()
  }
}
api.setAutofill(autofill)

function finish() {
  const { n, accuracy, bonus } = computeBonus()
  api.recordPageData({
    phase: 'traceJudgmentBonus',
    nScored: n,
    nCorrect: api.persist.nCorrect,
    accuracy,
    bonus,
    maxBonus: MAX_BONUS,
  })
  api.saveData(true)
  api.goNextView()
}
</script>

<template>
  <ConstrainedTaskWindow
    variant="ghost"
    :responsiveUI="api.config.responsiveUI"
    :width="api.config.windowsizerRequest.width"
    :height="api.config.windowsizerRequest.height"
  >
    <div v-if="api.path[0] !== 'summary'" class="text-left w-full h-full overflow-y-auto px-2">
      <div class="flex justify-end mb-1">
        <span class="text-xs text-muted-foreground">{{ api.stepIndex + 1 }} of {{ form.length }}</span>
      </div>
      <p class="text-muted-foreground mb-1">Expression given to {{ api.stepData.student_name }}:</p>
      <p class="text-2xl font-bold mb-5 font-mono">{{ api.stepData.expression }}</p>

      <p class="text-muted-foreground mb-2">
        Here is the final answer {{ api.stepData.student_name }} produced, along with their work:
      </p>
      <div class="font-mono text-base mb-5 space-y-1">
        <p v-for="(step, i) in api.stepData.trace.slice(1)" :key="i">= {{ step }}</p>
      </div>

      <div class="border border-yellow-300 bg-yellow-50 rounded-lg p-4 mb-5">
        <p class="italic text-yellow-800">{{ api.stepData.belief_statement }}</p>
      </div>

      <p class="font-semibold mb-3">
        How much do you agree that this is what the student believes?
        <span v-if="locked" class="ml-1 text-sm font-normal text-muted-foreground">
          (please read the work above — {{ remaining }}s)
        </span>
      </p>
      <div class="grid grid-cols-6 gap-2 mb-6">
        <button
          v-for="opt in LIKERT"
          :key="opt.value"
          type="button"
          :disabled="locked"
          @click="selectResponse(opt.value)"
          class="flex flex-col items-center p-2 rounded border transition-colors text-center"
          :class="[
            selected === opt.value ? 'border-primary bg-primary/10' : 'border-gray-300 hover:bg-gray-50',
            locked ? 'opacity-40 cursor-not-allowed' : '',
          ]"
        >
          <span
            class="w-4 h-4 rounded-full border-2 mb-2"
            :class="selected === opt.value ? 'border-primary bg-primary' : 'border-gray-400'"
          />
          <span class="text-xs">{{ opt.label }}</span>
        </button>
      </div>

      <div class="flex justify-end">
        <Button :disabled="selected === null || locked" @click="submit()">Submit</Button>
      </div>
    </div>

    <div class="text-center" v-else>
      <p class="text-lg text-muted-foreground mb-4" id="prompt">
        Thanks! You are finished with this task and can move on.
      </p>
      <Button variant="default" size="lg" id="finish" @click="finish()">
        Continue
        <svg class="w-4 h-4 ml-2" fill="currentColor" viewBox="0 0 20 20">
          <path
            fill-rule="evenodd"
            d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z"
            clip-rule="evenodd"
          />
        </svg>
      </Button>
    </div>
  </ConstrainedTaskWindow>
</template>
