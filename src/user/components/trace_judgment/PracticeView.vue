<script setup>
import { ref, watch, nextTick, onBeforeUnmount } from 'vue'
import useViewAPI from '@/core/composables/useViewAPI'
import { Button } from '@/uikit/components/ui/button'
import { ConstrainedTaskWindow } from '@/uikit/layouts'
import practiceItems from '@/user/data/practice_items.json'

const api = useViewAPI()

const LIKERT = [
  { value: 1, label: 'Strongly Disagree' },
  { value: 2, label: 'Disagree' },
  { value: 3, label: 'Somewhat Disagree' },
  { value: 4, label: 'Somewhat Agree' },
  { value: 5, label: 'Agree' },
  { value: 6, label: 'Strongly Agree' },
]

api.steps.append(practiceItems.map((item) => ({ ...item })))

const selected = ref(null)
// after a response is submitted we stay on the trial and show the explanation:
// the erroneous step(s) highlighted in the trace plus the feedback statement.
// The feedback never says whether the participant's own choice was right or
// wrong; it only explains what the right answer would be.
const feedbackShown = ref(false)
const feedbackBox = ref(null)

// ── read-before-answer lock (same as the main task) ─────────────────
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

watch(
  () => api.stepIndex,
  () => {
    selected.value = null
    feedbackShown.value = false
    startLock()
  },
  { immediate: true }
)

onBeforeUnmount(clearLockTimers)

if (!api.isTimerStarted()) {
  api.startTimer()
}

function selectResponse(value) {
  if (locked.value || feedbackShown.value) return
  selected.value = value
}

// the trace is rendered from index 1 (index 0 is the expression itself), so
// displayed line i corresponds to full-trace index i + 1
function isErrorStep(displayIndex) {
  return feedbackShown.value && api.stepData.error_steps.some((e) => e.trace_index === displayIndex + 1)
}

function errorNote(displayIndex) {
  const e = api.stepData.error_steps.find((e) => e.trace_index === displayIndex + 1)
  return e ? e.note : ''
}

function submit() {
  if (selected.value === null || feedbackShown.value) return
  api.stepData.response = selected.value
  api.stepData.responded_agree = selected.value >= 4
  api.stepData.correct_agree = api.stepData.statement_correct === true
  api.stepData.is_correct = api.stepData.responded_agree === api.stepData.correct_agree
  api.stepData.rt = api.elapsedTime()
  api.recordStep()
  feedbackShown.value = true
  nextTick(() => feedbackBox.value?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }))
}

function next() {
  if (api.isLastStep()) {
    api.saveData(true)
    api.goNextView()
    return
  }
  api.startTimer()
  api.goNextStep()
}

function autofill() {
  while (api.stepIndex < api.nSteps) {
    const value = api.faker.rchoice([1, 2, 3, 4, 5, 6])
    api.stepData.response = value
    api.stepData.rt = api.faker.rnorm(4000, 800)
    api.recordStep()
    if (api.goNextStep() === null) break
  }
  api.goNextView()
}
api.setAutofill(autofill)
</script>

<template>
  <ConstrainedTaskWindow
    variant="ghost"
    :responsiveUI="api.config.responsiveUI"
    :width="api.config.windowsizerRequest.width"
    :height="api.config.windowsizerRequest.height"
  >
    <div class="text-left w-full h-full overflow-y-auto px-2">
      <div class="flex items-center justify-between mb-1">
        <span class="text-xs font-semibold uppercase tracking-wide text-blue-600">Practice</span>
        <span class="text-xs text-muted-foreground">{{ api.stepIndex + 1 }} of {{ practiceItems.length }}</span>
      </div>
      <p class="text-muted-foreground mb-1">Expression given to {{ api.stepData.student_name }}:</p>
      <p class="text-2xl font-bold mb-5 font-mono">{{ api.stepData.expression }}</p>

      <p class="text-muted-foreground mb-2">
        Here is the final answer {{ api.stepData.student_name }} produced, along with their work:
      </p>
      <div class="font-mono text-base mb-5 space-y-1">
        <p
          v-for="(step, i) in api.stepData.trace.slice(1)"
          :key="i"
          class="rounded px-1 -mx-1 transition-colors"
          :class="isErrorStep(i) ? 'bg-amber-200' : ''"
        >
          = {{ step }}
          <span v-if="isErrorStep(i)" class="ml-2 text-sm italic text-amber-800 font-sans">
            &larr; {{ errorNote(i) }}
          </span>
        </p>
      </div>

      <div class="border border-yellow-300 bg-yellow-50 rounded-lg p-4 mb-5">
        <p class="italic text-yellow-800">{{ api.stepData.belief_statement }}</p>
      </div>

      <p class="font-semibold mb-3">
        How much do you agree that this is what the student believes?
        <span v-if="locked" class="ml-1 text-sm font-normal text-muted-foreground">
          (please read the work above: {{ remaining }}s)
        </span>
      </p>
      <div class="grid grid-cols-6 gap-2 mb-6">
        <button
          v-for="opt in LIKERT"
          :key="opt.value"
          type="button"
          :disabled="locked || feedbackShown"
          @click="selectResponse(opt.value)"
          class="flex flex-col items-center p-2 rounded border transition-colors text-center"
          :class="[
            selected === opt.value ? 'border-primary bg-primary/10' : 'border-gray-300',
            locked ? 'opacity-40 cursor-not-allowed' : '',
            !locked && !feedbackShown ? 'hover:bg-gray-50' : '',
          ]"
        >
          <span
            class="w-4 h-4 rounded-full border-2 mb-2"
            :class="selected === opt.value ? 'border-primary bg-primary' : 'border-gray-400'"
          />
          <span class="text-xs">{{ opt.label }}</span>
        </button>
      </div>

      <div v-if="feedbackShown" ref="feedbackBox" class="border border-blue-300 bg-blue-50 rounded-lg p-4 mb-6">
        <p class="text-blue-900">{{ api.stepData.feedback }}</p>
      </div>

      <div class="flex justify-end mb-4">
        <Button v-if="!feedbackShown" :disabled="selected === null || locked" @click="submit()">Submit</Button>
        <Button v-else @click="next()">
          {{ api.isLastStep() ? 'Start the task' : 'Next practice question' }}
          <i-fa6-solid-arrow-right />
        </Button>
      </div>
    </div>
  </ConstrainedTaskWindow>
</template>
