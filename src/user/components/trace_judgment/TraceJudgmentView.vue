<script setup>
import { ref } from 'vue'
import useViewAPI from '@/core/composables/useViewAPI'
import { Button } from '@/uikit/components/ui/button'
import { ConstrainedTaskWindow } from '@/uikit/layouts'
import pool from '@/user/data/stimulus_pool.json'
import { sampleForm, randomSeed } from '@/user/utils/sampleForm'

const api = useViewAPI()

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

const selected = ref(null)

if (!api.isTimerStarted()) {
  api.startTimer()
}

function selectResponse(value) {
  selected.value = value
}

function submit() {
  if (selected.value === null) return
  api.stepData.response = selected.value
  api.stepData.rt = api.elapsedTime()
  api.recordStep()
  selected.value = null
  api.startTimer()
  api.goNextStep()
}

function autofill() {
  while (api.stepIndex < api.nSteps) {
    if (api.path[0] !== 'summary') {
      api.stepData.response = api.faker.rchoice([1, 2, 3, 4, 5, 6])
      api.stepData.rt = api.faker.rnorm(4000, 800)
    }
    api.recordStep()
    api.goNextStep()
  }
}
api.setAutofill(autofill)

function finish() {
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

      <p class="font-semibold mb-3">How much do you agree that this is what the student believes?</p>
      <div class="grid grid-cols-6 gap-2 mb-6">
        <button
          v-for="opt in LIKERT"
          :key="opt.value"
          type="button"
          @click="selectResponse(opt.value)"
          class="flex flex-col items-center p-2 rounded border transition-colors text-center"
          :class="selected === opt.value ? 'border-primary bg-primary/10' : 'border-gray-300 hover:bg-gray-50'"
        >
          <span
            class="w-4 h-4 rounded-full border-2 mb-2"
            :class="selected === opt.value ? 'border-primary bg-primary' : 'border-gray-400'"
          />
          <span class="text-xs">{{ opt.label }}</span>
        </button>
      </div>

      <div class="flex justify-end">
        <Button :disabled="selected === null" @click="submit()">Submit</Button>
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
