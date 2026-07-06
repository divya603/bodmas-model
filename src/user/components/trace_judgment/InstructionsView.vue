<script setup>
import useViewAPI from '@/core/composables/useViewAPI'
import { Button } from '@/uikit/components/ui/button'
import { ConstrainedTaskWindow } from '@/uikit/layouts'

const api = useViewAPI()

function finish() {
  api.goFirstStep()
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
    <div class="w-[80%] h-[80%] overflow-y-auto">
      <h1 class="text-2xl font-bold mb-4">
        <i-material-symbols-integration-instructions class="inline-block mr-2 text-3xl" /> Instructions
      </h1>

      <p class="text-left text-lg mb-4">
        In this task, you will see a math problem along with the step-by-step work a student produced while solving
        it.
      </p>

      <p class="text-left text-lg mb-4">
        Below the work, you'll see a statement about what the student <strong>believes</strong> about the order of
        operations &mdash; for example, that they think addition should always come before multiplication.
      </p>

      <p class="text-left text-lg mb-4">
        Your job is to judge <strong>how well that statement explains the work shown</strong>, using the student's
        steps as your evidence &mdash; not just whether their final answer happens to be right or wrong.
      </p>

      <p class="text-left text-lg mb-4">
        For each problem, rate how much you agree with the statement on a 6-point scale from
        <strong>Strongly Disagree</strong> to <strong>Strongly Agree</strong>.
      </p>

      <hr class="border-gray-300 my-4" />

      <div class="flex justify-end">
        <Button variant="default" @click="finish()">
          Start practice quiz
          <i-fa6-solid-arrow-right />
        </Button>
      </div>
    </div>
  </ConstrainedTaskWindow>
</template>
