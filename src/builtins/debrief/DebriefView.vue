<script setup>
/**
 * @fileoverview DebriefView component for displaying study completion information
 * @description This component shows debrief text to participants and provides navigation
 * to the next view in the experiment flow.
 */

import useViewAPI from '@/core/composables/useViewAPI'
import { Button } from '@/uikit/components/ui/button'
import { ConstrainedPage, ConstrainedTaskWindow } from '@/uikit/layouts'

/**
 * Initialize the Smile API for navigation and configuration access
 * @type {import('@/core/composables/useViewAPI').default}
 */
const api = useViewAPI()

/**
 * Component props definition
 * @typedef {Object} Props
 * @property {Object} debriefText - The debrief text component to display
 */
const props = defineProps({
  debriefText: {
    type: Object,
    default: null,
  },
  debriefPdfUrl: {
    type: String,
    default: null,
  },
})

/**
 * Handles the completion of the debrief view
 * @description Advances to the next view in the experiment flow
 * @returns {void}
 */
function finish() {
  api.goNextView()
}
</script>

<template>
  <!-- PDF layout -->
  <ConstrainedPage
    v-if="debriefPdfUrl"
    :responsiveUI="api.config.responsiveUI"
    :width="api.config.windowsizerRequest.width"
    :height="api.config.windowsizerRequest.height"
  >
    <h1 class="text-2xl font-bold mb-3">What is this study about?</h1>
    <iframe
      :src="`${debriefPdfUrl}#toolbar=0&navpanes=0`"
      class="w-full rounded"
      :style="`height: ${api.config.windowsizerRequest.height - 120}px; border: none;`"
    />
    <div class="flex justify-end pt-3 px-1">
      <Button variant="default" @click="finish()">
        Continue
        <i-fa6-solid-arrow-right />
      </Button>
    </div>
  </ConstrainedPage>

  <!-- Text component layout -->
  <ConstrainedTaskWindow
    v-else
    variant="ghost"
    :responsiveUI="api.config.responsiveUI"
    :width="api.config.windowsizerRequest.width"
    :height="api.config.windowsizerRequest.height"
    class="p-8"
  >
    <div class="w-[80%] h-[80%]">
      <component :is="debriefText" />
      <hr class="border-border my-6" />
      <div class="flex justify-end mt-4">
        <Button variant="default" @click="finish()">
          next
          <i-fa6-solid-arrow-right />
        </Button>
      </div>
    </div>
  </ConstrainedTaskWindow>
</template>
