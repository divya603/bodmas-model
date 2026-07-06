<script setup>
import { onMounted, onBeforeUnmount, ref } from 'vue'
import useViewAPI from '@/core/composables/useViewAPI'
import { ConstrainedTaskWindow } from '@/uikit/layouts'

const props = defineProps({
  siteKey: {
    type: String,
    required: true,
  },
})

const api = useViewAPI()
const captchaContainer = ref(null)

function onVerified() {
  api.goNextView()
}

onMounted(() => {
  window.__recaptchaCallback = () => {
    window.grecaptcha.render(captchaContainer.value, {
      sitekey: props.siteKey,
      callback: onVerified,
    })
  }

  const script = document.createElement('script')
  script.src = 'https://www.google.com/recaptcha/api.js?onload=__recaptchaCallback&render=explicit'
  script.async = true
  script.defer = true
  document.head.appendChild(script)
})

onBeforeUnmount(() => {
  delete window.__recaptchaCallback
  const script = document.querySelector('script[src*="recaptcha/api.js"]')
  if (script) script.remove()
})
</script>

<template>
  <ConstrainedTaskWindow
    variant="ghost"
    :responsiveUI="api.config.responsiveUI"
    :width="api.config.windowsizerRequest.width"
    :height="api.config.windowsizerRequest.height"
  >
    <div class="flex flex-col items-center justify-center gap-6 text-center">
      <h1 class="text-2xl font-bold">Before we begin</h1>
      <p class="text-muted-foreground">Please confirm you are human to continue.</p>
      <div ref="captchaContainer" />
    </div>
  </ConstrainedTaskWindow>
</template>
