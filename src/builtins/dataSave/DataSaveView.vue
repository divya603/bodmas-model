<script setup>
import { onMounted, ref } from 'vue'
import { Progress } from '@/uikit/components/ui/progress'
import useViewAPI from '@/core/composables/useViewAPI'

const api = useViewAPI()
const uploadProgress = ref(0)

onMounted(async () => {
  const startTime = Date.now()
  const animateProgress = () => {
    const t = Date.now() - startTime
    uploadProgress.value = Math.round(99 * (1 - Math.exp(-t / 4000)))
    if (uploadProgress.value < 99) requestAnimationFrame(animateProgress)
  }
  requestAnimationFrame(animateProgress)

  await api.saveData(true)

  uploadProgress.value = 100
  setTimeout(() => api.goNextView(), 400)
})
</script>

<template>
  <div class="w-full h-screen flex flex-col items-center justify-center">
    <div class="w-4/5 max-w-md text-center">
      <h1 class="text-3xl font-bold mb-4">Saving Your Data</h1>
      <p class="text-lg text-muted-foreground mb-8">Do not close your browser window yet!</p>
      <Progress :model-value="uploadProgress" class="h-3 mb-4" />
      <p class="text-sm text-muted-foreground">{{ uploadProgress }}%</p>
    </div>
  </div>
</template>
