import { onBeforeUnmount } from 'vue'

/**
 * Lightweight mouse-movement recorder for bot detection.
 *
 * While active, it samples the pointer position at most once per `sampleMs`
 * and buffers `{ x, y, t }` points (t = ms since the current trial's tracking
 * started). Human cursor motion is curved and jittery; many automated/AI
 * agents either don't move the mouse at all or move in near-perfect straight
 * lines, so the raw path per trial is enough to flag suspicious sessions
 * offline. Kept deliberately minimal — throttled sampling and a hard cap on
 * points keep the recorded data to a few KB per participant.
 *
 * Usage in a View:
 *   const mouse = useMouseTracking()
 *   mouse.start()                          // begin the first trial
 *   // on submit:
 *   api.stepData.mouse = mouse.getPoints() // save this trial's path
 *   mouse.reset()                          // start fresh for the next trial
 */
export function useMouseTracking({ sampleMs = 50, maxPoints = 2000 } = {}) {
  let points = []
  let startT = 0
  let lastT = 0
  let active = false

  function onMove(e) {
    if (!active) return
    const now = performance.now()
    if (now - lastT < sampleMs) return
    lastT = now
    if (points.length < maxPoints) {
      points.push({
        x: Math.round(e.clientX),
        y: Math.round(e.clientY),
        t: Math.round(now - startT),
      })
    }
  }

  window.addEventListener('mousemove', onMove, { passive: true })
  onBeforeUnmount(() => window.removeEventListener('mousemove', onMove))

  function reset() {
    points = []
    startT = performance.now()
    lastT = 0
  }

  return {
    start() {
      reset()
      active = true
    },
    stop() {
      active = false
    },
    reset,
    getPoints() {
      return points
    },
  }
}
