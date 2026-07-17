/**
 * @file design.js
 * @description Configures the overall logic and timeline of the experiment.
 * The timeline defines the sequence of phases that the experiment goes through.
 * This file configures which phases occur in what order.
 *
 * Key documentation:
 * - Views: https://smile.gureckislab.org/views.html
 * - Timeline: https://smile.gureckislab.org/timeline.html
 * - Randomization: https://smile.gureckislab.org/randomization.html
 *
 * @module design
 */

import { processQuery, initService } from '@/core/utils/utils'

// 1. Import main built-in View components
import AdvertisementView from '@/builtins/advertisement/AdvertisementView.vue'
import MTurkRecruitView from '@/builtins/mturk/MTurkRecruitView.vue'
import InformedConsentView from '@/builtins/informedConsent/InformedConsentView.vue'
import DemographicSurveyView from '@/builtins/demographicSurvey/DemographicSurveyMinimalView.vue'
import InstructionsQuizView from '@/builtins/instructionsQuiz/InstructionsQuiz.vue'
import DebriefView from '@/builtins/debrief/DebriefView.vue'
import DataSaveView from '@/builtins/dataSave/DataSaveView.vue'
import ThanksView from '@/builtins/thanks/ThanksView.vue'
import WithdrawView from '@/builtins/withdraw/WithdrawView.vue'
import WindowSizerView from '@/builtins/windowSizer/WindowSizerView.vue'

// 2. Import user View components
import InstructionsView from '@/user/components/trace_judgment/InstructionsView.vue'
import PracticeView from '@/user/components/trace_judgment/PracticeView.vue'
import TraceJudgmentView from '@/user/components/trace_judgment/TraceJudgmentView.vue'
import StrategyQuestionView from '@/user/components/trace_judgment/StrategyQuestionView.vue'
import TaskFeedbackSurveyView from '@/user/components/TaskFeedbackSurveyView.vue'

// #3. Import smile API and timeline
import useAPI from '@/core/composables/useAPI'
const api = useAPI()

import Timeline from '@/core/timeline/Timeline'
const timeline = new Timeline(api)

// #4.  Set runtime configuration options
//      See http://smile.gureckislab.org/configuration.html#experiment-options-env
api.setRuntimeConfig('allowRepeats', false)

api.setRuntimeConfig('colorMode', 'light')
api.setRuntimeConfig('responsiveUI', true)

api.setRuntimeConfig('windowsizerRequest', { width: 800, height: 600 })
api.setRuntimeConfig('windowsizerAggressive', true)

api.setRuntimeConfig('anonymousMode', false)
api.setRuntimeConfig('labURL', 'https://codec-lab.github.io/')
api.setRuntimeConfig('brandLogoFn', 'universitylogo.png')

api.setRuntimeConfig('maxWrites', 1000)
api.setRuntimeConfig('minWriteInterval', 2000)
api.setRuntimeConfig('autoSave', true)

api.setRuntimeConfig('payrate', '$15USD/hour prorated for estimated completition time + performance related bonus')
api.setRuntimeConfig('estimated_time', '30-40 minutes')

// Consent PDF
api.setRuntimeConfig('consentPdfUrl', `${import.meta.env.BASE_URL}consent-form.pdf`)

// Debrief PDF
api.setRuntimeConfig('debriefPdfUrl', `${import.meta.env.BASE_URL}debrief.pdf`)

// #5. Add between-subjects condition assignment
// This is where you can define conditions to which each participant should be assigned

// You can assign conditions by passing a javascript object to api.randomAssignCondition(),
// where the key is the condition name and the value is an array of possible condition values.
// Each unique condition manipulation should be assigned via a separate call to setConditions.

// EXAMPLE: set a between-subjects condition called taskOrder (AB or BA)
// api.randomAssignCondition({
//   taskOrder: ['AB', 'BA'],
// })

// you can also optionally set randomization weights for each condition. For
// example, if you want twice as many participants to be assigned to instructions
// version 1 compared to versions 2 and 3, you can set the weights as follows:
api.randomAssignCondition({
  instructionsVersion: ['1', '2', '3'],
  weights: [2, 1, 1], // weights are automatically normalized, so [4, 2, 2] would be the same
})

// #6. Define and add some routes to the timeline
// Each route should map to a View component.
// Each needs a name
// but for most experiments they go in sequence from the begining
// to the end of this list

// by default routes have meta.requiresConsent = true (unless you manually override it)
// by default routes have meta.requiresDone = false (unless you manually override it)

// IMPORTANT: A least one route needs to be called 'welcome_anonymous'
// to handle the landing case for someone not coming from a recruitment service

// First welcome screen for non-referral
timeline.pushSeqView({
  path: '/welcome',
  name: 'welcome_anonymous',
  component: AdvertisementView,
  meta: {
    prev: undefined,
    next: 'consent',
    allowAlways: true,
    requiresConsent: false,
  }, // override what is next
  beforeEnter: (to) => {
    api.getBrowserFingerprint()
  },
})

// welcome screen for referral from a service (e.g., prolific)
timeline.pushSeqView({
  path: '/welcome/:service',
  name: 'welcome_referred',
  component: AdvertisementView,
  meta: {
    prev: undefined,
    next: 'consent',
    allowAlways: true,
    requiresConsent: false,
  },
  beforeEnter: (to) => {
    // handle any service-specific initialization before processing URL params
    if (initService(to.params.service) === false) return false
    // processes info to get the service-specific
    // participant info (e.g., Profilic ID)
    processQuery(to.query, to.params.service)
    api.getBrowserFingerprint()
  },
})

// this is a the special page that loads in the iframe on mturk.com
timeline.registerView({
  name: 'mturk',
  component: MTurkRecruitView,
  props: {
    estimated_time: api.getConfig('estimated_time'),
    payrate: api.getConfig('payrate'),
  },
  meta: { allowAlways: true, requiresConsent: false },
  beforeEnter: (to) => {
    processQuery(to.query, 'mturk')
  },
})

// consent
timeline.pushSeqView({
  name: 'consent',
  component: InformedConsentView,
  props: {
    consentPdfUrl: api.getConfig('consentPdfUrl'),
  },
  meta: {
    requiresConsent: false,
    setConsented: true,
  },
})

// windowsizer
timeline.pushSeqView({
  name: 'windowsizer',
  component: WindowSizerView,
})

// instructions
timeline.pushSeqView({
  name: 'instructions',
  component: InstructionsView,
})

// import the quiz questions
import { QUIZ_QUESTIONS } from './components/quizQuestions'
// instructions quiz
timeline.pushSeqView({
  name: 'quiz',
  component: InstructionsQuizView,
  props: {
    questions: QUIZ_QUESTIONS,
    returnTo: 'instructions',
    randomizeQandA: true,
  },
})

// practice trials: 3 fixed items answered like real trials, each followed by
// feedback (erroneous steps highlighted + the right answer explained). Not
// scored toward the bonus.
timeline.pushSeqView({
  name: 'practice',
  component: PracticeView,
})

// main experiment
// note: by default, the path will be set to the name of the view
// however, you can override this by setting the path explicitly
timeline.pushSeqView({
  name: 'exp',
  path: '/experiment',
  component: TraceJudgmentView,
})

// strategy free-response (its own screen, right after the task)
timeline.pushSeqView({
  name: 'strategy',
  component: StrategyQuestionView,
})

// feedback survey
timeline.pushSeqView({
  name: 'feedback',
  component: TaskFeedbackSurveyView,
})

// demographic survey
timeline.pushSeqView({
  name: 'demograph',
  component: DemographicSurveyView,
  meta: { setDone: true },
})

// save data
timeline.pushSeqView({
  name: 'save',
  component: DataSaveView,
  meta: { requiresDone: true },
})

// debriefing form
timeline.pushSeqView({
  name: 'debrief',
  component: DebriefView,
  props: {
    debriefPdfUrl: api.getConfig('debriefPdfUrl'),
  },
})

// thanks/submit page
timeline.pushSeqView({
  name: 'thanks',
  component: ThanksView,
  meta: {
    requiresDone: true,
    resetApp: api.getConfig('allowRepeats'),
  },
})

// this is a special page that is for a withdraw
timeline.registerView({
  name: 'withdraw',
  meta: {
    requiresWithdraw: true,
    resetApp: api.getConfig('allowRepeats'),
  },
  component: WithdrawView,
})

timeline.build()

export default timeline
