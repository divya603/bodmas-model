export const QUIZ_QUESTIONS = [
  {
    id: 'pg1',
    questions: [
      {
        id: 'q1',
        question: 'What should your rating be based on?',
        multiSelect: false,
        answers: [
          'Whether the final answer is correct',
          "How well the statement explains the student's step-by-step work",
          'How many steps the student used',
        ],
        correctAnswer: ["How well the statement explains the student's step-by-step work"],
      },
      {
        id: 'q2',
        question:
          "A student's work shows they multiplied before adding, exactly as they should have. The statement says " +
          '"the student believes addition should come before multiplication." How much should you agree?',
        multiSelect: false,
        answers: ['Strongly Agree', 'Strongly Disagree', 'Somewhat Agree'],
        correctAnswer: ['Strongly Disagree'],
      },
    ],
  },
]
