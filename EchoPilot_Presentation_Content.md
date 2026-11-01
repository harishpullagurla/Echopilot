# EchoPilot — presentation content

**Suggested length:** 7–8 minutes
**Audience:** College project presentation
**Core message:** EchoPilot turns a student's own PDF into practice, tracks topic-level progress, and uses previous answers to guide later practice.

## Slide 1 — EchoPilot: Your study material, your learning path

**Put on the slide**

- AI-powered personal learning tutor
- Upload a PDF → take a quiz → see progress → practice again
- Quiz generation: Gemini API

**Say:** “EchoPilot helps students practice from their own study files. It records how they do by topic and uses that history to guide a later quiz.”

**Visual:** App screenshot with a material selected, or a simple PDF-to-quiz illustration.

## Slide 2 — The learning problem

**Put on the slide**

- Students have notes, slides, and PDFs, but need focused practice
- A fixed quiz can repeat strengths and overlook weak topics
- Progress should carry from one quiz to the next

**Say:** “The project focuses on turning existing course material into practice and making the next practice session respond to the learner's previous answers.”

**Visual:** Three simple labels: *Study material → Practice → Targeted review*.

## Slide 3 — What a student does

**Put on the slide**

1. Upload and select a PDF
2. Start a quiz for that material
3. Answer five multiple-choice questions
4. Review score, explanations, and source excerpts
5. Return to see quizzes and progress for that material

**Say:** “The flow stays centered on materials, quizzes, and progress. Quiz history and mastery are kept separately for each uploaded material.”

**Visual:** Use three actual app screenshots: Materials, Quiz, Progress.

## Slide 4 — How the AI parts work together

**Put on the slide**

```text
PDF → extract text → retrieve relevant passages → Gemini writes quiz
                                      ↓
Student answers → deterministic grading → BKT progress update
                                      ↓
                              choose next topic
```

- Local retrieval: embeddings + BM25 + reranking
- Gemini: writes quiz questions from selected excerpts
- Server: grades against the answer key and saves progress

**Say:** “Gemini writes the questions. Local models help find useful passages and check whether the source appears to support an answer. The server grades by comparing the selected option to the saved answer index.”

**Visual:** Keep the flow diagram large; use one color for local steps and another for Gemini.

## Slide 5 — How progress adapts

**Put on the slide**

- BKT estimates the probability that the learner knows a topic
- Correct and incorrect answers update that probability
- Guessing and careless mistakes are represented by `P(G)` and `P(S)`
- Next-topic priority combines uncertainty, review timing, recent errors, and mastery

**Say:** “Bayesian Knowledge Tracing, or BKT, is the learner model. It treats knowledge as an estimate that can change after each answer. The topic scheduler then looks at that estimate alongside recent mistakes and whether review is due.”

**Visual:** Show a simple line of topic mastery changing after a correct answer and an incorrect answer. Label it “illustrative estimate,” not a grade.

## Slide 6 — Dataset and training

**Put on the slide**

- Dataset: corrected/collapsed ASSISTments 2009–2010 Skill Builder
- Historical math mastery-learning responses
- 346,860 valid interactions · 4,217 students · 113 skill labels
- 80/20 split by student: 3,374 train · 843 holdout
- Baum-Welch EM fits shared BKT parameters on 6,000 training sequences

**Say:** “The public dataset was used only to fit the learner model's shared starting values. Students were split before fitting, so no student is shared between training and holdout. Gemini quiz generation was not trained on this data.”

**Visual:** Simple dataset → student split → BKT parameter fit diagram. Cite the [official ASSISTments dataset page](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010) in small text on the slide.

## Slide 7 — Evaluation: what improved

**Put on the slide**

| Held-out metric | Previous settings | Fitted BKT | Result |
|---|---:|---:|---:|
| Brier score ↓ | 0.221969 | **0.206817** | 6.83% lower |
| Log loss ↓ | 0.653067 | **0.603175** | 7.64% lower |
| AUC-ROC ↑ | 0.656065 | **0.659295** | +0.003230 |
| Calibration error (ECE) ↓ | 0.104846 | **0.016676** | 84.09% lower |

- 72,252 held-out responses
- Lower Brier/log loss/ECE is better; higher AUC is better

**Say:** “All four response-prediction metrics improved on students the model did not train on. The largest improvement was calibration: the predicted confidence better matched observed correctness. This evaluates response prediction on historical math data, not learning gains.”

**Visual:** Use four paired horizontal bars. Keep the lower-is-better and higher-is-better arrows visible.

## Slide 8 — Limits and conclusion

**Put on the slide**

- Historical dataset is math-only and differs from EchoPilot quizzes
- No claim that the app improves learning outcomes yet
- Gemini can still generate an incorrect question or answer
- Next evaluation: representative PDFs, human review, and consented EchoPilot responses

**Say:** “The project demonstrates a complete learning loop and a measured improvement in BKT response prediction. It does not yet show that students learn more. The next step is evaluating quiz quality and adaptation with representative course material and learner data.”

**Closing line:** “EchoPilot makes practice from a student's own material and carries learning progress into what to practice next.”

## Likely questions

**Why use Gemini if the project has machine learning?**
Gemini writes the quiz. The app also uses local retrieval and a BKT learner model; this dataset trains only BKT's shared starting parameters.

**What do the metrics prove?**
They show better prediction and calibration of responses on held-out ASSISTments students. They do not prove better learning or better quiz correctness.

**Why split by student?**
It prevents the model from being evaluated on response histories from students it already saw during fitting, giving a more useful check on unseen learners.

**What are Brier score and ECE in plain terms?**
Brier score measures the average squared error in predicted response probabilities. ECE measures how far predicted confidence differs from actual correctness across confidence groups. Lower is better for both.

**Does the dataset contain students' actual EchoPilot answers?**
No. It is a public historical math dataset. EchoPilot stores its own progress locally, and those personal responses were not used for this fit.

## Presenter notes

- Use screenshots from the current app for Slides 1 and 3.
- Keep equations out of the main slides; explain BKT in words and put formulas in speaker notes only if asked.
- Do not call the holdout results “learning improvement.” Call them “response-prediction and calibration improvement on held-out ASSISTments students.”
- Put the raw dataset file only on the local machine. The tracked JSON result contains aggregate counts and metrics, not student IDs or answer rows.
