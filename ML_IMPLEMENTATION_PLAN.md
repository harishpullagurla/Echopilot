# EchoPilot: current AI and adaptive-learning implementation

## 1. What this document describes

This is the current college-project implementation, not a proposed transition to an offline language model. Gemini remains responsible for writing quizzes. Local models handle passage search and evidence checks. A public student-response dataset was used to fit the shared starting values for the app's Bayesian Knowledge Tracing (BKT) learner model.

## 2. Current AI roles

| App task | Method | Current role |
|---|---|---|
| Quiz generation | Gemini API | Writes five questions and answer keys from selected passages of the chosen PDF. It is not trained by the dataset discussed below. |
| Passage search | Local BGE embeddings, BM25, reciprocal-rank fusion, and a MiniLM cross-encoder | Finds candidate passages from the learner's selected PDF. |
| Source-support hint | Local MobileBERT NLI classifier | Estimates whether a passage supports the answer and explanation; advisory only. |
| Learner modeling | Bayesian Knowledge Tracing (BKT) | Updates the probability that the learner knows a topic after each correct or incorrect answer. |
| Topic selection | Expected information gain, review due-ness, recent errors, and mastery | Selects a stored topic to practice; this is a transparent heuristic, not a trained reinforcement-learning policy. |

The app saves separate progress for each material and topic. The shared BKT parameters provide a starting population prior; a learner's answer history then updates that learner's `pKnown` state. Existing saved `pKnown` values are preserved when shared parameters change.

## 3. Public dataset used for BKT calibration

**Dataset:** Corrected/collapsed ASSISTments 2009–2010 Skill Builder dataset, from the [official ASSISTments page](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010).

The dataset page describes Skill Builder as math mastery-learning problem sets and says the data is free to use. The corrected file is important because the original linked files contained duplicate records. The corrected file stores one row per student-problem; if a problem has multiple skill tags, the tags are kept in a composite label such as `skill1_skill2`. The training script treats this whole label as a group, avoiding response duplication. The `correct` target means correct on the first attempt (`1`) versus incorrect on the first attempt or help requested (`0`).

### Preprocessing and split

1. Read the corrected CSV while preserving its legacy text encoding.
2. Keep rows with a student ID, a skill label, and a binary correctness outcome.
3. Group responses by student and skill label, and order each sequence by `order_id`, which the dataset describes as chronological problem-log order.
4. Split by student with fixed seed 42: 3,374 students for training and 843 for holdout. The same student cannot appear in both splits.
5. Fit on a deterministic sample of 6,000 student-skill sequences from 31,322 available training sequences.
6. Evaluate the model on all 7,958 held-out sequences (72,252 responses).

This run processed 346,860 valid interactions from 4,217 students and 113 observed skill labels. The raw CSV remains in ignored `ml_research/data/raw/`; results contain aggregate values only, no student IDs or response rows.

## 4. How BKT is fit and applied

BKT represents knowledge of one skill as a hidden state: known or not yet known. It has four probabilities:

- `P(L0)`: probability the learner knows the topic before practice.
- `P(T)`: probability of learning after an opportunity, if not already known.
- `P(G)`: chance of a correct answer while not knowing the topic.
- `P(S)`: chance of an incorrect answer while knowing the topic.

The trainer uses Baum-Welch Expectation-Maximization (EM). The expectation step estimates the hidden known/not-known state probabilities that best explain each response sequence. The maximization step updates the four probabilities. This repeats until the fit converges or reaches its iteration limit.

At app runtime, EchoPilot uses the response and the guess/slip probabilities to update a topic's `pKnown`, then applies the learning transition. Its topic scheduler combines expected information gain, due review, recent wrong answers, and current mastery. Gemini receives mastery and recent accuracy as guidance for question difficulty, but still writes the quiz.

## 5. Holdout metrics and measured change

The baseline was the app's former fixed values (`P(L0)=0.30`, `P(T)=0.15`, `P(G)=0.25`, `P(S)=0.10`). The fitted shared values are `P(L0)=0.7306`, `P(T)=0.0311`, `P(G)=0.3193`, and `P(S)=0.2313`.

| Metric | Former values | Fitted BKT | Change | What it measures |
|---|---:|---:|---:|---|
| Brier score | 0.221969 | **0.206817** | 6.83% lower | Squared error between predicted correctness probability and actual result. Lower is better. |
| Log loss | 0.653067 | **0.603175** | 7.64% lower | Penalty for assigning low probability to the result that occurred; punishes confident mistakes. Lower is better. |
| AUC-ROC | 0.656065 | **0.659295** | +0.003230 | How well response probabilities rank correct outcomes above incorrect outcomes. Higher is better. |
| ECE | 0.104846 | **0.016676** | 84.09% lower | Average difference between predicted confidence and observed accuracy across probability bins. Lower is better. |

All four metrics improved on held-out students. The strongest gain was calibration (ECE), and ranking improved modestly (AUC). The updated parameters are saved in `server/bkt-parameters.json`; the detailed run is in `ml_research/results/assistments_bkt.json`.

### Limits of this evaluation

This measures next-response prediction on historical math tutoring records. It does not measure correctness of Gemini-generated quizzes, learning gains, or how well the prior fits non-math subjects. ASSISTments marks help requests as incorrect, and its mastery-learning workflow differs from EchoPilot's quiz flow. Treat the fitted values as a useful project starting prior, not proof of educational effectiveness. A future evaluation should collect consented EchoPilot response data and compare predictions on held-out learners, then separately study whether the schedule helps learning.

## 6. Reproduce the experiment

From the project root, with the corrected CSV placed in the ignored raw-data directory:

```bash
python -m pip install -r ml_research/requirements.txt
python ml_research/train_assistments_bkt.py
```

The same complete workflow is available cell by cell in `ml_research/notebooks/01_assistments_bkt_calibration.ipynb`; its saved outputs show the latest executed run.

The script writes aggregate metrics to `ml_research/results/assistments_bkt.json` and updates the runtime parameter file only when held-out Brier score improves and log loss is no worse than the previous app baseline. Raw learner records stay local and should not be pushed to GitHub.
