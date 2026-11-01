# EchoPilot learner-model calibration

This folder contains only the Python code and artifacts used to fit and evaluate the app's Bayesian Knowledge Tracing (BKT) starting parameters, including an executed Jupyter notebook with cell outputs. It does not generate quizzes: Gemini remains the quiz writer.

## Contents

```text
ml_research/
├── data/
│   └── raw/skill_builder_data_corrected_collapsed.csv  # local only; ignored by Git
├── notebooks/
│   └── 01_assistments_bkt_calibration.ipynb            # runnable, cell-by-cell training walkthrough
├── results/assistments_bkt.json                        # aggregate data and metrics; no student IDs
├── src/
│   ├── __init__.py
│   └── bkt.py                                           # BKT forward updates and EM fitting
├── train_assistments_bkt.py                            # preprocessing, training, evaluation
└── requirements.txt                                    # NumPy, scikit-learn, and IPython kernel
```

The Express app reads the fitted values from `server/bkt-parameters.json` at startup. The raw CSV stays local and is excluded by the root `.gitignore`.

## Dataset

The model was calibrated using the corrected/collapsed **ASSISTments 2009–2010 Skill Builder** file from the [official ASSISTments dataset page](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010). The original linked files had duplicate records; the corrected file has one row per student-problem. When one problem has multiple skill tags, the corrected dataset preserves them as a composite label such as `skill1_skill2`. This run treats that complete label as one skill category instead of splitting the label and duplicating a response.

The dataset is historical math tutoring data. The `correct` field is 1 for correct on the first attempt and 0 for an incorrect first attempt or help request. It includes 346,860 valid responses from 4,217 students and 113 observed skill labels in this run. The original dataset page says the data is free to use and describes Skill Builder as mastery-learning problem sets; use the page URL when citing the dataset.

## Training and holdout method

Open `ml_research/notebooks/01_assistments_bkt_calibration.ipynb` in Jupyter or VS Code and run its cells from top to bottom. It contains the complete preprocessing, student split, model fit, holdout evaluation, and artifact export, with outputs saved from the latest run.

For a command-line repeat of the same steps, from the project root install the Python dependencies and run:

```bash
python -m pip install -r ml_research/requirements.txt
python ml_research/train_assistments_bkt.py
```

The script:

1. Reads the corrected CSV using its legacy-compatible encoding, keeping valid binary `correct` outcomes with a student ID and skill label.
2. Groups rows by student and skill label, then orders each response sequence by `order_id` (the dataset's chronological problem-log order).
3. Uses seed 42 to split students 80/20. No student appears on both sides: 3,374 train students and 843 held-out students.
4. Fits one shared four-parameter BKT model with Baum-Welch Expectation-Maximization, using a deterministic random sample of up to 6,000 training student-skill sequences from 31,322 available sequences. The held-out evaluation uses all 7,958 sequences for the 843 held-out students.
5. Compares against the app's former settings (`P(L0)=0.30`, `P(T)=0.15`, `P(G)=0.25`, `P(S)=0.10`).
6. Always writes aggregate experiment details to `results/assistments_bkt.json`. It updates `server/bkt-parameters.json` only when the holdout Brier score improves and log loss is no worse than the former settings.

To use another local CSV or result location, pass `--data` and `--output`. The script does not export student identifiers or raw answer rows.

## Results

The reported metrics use 72,252 held-out responses:

| Metric | Former settings | Fitted BKT | Change | Meaning |
|---|---:|---:|---:|---|
| Brier score | 0.221969 | **0.206817** | 6.83% lower | Mean squared error of predicted response probabilities; lower is better. |
| Log loss | 0.653067 | **0.603175** | 7.64% lower | Penalizes confident wrong predictions; lower is better. |
| AUC-ROC | 0.656065 | **0.659295** | +0.003230 | Ability to rank a correct response above an incorrect one; higher is better. |
| Expected Calibration Error (ECE) | 0.104846 | **0.016676** | 84.09% lower | Average gap between predicted confidence and observed correctness; lower is better. |

Fitted parameters are `P(L0)=0.7306`, `P(T)=0.0311`, `P(G)=0.3193`, and `P(S)=0.2313`. The probability metrics improved on held-out students, with the largest change in calibration (ECE); AUC improved modestly. This supports using the fitted values as a shared starting prior for this college-project app.

## How to explain the result responsibly

This is a held-out **response-prediction** comparison on old math tutoring records. It does not test whether EchoPilot's Gemini questions are correct, whether its quizzes improve learning, or whether the same parameters fit other school subjects. The app continues to update each learner's per-material/topic BKT state from their own quiz answers. Existing stored `pKnown` values are not reset when the shared starting parameters change. Use the official dataset page as the source and describe these limits in presentations.
