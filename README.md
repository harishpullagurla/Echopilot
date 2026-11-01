# EchoPilot

EchoPilot is a college-project study tutor built around a student's own PDFs. Students upload materials, take material-specific quizzes, get answer feedback, and see per-topic progress adapt over time.

## Run locally

1. Install Node.js 20 or newer.
2. In this folder, run `npm install`.
3. Copy `.env.example` to `.env` and set `GEMINI_API_KEY`. The key is used only by the server for quiz generation.
4. Run `npm run dev` and open the Vite URL shown in the terminal. The API uses `PORT` from `.env` (default `3001`; this local setup currently uses `3002`).

The first material indexing run downloads the configured small ONNX models from Hugging Face and caches them under `server/data/model-cache/`. Set `LOCAL_EMBEDDING_MODEL`, `LOCAL_RERANKER_MODEL`, or `LOCAL_NLI_MODEL` to change the defaults. Local inference can then work offline. `GEMINI_MODEL` defaults to `gemini-2.5-flash` and is used only to author quizzes.

## AI flow

- `pdf-parse` extracts readable text from each uploaded PDF.
- The backend chunks text and creates normalized 384-dimensional local embeddings with `Xenova/bge-small-en-v1.5` using Transformers.js/ONNX.
- Passage retrieval combines semantic similarity and BM25 keyword relevance with reciprocal rank fusion (RRF), then reranks the strongest candidates with the local `Xenova/ms-marco-MiniLM-L-6-v2` cross-encoder.
- Per-topic Bayesian Knowledge Tracing (BKT) updates a probability of mastery after every answer. The next topic is selected by expected information gain, spaced-review due-ness, and recent mistakes rather than a fixed mastery cutoff. Gemini receives mastery and recent accuracy as continuous signals for question difficulty.
- Gemini creates a structured five-question quiz from the retrieved PDF excerpts. The server validates structure, options, answer-index bounds, and topic focus; a local MobileBERT NLI model checks whether each answer/explanation appears supported by its selected passage.
- Answers are scored deterministically against the quiz key. Quiz history, evidence quotes, source-check results, and per-material progress are saved locally.

The evidence model is an advisory cross-check, not a guarantee of correctness. If a model download or local model inference fails, retrieval falls back to BM25 and evidence checks are marked unavailable. Without a Gemini key, built-in demo questions can be shown; they are not PDF-grounded quizzes.

## Local data and limits

PDFs are stored in `server/data/uploads/`, learner state in `server/data/db.json`, local model weights in `server/data/model-cache/`, and passage vectors in `server/data/embeddings/`. PDF text stays local for embedding, retrieval, and evidence checks; retrieved passages are sent to Google as context for Gemini quiz generation. This is a single-user local project, not a production or multi-user service. Scanned/image-only PDF OCR and model fine-tuning are not included.

Retrieval quality can be measured with the synthetic labeled cases in `server/evaluation/retrieval-cases.json` and `npm run evaluate:retrieval`; the report includes Recall@7 and MRR. The adaptive BKT parameters are still project defaults and have not been calibrated on a real learner dataset.
