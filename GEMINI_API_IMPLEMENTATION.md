# EchoPilot AI and Gemini Implementation

## What the app does

EchoPilot lets a learner upload PDFs, take a five-question quiz grounded in one selected file, review feedback, and see the next quiz adapt to the learner's stored progress. The project is a local, single-user college app.

## Which parts use Gemini

Gemini is used only to write quiz questions from retrieved excerpts. The server sends the quiz prompt and selected source passages to Gemini `generateContent` using `GEMINI_MODEL` (default `gemini-2.5-flash`). The browser never receives the API key. Gemini does not create passage embeddings or grade submitted answers.

## AI and information-retrieval pipeline

1. **PDF extraction:** `pdf-parse` extracts text. Image-only documents are rejected with a readable error; OCR is not implemented.
2. **Passage splitting:** the backend creates passages up to about 1,200 characters with overlap for long blocks. This keeps context manageable while retaining nearby sentences.
3. **Local semantic embeddings:** Transformers.js runs `Xenova/bge-small-en-v1.5` locally through ONNX. It produces normalized 384-dimensional vectors for passages and search queries. The model weights are downloaded on first use and cached in `server/data/model-cache/`; the per-material vectors are cached in `server/data/embeddings/`. Existing quiz history and learner records are not cleared when an embedding cache is rebuilt.
4. **Hybrid retrieval:** BM25 ranks literal term matches. Dense cosine similarity ranks passages with related meaning. Reciprocal Rank Fusion combines these independent rankings without requiring their raw scores to share a scale.
5. **Cross-encoder reranking:** `Xenova/ms-marco-MiniLM-L-6-v2` scores query/passage pairs among the strongest candidates. This pair model reads both texts together and reorders results more precisely than embedding similarity alone. The best seven passages are sent to Gemini.
6. **Quiz generation and validation:** Gemini is told to use only the cited passages and create five distinct multiple-choice questions with one defensible key and four distinct options. The server validates schema, question/option uniqueness, answer indexes, difficulty labels, and target-topic consistency. Invalid drafts can be repaired by Gemini up to two times after the initial request.
7. **Evidence attachment and support check:** the local cross-encoder chooses a passage for each question, and the server attaches a quote taken directly from that PDF passage. `Xenova/mobilebert-uncased-mnli` compares the passage (premise) with a statement containing the keyed answer and explanation (hypothesis). Its entailment score is used for an advisory `supported` or `review suggested` label. If the model cannot run, the label is `not available`. This classifier is not a proof of correctness.
8. **Deterministic grading:** on submission, the server compares each selected option index with the quiz's answer index and saves the attempt, question details, evidence quote, and support-check status.

## Adaptive learning

Progress is maintained separately for each material and topic. The learner state uses Bayesian Knowledge Tracing (BKT), with initial knowledge 0.30, learning transition 0.15, guess 0.25, and slip 0.10. After each answer, BKT updates the estimated probability of knowing the topic. These are transparent project defaults, not values trained on EchoPilot users.

To choose a topic for the next quiz, EchoPilot estimates expected information gain: how much a new answer could reduce uncertainty in the current mastery estimate. It boosts topics whose review interval is due and topics with recent incorrect answers. Correct responses lengthen a simple per-topic review interval; incorrect responses reset it. This removes the old single mastery cutoff. The selected topic remains stable for that quiz. Gemini receives the mastery estimate and recent accuracy and is prompted to adjust recall/application/stretch level smoothly; it is still a generative model and may not follow the desired level perfectly.

The topic selector is an adaptive scheduling heuristic using learner-modeling and information-theoretic ideas. It does not train a policy, perform reinforcement learning, or guarantee optimal learning gains. BKT parameters and scheduling weights need calibration against real learner response data before making empirical claims.

## Data, privacy, and fallback behavior

- `server/data/uploads/` stores original PDFs.
- `server/data/db.json` stores extracted text, material metadata, attempts, answers, evidence, and mastery states.
- `server/data/embeddings/` stores per-material local vector indexes.
- `server/data/model-cache/` stores downloaded local model files. These and other runtime data are ignored by Git.
- Local embedding, retrieval, reranking, and NLI inference run on the computer after initial model download. Only retrieved quiz context and instructions are sent to Google for Gemini generation.
- If local embedding or reranking is unavailable, the server falls back to BM25 keyword ranking. If NLI is unavailable, evidence is marked unchecked. If Gemini is missing, the UI can show generic demo questions, which must not be treated as PDF-grounded quizzes.

## Retrieval evaluation

`server/evaluation/retrieval-cases.json` contains synthetic labeled passages and queries. Run `npm run evaluate:retrieval` to compare BM25 with the local hybrid retriever using Recall@7 and Mean Reciprocal Rank (MRR). The first evaluation may download the local embedding and reranker models. This small synthetic set is a smoke measure for ranking behavior, not evidence that retrieval is accurate for arbitrary course PDFs. For meaningful evaluation, create a held-out set from representative materials with human-labeled relevant passages.

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `GEMINI_API_KEY` | Server-only key for quiz generation | none |
| `GEMINI_MODEL` | Gemini quiz-generation model | `gemini-2.5-flash` |
| `LOCAL_EMBEDDING_MODEL` | Local passage/query embeddings | `Xenova/bge-small-en-v1.5` |
| `LOCAL_RERANKER_MODEL` | Local query/passage cross-encoder | `Xenova/ms-marco-MiniLM-L-6-v2` |
| `LOCAL_NLI_MODEL` | Local answer-support classifier | `Xenova/mobilebert-uncased-mnli` |
| `PORT` | Express server port | `3001` |

## Main code locations

- `server/index.js` — PDF upload/extraction, Gemini prompt and validation, adaptive topic selection, deterministic grading, persistence, and API routes.
- `server/ai.js` — local models, chunking/indexing, BM25 and hybrid retrieval, reranking, evidence checks, BKT, and adaptive scoring.
- `server/evaluate-retrieval.js` — synthetic ranking evaluation runner.
- `server/evaluation/retrieval-cases.json` — small synthetic labeled retrieval corpus.
- `src/App.jsx` — material-first quiz flow, progress view, and source-check feedback.
