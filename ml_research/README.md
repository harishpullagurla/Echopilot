# EchoPilot: Machine Learning Research & Local Model Transition Suite

This folder contains the complete, self-contained research and development implementation designed to transition EchoPilot from cloud API calls (Gemini) to **100% offline, local open-source machine learning models**.

> **Important**: This research suite is completely decoupled from the production Express/React application. The active web application continues running on the Gemini API without modification.

---

## 1. Machine Learning Architecture Overview

| Task / Domain | Proposed Local ML Model | Role & Function | Key Research Justification |
|---|---|---|---|
| **Semantic Retrieval & Indexing** | `sentence-transformers/all-MiniLM-L6-v2` | Extracts 384-dimensional dense vectors from PDF text chunks; runs cosine similarity with strict material-level isolation. | 22M parameters, low memory footprint, runs in sub-10ms on CPU, high Recall@K on academic passages. |
| **Grounded Question Generation** | `Qwen/Qwen2.5-3B-Instruct` (+ 4-bit QLoRA adapter) | Turns retrieved evidence passages, target skill, and desired difficulty into structured 4-option multiple-choice questions. | State-of-the-art open weights, Apache-2.0 license, fits in <3 GB VRAM with 4-bit NF4 quantization, robust ChatML formatting. |
| **Knowledge Tracing (Mastery Tracking)** | **Bayesian Knowledge Tracing (BKT)** (Corbett & Anderson HMM) | Models latent cognitive mastery $P(L_t \in \{0, 1\})$ over sequential quiz attempts with parameters $P(L_0), P(T), P(G), P(S)$. | Mathematically interpretable, provably calibrates on small student cohort sizes (unlike uninterpretable black-box deep nets), estimates real posterior probabilities. |
| **Benchmark Baseline** | **Deep Knowledge Tracing (DKT)** (LSTM) | PyTorch recurrent sequence baseline comparing BKT against neural sequence learning. | Used for comparative empirical research; benchmarks show BKT yields superior calibration on sparse student records. |

---

## 2. Directory Structure

```
ml_research/
├── notebooks/
│   ├── 01_document_parsing_and_chunking.ipynb      # PDF text extraction, page attribution, semantic sliding chunker
│   ├── 02_semantic_retrieval_and_indexing.ipynb    # Local vector store, cosine similarity, Recall@K & MRR@K evaluation
│   ├── 03_qlora_question_generation_finetuning.ipynb # QLoRA SFT training pipeline for Qwen2.5-3B, ChatML dataset formatting
│   ├── 04_knowledge_tracing_bkt_and_dkt.ipynb      # BKT Baum-Welch EM fitting, forward updates, DKT LSTM comparison
│   └── 05_end_to_end_local_pipeline_eval.ipynb     # Complete offline simulation (Upload -> Chunk -> Retrieve -> QGen -> BKT Update)
├── src/
│   ├── chunker.py                                  # Page-aware text extraction and sliding window chunking
│   ├── retriever.py                                # MaterialVectorStore with dense embeddings & evaluation metrics
│   ├── generator.py                                # Prompt templates, schema validation, and JSON output parsing
│   ├── bkt.py                                      # Production BKT engine with Baum-Welch EM and forward updating
│   └── dkt.py                                      # PyTorch Deep Knowledge Tracing (LSTM) model
├── data/
│   ├── sample_training_data.jsonl                  # Curated gold standard MCQ items with evidence spans & skills
│   ├── sample_interactions.csv                     # 1,200+ simulated student response interactions across 3 skills
│   └── retrieval_benchmarks.json                   # Ground-truth evaluation queries for measuring Recall@K & MRR@K
├── service/
│   ├── app.py                                      # FastAPI drop-in microservice matching Express API schemas
│   └── test_service.py                             # Automated pytest suite confirming frontend compatibility
└── requirements.txt                                # Python package dependencies
```

---

## 3. The 5 Research Notebooks

1. **`01_document_parsing_and_chunking.ipynb`**:
   - Inspects PDF uploads (`Deadlock.pdf`, `Slide 1 -Sets and Relations .pdf`).
   - Extracts page-level text, removes non-informative characters, and segments text into 350-word semantic chunks with 50-word overlap.
   - Exports metadata-tagged chunks (`page_number`, `section_heading`, `word_count`, `material_id`) ready for indexing.

2. **`02_semantic_retrieval_and_indexing.ipynb`**:
   - Encodes chunks into 384-d normalized vectors using `sentence-transformers/all-MiniLM-L6-v2`.
   - Enforces strict material-level isolation so that queries for one PDF cannot retrieve chunks from another.
   - Evaluates Recall@K and MRR@K on benchmark queries.

3. **`03_qlora_question_generation_finetuning.ipynb`**:
   - Prepares ChatML instruction datasets from reviewed multiple-choice items.
   - Sets up `bitsandbytes` 4-bit NF4 double quantization and `peft.LoraConfig` targeting attention and MLP projections.
   - Outlines `TRL` `SFTTrainer` execution and verifies JSON output schema adherence.

4. **`04_knowledge_tracing_bkt_and_dkt.ipynb`**:
   - Implements Baum-Welch Expectation-Maximization to learn parameters $P(L_0), P(T), P(G), P(S)$ from student interaction logs.
   - Traces question-by-question forward Bayesian belief updates.
   - Trains a PyTorch Deep Knowledge Tracing (DKT) LSTM model and compares AUC-ROC, Brier score, and calibration.

5. **`05_end_to_end_local_pipeline_eval.ipynb`**:
   - Simulates the entire offline pipeline end-to-end:
     `PDF Ingestion -> Chunking -> Vector Indexing -> Weak-Skill Adaptive Policy -> Evidence Retrieval -> Structured QGen -> Simulated Quiz Submission -> BKT Mastery Update`.

---

## 4. How to Run the Local ML Microservice (FastAPI)

When ready to test EchoPilot locally without the Gemini API:

```bash
# 1. Start the local ML microservice (runs on port 8000)
python ml_research/service/app.py

# 2. Run the automated compatibility test suite
python ml_research/service/test_service.py
```

This service provides the identical API contract (`/api/health`, `/api/quiz`, `/api/quiz/submit`) used by the React frontend.
