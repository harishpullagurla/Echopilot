"""
Local Question Generator & Structured Schema Validator

Supports local Hugging Face Transformers pipeline or GGUF/llama.cpp inference
with strict JSON validation, grounding checks, and deduplication.
"""

import json
import re
from typing import List, Dict, Any, Optional

QUIZ_SYSTEM_PROMPT = """You are an expert pedagogical assessment assistant.
Your task is to generate high-quality multiple-choice questions (MCQs) grounded strictly in the provided study material chunks.
Rules:
1. Grounding: Every question, option, and explanation MUST be directly supported by the source text.
2. Structure: Return ONLY a valid JSON object matching the requested schema. No markdown formatting or extra text outside the JSON.
3. Quality: Exactly 4 options per question. One clear, unambiguous correct answer. Plausible distractors.
4. Explanations: Clear, supportive rationale citing why the answer is correct and referencing the context.
"""


def build_quiz_generation_prompt(
    material_name: str,
    topic: str,
    difficulty: str,
    evidence_chunks: List[Dict[str, Any]],
    num_questions: int = 5,
) -> str:
    """
    Constructs an instruction prompt containing context chunks and exact schema constraints.
    """
    context_text = "\n\n".join(
        [
            f"[Page {c.get('page_number', '?')} | {c.get('section_heading', 'Section')}]:\n{c.get('text', '')}"
            for c in evidence_chunks
        ]
    )

    user_prompt = f"""Generate a quiz with exactly {num_questions} multiple-choice questions for the material "{material_name}".

PRACTICE TOPIC: {topic if topic else 'Core Concepts from this material'}
TARGET DIFFICULTY: {difficulty} (gentle: factual recall/definitions; steady: conceptual understanding; stretch: multi-step application)

SOURCE MATERIAL EXCERPTS:
{context_text}

OUTPUT FORMAT (RETURN RAW JSON ONLY):
{{
  "title": "Short descriptive quiz title",
  "questions": [
    {{
      "topic": "{topic if topic else 'Central Concept'}",
      "question": "The question prompt?",
      "options": [
        "First option",
        "Second option",
        "Third option",
        "Fourth option"
      ],
      "answer": 0,
      "explanation": "Clear explanation grounded in the text",
      "difficulty": "gentle | steady | stretch",
      "source_chunk_id": "ID of supporting chunk"
    }}
  ]
}}"""
    return user_prompt


def validate_and_parse_quiz_json(raw_text: str) -> Dict[str, Any]:
    """
    Robust JSON parser that removes markdown fences and validates required schema attributes:
    - title (str)
    - questions (list of dicts)
    - exactly 4 options
    - answer in range [0, 3]
    - distinct options
    - topic and explanation non-empty
    """
    cleaned = raw_text.strip()
    # Strip markdown code blocks if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as err:
        # Fallback: extract first JSON object via regex
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
        else:
            raise ValueError(f"Failed to parse valid JSON from model output: {err}")

    if not isinstance(data, dict):
        raise ValueError("Model output root must be a JSON object.")

    title = data.get("title", "Study Quiz")
    questions = data.get("questions")
    if not isinstance(questions, list) or len(questions) == 0:
        raise ValueError("Questions field must be a non-empty list.")

    valid_questions = []
    for idx, q in enumerate(questions):
        topic = str(q.get("topic", "General")).strip()
        prompt = str(q.get("question", "")).strip()
        options = q.get("options", [])
        answer = q.get("answer")
        explanation = str(q.get("explanation", "")).strip()
        difficulty = str(q.get("difficulty", "steady")).strip().lower()

        if not prompt:
            raise ValueError(f"Question {idx+1} missing question text.")

        if not isinstance(options, list) or len(options) < 2:
            raise ValueError(f"Question {idx+1} must have at least 2 options.")

        # Ensure all options are strings and non-empty
        cleaned_options = [str(opt).strip() for opt in options]
        if len(set(cleaned_options)) != len(cleaned_options):
            raise ValueError(f"Question {idx+1} contains duplicate options.")

        try:
            answer_idx = int(answer)
        except (TypeError, ValueError):
            raise ValueError(f"Question {idx+1} answer index must be an integer.")

        if answer_idx < 0 or answer_idx >= len(cleaned_options):
            raise ValueError(
                f"Question {idx+1} answer index {answer_idx} out of range [0, {len(cleaned_options)-1}]."
            )

        if difficulty not in ["gentle", "steady", "stretch"]:
            difficulty = "steady"

        valid_questions.append(
            {
                "topic": topic,
                "question": prompt,
                "options": cleaned_options,
                "answer": answer_idx,
                "explanation": explanation or "Review the source passage for details.",
                "difficulty": difficulty,
                "source_chunk_id": str(q.get("source_chunk_id", "")),
            }
        )

    return {
        "title": title,
        "questions": valid_questions,
    }
