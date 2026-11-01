"""
Automated Compatibility Tests for Local ML Service

Verifies that the local ML microservice responds with the exact JSON shapes
and types expected by the React frontend (`src/App.jsx`).
"""

import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health_check():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert "models" in data


def test_generate_quiz():
    payload = {"materialId": "test_material_123", "topic": "Deadlock Avoidance"}
    res = client.post("/api/quiz", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "id" in data
    assert "materialId" in data
    assert data["materialId"] == "test_material_123"
    assert "questions" in data
    assert len(data["questions"]) == 5

    # Check question schema
    for q in data["questions"]:
        assert "topic" in q
        assert "question" in q
        assert "options" in q
        assert len(q["options"]) == 4
        assert "answer" in q
        assert 0 <= q["answer"] < 4
        assert "explanation" in q


def test_submit_quiz_and_bkt_update():
    # 1. Generate quiz
    gen_res = client.post("/api/quiz", json={"materialId": "test_mat_bkt"})
    quiz = gen_res.json()

    # 2. Submit answers: 3 correct, 2 incorrect
    answers = [
        quiz["questions"][0]["answer"],
        quiz["questions"][1]["answer"],
        999,
        quiz["questions"][3]["answer"],
        999,
    ]

    submit_payload = {
        "materialId": "test_mat_bkt",
        "title": "BKT Test Attempt",
        "questions": quiz["questions"],
        "answers": answers,
    }

    res = client.post("/api/quiz/submit", json=submit_payload)
    assert res.status_code == 200
    data = res.json()

    assert "attempt" in data
    assert "mastery" in data
    attempt = data["attempt"]
    assert attempt["correct"] == 3
    assert attempt["total"] == 5
    assert attempt["score"] == 60

    # Ensure mastery returns scores mapped to percentages for React UI
    for topic, info in data["mastery"].items():
        assert "score" in info
        assert "attempts" in info
        assert 0 <= info["score"] <= 100


if __name__ == "__main__":
    pytest.main(["-v", __file__])
