"""Fit and evaluate EchoPilot's shared BKT prior on ASSISTments Skill Builder.

The source data stays local. The generated JSON contains model parameters and
aggregate metrics only; it never contains student IDs or interaction rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.bkt import BKTModel


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "ml_research" / "data" / "raw" / "skill_builder_data_corrected_collapsed.csv"
DEFAULT_RESULTS = ROOT / "ml_research" / "results" / "assistments_bkt.json"
DEFAULT_PARAMETERS = ROOT / "server" / "bkt-parameters.json"
APP_BASELINE = {"p_l0": 0.30, "p_t": 0.15, "p_g": 0.25, "p_s": 0.10}
DATASET_SOURCE = "ASSISTments 2009-2010 Skill Builder, corrected/collapsed version"
DATASET_URL = "https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010"


def load_sequences(path: Path) -> tuple[dict[str, list[list[int]]], int, int]:
    """Return user -> skill sequence list, ordered by the dataset's order_id."""
    grouped: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    unique_skills: set[str] = set()
    valid_rows = 0
    # The historical CSV contains non-UTF-8 bytes; latin-1 preserves all fields.
    with path.open("r", encoding="latin-1", newline="") as source:
        reader = csv.DictReader(source)
        required = {"user_id", "skill_name", "skill_id", "correct", "order_id"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

        for row in reader:
            user = (row.get("user_id") or "").strip()
            skill = (row.get("skill_name") or "").strip() or (row.get("skill_id") or "").strip()
            response = (row.get("correct") or "").strip()
            if not user or not skill or response not in {"0", "1"}:
                continue
            try:
                order = int(float(row.get("order_id") or 0))
            except ValueError:
                order = valid_rows
            grouped[(user, skill)].append((order, int(response)))
            unique_skills.add(skill)
            valid_rows += 1

    by_user: dict[str, list[list[int]]] = defaultdict(list)
    for (user, _skill), rows in grouped.items():
        rows.sort(key=lambda item: item[0])
        by_user[user].append([response for _order, response in rows])
    return dict(by_user), valid_rows, len(unique_skills)


def split_users(by_user: dict[str, list[list[int]]], seed: int, test_fraction: float):
    users = sorted(by_user)
    random.Random(seed).shuffle(users)
    test_count = max(1, round(len(users) * test_fraction))
    test_users = set(users[:test_count])
    train_users = set(users[test_count:])
    train = [seq for user in sorted(train_users) for seq in by_user[user]]
    test = [seq for user in sorted(test_users) for seq in by_user[user]]
    return train_users, test_users, train, test


def sample_sequences(sequences: list[list[int]], limit: int, seed: int) -> list[list[int]]:
    if len(sequences) <= limit:
        return sequences
    return random.Random(seed).sample(sequences, limit)


def predict(model: BKTModel, sequences: list[list[int]]):
    labels: list[int] = []
    predictions: list[float] = []
    for sequence in sequences:
        current = model.p_l0
        for response in sequence:
            predictions.append(model.predict_correct(current))
            labels.append(response)
            current = model.update_mastery(current, bool(response))
    return np.asarray(labels), np.clip(np.asarray(predictions), 1e-7, 1 - 1e-7)


def metrics(model: BKTModel, sequences: list[list[int]]) -> dict[str, Any]:
    labels, predictions = predict(model, sequences)
    if not len(labels):
        raise ValueError("The held-out split contains no usable responses.")
    result: dict[str, Any] = {
        "brier_score": round(float(brier_score_loss(labels, predictions)), 6),
        "log_loss": round(float(log_loss(labels, predictions, labels=[0, 1])), 6),
        "n_observations": int(len(labels)),
    }
    result["auc_roc"] = (
        round(float(roc_auc_score(labels, predictions)), 6)
        if len(np.unique(labels)) == 2
        else None
    )

    # Expected Calibration Error: weighted absolute gap over ten probability bins.
    ece = 0.0
    for lower in np.linspace(0, 1, 11)[:-1]:
        upper = lower + 0.1
        mask = (predictions >= lower) & (predictions < upper if upper < 1 else predictions <= upper)
        if mask.any():
            ece += float(mask.mean()) * abs(float(labels[mask].mean() - predictions[mask].mean()))
    result["ece"] = round(ece, 6)
    return result


def make_model(params: dict[str, float]) -> BKTModel:
    return BKTModel(
        p_l0=params["p_l0"], p_t=params["p_t"], p_g=params["p_g"], p_s=params["p_s"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--parameters-output", type=Path, default=DEFAULT_PARAMETERS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--max-train-sequences", type=int, default=6000)
    parser.add_argument("--max-iterations", type=int, default=30)
    args = parser.parse_args()

    if not 0.05 <= args.test_fraction <= 0.5:
        parser.error("--test-fraction must be between 0.05 and 0.5")
    if not args.data.is_file():
        parser.error(f"Dataset file not found: {args.data}")

    by_user, valid_rows, skill_count = load_sequences(args.data)
    train_users, test_users, train_sequences, test_sequences = split_users(
        by_user, args.seed, args.test_fraction
    )
    sampled_train = sample_sequences(train_sequences, args.max_train_sequences, args.seed)

    fitted_model = make_model(APP_BASELINE)
    fitted_params = fitted_model.fit_em(sampled_train, max_iter=args.max_iterations)
    baseline_metrics = metrics(make_model(APP_BASELINE), test_sequences)
    fitted_metrics = metrics(fitted_model, test_sequences)

    # Calibration quality is the primary criterion. AUC is reported as a
    # secondary ranking metric; lower Brier and log loss mean better forecasts.
    improves_calibration = (
        fitted_metrics["brier_score"] < baseline_metrics["brier_score"]
        and fitted_metrics["log_loss"] <= baseline_metrics["log_loss"]
    )
    result = {
        "dataset": {
            "name": DATASET_SOURCE,
            "source": DATASET_URL,
            "valid_interactions": valid_rows,
            "students": len(by_user),
            "skills": skill_count,
            "split": "deterministic student-level holdout",
            "seed": args.seed,
            "test_fraction": args.test_fraction,
            "train_students": len(train_users),
            "test_students": len(test_users),
            "train_sequences_available": len(train_sequences),
            "train_sequences_used": len(sampled_train),
            "test_sequences": len(test_sequences),
        },
        "baseline_parameters": APP_BASELINE,
        "fitted_parameters": fitted_params,
        "held_out_metrics": {"baseline": baseline_metrics, "fitted": fitted_metrics},
        "integration_decision": "use_fitted_parameters" if improves_calibration else "keep_app_defaults",
        "evaluation_note": "Lower Brier score and log loss indicate better probability forecasts; AUC and ECE are supplementary.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if improves_calibration:
        parameters = {
            "parameters": {
                "initialKnowledge": fitted_params["p_l0"],
                "learning": fitted_params["p_t"],
                "guess": fitted_params["p_g"],
                "slip": fitted_params["p_s"],
            },
            "source": DATASET_SOURCE,
            "source_url": DATASET_URL,
            "trained_at_utc": result["created_at_utc"],
            "evaluation": result["held_out_metrics"],
        }
        args.parameters_output.write_text(json.dumps(parameters, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
