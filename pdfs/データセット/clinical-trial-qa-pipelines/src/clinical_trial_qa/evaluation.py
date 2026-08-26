"""Legacy-label compatibility and abstention-aware prediction metrics."""

from __future__ import annotations

from collections import Counter
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .questions import get_question_spec


class EvaluationError(ValueError):
    """Raised when gold or prediction records cannot be joined unambiguously."""


def evaluate_predictions(gold_path: Path | str, prediction_path: Path | str, tolerance: float) -> dict[str, Any]:
    """Evaluate JSONL predictions against the legacy CSV with explicit denominators."""
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance < 0:
        raise EvaluationError("tolerance must be a finite non-negative number")
    gold = _read_gold(Path(gold_path))
    predictions = _read_predictions(Path(prediction_path))
    records = [_score(row, predictions.get(row["key"]), float(tolerance)) for row in gold]
    by_criterion: dict[str, dict[str, Any]] = {}
    for criterion in sorted({str(row["criterion"]) for row in gold}):
        by_criterion[criterion] = _summary(record for record in records if record["criterion"] == criterion)
    overall = _summary(records)
    criterion_accuracies = [value["accuracy_with_abstention_wrong"] for value in by_criterion.values()]
    overall["criterion_macro_accuracy"] = _mean(criterion_accuracies)
    overall["criterion_macro_accuracy_numerator"] = sum(criterion_accuracies)
    overall["criterion_macro_accuracy_denominator"] = len(criterion_accuracies)
    overall["gold_rows"] = len(gold)
    overall["prediction_rows"] = len(predictions)
    gold_keys = {row["key"] for row in gold}
    overall["matched_prediction_rows"] = sum(1 for key in predictions if key in gold_keys)
    overall["extra_prediction_rows"] = sum(1 for key in predictions if key not in gold_keys)
    mapping_counts = Counter(row["legacy_mapping"] for row in gold)
    return {
        "overall": overall,
        "by_criterion": by_criterion,
        "boolean": _boolean_metrics(record for record in records if record["question_type"] == "yes"),
        "numeric": _numeric_metrics(
            (record for record in records if record["question_type"] == "numeric"),
            float(tolerance),
        ),
        "tolerance": float(tolerance),
        "limitations": {
            "evidence_recall_available": False,
            "reason": "expert_span_annotations_required",
        },
        "legacy_mapping": {
            "boolean_rows": mapping_counts["boolean"],
            "numeric_rows": sum(row["question_type"] == "numeric" for row in gold),
            "numeric_value_rows": mapping_counts["numeric_value"],
            "numeric_missing_rows": mapping_counts["numeric_missing_flag"]
            + mapping_counts["numeric_blank_without_flag"]
            + mapping_counts["numeric_nonblank_with_flag"],
            "numeric_blank_without_not_specified_flag": mapping_counts["numeric_blank_without_flag"],
            "numeric_nonblank_with_not_specified_flag": mapping_counts["numeric_nonblank_with_flag"],
        },
    }


def _read_gold(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"note_id", "criterion", "question_type", "answer", "not_specified"}
            if not required.issubset(set(reader.fieldnames or ())):
                raise EvaluationError("gold CSV is missing required columns")
            rows: list[dict[str, Any]] = []
            seen: set[tuple[str, str]] = set()
            for row_number, row in enumerate(reader, 2):
                note_id = row.get("note_id") or ""
                criterion = row.get("criterion") or ""
                question_type = (row.get("question_type") or "").strip().casefold()
                key = (note_id, criterion)
                if not note_id or not criterion or key in seen:
                    raise EvaluationError(f"gold row {row_number} has an invalid or duplicate key")
                seen.add(key)
                try:
                    spec = get_question_spec(criterion)
                except KeyError as exc:
                    raise EvaluationError(f"gold row {row_number} has an unknown criterion") from exc
                if question_type != spec.question_type.value:
                    raise EvaluationError(f"gold row {row_number} has a question type mismatch")
                not_specified = _parse_not_specified(row.get("not_specified") or "", row_number)
                if question_type == "yes":
                    answer = (row.get("answer") or "").strip().casefold()
                    if answer not in {"yes", "no"}:
                        raise EvaluationError(f"gold row {row_number} has an invalid boolean label")
                    target: Any = answer
                    legacy_mapping = "boolean"
                elif question_type == "numeric":
                    missing = not_specified
                    raw_answer = (row.get("answer") or "").strip()
                    if missing:
                        target = ("missing", None)
                        legacy_mapping = "numeric_missing_flag" if not raw_answer else "numeric_nonblank_with_flag"
                    elif not raw_answer:
                        target = ("missing", None)
                        legacy_mapping = "numeric_blank_without_flag"
                    else:
                        try:
                            numeric = float(raw_answer)
                        except ValueError as exc:
                            raise EvaluationError(f"gold row {row_number} has an invalid numeric label") from exc
                        if not math.isfinite(numeric):
                            raise EvaluationError(f"gold row {row_number} has an invalid numeric label")
                        target = ("value", numeric)
                        legacy_mapping = "numeric_value"
                else:
                    raise EvaluationError(f"gold row {row_number} has an invalid question type")
                rows.append(
                    {
                        "key": key,
                        "criterion": criterion,
                        "question_type": question_type,
                        "target": target,
                        "legacy_mapping": legacy_mapping,
                    }
                )
            return rows
    except EvaluationError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise EvaluationError(f"gold CSV could not be read: {exc.__class__.__name__}") from exc


def _read_predictions(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    predictions: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for row_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise EvaluationError(f"prediction row {row_number} must be an object")
                note_id = value.get("note_id")
                criterion = value.get("criterion")
                if not isinstance(note_id, str) or not note_id or not isinstance(criterion, str) or not criterion:
                    raise EvaluationError(f"prediction row {row_number} has an invalid key")
                key = (note_id, criterion)
                if key in predictions:
                    raise EvaluationError(f"prediction row {row_number} duplicates a key")
                predictions[key] = value
        return predictions
    except EvaluationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"predictions could not be read: {exc.__class__.__name__}") from exc


def _score(gold: dict[str, Any], prediction: dict[str, Any] | None, tolerance: float) -> dict[str, Any]:
    question_type = gold["question_type"]
    observed = _boolean_prediction(prediction) if question_type == "yes" else _numeric_prediction(prediction)
    covered = observed is not None
    if not covered:
        correct = False
    elif question_type == "yes":
        correct = observed == gold["target"]
    else:
        gold_kind, gold_value = gold["target"]
        observed_kind, observed_value = observed
        correct = gold_kind == observed_kind and (
            gold_kind == "missing" or abs(float(gold_value) - float(observed_value)) <= tolerance
        )
    evidence_count, evidence_valid_count = _evidence_counts(prediction)
    return {
        **gold,
        "observed": observed,
        "covered": covered,
        "correct": correct,
        "evidence_count": evidence_count,
        "evidence_valid_count": evidence_valid_count,
    }


def _boolean_prediction(prediction: dict[str, Any] | None) -> str | None:
    status = prediction.get("document_status") if prediction else None
    if status == "yes":
        return "yes"
    if status in {"no", "not_documented"}:
        return "no"
    return None


def _numeric_prediction(prediction: dict[str, Any] | None) -> tuple[str, float | None] | None:
    status = prediction.get("document_status") if prediction else None
    if status == "not_documented":
        return ("missing", None)
    if status != "value_available":
        return None
    answer = prediction.get("answer")
    if isinstance(answer, bool) or not isinstance(answer, (int, float)) or not math.isfinite(answer):
        return None
    return ("value", float(answer))


def _summary(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(records)
    covered = sum(bool(record["covered"]) for record in values)
    correct = sum(bool(record["correct"]) for record in values)
    total = len(values)
    evidence_count = sum(record["evidence_count"] for record in values)
    evidence_valid_count = sum(record["evidence_valid_count"] for record in values)
    return {
        "coverage": _ratio(covered, total),
        "coverage_numerator": covered,
        "coverage_denominator": total,
        "selective_accuracy": _ratio(correct, covered),
        "selective_accuracy_numerator": correct,
        "selective_accuracy_denominator": covered,
        "accuracy_with_abstention_wrong": _ratio(correct, total),
        "accuracy_with_abstention_wrong_numerator": correct,
        "accuracy_with_abstention_wrong_denominator": total,
        "abstentions": total - covered,
        "evidence_exact_match_validity": _ratio(evidence_valid_count, evidence_count),
        "evidence_exact_match_validity_numerator": evidence_valid_count,
        "evidence_exact_match_validity_denominator": evidence_count,
    }


def _evidence_counts(prediction: dict[str, Any] | None) -> tuple[int, int]:
    if prediction is None:
        return 0, 0
    count = prediction.get("evidence_count", 0)
    valid = prediction.get("evidence_valid_count", 0)
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        or isinstance(valid, bool)
        or not isinstance(valid, int)
        or not 0 <= valid <= count
    ):
        raise EvaluationError("prediction has invalid evidence counts")
    return count, valid


def _boolean_metrics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(records)
    classes = sorted({record["target"] for record in values})
    f1_values: list[float] = []
    recall_values: list[float] = []
    for label in classes:
        tp = sum(record["target"] == label and record["observed"] == label for record in values)
        fp = sum(record["target"] != label and record["observed"] == label for record in values)
        fn = sum(record["target"] == label and record["observed"] != label for record in values)
        f1_values.append(_ratio(2 * tp, 2 * tp + fp + fn) or 0.0)
        recall_values.append(_ratio(tp, tp + fn) or 0.0)
    return {
        "gold_rows": len(values),
        "covered_rows": sum(bool(record["covered"]) for record in values),
        "macro_f1": _mean(f1_values),
        "macro_f1_sum": sum(f1_values),
        "macro_f1_class_denominator": len(classes),
        "balanced_accuracy": _mean(recall_values),
        "balanced_accuracy_sum": sum(recall_values),
        "balanced_accuracy_class_denominator": len(classes),
    }


def _numeric_metrics(records: Iterable[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    values = list(records)
    pairs = [
        (float(record["target"][1]), float(record["observed"][1]))
        for record in values
        if record["target"][0] == "value" and record["observed"] is not None and record["observed"][0] == "value"
    ]
    absolute_errors = [abs(gold - observed) for gold, observed in pairs]
    within = sum(error <= tolerance for error in absolute_errors)
    missing_correct = sum(
        record["observed"] is not None and record["target"][0] == record["observed"][0]
        for record in values
    )
    return {
        "gold_rows": len(values),
        "covered_rows": sum(bool(record["covered"]) for record in values),
        "value_pair_denominator": len(pairs),
        "mae": _mean(absolute_errors),
        "mae_absolute_error_sum": sum(absolute_errors),
        "mae_denominator": len(absolute_errors),
        "within_tolerance": _ratio(within, len(pairs)),
        "within_tolerance_numerator": within,
        "within_tolerance_denominator": len(pairs),
        "missingness_accuracy": _ratio(missing_correct, len(values)),
        "missingness_accuracy_numerator": missing_correct,
        "missingness_accuracy_denominator": len(values),
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _parse_not_specified(value: str, row_number: int) -> bool:
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise EvaluationError(f"gold row {row_number} has an invalid not_specified token")
