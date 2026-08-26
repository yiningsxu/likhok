import csv
import json


_GOLD_FIELDS = (
    "text",
    "note_id",
    "hadm_id",
    "criterion",
    "question_type",
    "question",
    "answer",
    "not_specified",
)


def _write_gold(tmp_path, rows):
    path = tmp_path / "gold.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_GOLD_FIELDS)
        writer.writeheader()
        for note_id, criterion, question_type, answer, not_specified in rows:
            writer.writerow(
                {
                    "text": "sensitive gold note",
                    "note_id": note_id,
                    "hadm_id": f"h-{note_id}",
                    "criterion": criterion,
                    "question_type": question_type,
                    "question": "fixture question",
                    "answer": answer,
                    "not_specified": not_specified,
                }
            )
    return path


def _write_predictions(tmp_path, rows):
    path = tmp_path / "predictions.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for note_id, criterion, status, answer in rows:
            handle.write(
                json.dumps(
                    {
                        "note_id": note_id,
                        "criterion": criterion,
                        "document_status": status,
                        "answer": answer,
                    }
                )
                + "\n"
            )
    return path


def test_selective_metrics_separate_accuracy_from_coverage(tmp_path):
    """Would fail if abstentions inflated selective accuracy or disappeared from coverage."""
    from clinical_trial_qa.evaluation import evaluate_predictions

    gold = _write_gold(
        tmp_path,
        [("n1", "afib", "yes", "Yes", "0"), ("n2", "afib", "yes", "No", "0")],
    )
    pred = _write_predictions(
        tmp_path,
        [("n1", "afib", "yes", "yes"), ("n2", "afib", "indeterminate", None)],
    )

    metrics = evaluate_predictions(gold, pred, tolerance=1e-6)

    assert metrics["overall"]["coverage"] == 0.5
    assert metrics["overall"]["coverage_numerator"] == 1
    assert metrics["overall"]["coverage_denominator"] == 2
    assert metrics["overall"]["selective_accuracy"] == 1.0
    assert metrics["overall"]["selective_accuracy_denominator"] == 1
    assert metrics["overall"]["accuracy_with_abstention_wrong"] == 0.5
    assert metrics["overall"]["accuracy_with_abstention_wrong_denominator"] == 2
    assert metrics["overall"]["criterion_macro_accuracy_numerator"] == 0.5
    assert metrics["overall"]["criterion_macro_accuracy_denominator"] == 1


def test_legacy_boolean_no_maps_from_no_or_not_documented(tmp_path):
    """Would fail if the legacy Yes/No CSV were confused with the four-state document model."""
    from clinical_trial_qa.evaluation import evaluate_predictions

    gold = _write_gold(
        tmp_path,
        [("n1", "afib", "yes", "No", "0"), ("n2", "afib", "yes", "No", "0")],
    )
    pred = _write_predictions(
        tmp_path,
        [("n1", "afib", "no", "no"), ("n2", "afib", "not_documented", None)],
    )

    metrics = evaluate_predictions(gold, pred, tolerance=1e-6)

    assert metrics["overall"]["accuracy_with_abstention_wrong"] == 1.0
    assert metrics["boolean"]["macro_f1"] == 1.0
    assert metrics["boolean"]["balanced_accuracy"] == 1.0


def test_numeric_metrics_distinguish_values_missingness_and_abstention(tmp_path):
    """Would fail if blank legacy numeric labels were parsed as zero or excluded silently."""
    from clinical_trial_qa.evaluation import evaluate_predictions

    gold = _write_gold(
        tmp_path,
        [
            ("n1", "PLT", "numeric", "91", "0"),
            ("n2", "PLT", "numeric", "", "1"),
            ("n3", "PLT", "numeric", "100", "0"),
        ],
    )
    pred = _write_predictions(
        tmp_path,
        [
            ("n1", "PLT", "value_available", 91.5),
            ("n2", "PLT", "not_documented", None),
            ("n3", "PLT", "indeterminate", None),
        ],
    )

    metrics = evaluate_predictions(gold, pred, tolerance=1.0)

    assert metrics["numeric"]["value_pair_denominator"] == 1
    assert metrics["numeric"]["mae"] == 0.5
    assert metrics["numeric"]["mae_absolute_error_sum"] == 0.5
    assert metrics["numeric"]["mae_denominator"] == 1
    assert metrics["numeric"]["within_tolerance"] == 1.0
    assert metrics["numeric"]["missingness_accuracy"] == 2 / 3
    assert metrics["numeric"]["missingness_accuracy_denominator"] == 3
    assert metrics["by_criterion"]["PLT"]["coverage_denominator"] == 3


def test_evaluation_rejects_duplicate_prediction_keys_and_negative_tolerance(tmp_path):
    """Would fail if ambiguous joins or nonsensical numeric tolerance were accepted."""
    from clinical_trial_qa.evaluation import EvaluationError, evaluate_predictions

    gold = _write_gold(tmp_path, [("n1", "afib", "yes", "Yes", "0")])
    pred = _write_predictions(
        tmp_path,
        [("n1", "afib", "yes", "yes"), ("n1", "afib", "no", "no")],
    )

    for tolerance in (1e-6, -1.0):
        try:
            evaluate_predictions(gold, pred, tolerance=tolerance)
        except EvaluationError:
            pass
        else:
            raise AssertionError("EvaluationError was not raised")


def test_evidence_validity_uses_redacted_counts_with_explicit_denominator(tmp_path):
    """Would fail if exact-match validity lacked a denominator or required quote persistence."""
    from clinical_trial_qa.evaluation import evaluate_predictions

    gold = _write_gold(tmp_path, [("n1", "afib", "yes", "Yes", "0")])
    pred = tmp_path / "predictions.jsonl"
    pred.write_text(
        json.dumps(
            {
                "note_id": "n1",
                "criterion": "afib",
                "document_status": "yes",
                "answer": "yes",
                "evidence_count": 2,
                "evidence_valid_count": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = evaluate_predictions(gold, pred, tolerance=1e-6)

    assert metrics["overall"]["evidence_exact_match_validity"] == 0.5
    assert metrics["overall"]["evidence_exact_match_validity_numerator"] == 1
    assert metrics["overall"]["evidence_exact_match_validity_denominator"] == 2
    assert metrics["by_criterion"]["afib"]["evidence_exact_match_validity_denominator"] == 2


def test_blank_numeric_legacy_label_is_missing_even_when_flag_is_zero(tmp_path):
    """Would fail on the two known blank/flag mismatches in the attached legacy CSV."""
    from clinical_trial_qa.evaluation import evaluate_predictions

    gold = _write_gold(tmp_path, [("n1", "BILI", "numeric", "", "0")])
    pred = _write_predictions(tmp_path, [("n1", "BILI", "not_documented", None)])

    metrics = evaluate_predictions(gold, pred, tolerance=1e-6)

    assert metrics["overall"]["accuracy_with_abstention_wrong"] == 1.0
    assert metrics["legacy_mapping"]["numeric_missing_rows"] == 1
    assert metrics["legacy_mapping"]["numeric_blank_without_not_specified_flag"] == 1


def test_gold_rejects_unknown_or_question_type_mismatched_criteria_before_metric_keys(tmp_path):
    """Would fail if PHI-like criterion text could become an output dictionary key."""
    from clinical_trial_qa.evaluation import EvaluationError, evaluate_predictions

    sensitive_criterion = "patient metastatic diagnosis private sentence"
    cases = (
        [("n1", sensitive_criterion, "yes", "Yes", "0")],
        [("n1", "PLT", "yes", "Yes", "0")],
    )
    pred = _write_predictions(tmp_path, [])
    for index, rows in enumerate(cases):
        gold = _write_gold(tmp_path, rows)
        try:
            evaluate_predictions(gold, pred, tolerance=1e-6)
        except EvaluationError as exc:
            assert sensitive_criterion not in str(exc)
        else:
            raise AssertionError(f"EvaluationError was not raised for case {index}")


def test_gold_rejects_unknown_not_specified_tokens_even_for_boolean_rows(tmp_path):
    """Would fail if an unknown missingness token silently became false."""
    from clinical_trial_qa.evaluation import EvaluationError, evaluate_predictions

    sensitive_token = "private/path token"
    gold = _write_gold(tmp_path, [("n1", "afib", "yes", "Yes", sensitive_token)])
    pred = _write_predictions(tmp_path, [])

    try:
        evaluate_predictions(gold, pred, tolerance=1e-6)
    except EvaluationError as exc:
        assert sensitive_token not in str(exc)
    else:
        raise AssertionError("EvaluationError was not raised")


def test_boolean_macro_metrics_publish_aggregation_sums_and_counts(tmp_path):
    """Would fail if macro rates could not be reconstructed from reported aggregates."""
    from clinical_trial_qa.evaluation import evaluate_predictions

    gold = _write_gold(
        tmp_path,
        [("n1", "afib", "yes", "Yes", "0"), ("n2", "afib", "yes", "No", "0")],
    )
    pred = _write_predictions(
        tmp_path,
        [("n1", "afib", "yes", "yes"), ("n2", "afib", "no", "no")],
    )

    metrics = evaluate_predictions(gold, pred, tolerance=1e-6)["boolean"]

    assert metrics["macro_f1_sum"] == 2.0
    assert metrics["macro_f1_class_denominator"] == 2
    assert metrics["balanced_accuracy_sum"] == 2.0
    assert metrics["balanced_accuracy_class_denominator"] == 2
