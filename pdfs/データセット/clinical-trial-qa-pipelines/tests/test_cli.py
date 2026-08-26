import csv
import json
from pathlib import Path

from clinical_trial_qa.models import NoteCase, QuestionItem, QuestionResult, QuestionType
from clinical_trial_qa.questions import all_question_specs, get_question_spec


_FIELDS = ("text", "note_id", "hadm_id", "criterion", "question_type", "question", "answer", "not_specified")


def _draft(status="not_documented", *, quote=None, inference=None):
    evidence = [] if quote is None else [{"quote": quote, "start_char": None, "end_char": None}]
    return {
        "document_status": status,
        "answer": "yes" if status == "yes" else None,
        "unit": None,
        "evidence": evidence,
        "candidate_values": [],
        "inference": inference,
        "confidence": 0.75,
    }


def _write_config(tmp_path, script):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "pipeline": "p1",
                "seed": 42,
                "debate_rounds": 0,
                "clients": {
                    "primary": [{"provider": "scripted", "model": "fixture-model", "script": script}]
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_p5_config(tmp_path, *, model="fixture-model"):
    path = tmp_path / "p5-config.json"
    verifier_approval = {"approved": True, "result": {"selected_evidence_ids": []}}
    path.write_text(
        json.dumps(
            {
                "pipeline": "p5",
                "seed": 42,
                "debate_rounds": 0,
                "clients": {
                    "primary": [
                        {"provider": "scripted", "model": model, "script": [_draft()] * 23},
                        {"provider": "scripted", "model": model, "script": [_draft()] * 23},
                    ],
                    "verifier": {
                        "provider": "scripted",
                        "model": model,
                        "script": [verifier_approval] * 23,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _case(text="secret note"):
    return NoteCase("n1", "h1", text, (QuestionItem(get_question_spec("bipolar")),))


def _write_complete_csv(tmp_path, text="sensitive clinical sentence"):
    path = tmp_path / "input.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDS)
        writer.writeheader()
        for spec in all_question_specs():
            writer.writerow(
                {
                    "text": text,
                    "note_id": "n1",
                    "hadm_id": "h1",
                    "criterion": spec.criterion,
                    "question_type": spec.question_type.value,
                    "question": spec.question,
                    "answer": "No" if spec.question_type is QuestionType.BOOLEAN else "",
                    "not_specified": "0" if spec.question_type is QuestionType.BOOLEAN else "1",
                }
            )
    return path


def test_validate_data_cli_reports_counts_without_note_text(tmp_path, capsys):
    """Would fail if read-only validation printed a clinical sentence or diagnostics."""
    from clinical_trial_qa.cli import main

    path = _write_complete_csv(tmp_path)
    before = path.read_bytes()

    rc = main(["validate-data", "--input", str(path)])
    output = capsys.readouterr().out

    assert rc == 0
    assert '"rows": 23' in output
    assert '"notes": 1' in output
    assert '"criteria": 23' in output
    assert "sensitive clinical sentence" not in output
    assert path.read_bytes() == before


def test_runner_jsonl_and_manifest_never_contain_note_prompt_key_or_exception_text(tmp_path):
    """Would fail if persisted artifacts included any free-text model or note field."""
    from clinical_trial_qa.config import load_config
    from clinical_trial_qa.runner import run_dataset

    note_text = "secret note"
    config = load_config(_write_config(tmp_path, [_draft("yes", quote=note_text, inference=note_text)]))
    summary = run_dataset(config, [_case(note_text)], tmp_path / "outputs", limit_notes=1)
    predictions = summary.predictions_path.read_text(encoding="utf-8")
    manifest = summary.manifest_path.read_text(encoding="utf-8")

    for forbidden in (
        note_text,
        "quote",
        "inference",
        "candidate_values",
        "super-secret-key",
        "Return JSON only",
    ):
        assert forbidden not in predictions
        assert forbidden not in manifest
    record = json.loads(predictions)
    metadata = json.loads(manifest)
    assert record["answer"] == "yes"
    assert record["evidence_count"] == 1
    assert record["evidence_valid_count"] == 1
    assert metadata["pipeline"] == "p1"
    assert metadata["seed"] == 42
    assert metadata["model_names"] == ["fixture-model"]
    assert metadata["prompt_version"] == "clinical-trial-qa-v1"
    assert metadata["call_counts"]["total"] == 1
    assert len(metadata["config_hash"]) == 64


def test_runner_redacts_arbitrary_result_strings_to_safe_counts(tmp_path):
    """Would fail if a compromised pipeline could persist free-text errors or answers."""
    from clinical_trial_qa.config import PipelineRuntime
    from clinical_trial_qa.runner import run_dataset

    secret = "SECRET EXCEPTION MESSAGE AND NOTE TEXT"

    class Pipeline:
        def run_case(self, case):
            return [
                QuestionResult(
                    note_id=secret,
                    criterion=secret,
                    question_type=secret,
                    document_status=secret,
                    answer=secret,
                    inference=secret,
                    provenance=(secret,),
                    validation_errors=(secret,),
                )
            ]

    class Config:
        pipeline = "p1"
        seed = 42
        config_hash = "a" * 64
        model_names = ("safe-model",)

        def build_runtime(self):
            return PipelineRuntime(Pipeline(), ())

    summary = run_dataset(Config(), [_case(secret)], tmp_path / "safe", limit_notes=1)
    persisted = summary.predictions_path.read_text() + summary.manifest_path.read_text()

    assert secret not in persisted
    assert json.loads(summary.predictions_path.read_text())["validation_error_count"] == 1


def test_atomic_prediction_write_preserves_existing_file_on_replace_failure(tmp_path, monkeypatch):
    """Would fail if a failed final rename truncated a previously complete prediction file."""
    from clinical_trial_qa.config import load_config
    from clinical_trial_qa.runner import run_dataset
    import clinical_trial_qa.runner as runner

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    existing = output_dir / "p1.jsonl"
    existing.write_text("previous-complete-output\n", encoding="utf-8")
    config = load_config(_write_config(tmp_path, [_draft()]))

    real_replace = runner.os.replace

    def fail_replace(source, destination):
        if Path(source).is_dir():
            raise OSError("secret operating system message")
        return real_replace(source, destination)

    monkeypatch.setattr(runner.os, "replace", fail_replace)
    try:
        run_dataset(config, [_case()], output_dir, limit_notes=1)
    except OSError:
        pass
    else:
        raise AssertionError("OSError was not raised")

    assert existing.read_text(encoding="utf-8") == "previous-complete-output\n"
    assert list(output_dir.iterdir()) == [existing]


def test_split_run_and_evaluate_cli_dry_run(tmp_path, capsys):
    """Would fail if any documented subcommand could not complete without a provider key."""
    from clinical_trial_qa.cli import main

    input_path = _write_complete_csv(tmp_path, text="dry run private note")
    config_path = _write_config(tmp_path, [_draft()] * 23)
    split_dir = tmp_path / "splits"
    output_dir = tmp_path / "outputs"

    assert main(["split", "--input", str(input_path), "--output-dir", str(split_dir), "--seed", "7"]) == 0
    split_output = capsys.readouterr().out
    assert "dry run private note" not in split_output
    assert main(
        [
            "run",
            "--config",
            str(config_path),
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--limit-notes",
            "1",
        ]
    ) == 0
    run_output = capsys.readouterr().out
    assert "dry run private note" not in run_output
    run_metadata = json.loads(run_output)
    prediction_path = output_dir / run_metadata["run_directory"] / "predictions.jsonl"
    assert main(
        [
            "evaluate",
            "--gold",
            str(input_path),
            "--predictions",
            str(prediction_path),
        ]
    ) == 0
    evaluation = json.loads(capsys.readouterr().out)
    assert evaluation["overall"]["accuracy_with_abstention_wrong"] == 1.0
    assert (split_dir / "test.json").exists()


def test_successful_run_publishes_predictions_and_manifest_in_one_unique_directory(tmp_path):
    """Would fail if a run exposed independently replaceable top-level artifact files."""
    from clinical_trial_qa.config import load_config
    from clinical_trial_qa.runner import run_dataset

    output_dir = tmp_path / "outputs"
    config = load_config(_write_config(tmp_path, [_draft()]))

    summary = run_dataset(config, [_case()], output_dir, limit_notes=1)

    assert summary.run_dir == summary.predictions_path.parent == summary.manifest_path.parent
    assert summary.run_dir.parent == output_dir
    assert summary.run_dir.name.startswith("p1-")
    assert summary.predictions_path.name == "predictions.jsonl"
    assert summary.manifest_path.name == "manifest.json"
    assert sorted(path.name for path in output_dir.iterdir()) == [summary.run_dir.name]


def test_final_run_directory_rename_failure_leaves_no_partial_artifact_set(tmp_path, monkeypatch):
    """Would fail if either artifact became final before the run directory publication point."""
    from clinical_trial_qa.config import load_config
    from clinical_trial_qa.runner import run_dataset
    import clinical_trial_qa.runner as runner

    output_dir = tmp_path / "outputs"
    config = load_config(_write_config(tmp_path, [_draft()]))
    real_replace = runner.os.replace

    def fail_final_directory_replace(source, destination):
        if Path(source).is_dir():
            raise OSError("private final rename failure")
        return real_replace(source, destination)

    monkeypatch.setattr(runner.os, "replace", fail_final_directory_replace)
    try:
        run_dataset(config, [_case()], output_dir, limit_notes=1)
    except OSError:
        pass
    else:
        raise AssertionError("OSError was not raised")

    assert output_dir.exists()
    assert list(output_dir.iterdir()) == []


def test_runner_normalizes_contradictory_status_answer_pairs(tmp_path):
    """Would fail if a persisted status and answer could contradict one another."""
    from clinical_trial_qa.config import PipelineRuntime
    from clinical_trial_qa.runner import run_dataset

    class Pipeline:
        def run_case(self, case):
            values = (
                ("afib", "yes", "no"),
                ("bipolar", "no", "yes"),
                ("bleeding", "not_documented", "yes"),
                ("PLT", "value_available", float("inf")),
                ("HGB", "value_available", 91),
            )
            return [
                QuestionResult(
                    note_id=case.note_id,
                    criterion=criterion,
                    question_type=get_question_spec(criterion).question_type,
                    document_status=status,
                    answer=answer,
                )
                for criterion, status, answer in values
            ]

    class Config:
        pipeline = "p1"
        seed = 42
        config_hash = "a" * 64
        model_names = ("safe-model",)

        def build_runtime(self):
            return PipelineRuntime(Pipeline(), ())

    summary = run_dataset(Config(), [_case()], tmp_path / "outputs", limit_notes=1)
    records = {
        value["criterion"]: value
        for value in map(json.loads, summary.predictions_path.read_text().splitlines())
    }

    assert records["afib"]["document_status"] == "yes"
    assert records["afib"]["answer"] == "yes"
    assert records["bipolar"]["document_status"] == "no"
    assert records["bipolar"]["answer"] == "no"
    assert records["bleeding"]["answer"] is None
    assert records["PLT"]["document_status"] == "indeterminate"
    assert records["PLT"]["answer"] is None
    assert records["HGB"]["document_status"] == "value_available"
    assert records["HGB"]["answer"] == 91.0


def test_invalid_cli_arguments_return_class_only_json_without_echoing_argv(capsys):
    """Would fail if argparse printed a sensitive invalid value or raised SystemExit."""
    from clinical_trial_qa.cli import main

    sensitive = "PRIVATE_PATIENT_SENTENCE"
    cases = (
        ["run", "--config", "config.json", "--input", "input.csv", "--pipeline", sensitive],
        ["split", "--input", "input.csv", "--output-dir", "splits", "--seed", sensitive],
        ["validate-data", "--input", "input.csv", f"--{sensitive}"],
        [],
    )
    for argv in cases:
        rc = main(argv)
        captured = capsys.readouterr()

        assert rc == 2
        assert captured.out == ""
        assert json.loads(captured.err) == {"error_class": "CLIArgumentError"}
        assert sensitive not in captured.err


def test_cli_help_returns_zero_without_system_exit(capsys):
    """Would fail if the callable main API leaked argparse's process exit."""
    from clinical_trial_qa.cli import main

    rc = main(["--help"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "clinical-trial-qa" in captured.out
    assert captured.err == ""


def test_cli_pipeline_override_revalidates_before_clients_or_artifacts(tmp_path, capsys, monkeypatch):
    """Would fail if a P5 config overridden to P1 constructed or called unused clients."""
    from clinical_trial_qa.cli import main
    from clinical_trial_qa.config import ClientConfig
    from clinical_trial_qa.llm import ScriptedLLMClient

    sensitive = "PRIVATE_PATIENT_TEXT"
    input_path = _write_complete_csv(tmp_path, text=sensitive)
    config_path = _write_p5_config(tmp_path, model=sensitive)
    output_dir = tmp_path / "invalid-override-output"
    builds = []
    generations = []
    original_build = ClientConfig.build
    original_generate = ScriptedLLMClient.generate

    def tracked_build(client_config):
        builds.append(client_config.model)
        return original_build(client_config)

    def tracked_generate(client, request):
        generations.append(request.task)
        return original_generate(client, request)

    monkeypatch.setattr(ClientConfig, "build", tracked_build)
    monkeypatch.setattr(ScriptedLLMClient, "generate", tracked_generate)

    rc = main(
        [
            "run",
            "--config",
            str(config_path),
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--pipeline",
            "p1",
            "--limit-notes",
            "1",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 2
    assert captured.out == ""
    assert json.loads(captured.err) == {"error_class": "ConfigurationError"}
    assert sensitive not in captured.out + captured.err
    assert builds == []
    assert generations == []
    assert not output_dir.exists()


def test_cli_pipeline_override_with_same_value_keeps_valid_topology(tmp_path, capsys):
    """Would fail if a no-op pipeline override were rejected with its matching P5 clients."""
    from clinical_trial_qa.cli import main

    input_path = _write_complete_csv(tmp_path, text="same override private note")
    config_path = _write_p5_config(tmp_path)
    output_dir = tmp_path / "valid-override-output"

    rc = main(
        [
            "run",
            "--config",
            str(config_path),
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--pipeline",
            "p5",
            "--limit-notes",
            "1",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    metadata = json.loads(captured.out)
    assert metadata["pipeline"] == "p5"
    assert (output_dir / metadata["run_directory"] / "manifest.json").exists()
    assert captured.err == ""
