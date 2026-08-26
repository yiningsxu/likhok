import csv

from clinical_trial_qa.dataset import DatasetValidationError, load_cases, split_note_ids, validate_dataset
from clinical_trial_qa.models import NoteCase
from clinical_trial_qa.questions import all_question_specs


FIELDS = ("text", "note_id", "hadm_id", "criterion", "question_type", "question", "answer", "not_specified")


def rows_for(note_id: str, hadm_id: str, text: str) -> list[dict[str, str]]:
    return [
        {
            "text": text,
            "note_id": note_id,
            "hadm_id": hadm_id,
            "criterion": spec.criterion,
            "question_type": spec.question_type.value,
            "question": spec.question,
            "answer": "no" if spec.question_type.value == "yes" else "NA",
            "not_specified": "false",
        }
        for spec in all_question_specs()
    ]


def two_complete_note_fixtures() -> list[dict[str, str]]:
    return rows_for("n1", "h1", "first fixture note") + rows_for("n2", "h2", "second fixture note")


def write_csv(tmp_path, rows, fields=FIELDS):
    path = tmp_path / "cases.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])
    return path


def write_mutated_csv(tmp_path, mutation):
    rows = rows_for("n1", "h1", "private fixture note")
    fields = FIELDS
    if mutation == "drop_text_column":
        fields = tuple(field for field in FIELDS if field != "text")
    elif mutation == "change_second_row_text":
        rows[1]["text"] = "different private fixture note"
    elif mutation == "duplicate_first_criterion":
        rows[1]["criterion"] = rows[0]["criterion"]
    elif mutation == "drop_last_row":
        rows.pop()
    elif mutation == "unexpected_named_column":
        fields = FIELDS + ("unexpected",)
    elif mutation == "duplicate_header":
        fields = FIELDS + ("text",)
    elif mutation == "surplus_row_cells":
        path = tmp_path / "cases.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(FIELDS)
            writer.writerow([rows[0][field] for field in FIELDS] + ["surplus"])
            writer.writerows([[row[field] for field in FIELDS] for row in rows[1:]])
        return path
    else:
        raise ValueError(mutation)
    return write_csv(tmp_path, rows, fields)


def make_case(note_id: str) -> NoteCase:
    return NoteCase(note_id=note_id, hadm_id=f"h-{note_id}", text="test note", questions=())


def test_loader_groups_rows_without_copying_a_note_across_cases(tmp_path):
    """Would fail if rows were returned separately rather than aggregated by note."""
    path = write_csv(tmp_path, two_complete_note_fixtures())

    cases = load_cases(path)

    assert [case.note_id for case in cases] == ["n1", "n2"]
    assert all(len(case.questions) == 23 for case in cases)
    assert [item.criterion for item in cases[0].questions] == [spec.criterion for spec in all_question_specs()]


def test_note_level_split_has_no_identifier_overlap():
    """Would fail if rows of a note could enter more than one partition."""
    cases = tuple(make_case(str(index)) for index in range(20))

    split = split_note_ids(cases, seed=42, train_fraction=.6, validation_fraction=.2)

    assert set(split.train).isdisjoint(split.validation)
    assert set(split.train).isdisjoint(split.test)
    assert set(split.validation).isdisjoint(split.test)
    assert set(split.train) | set(split.validation) | set(split.test) == {str(index) for index in range(20)}


def test_loader_rejects_structural_data_errors(tmp_path):
    """Would fail if malformed data were silently grouped into a case."""
    cases = (
        ("drop_text_column", "missing columns"),
        ("change_second_row_text", "inconsistent text"),
        ("duplicate_first_criterion", "duplicate criterion"),
        ("drop_last_row", "expected 23 criteria"),
        ("unexpected_named_column", "unexpected columns"),
        ("duplicate_header", "duplicate columns"),
        ("surplus_row_cells", "surplus cells"),
    )
    for mutation, error_fragment in cases:
        path = write_mutated_csv(tmp_path, mutation)
        report = validate_dataset(path)
        assert error_fragment in " ".join(report.errors)
        assert "private fixture note" not in " ".join(report.errors)
        _assert_raises(DatasetValidationError, lambda: load_cases(path), error_fragment)


def test_validation_report_keeps_note_text_out_of_errors(tmp_path):
    """Would fail if diagnostics leaked clinical-note text into reports."""
    path = write_mutated_csv(tmp_path, "change_second_row_text")

    report = validate_dataset(path)

    assert report.errors
    assert "private fixture note" not in " ".join(report.errors)


def test_header_diagnostics_do_not_echo_sensitive_extra_header(tmp_path):
    """Would fail if invalid header text were copied into a validation diagnostic."""
    sensitive_header = "Patient has metastatic cancer and needs chemotherapy."
    path = write_csv(
        tmp_path,
        rows_for("n1", "h1", "private fixture note"),
        FIELDS + (sensitive_header,),
    )

    report = validate_dataset(path)

    assert "unexpected columns" in " ".join(report.errors)
    assert sensitive_header not in " ".join(report.errors)
    _assert_raises(DatasetValidationError, lambda: load_cases(path), "unexpected columns")


def _assert_raises(error_type, action, fragment):
    try:
        action()
    except error_type as exc:
        assert fragment in str(exc)
        return exc
    raise AssertionError(f"{error_type.__name__} was not raised")
