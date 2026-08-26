"""CSV validation, note-level aggregation, and leakage-safe data splitting."""

from __future__ import annotations

import csv
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from .models import NoteCase, QuestionItem
from .questions import all_question_specs, get_question_spec


REQUIRED_COLUMNS = (
    "text", "note_id", "hadm_id", "criterion", "question_type", "question", "answer", "not_specified",
)


class DatasetValidationError(ValueError):
    """Raised when a CSV cannot be safely converted into complete note cases."""


@dataclass(frozen=True)
class DatasetReport:
    row_count: int
    note_count: int
    criterion_count: int
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class DatasetSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]


def _read_rows(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...], tuple[str, ...]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = tuple(reader.fieldnames or ())
            missing = tuple(column for column in REQUIRED_COLUMNS if column not in fieldnames)
            unexpected = tuple(column for column in fieldnames if column not in REQUIRED_COLUMNS)
            duplicates = tuple(sorted({column for column in fieldnames if fieldnames.count(column) > 1}))
            header_errors: list[str] = []
            if missing:
                header_errors.append(f"missing columns: {', '.join(missing)}")
            if unexpected:
                header_errors.append(f"unexpected columns: {len(unexpected)}")
            if duplicates:
                header_errors.append(f"duplicate columns: {len(duplicates)}")
            if header_errors:
                return [], fieldnames, tuple(header_errors)
            rows: list[dict[str, str]] = []
            row_errors: list[str] = []
            for row_number, row in enumerate(reader, start=2):
                if None in row:
                    row.pop(None)
                    row_errors.append(f"row {row_number}: surplus cells")
                rows.append(dict(row))
            return rows, fieldnames, tuple(row_errors)
    except OSError as exc:
        return [], (), (f"could not read dataset: {exc.__class__.__name__}",)


def _group_and_validate(rows: list[dict[str, str]]) -> tuple[OrderedDict[str, list[dict[str, str]]], tuple[str, ...]]:
    grouped: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    errors: list[str] = []
    known_criteria = {spec.criterion for spec in all_question_specs()}
    for row_number, row in enumerate(rows, start=2):
        note_id = row.get("note_id") or ""
        criterion = row.get("criterion") or ""
        if not note_id:
            errors.append(f"row {row_number}: missing note_id")
            continue
        if not criterion:
            errors.append(f"note_id {note_id}: missing criterion")
        elif criterion not in known_criteria:
            errors.append(f"note_id {note_id}: unknown criterion {criterion}")
        grouped.setdefault(note_id, []).append(row)
    for note_id, note_rows in grouped.items():
        first = note_rows[0]
        text = first.get("text") or ""
        hadm_id = first.get("hadm_id") or ""
        seen: set[str] = set()
        for row in note_rows:
            if (row.get("text") or "") != text:
                errors.append(f"note_id {note_id}: inconsistent text across rows")
                break
        for row in note_rows:
            if (row.get("hadm_id") or "") != hadm_id:
                errors.append(f"note_id {note_id}: inconsistent hadm_id across rows")
                break
        for row in note_rows:
            criterion = row.get("criterion") or ""
            if criterion in seen:
                errors.append(f"note_id {note_id}: duplicate criterion {criterion}")
                break
            seen.add(criterion)
        if len(seen) != len(all_question_specs()):
            errors.append(f"note_id {note_id}: expected 23 criteria, found {len(seen)}")
        if seen and seen != known_criteria:
            errors.append(f"note_id {note_id}: criteria do not match approved registry")
        for row in note_rows:
            criterion = row.get("criterion") or ""
            if criterion in known_criteria and (row.get("question_type") or "") != get_question_spec(criterion).question_type.value:
                errors.append(f"note_id {note_id}: question_type mismatch for {criterion}")
    return grouped, tuple(errors)


def validate_dataset(path: Path) -> DatasetReport:
    """Validate the source structure without returning or exposing note text."""
    rows, _fieldnames, read_errors = _read_rows(Path(path))
    if read_errors:
        return DatasetReport(0, 0, 0, errors=read_errors)
    grouped, validation_errors = _group_and_validate(rows)
    criterion_count = len({row.get("criterion") or "" for row in rows})
    return DatasetReport(len(rows), len(grouped), criterion_count, errors=validation_errors)


def load_cases(path: Path) -> list[NoteCase]:
    """Load complete 23-question records, rejecting structural inconsistencies."""
    report = validate_dataset(path)
    if report.errors:
        raise DatasetValidationError("; ".join(report.errors))
    rows, _fieldnames, _read_errors = _read_rows(Path(path))
    grouped, _validation_errors = _group_and_validate(rows)
    cases: list[NoteCase] = []
    for note_id, note_rows in grouped.items():
        first = note_rows[0]
        questions = tuple(
            QuestionItem(
                spec=get_question_spec(row["criterion"]),
                answer=row.get("answer") or None,
                not_specified=_parse_not_specified(row.get("not_specified") or ""),
                question=row.get("question") or None,
            )
            for row in note_rows
        )
        cases.append(NoteCase(note_id, first.get("hadm_id") or "", first.get("text") or "", questions))
    return cases


def _parse_not_specified(value: str) -> bool:
    return value.strip().casefold() in {"1", "true", "yes", "y"}


def split_note_ids(cases: tuple[NoteCase, ...] | list[NoteCase], seed: int, train_fraction: float, validation_fraction: float) -> DatasetSplit:
    """Split only whole note identifiers using a reproducible shuffled holdout."""
    if not 0 <= train_fraction <= 1 or not 0 <= validation_fraction <= 1:
        raise ValueError("split fractions must be between 0 and 1")
    if train_fraction + validation_fraction > 1:
        raise ValueError("train_fraction + validation_fraction must not exceed 1")
    note_ids = [str(case.note_id) for case in cases]
    if len(note_ids) != len(set(note_ids)):
        raise ValueError("cases must have unique note_id values")
    random.Random(seed).shuffle(note_ids)
    train_end = int(len(note_ids) * train_fraction)
    validation_end = train_end + int(len(note_ids) * validation_fraction)
    return DatasetSplit(tuple(note_ids[:train_end]), tuple(note_ids[train_end:validation_end]), tuple(note_ids[validation_end:]))
