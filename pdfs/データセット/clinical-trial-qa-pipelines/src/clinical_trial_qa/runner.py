"""PHI-safe dataset execution with atomic prediction and manifest writes."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable
from uuid import uuid4

from .config import AppConfig
from .models import QuestionResult, QuestionType
from .prompts import PROMPT_VERSION
from .questions import get_question_spec


@dataclass(frozen=True)
class RunSummary:
    run_dir: Path
    predictions_path: Path
    manifest_path: Path
    note_count: int
    result_count: int


def run_dataset(
    config: AppConfig,
    cases: Iterable[Any],
    output_dir: Path | str,
    limit_notes: int | None = None,
) -> RunSummary:
    """Run complete cases and persist only a redacted evaluation projection."""
    if limit_notes is not None and (isinstance(limit_notes, bool) or not isinstance(limit_notes, int) or limit_notes < 1):
        raise ValueError("limit_notes must be a positive integer or None")
    runtime = config.build_runtime()
    selected = []
    for case in cases:
        if limit_notes is not None and len(selected) >= limit_notes:
            break
        selected.append(case)

    started = _utc_now()
    safe_results: list[dict[str, Any]] = []
    statuses: Counter[str] = Counter()
    for case in selected:
        for result in runtime.pipeline.run_case(case):
            safe = _safe_result(result, case.note_id, case.text)
            safe_results.append(safe)
            statuses[safe["document_status"]] += 1
    finished = _utc_now()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex
    run_dir = output_path / f"{config.pipeline}-{run_id}"
    jsonl = "".join(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n" for value in safe_results)
    call_counts = {
        tracked.purpose: len(getattr(tracked.client, "calls", ()))
        for tracked in runtime.clients
    }
    call_counts["total"] = sum(call_counts.values())
    manifest = {
        "pipeline": config.pipeline,
        "seed": config.seed,
        "prompt_version": PROMPT_VERSION,
        "model_names": list(config.model_names),
        "started_utc": started,
        "finished_utc": finished,
        "call_counts": call_counts,
        "config_hash": config.config_hash,
        "run_id": run_id,
        "notes": len(selected),
        "results": len(safe_results),
        "result_status_counts": dict(sorted(statuses.items())),
    }
    temporary_run_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{config.pipeline}-{run_id}.",
            suffix=".tmp",
            dir=output_path,
        )
    )
    temporary_predictions = temporary_run_dir / "predictions.jsonl"
    temporary_manifest = temporary_run_dir / "manifest.json"
    try:
        _write_complete_file(temporary_predictions, jsonl)
        _write_complete_file(
            temporary_manifest,
            json.dumps(manifest, sort_keys=True, indent=2, separators=(",", ": ")) + "\n",
        )
        _fsync_directory(temporary_run_dir)
        os.replace(temporary_run_dir, run_dir)
        _fsync_directory(output_path)
    except BaseException:
        _remove_temporary_run(temporary_run_dir)
        raise
    return RunSummary(
        run_dir,
        run_dir / temporary_predictions.name,
        run_dir / temporary_manifest.name,
        len(selected),
        len(safe_results),
    )


def _safe_result(result: QuestionResult, case_note_id: str, note_text: str) -> dict[str, Any]:
    """Project a rich in-memory result to identifiers, enums, numbers, and counts only."""
    try:
        spec = get_question_spec(str(result.criterion))
    except KeyError:
        criterion = "unknown"
        question_type = "unknown"
    else:
        criterion = spec.criterion
        question_type = spec.question_type.value
    raw_status = result.document_status.value if hasattr(result.document_status, "value") else str(result.document_status)
    allowed_statuses = (
        {"yes", "no", "not_documented", "indeterminate"}
        if question_type == QuestionType.BOOLEAN.value
        else {"value_available", "not_documented", "indeterminate"}
        if question_type == QuestionType.NUMERIC.value
        else {"indeterminate"}
    )
    status = raw_status if raw_status in allowed_statuses else "indeterminate"
    status, answer = _normalized_status_answer(result, question_type, status)
    return {
        "note_id": str(case_note_id),
        "criterion": criterion,
        "question_type": question_type,
        "document_status": status,
        "answer": answer,
        "confidence": _safe_confidence(result.confidence),
        "evidence_count": len(result.evidence),
        "evidence_valid_count": sum(_span_is_exact(span, note_text) for span in result.evidence),
        "candidate_value_count": len(result.candidate_values),
        "provenance_count": len(result.provenance),
        "validation_error_count": len(result.validation_errors),
    }


def _span_is_exact(span: Any, note_text: str) -> bool:
    quote = getattr(span, "quote", None)
    start = getattr(span, "start_char", None)
    end = getattr(span, "end_char", None)
    return (
        isinstance(quote, str)
        and bool(quote)
        and isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(end, int)
        and not isinstance(end, bool)
        and 0 <= start <= end <= len(note_text)
        and note_text[start:end] == quote
    )


def _normalized_status_answer(
    result: QuestionResult,
    question_type: str,
    status: str,
) -> tuple[str, str | float | None]:
    if question_type == QuestionType.BOOLEAN.value:
        return (status, status) if status in {"yes", "no"} else (status, None)
    if question_type == QuestionType.NUMERIC.value and status == "value_available":
        answer = result.answer
        if not isinstance(answer, bool) and isinstance(answer, (int, float)) and math.isfinite(answer):
            return status, float(answer)
        return "indeterminate", None
    return status, None


def _safe_confidence(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value) if 0 <= value <= 1 else None


def _atomic_write_text(path: Path, content: str) -> None:
    """Replace one destination only after its complete sibling temporary file is durable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _write_complete_file(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_temporary_run(path: Path) -> None:
    if not path.exists():
        return
    for filename in ("predictions.jsonl", "manifest.json"):
        try:
            (path / filename).unlink(missing_ok=True)
        except OSError:
            pass
    try:
        path.rmdir()
    except OSError:
        pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
