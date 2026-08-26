"""Command-line interface for validation, splitting, execution, and evaluation."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Sequence

from .config import load_config
from .dataset import load_cases, split_note_ids, validate_dataset
from .evaluation import evaluate_predictions
from .runner import _atomic_write_text, run_dataset


class CLIArgumentError(ValueError):
    """Fixed parser failure whose string representation never contains argv text."""


class _ParserExit(Exception):
    def __init__(self, status: int) -> None:
        super().__init__()
        self.status = status


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise CLIArgumentError("invalid command arguments")

    def exit(self, status: int = 0, message: str | None = None) -> None:
        if message and status == 0:
            self._print_message(message)
        raise _ParserExit(status)


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        prog="clinical-trial-qa",
        description="Evidence-grounded clinical trial QA research pipelines",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    validate = subcommands.add_parser("validate-data", help="validate a source CSV without printing note text")
    validate.add_argument("--input", required=True, type=Path)

    split = subcommands.add_parser("split", help="write reproducible note-level split identifiers")
    split.add_argument("--input", required=True, type=Path)
    split.add_argument("--output-dir", required=True, type=Path)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--train-fraction", type=float, default=0.7)
    split.add_argument("--validation-fraction", type=float, default=0.15)

    run = subcommands.add_parser("run", help="run a configured pipeline")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--input", required=True, type=Path)
    run.add_argument("--output-dir", type=Path, default=Path("outputs"))
    run.add_argument("--pipeline", choices=("p1", "p2", "p3", "p4", "p5", "p6"))
    run.add_argument("--limit-notes", type=int)

    evaluate = subcommands.add_parser("evaluate", help="evaluate redacted JSONL predictions")
    evaluate.add_argument("--gold", required=True, type=Path)
    evaluate.add_argument("--predictions", required=True, type=Path)
    evaluate.add_argument("--tolerance", type=float, default=1e-6)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except _ParserExit as exc:
        return exc.status
    except CLIArgumentError as exc:
        print(json.dumps({"error_class": exc.__class__.__name__}, sort_keys=True), file=sys.stderr)
        return 2
    try:
        if args.command == "validate-data":
            report = validate_dataset(args.input)
            _print_json(
                {
                    "criteria": report.criterion_count,
                    "errors": len(report.errors),
                    "notes": report.note_count,
                    "rows": report.row_count,
                    "valid": report.is_valid,
                    "warnings": len(report.warnings),
                }
            )
            return 0 if report.is_valid else 1
        if args.command == "split":
            cases = load_cases(args.input)
            split = split_note_ids(cases, args.seed, args.train_fraction, args.validation_fraction)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            for name, identifiers in (("train", split.train), ("validation", split.validation), ("test", split.test)):
                _atomic_write_text(args.output_dir / f"{name}.json", json.dumps(list(identifiers), indent=2) + "\n")
            _print_json({"test": len(split.test), "train": len(split.train), "validation": len(split.validation)})
            return 0
        if args.command == "run":
            config = load_config(args.config)
            if args.pipeline:
                config = replace(config, pipeline=args.pipeline)
                config.build_runtime()
            summary = run_dataset(config, load_cases(args.input), args.output_dir, args.limit_notes)
            _print_json(
                {
                    "notes": summary.note_count,
                    "pipeline": config.pipeline,
                    "results": summary.result_count,
                    "run_directory": summary.run_dir.name,
                }
            )
            return 0
        if args.command == "evaluate":
            _print_json(evaluate_predictions(args.gold, args.predictions, args.tolerance))
            return 0
    except Exception as exc:
        print(json.dumps({"error_class": exc.__class__.__name__}, sort_keys=True), file=sys.stderr)
        return 2
    return 2


def _print_json(value) -> None:
    print(json.dumps(value, sort_keys=True, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
