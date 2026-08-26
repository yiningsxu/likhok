#!/usr/bin/env python3
"""Run the complete test suite using only the Python standard library."""

from __future__ import annotations

import contextlib
import importlib.util
import inspect
import io
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class _Captured:
    def __init__(self, out: str, err: str) -> None:
        self.out = out
        self.err = err


class _Capture:
    def __init__(self) -> None:
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()

    def readouterr(self) -> _Captured:
        captured = _Captured(self.stdout.getvalue(), self.stderr.getvalue())
        for stream in (self.stdout, self.stderr):
            stream.seek(0)
            stream.truncate(0)
        return captured


class _MonkeyPatch:
    def __init__(self) -> None:
        self._undo: list[Callable[[], None]] = []

    def setattr(self, target: Any, name: str, value: Any) -> None:
        previous = getattr(target, name)
        setattr(target, name, value)
        self._undo.append(lambda: setattr(target, name, previous))

    def setenv(self, name: str, value: str) -> None:
        previous = os.environ.get(name)
        os.environ[name] = value
        self._undo.append(lambda: _restore_environment(name, previous))

    def delenv(self, name: str, raising: bool = True) -> None:
        if name not in os.environ:
            if raising:
                raise KeyError(name)
            return
        previous = os.environ.pop(name)
        self._undo.append(lambda: os.environ.__setitem__(name, previous))

    def undo(self) -> None:
        for action in reversed(self._undo):
            action()
        self._undo.clear()


def _restore_environment(name: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


def _load(path: Path) -> ModuleType:
    module_name = f"stdlib_tests_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("test module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tests(module: ModuleType) -> list[tuple[str, Callable[..., None]]]:
    return sorted(
        (name, value)
        for name, value in vars(module).items()
        if name.startswith("test_") and inspect.isfunction(value) and value.__module__ == module.__name__
    )


def _run_one(function: Callable[..., None]) -> None:
    capture = _Capture()
    monkeypatch = _MonkeyPatch()
    with tempfile.TemporaryDirectory(prefix="clinical-trial-qa-tests-") as directory:
        fixtures = {
            "tmp_path": Path(directory),
            "capsys": capture,
            "monkeypatch": monkeypatch,
        }
        parameters = inspect.signature(function).parameters
        unknown = set(parameters) - set(fixtures)
        if unknown:
            raise RuntimeError("unsupported standard-library test fixture")
        arguments = {name: fixtures[name] for name in parameters}
        try:
            with contextlib.redirect_stdout(capture.stdout), contextlib.redirect_stderr(capture.stderr):
                function(**arguments)
        finally:
            monkeypatch.undo()


def main() -> int:
    failures: list[tuple[str, str]] = []
    count = 0
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        try:
            module = _load(path)
        except Exception as exc:
            failures.append((path.name, exc.__class__.__name__))
            continue
        for name, function in _tests(module):
            count += 1
            try:
                _run_one(function)
            except Exception as exc:
                failures.append((f"{path.name}::{name}", exc.__class__.__name__))
    if failures:
        for test_name, error_class in failures:
            print(f"FAIL {test_name} [{error_class}]")
        print(f"FAILED {len(failures)} of {count} tests")
        return 1
    print(f"PASS {count} tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
