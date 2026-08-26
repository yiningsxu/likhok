# Clinical Trial QA Pipelines

Research code for comparing six evidence-grounded LLM pipelines on the same clinical notes and 23 trial criteria. It is not a medical device and must not be used to make enrollment or treatment decisions.

## Install and test

Python 3.11 or newer is required. Runtime code uses only the standard library.

```bash
python -m pip install -e .
PYTHONPATH=src python tests/run_all.py
```

`pytest` remains an optional development extra:

```bash
python -m pip install -e '.[dev]'
PYTHONPATH=src pytest -q
```

The standard-library runner exercises the complete suite, including temporary-path, output-capture, environment, and monkeypatch-dependent tests. It exits nonzero on any failure and does not provide a fake `pytest` package.

## Source CSV

The input is UTF-8 CSV with exactly these eight columns:

| Column | Meaning |
|---|---|
| `text` | Full clinical note; never printed by validation or the runner |
| `note_id` | Note identifier and split unit |
| `hadm_id` | Admission identifier |
| `criterion` | One of the approved 23 criterion identifiers |
| `question_type` | `yes` or `numeric` |
| `question` | Question wording |
| `answer` | Legacy expert label |
| `not_specified` | Legacy missing-value flag |

Each note must contain all 23 criteria exactly once. `text` and `hadm_id` must be consistent within a note. Validation and splitting are read-only with respect to the source CSV; splitting is always by `note_id`, never by row.

## Pipelines

| Pipeline | Context | Primary inference | Aggregation | Final audit |
|---|---|---|---|---|
| P1 | Full note | One client | None | Common postprocessor |
| P2 | Full note | Independent clients | Evidence-gated | Common postprocessor |
| P3 | Full note | Error-role panel | Bounded adjudication | Common postprocessor |
| P4 | Routed sections | One client | None | Full-note verifier |
| P5 | Routed sections | Independent clients | Evidence-gated | Full-note verifier |
| P6 | Routed sections | Error-role panel | Bounded adjudication | Full-note verifier |

Start with **P5** when accuracy and independent-model corroboration are the priority. **P4** is the lower-cost accuracy alternative because it makes one routed primary call while retaining the mandatory full-note audit. P3 and P6 permit only 0–2 review rounds.

## Configuration

[`configs/example.json`](configs/example.json) is a P5 template. Configuration is strict JSON: unknown fields and pipelines, missing client fields, non-finite numbers (including exponent overflow), too-small P2/P5 ensembles, invalid debate rounds, and note-like model labels are rejected. P1/P4 require exactly one primary client; P2/P5 require at least two; P3/P6 require a role client; and P4–P6 require a verifier. Router-label clients are allowed only for routed pipelines, while aggregator clients are allowed only where aggregation or adjudication participates. Configured-but-unused roles are rejected. The evidence-gated aggregator itself is always present for P2/P5 and may be deterministic when no optional aggregator client is configured. Every ensemble entry constructs a fresh stateful client, even when two entries use the same model name.

Provider credentials must never appear in JSON. `api_key_env` must match `[A-Za-z_][A-Za-z0-9_]*` and names an environment variable; its value is read only when the HTTP request is made. Token-like, whitespace-containing, and path-like names are rejected, as are plaintext `api_key` fields. The manifest hash is computed from credential-safe configuration fields and cannot depend on the environment variable value.

The supported runtime provider is `openai_compatible`, targeting a JSON-compatible `/chat/completions` endpoint. A deterministic `scripted` provider is also accepted for tests and dry runs; its response bodies are represented in hashes and counts only, never copied into manifests.

## Commands

```bash
clinical-trial-qa validate-data --input annotated_apixaban_combined.csv
clinical-trial-qa split --input annotated_apixaban_combined.csv --output-dir splits --seed 42
clinical-trial-qa run --config configs/example.json --input annotated_apixaban_combined.csv --pipeline p5 --limit-notes 1
clinical-trial-qa evaluate --gold annotated_apixaban_combined.csv --predictions outputs/<run-directory>/predictions.jsonl --tolerance 1e-6
```

Equivalent source-checkout commands use `PYTHONPATH=src python -m clinical_trial_qa.cli ...`.

`validate-data` prints counts and validity only—not note text or diagnostic contents. `split` atomically writes `train.json`, `validation.json`, and `test.json` containing note identifiers. `run` prints a safe generated `run_directory` name. It writes `predictions.jsonl` and `manifest.json` into a hidden sibling temporary directory, flushes both files and the directory, then atomically renames that directory to `outputs/<pipeline>-<run-id>/`. The two artifacts therefore become visible as one unique run-level unit; a failure before the final directory rename leaves neither artifact final. Invalid CLI arguments return class-only JSON without echoing argv values or raising `SystemExit` from the callable `main` API.

`run --pipeline` may override the configured pipeline only when the existing client-role configuration is an exact valid topology for the requested value. Repeating the same pipeline value is valid. An incompatible override fails with class-only `ConfigurationError` before any client is constructed or called and before an output directory is created.

## Persisted results and reproducibility

Rich `QuestionResult` objects exist in memory with validated evidence and numeric candidates. The runner deliberately persists a redacted evaluation projection:

```json
{
  "note_id": "n1",
  "criterion": "PLT",
  "question_type": "numeric",
  "document_status": "value_available",
  "answer": 91.0,
  "confidence": 0.91,
  "evidence_count": 2,
  "evidence_valid_count": 2,
  "candidate_value_count": 3,
  "provenance_count": 4,
  "validation_error_count": 0
}
```

Persisted predictions and manifests never contain note text, evidence quotations, inference, prompts, API-key values, exception messages, or free-text validation errors. Boolean `yes`/`no` statuses persist the matching canonical answer regardless of an untrusted contradictory answer; abstaining statuses persist `null`. Numeric `value_available` persists only with a finite numeric answer and otherwise becomes `indeterminate` with `null`. Everything else is represented by safe counts.

The manifest records pipeline, seed, prompt version, compact identifiers for participating models only, start/finish UTC timestamps, per-purpose and total call counts, result-status counts, note/result counts, a run ID, and a SHA-256 configuration hash. It does not store the configuration document itself.

## Evidence and numeric guarantees

Evidence is accepted in memory only when the quotation is an exact substring of the full note; offsets are recomputed in Python. Aggregators and verifiers may select only entries already present in the validated evidence ledger. A failed routed component still reaches the mandatory full-note verifier in P4–P6, and question-level failures do not abort the rest of a note.

Numeric answers are never trusted directly from model prose. Python validates every candidate token and unit, retains all compatible candidates, then applies the registered reducer: `chads2`, `blood_glucose`, `CREAT`, `AST`, and `BILI` use the maximum; `lvef`, `PLT`, and `HGB` use the minimum. Mixed incompatible units abstain. For compatibility with the legacy dataset, LVEF minima of at least 55 are emitted as 55 while original candidates remain available in memory.

## Evaluation semantics and limitations

Evaluation validates every gold criterion against the approved 23-spec registry, requires the CSV `question_type` to match that registry, and rejects unknown or duplicate keys before they can become metric dictionary keys. `not_specified` accepts only explicit `0`/`1` or documented boolean spellings; unknown tokens are errors. Legacy boolean `Yes` maps to document status `yes`; legacy `No` is compatible with either `no` or `not_documented`. For numeric rows, `not_specified` marks the legacy missing target; a blank numeric answer is also treated as missing and counted separately when the flag is zero, so it is never parsed as zero. `indeterminate`, invalid predictions, and missing prediction rows are abstentions. A `legacy_mapping` section reports all mapping counts, including blank/flag mismatches.

Every rate reports its numerator and denominator. Overall output separates coverage, selective accuracy among covered predictions, and accuracy with abstentions counted wrong. It also includes criterion-macro accuracy, boolean macro-F1 and balanced accuracy (with aggregation sums and class counts), numeric MAE and tolerance accuracy, numeric missingness accuracy, and evidence exact-match validity computed from redacted valid/total counts. Per-criterion coverage, accuracy, and evidence validity use the same explicit-denominator convention.

The existing CSV has no expert evidence-span annotations. Automated checks can guarantee that accepted quotes existed in the source note, but **source evidence recall cannot be measured until expert span annotations are added**. Numeric candidate completeness likewise requires annotated candidate spans. Metrics do not establish clinical safety or prospective trial performance.

## Data governance

Do not commit or distribute source CSVs, predictions, caches, `.env` files, or credentials. Before using an external API, confirm the data-use agreement, de-identification policy, provider retention policy, permitted region, and institutional approval. This package does not anonymize notes. Prefer local or institutionally approved endpoints, begin with a one-note limit, and review redacted artifacts before scaling.

## Extension points

- Implement `LLMClient.generate(request)` for another JSON-only provider boundary.
- Add deterministic section labels through `RecallFirstRouter` without weakening full-note fallback.
- Extend the approved `QuestionSpec` registry with explicit aliases, units, routing labels, and reducers.
- Supply a custom evidence-gated aggregator or verifier through `PipelineComponents`.
- Add expert span columns and an evaluator extension for evidence recall and numeric candidate completeness.

Keep all extensions behind the shared `Pipeline.run_case(case) -> list[QuestionResult]` contract and preserve the persisted redaction boundary.
