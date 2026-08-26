# Clinical Trial QA Pipelines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a vendor-neutral, evidence-grounded Python package that runs and evaluates P1–P6 on the supplied 23-question clinical-note CSV.

**Architecture:** A standard-library-first package turns each CSV note into a validated case, sends one question at a time through interchangeable LLM clients, and postprocesses every draft through exact-quote and deterministic-numeric gates. P1–P6 compose the same inference, routing, aggregation, role-panel, and full-document verification units so experimental differences remain explicit and testable.

**Tech Stack:** Python 3.11+, standard library runtime, pytest development tests, JSON/JSONL configuration and output, OpenAI-compatible HTTP via `urllib.request`.

**Spec:** `docs/superpowers/specs/2026-08-24-clinical-trial-qa-pipelines-design.md`

## Global Constraints

- Support exactly the 8 source columns and 23 criteria in the approved spec; never modify the input CSV.
- Use only Python 3.11+ standard-library modules at runtime; `pytest>=8` is a development extra.
- Never log or package clinical-note text, API keys, prediction outputs, caches, or the attached CSV.
- Accept a final evidence quote only when it is an exact substring of the full note; recompute offsets in code.
- Compute every numeric minimum/maximum in Python from all validated candidates, retaining raw candidates and source quotes.
- Limit role debate to `0..2` review rounds and prefer abstention on unresolved conflict.
- Split data only by `note_id`; never split the 23 rows of one note across partitions.
- Keep all six pipelines behind `Pipeline.run_case(case) -> list[QuestionResult]`.
- Preserve inference separately from verbatim evidence.

---

## File Map

| File | Responsibility |
|---|---|
| `pyproject.toml` | package metadata, Python floor, CLI entry point, pytest config |
| `.gitignore` | exclude inputs, outputs, secrets, caches, and SDD artifacts |
| `src/clinical_trial_qa/models.py` | enums and immutable data/result models with JSON conversion |
| `src/clinical_trial_qa/questions.py` | all 23 question specifications and lookup |
| `src/clinical_trial_qa/dataset.py` | CSV validation, note aggregation, note-level split |
| `src/clinical_trial_qa/evidence.py` | exact quote validation and immutable evidence ledger |
| `src/clinical_trial_qa/numeric.py` | numeric parsing, unit compatibility, deterministic reducer |
| `src/clinical_trial_qa/postprocess.py` | convert untrusted model drafts to grounded results |
| `src/clinical_trial_qa/llm.py` | client protocol, scripted test client, OpenAI-compatible HTTP client |
| `src/clinical_trial_qa/prompts.py` | versioned JSON-only prompts for answering, roles, routing, aggregation, verification |
| `src/clinical_trial_qa/sections.py` | section splitting, multi-label classification, recall-first routing |
| `src/clinical_trial_qa/aggregation.py` | evidence-gated deterministic/LLM-assisted aggregation |
| `src/clinical_trial_qa/verification.py` | bounded role panel and full-document verifier |
| `src/clinical_trial_qa/pipelines/base.py` | shared pipeline interface and per-question failure isolation |
| `src/clinical_trial_qa/pipelines/full.py` | P1, P2, P3 |
| `src/clinical_trial_qa/pipelines/routed.py` | P4, P5, P6 |
| `src/clinical_trial_qa/pipelines/factory.py` | pipeline construction and name validation |
| `src/clinical_trial_qa/config.py` | strict JSON config loading and client construction |
| `src/clinical_trial_qa/runner.py` | run manifest and JSONL execution without note text |
| `src/clinical_trial_qa/evaluation.py` | legacy-label and selective metrics |
| `src/clinical_trial_qa/cli.py` | validate-data, split, run, evaluate commands |
| `configs/example.json` | safe environment-variable-based example |
| `README.md` | setup, P1–P6 semantics, commands, output contract, data governance |

---

### Task 1: Domain models, 23 question specifications, and note-safe dataset layer

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `src/clinical_trial_qa/__init__.py`
- Create: `src/clinical_trial_qa/models.py`
- Create: `src/clinical_trial_qa/questions.py`
- Create: `src/clinical_trial_qa/dataset.py`
- Create: `tests/test_dataset.py`
- Create: `tests/test_questions.py`

**Interfaces:**
- Produces: `DocumentStatus`, `QuestionType`, `Reducer`, `EvidenceSpan`, `QuestionSpec`, `QuestionItem`, `NoteCase`, `ModelDraft`, `QuestionResult`.
- Produces: `get_question_spec(criterion: str) -> QuestionSpec`, `all_question_specs() -> tuple[QuestionSpec, ...]`.
- Produces: `load_cases(path: Path) -> list[NoteCase]`, `validate_dataset(path: Path) -> DatasetReport`, `split_note_ids(cases, seed, train_fraction, validation_fraction) -> DatasetSplit`.

- [ ] **Step 1: Write failing tests for the 23-spec registry**

```python
def test_registry_has_15_boolean_and_8_numeric_specs():
    specs = all_question_specs()
    assert len(specs) == 23
    assert sum(s.question_type is QuestionType.BOOLEAN for s in specs) == 15
    assert sum(s.question_type is QuestionType.NUMERIC for s in specs) == 8
    assert get_question_spec("PLT").reducer is Reducer.MIN
    assert get_question_spec("BILI").reducer is Reducer.MAX
```

- [ ] **Step 2: Run the registry test and verify RED**

Run: `PYTHONPATH=src pytest tests/test_questions.py -q`

Expected: collection fails because `clinical_trial_qa.questions` does not exist.

- [ ] **Step 3: Implement enums, dataclasses, JSON conversion, and all 23 specs**

Use frozen dataclasses where feasible. `QuestionResult.to_dict()` must emit only serializable primitives and never include note text. Encode LVEF compatibility clipping as `QuestionSpec.output_floor_or_cap`, not in prompt prose alone.

- [ ] **Step 4: Run the registry test and verify GREEN**

Run: `PYTHONPATH=src pytest tests/test_questions.py -q`

Expected: 2 or more tests pass.

- [ ] **Step 5: Write failing dataset tests**

```python
def test_loader_groups_rows_without_copying_a_note_across_cases(tmp_path):
    path = write_csv(tmp_path, two_complete_note_fixtures())
    cases = load_cases(path)
    assert [case.note_id for case in cases] == ["n1", "n2"]
    assert all(len(case.questions) == 23 for case in cases)

def test_note_level_split_has_no_identifier_overlap():
    cases = tuple(make_case(str(i)) for i in range(20))
    split = split_note_ids(cases, seed=42, train_fraction=.6, validation_fraction=.2)
    assert set(split.train).isdisjoint(split.validation)
    assert set(split.train).isdisjoint(split.test)
    assert set(split.validation).isdisjoint(split.test)
    assert set(split.train) | set(split.validation) | set(split.test) == {str(i) for i in range(20)}
```

```python
@pytest.mark.parametrize("mutation,error_fragment", [
    ("drop_text_column", "missing columns"),
    ("change_second_row_text", "inconsistent text"),
    ("duplicate_first_criterion", "duplicate criterion"),
    ("drop_last_row", "expected 23 criteria"),
])
def test_loader_rejects_structural_data_errors(tmp_path, mutation, error_fragment):
    path = write_mutated_csv(tmp_path, mutation)
    with pytest.raises(DatasetValidationError, match=error_fragment):
        load_cases(path)
```

- [ ] **Step 6: Run dataset tests and verify RED**

Run: `PYTHONPATH=src pytest tests/test_dataset.py -q`

Expected: fails because dataset functions are absent.

- [ ] **Step 7: Implement CSV validation, aggregation, and note-level splitting**

Read with `csv.DictReader(encoding="utf-8-sig")`. Keep identifiers as strings. Return a report containing row count, note count, criterion count, warnings, and errors; never include note text in errors.

- [ ] **Step 8: Run Task 1 tests and verify GREEN**

Run: `PYTHONPATH=src pytest tests/test_questions.py tests/test_dataset.py -q`

Expected: all Task 1 tests pass.

- [ ] **Step 9: Add packaging metadata and exclusions**

Set `requires-python = ">=3.11"`, register `clinical-trial-qa = "clinical_trial_qa.cli:main"`, and exclude `upload/`, `inputs/`, `outputs/`, `.env*`, `*.csv`, `__pycache__/`, `.pytest_cache/`, `.superpowers/`.

- [ ] **Step 10: Commit Task 1**

```bash
git add .gitignore pyproject.toml src tests
git commit -m "feat: add clinical QA data model and dataset loader"
```

---

### Task 2: Exact evidence gate and deterministic numeric postprocessing

**Files:**
- Create: `src/clinical_trial_qa/evidence.py`
- Create: `src/clinical_trial_qa/numeric.py`
- Create: `src/clinical_trial_qa/postprocess.py`
- Create: `tests/test_evidence.py`
- Create: `tests/test_numeric.py`
- Create: `tests/test_postprocess.py`

**Interfaces:**
- Consumes: Task 1 models and `QuestionSpec`.
- Produces: `EvidenceValidator.validate(note_text: str, span: EvidenceSpan) -> EvidenceSpan | None`.
- Produces: `EvidenceLedger.from_spans(spans)`, `EvidenceLedger.contains(span)`.
- Produces: `parse_numeric(raw: str) -> float | None`, `reduce_candidates(spec, candidates, related_mention_present) -> NumericDecision`.
- Produces: `postprocess_draft(case, item, draft, source_scope) -> QuestionResult`.

- [ ] **Step 1: Write exact-evidence failing tests**

```python
def test_exact_quote_recomputes_wrong_offsets():
    text = "Labs: PLT 143 K/uL. Later PLT 91 K/uL."
    span = EvidenceSpan(quote="PLT 91 K/uL", start_char=0, end_char=3)
    fixed = EvidenceValidator().validate(text, span)
    assert fixed is not None
    assert text[fixed.start_char:fixed.end_char] == fixed.quote

def test_nonexistent_quote_is_rejected():
    assert EvidenceValidator().validate("PLT 91", EvidenceSpan(quote="PLT 19")) is None
```

- [ ] **Step 2: Run evidence tests and verify RED**

Run: `PYTHONPATH=src pytest tests/test_evidence.py -q`

Expected: fails because evidence validator is absent.

- [ ] **Step 3: Implement exact matching and evidence ledger**

For duplicate quotes, choose the match closest to a valid model-supplied start offset, otherwise the earliest. Rebuild immutable offsets and preserve raw quote exactly.

- [ ] **Step 4: Run evidence tests and verify GREEN**

Run: `PYTHONPATH=src pytest tests/test_evidence.py -q`

Expected: all evidence tests pass.

- [ ] **Step 5: Write numeric reducer failing tests**

```python
def test_plt_minimum_uses_all_validated_candidates():
    candidates = (candidate("PLT 143 K/uL", "143", "K/uL"), candidate("PLT 91 K/uL", "91", "K/uL"))
    decision = reduce_candidates(get_question_spec("PLT"), candidates, related_mention_present=True)
    assert decision.status is DocumentStatus.VALUE_AVAILABLE
    assert decision.answer == 91.0
    assert len(decision.candidates) == 2

def test_lvef_at_or_above_55_is_legacy_clipped_but_raw_is_retained():
    decision = reduce_candidates(get_question_spec("lvef"), (candidate("EF 60%", "60", "%"),), True)
    assert decision.answer == 55.0
    assert decision.candidates[0].normalized_value == 60.0

def test_incompatible_units_abstain_without_dropping_candidates():
    decision = reduce_candidates(get_question_spec("CREAT"), (candidate("Creat 1 mg/dL", "1", "mg/dL"), candidate("Creat 90 umol/L", "90", "umol/L")), True)
    assert decision.status is DocumentStatus.INDETERMINATE
    assert len(decision.candidates) == 2
```

- [ ] **Step 6: Run numeric tests and verify RED**

Run: `PYTHONPATH=src pytest tests/test_numeric.py -q`

Expected: fails because numeric reducer is absent.

- [ ] **Step 7: Implement numeric parsing and reducer**

Parse optional `<`, `>`, `<=`, `>=`, sign, commas, decimals, and percentages while retaining `raw_value`. Treat one explicit unit plus missing units as compatible; treat two different explicit normalized units as incompatible. Do not convert units.

- [ ] **Step 8: Write postprocessor failing tests**

```python
def test_boolean_yes_without_valid_quote_becomes_indeterminate():
    result = postprocess_draft(case("No psychiatric history."), bipolar_item(), ModelDraft(status="yes", evidence=(EvidenceSpan(quote="Bipolar disorder"),)), "full")
    assert result.document_status is DocumentStatus.INDETERMINATE
    assert result.answer is None

def test_numeric_result_is_recomputed_not_trusted_from_model():
    draft = numeric_draft(model_answer=999, quotes=[("PLT 143", "143"), ("PLT 91", "91")])
    result = postprocess_draft(case("PLT 143; PLT 91"), plt_item(), draft, "full")
    assert result.answer == 91.0
```

- [ ] **Step 9: Run postprocessor tests and verify RED**

Run: `PYTHONPATH=src pytest tests/test_postprocess.py -q`

Expected: fails because postprocessor is absent.

- [ ] **Step 10: Implement grounded postprocessing and verify Task 2 GREEN**

Run: `PYTHONPATH=src pytest tests/test_evidence.py tests/test_numeric.py tests/test_postprocess.py -q`

Expected: all Task 2 tests pass.

- [ ] **Step 11: Commit Task 2**

```bash
git add src/clinical_trial_qa/evidence.py src/clinical_trial_qa/numeric.py src/clinical_trial_qa/postprocess.py tests
git commit -m "feat: ground evidence and reduce numeric answers deterministically"
```

---

### Task 3: LLM boundary, versioned prompts, and recall-first section routing

**Files:**
- Create: `src/clinical_trial_qa/llm.py`
- Create: `src/clinical_trial_qa/prompts.py`
- Create: `src/clinical_trial_qa/sections.py`
- Create: `tests/test_llm.py`
- Create: `tests/test_sections.py`

**Interfaces:**
- Consumes: Task 1 models/specs.
- Produces: `LLMRequest`, `LLMResponse`, `LLMClient` protocol, `ScriptedLLMClient`, `OpenAICompatibleClient`.
- Produces: `build_answer_request`, `build_role_request`, `build_aggregation_request`, `build_verification_request`, `build_section_label_request`.
- Produces: `SectionSplitter.split(text) -> tuple[Section, ...]`, `RecallFirstRouter.route(case, spec) -> RoutedContext`.

- [ ] **Step 1: Write LLM adapter failing tests**

```python
def test_scripted_client_records_metadata_without_prompt_text():
    client = ScriptedLLMClient([{"document_status": "not_documented", "evidence": []}])
    response = client.generate(LLMRequest(task="answer", messages=(("user", "sensitive note"),)))
    assert response.data["document_status"] == "not_documented"
    assert client.calls[0].task == "answer"
    assert "sensitive note" not in repr(client.calls[0])

def test_openai_client_requires_key_before_network_call(monkeypatch):
    monkeypatch.delenv("MISSING_TEST_KEY", raising=False)
    client = OpenAICompatibleClient(base_url="https://example.invalid/v1", model="m", api_key_env="MISSING_TEST_KEY")
    with pytest.raises(ConfigurationError):
        client.generate(request())
```

- [ ] **Step 2: Run LLM tests and verify RED**

Run: `PYTHONPATH=src pytest tests/test_llm.py -q`

Expected: fails because the LLM module is absent.

- [ ] **Step 3: Implement adapters and bounded retry**

Use `urllib.request.Request`; parse `choices[0].message.content` as a JSON object. Support `max_retries` in `0..2`, timeout, temperature, and response metadata. Call records may store prompt hash and character counts but not prompt content.

- [ ] **Step 4: Run LLM tests and verify GREEN**

Run: `PYTHONPATH=src pytest tests/test_llm.py -q`

Expected: all adapter tests pass.

- [ ] **Step 5: Write routing failing tests**

```python
def test_lab_question_routes_lab_section_and_preserves_full_offsets():
    text = "HISTORY:\nNo stroke.\n\nLABS:\nPLT 91 K/uL"
    routed = RecallFirstRouter(top_k=3).route(make_case(text), get_question_spec("PLT"))
    assert "PLT 91 K/uL" in routed.text
    assert routed.sections[0].start_char == text.index("LABS:")

def test_router_falls_back_to_full_text_when_no_label_matches():
    text = "Narrative without headings"
    routed = RecallFirstRouter(top_k=3).route(make_case(text), get_question_spec("bipolar"))
    assert routed.used_full_text_fallback is True
    assert routed.text == text
```

```python
def test_router_keeps_multiple_labels_and_respects_top_k():
    routed = RecallFirstRouter(top_k=2).route(multilabel_case(), get_question_spec("recent_stroke"))
    assert len(routed.sections) <= 2
    assert {"neurology", "history"} & set(routed.sections[0].labels)

def test_malformed_llm_labels_fall_back_to_heuristics():
    client = ScriptedLLMClient([{"labels": "not-a-list"}])
    routed = RecallFirstRouter(top_k=2, label_client=client).route(lab_case(), get_question_spec("PLT"))
    assert "laboratory" in routed.sections[0].labels
```

- [ ] **Step 6: Run section tests and verify RED**

Run: `PYTHONPATH=src pytest tests/test_sections.py -q`

Expected: fails because section routing is absent.

- [ ] **Step 7: Implement versioned prompts, splitter, labels, and router**

All answer prompts distinguish verbatim `evidence` from `inference`, demand all numeric candidates, and include a JSON field contract. Static `QuestionSpec.labels` are the default question labels; an optional label client may add labels but not remove static ones.

- [ ] **Step 8: Run Task 3 tests and verify GREEN**

Run: `PYTHONPATH=src pytest tests/test_llm.py tests/test_sections.py -q`

Expected: all Task 3 tests pass.

- [ ] **Step 9: Commit Task 3**

```bash
git add src/clinical_trial_qa/llm.py src/clinical_trial_qa/prompts.py src/clinical_trial_qa/sections.py tests
git commit -m "feat: add LLM adapters and recall-first routing"
```

---

### Task 4: Evidence-gated aggregation, bounded roles, full verifier, and P1–P6

**Files:**
- Create: `src/clinical_trial_qa/aggregation.py`
- Create: `src/clinical_trial_qa/verification.py`
- Create: `src/clinical_trial_qa/pipelines/__init__.py`
- Create: `src/clinical_trial_qa/pipelines/base.py`
- Create: `src/clinical_trial_qa/pipelines/full.py`
- Create: `src/clinical_trial_qa/pipelines/routed.py`
- Create: `src/clinical_trial_qa/pipelines/factory.py`
- Create: `tests/test_aggregation.py`
- Create: `tests/test_verification.py`
- Create: `tests/test_pipelines.py`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: `EvidenceGatedAggregator.aggregate(case, item, proposals) -> QuestionResult`.
- Produces: `RolePanel.answer(case, item, context, client) -> QuestionResult` with `max_rounds <= 2`.
- Produces: `FullDocumentVerifier.verify(case, item, candidate) -> QuestionResult`.
- Produces: `P1Pipeline` through `P6Pipeline`, `build_pipeline(name, components) -> Pipeline`.

- [ ] **Step 1: Write aggregation failing tests**

```python
def test_equal_boolean_votes_abstain():
    result = EvidenceGatedAggregator().aggregate(case, item, (yes_result(), no_result()))
    assert result.document_status is DocumentStatus.INDETERMINATE

def test_numeric_ensemble_reduces_union_of_valid_candidates():
    result = EvidenceGatedAggregator().aggregate(case, plt_item, (result_with_value(143), result_with_value(91)))
    assert result.answer == 91.0
    assert {c.normalized_value for c in result.candidate_values} == {91.0, 143.0}
```

```python
def test_llm_aggregator_cannot_select_evidence_outside_ledger():
    client = ScriptedLLMClient([{"selected_proposal_ids": ["unknown"], "confidence": 1.0}])
    result = EvidenceGatedAggregator(client).aggregate(case, item, (yes_result(), no_result()))
    assert result.document_status is DocumentStatus.INDETERMINATE
    assert "aggregator_selected_unknown_proposal" in result.validation_errors
```

- [ ] **Step 2: Run aggregation tests and verify RED**

Run: `PYTHONPATH=src pytest tests/test_aggregation.py -q`

Expected: fails because aggregation is absent.

- [ ] **Step 3: Implement deterministic and optional LLM-assisted aggregation**

Give the aggregation LLM opaque proposal/evidence IDs and require selected proposal IDs. Never accept free-text quotes from the aggregator. Fall back deterministically if its JSON is invalid.

- [ ] **Step 4: Run aggregation tests and verify GREEN**

Run: `PYTHONPATH=src pytest tests/test_aggregation.py -q`

Expected: all aggregation tests pass.

- [ ] **Step 5: Write role and verifier failing tests**

```python
def test_role_panel_never_exceeds_two_review_rounds():
    client = ScriptedLLMClient(script_for_role_panel())
    RolePanel(max_rounds=2).answer(case, item, full_context(case), client)
    assert sum(call.task == "role_review" for call in client.calls) == 2

def test_verifier_rejects_revision_with_invented_quote():
    verifier = FullDocumentVerifier(ScriptedLLMClient([revision_with_quote("not in note")]))
    checked = verifier.verify(case, item, candidate)
    assert checked.document_status is DocumentStatus.INDETERMINATE
    assert "verifier_revision_not_grounded" in checked.validation_errors
```

- [ ] **Step 6: Run verification tests and verify RED**

Run: `PYTHONPATH=src pytest tests/test_verification.py -q`

Expected: fails because role/verifier components are absent.

- [ ] **Step 7: Implement role panel and full-document verifier**

Boolean roles are assertion/negation/experiencer, temporality, and evidence fidelity. Numeric roles are numeric completeness, temporality, and evidence fidelity. Review rounds see the current result and may add a grounded proposal; aggregator adjudicates after each round.

- [ ] **Step 8: Write P1–P6 orchestration failing tests**

```python
@pytest.mark.parametrize("name,primary_calls,uses_router,uses_verifier", [
    ("p1", 1, False, False),
    ("p2", 3, False, False),
    ("p3", 3, False, False),
    ("p4", 1, True, True),
    ("p5", 3, True, True),
    ("p6", 3, True, True),
])
def test_pipeline_topology(name, primary_calls, uses_router, uses_verifier):
    components = scripted_components(name)
    results = build_pipeline(name, components).run_case(one_question_case())
    assert len(results) == 1
    assert components.observed_primary_calls >= primary_calls
    assert components.router.used is uses_router
    assert components.verifier.used is uses_verifier
```

```python
def test_one_question_failure_does_not_abort_remaining_questions():
    client = ScriptedLLMClient([{"bad": "shape"}, valid_not_documented_draft()])
    results = P1Pipeline(client).run_case(two_question_case())
    assert len(results) == 2
    assert results[0].validation_errors
    assert results[1].document_status is DocumentStatus.NOT_DOCUMENTED
```

- [ ] **Step 9: Run pipeline tests and verify RED**

Run: `PYTHONPATH=src pytest tests/test_pipelines.py -q`

Expected: fails because pipeline classes are absent.

- [ ] **Step 10: Implement shared base, P1–P3, P4–P6, and factory**

P2 and P5 require at least two primary clients. P3 and P6 require one role client and enforce `max_rounds <= 2`. P4–P6 route before primary inference and call the full-document verifier exactly once per question.

- [ ] **Step 11: Run Task 4 tests and verify GREEN**

Run: `PYTHONPATH=src pytest tests/test_aggregation.py tests/test_verification.py tests/test_pipelines.py -q`

Expected: all Task 4 tests pass.

- [ ] **Step 12: Commit Task 4**

```bash
git add src/clinical_trial_qa/aggregation.py src/clinical_trial_qa/verification.py src/clinical_trial_qa/pipelines tests
git commit -m "feat: implement six evidence-grounded QA pipelines"
```

---

### Task 5: Configuration, runner, evaluation, CLI, documentation, and attached-data validation

**Files:**
- Create: `src/clinical_trial_qa/config.py`
- Create: `src/clinical_trial_qa/runner.py`
- Create: `src/clinical_trial_qa/evaluation.py`
- Create: `src/clinical_trial_qa/cli.py`
- Create: `configs/example.json`
- Create: `README.md`
- Create: `tests/test_config.py`
- Create: `tests/test_evaluation.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: `load_config(path) -> AppConfig`, `run_dataset(config, cases, output_dir, limit_notes) -> RunSummary`, `evaluate_predictions(gold_path, prediction_path, tolerance) -> dict`.
- Produces: console entry point `clinical-trial-qa` with `validate-data`, `split`, `run`, `evaluate`.

- [ ] **Step 1: Write configuration and evaluation failing tests**

```python
def test_config_rejects_p5_with_only_one_primary_client(tmp_path):
    path = write_json(tmp_path, p5_config(primary_count=1))
    with pytest.raises(ConfigurationError):
        load_config(path)

def test_selective_metrics_separate_accuracy_from_coverage(tmp_path):
    gold = write_gold(tmp_path, [("n1", "afib", "Yes"), ("n2", "afib", "No")])
    pred = write_predictions(tmp_path, [("n1", "afib", "yes", "yes"), ("n2", "afib", "indeterminate", None)])
    metrics = evaluate_predictions(gold, pred, tolerance=1e-6)
    assert metrics["overall"]["coverage"] == .5
    assert metrics["overall"]["selective_accuracy"] == 1.0
    assert metrics["overall"]["accuracy_with_abstention_wrong"] == .5
```

- [ ] **Step 2: Run configuration/evaluation tests and verify RED**

Run: `PYTHONPATH=src pytest tests/test_config.py tests/test_evaluation.py -q`

Expected: fails because configuration and evaluation modules are absent.

- [ ] **Step 3: Implement strict JSON config and metrics**

Reject unknown pipeline names, missing client fields, plaintext `api_key`, invalid ensemble size, and debate rounds outside `0..2`. Emit overall and per-criterion metrics with explicit denominators.

- [ ] **Step 4: Run configuration/evaluation tests and verify GREEN**

Run: `PYTHONPATH=src pytest tests/test_config.py tests/test_evaluation.py -q`

Expected: all tests pass.

- [ ] **Step 5: Write runner and CLI failing tests**

```python
def test_validate_data_cli_reports_counts_without_note_text(csv_fixture, capsys):
    rc = main(["validate-data", "--input", str(csv_fixture)])
    output = capsys.readouterr().out
    assert rc == 0
    assert '"notes": 1' in output
    assert "sensitive clinical sentence" not in output

def test_runner_jsonl_never_contains_note_text(tmp_path):
    summary = run_dataset(scripted_config(), [case_with_text("secret note")], tmp_path, limit_notes=1)
    assert "secret note" not in summary.predictions_path.read_text()
    assert "secret note" not in summary.manifest_path.read_text()
```

- [ ] **Step 6: Run runner/CLI tests and verify RED**

Run: `PYTHONPATH=src pytest tests/test_cli.py -q`

Expected: fails because runner and CLI are absent.

- [ ] **Step 7: Implement runner, manifests, CLI, and safe example config**

Use atomic temporary-file rename for JSONL and manifest writes. Include pipeline, seed, prompt version, model names, started/finished UTC timestamps, call counts, and config hash; exclude note text and API key values.

- [ ] **Step 8: Run CLI tests and verify GREEN**

Run: `PYTHONPATH=src pytest tests/test_cli.py -q`

Expected: all CLI tests pass.

- [ ] **Step 9: Write README**

Document installation, attached CSV schema, P1–P6 table, recommended starting point P5 and cost/accuracy alternative P4, configuration, commands, result schema, evidence guarantees, numeric reducer, evaluation limitations, data governance, and extension points. State that source evidence recall cannot be measured until expert span annotations are added.

- [ ] **Step 10: Validate the supplied CSV read-only**

Run:

```bash
PYTHONPATH=src python -m clinical_trial_qa.cli validate-data --input ../upload/annotated_apixaban_combined.csv
```

Expected JSON includes `rows: 2300`, `notes: 100`, `criteria: 23`, and no note text.

- [ ] **Step 11: Run full verification suite**

Run:

```bash
PYTHONPATH=src pytest -q
python -m compileall -q src
PYTHONPATH=src python -m clinical_trial_qa.cli --help
```

Expected: all tests pass, compileall exits 0, CLI help exits 0.

- [ ] **Step 12: Commit Task 5**

```bash
git add src/clinical_trial_qa/config.py src/clinical_trial_qa/runner.py src/clinical_trial_qa/evaluation.py src/clinical_trial_qa/cli.py configs README.md tests pyproject.toml
git commit -m "feat: add reproducible runner evaluation and CLI"
```

---

### Task 6: Final review, release checks, and PHI-safe archive

**Files:**
- Modify only files required by final review findings.
- Create outside Git working tree: `../clinical-trial-qa-pipelines.zip`.

**Interfaces:**
- Consumes: complete implementation and test suite.
- Produces: reviewed Git tree and archive containing tracked source files only.

- [ ] **Step 1: Request whole-branch code review**

Review the complete diff from the design-spec commit through `HEAD` for spec compliance, evidence leakage, numeric correctness, pipeline topology, error isolation, and test quality.

- [ ] **Step 2: Fix Critical and Important findings with tests first**

For every bug, add a failing regression test, observe the intended failure, make the minimal correction, and re-run the scoped suite.

- [ ] **Step 3: Re-run fresh full verification**

Run:

```bash
PYTHONPATH=src pytest -q
python -m compileall -q src
PYTHONPATH=src python -m clinical_trial_qa.cli validate-data --input ../upload/annotated_apixaban_combined.csv
git status --short
```

Expected: tests and compile pass; validation reports the expected counts; only intentional tracked files remain.

- [ ] **Step 4: Build an archive from tracked files only**

Run `git archive --format=zip --output=../clinical-trial-qa-pipelines.zip HEAD`. Then inspect the member list and assert it contains no `.csv`, `.env`, `outputs/`, caches, or patient data.

- [ ] **Step 5: Record release evidence**

Capture test count, validation counts, final commit, archive SHA-256, and any remaining documented limitations in the handoff.
