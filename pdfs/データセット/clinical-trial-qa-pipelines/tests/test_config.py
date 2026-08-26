import json
from dataclasses import replace


def _write_json(tmp_path, payload):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _client(model="shared-model"):
    return {
        "provider": "openai_compatible",
        "base_url": "https://example.invalid/v1",
        "model": model,
        "api_key_env": "TEST_API_KEY",
    }


def _config(pipeline="p5", primary_count=2, debate_rounds=1):
    return {
        "pipeline": pipeline,
        "seed": 42,
        "debate_rounds": debate_rounds,
        "clients": {
            "primary": [_client() for _ in range(primary_count)],
            "verifier": _client("verifier-model"),
        },
    }


def _assert_configuration_error(path):
    from clinical_trial_qa.config import ConfigurationError, load_config

    try:
        load_config(path)
    except ConfigurationError:
        return
    raise AssertionError("ConfigurationError was not raised")


def test_config_rejects_p5_with_only_one_primary_client(tmp_path):
    """Would fail if P5 could silently run as a single-model pipeline."""
    _assert_configuration_error(_write_json(tmp_path, _config(primary_count=1)))


def test_config_rejects_unknown_fields_pipeline_and_plaintext_api_key(tmp_path):
    """Would fail if typos or a credential value survived strict parsing."""
    cases = []
    unknown_root = _config()
    unknown_root["temperature_policy"] = "hidden typo"
    cases.append(unknown_root)
    unknown_pipeline = _config()
    unknown_pipeline["pipeline"] = "p7"
    cases.append(unknown_pipeline)
    plaintext_key = _config()
    plaintext_key["clients"]["primary"][0]["api_key"] = "super-secret-key"
    cases.append(plaintext_key)

    for index, payload in enumerate(cases):
        path = tmp_path / f"invalid-{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        _assert_configuration_error(path)


def test_config_rejects_missing_client_fields_invalid_rounds_and_small_p2_ensemble(tmp_path):
    """Would fail if an invalid topology reached the pipeline factory."""
    cases = []
    missing_model = _config()
    del missing_model["clients"]["primary"][0]["model"]
    cases.append(missing_model)
    cases.append(_config(debate_rounds=-1))
    cases.append(_config(debate_rounds=3))
    small_p2 = _config(pipeline="p2", primary_count=1)
    small_p2["clients"].pop("verifier")
    cases.append(small_p2)

    for index, payload in enumerate(cases):
        path = tmp_path / f"topology-{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        _assert_configuration_error(path)


def test_config_builds_distinct_p2_and_p5_client_objects_even_for_same_model(tmp_path):
    """Would fail if repeated specs reused one stateful client object."""
    from clinical_trial_qa.config import load_config

    for pipeline in ("p2", "p5"):
        payload = _config(pipeline=pipeline)
        if pipeline == "p2":
            payload["clients"].pop("verifier")
        config = load_config(_write_json(tmp_path, payload))
        built = config.build_runtime().pipeline

        assert len(built.clients) == 2
        assert built.clients[0] is not built.clients[1]
        assert built.clients[0].model == built.clients[1].model == "shared-model"


def test_config_hash_never_depends_on_api_key_environment_value(tmp_path, monkeypatch):
    """Would fail if an environment credential were read into persisted configuration state."""
    from clinical_trial_qa.config import load_config

    path = _write_json(tmp_path, _config())
    monkeypatch.setenv("TEST_API_KEY", "first-secret")
    first = load_config(path).config_hash
    monkeypatch.setenv("TEST_API_KEY", "second-secret")
    second = load_config(path).config_hash

    assert first == second
    assert "first-secret" not in first
    assert "second-secret" not in second


def test_config_rejects_free_text_model_names_that_would_reach_manifests(tmp_path):
    """Would fail if an arbitrary note-like model label could be persisted verbatim."""
    payload = _config()
    payload["clients"]["primary"][0]["model"] = "patient says this whole sentence is private"

    _assert_configuration_error(_write_json(tmp_path, payload))


def test_config_rejects_nonfinite_json_numbers_and_unsafe_api_key_env_names(tmp_path):
    """Would fail if JSON extensions or path/token-like env names reached runtime state."""
    cases = []
    for timeout in (float("nan"), float("inf"), float("-inf"), 10**1000):
        payload = _config()
        payload["clients"]["primary"][0]["timeout"] = timeout
        cases.append(payload)
    for env_name in ("sk-secret-token", "OPENAI KEY", "../OPENAI_KEY", "9OPENAI_KEY", " OPENAI_KEY"):
        payload = _config()
        payload["clients"]["primary"][0]["api_key_env"] = env_name
        cases.append(payload)

    for index, payload in enumerate(cases):
        path = tmp_path / f"unsafe-{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        _assert_configuration_error(path)

    overflow_path = tmp_path / "overflow.json"
    overflow_payload = {
        "pipeline": "p1",
        "seed": 42,
        "debate_rounds": 0,
        "clients": {
            "primary": [
                {
                    "provider": "scripted",
                    "model": "fixture-model",
                    "script": [{"answer": 7}],
                }
            ]
        },
    }
    overflow_text = json.dumps(overflow_payload).replace('"answer": 7', '"answer": 1e999')
    overflow_path.write_text(overflow_text, encoding="utf-8")
    _assert_configuration_error(overflow_path)


def _topology_config(pipeline):
    clients = {}
    if pipeline in {"p1", "p4"}:
        clients["primary"] = [_client("primary-a")]
    elif pipeline in {"p2", "p5"}:
        clients["primary"] = [_client("primary-a"), _client("primary-b")]
        clients["aggregator"] = _client("aggregator-model")
    if pipeline in {"p3", "p6"}:
        clients["role"] = _client("role-model")
        clients["aggregator"] = _client("aggregator-model")
    if pipeline in {"p4", "p5", "p6"}:
        clients["verifier"] = _client("verifier-model")
        clients["router_label"] = _client("router-model")
    return {"pipeline": pipeline, "seed": 42, "debate_rounds": 1, "clients": clients}


def test_config_accepts_exact_role_matrix_and_reports_only_participating_models(tmp_path):
    """Would fail if a valid P1-P6 topology lost or invented a participating client."""
    from clinical_trial_qa.config import load_config

    for pipeline in ("p1", "p2", "p3", "p4", "p5", "p6"):
        path = tmp_path / f"{pipeline}.json"
        path.write_text(json.dumps(_topology_config(pipeline)), encoding="utf-8")
        config = load_config(path)
        runtime = config.build_runtime()

        assert set(config.model_names) == {client.model for client in runtime.clients}
        assert len(config.model_names) == len(runtime.clients)


def test_config_rejects_configured_but_unused_client_roles(tmp_path):
    """Would fail if irrelevant clients were constructed or listed in a manifest."""
    mutations = (
        ("p1", "primary", [_client("primary-a"), _client("primary-b")]),
        ("p1", "role", _client("role-model")),
        ("p1", "verifier", _client("verifier-model")),
        ("p1", "aggregator", _client("aggregator-model")),
        ("p1", "router_label", _client("router-model")),
        ("p2", "role", _client("role-model")),
        ("p2", "verifier", _client("verifier-model")),
        ("p2", "router_label", _client("router-model")),
        ("p3", "primary", [_client("primary-a")]),
        ("p3", "verifier", _client("verifier-model")),
        ("p3", "router_label", _client("router-model")),
        ("p4", "primary", [_client("primary-a"), _client("primary-b")]),
        ("p4", "role", _client("role-model")),
        ("p4", "aggregator", _client("aggregator-model")),
        ("p5", "role", _client("role-model")),
        ("p6", "primary", [_client("primary-a")]),
    )
    for index, (pipeline, role, value) in enumerate(mutations):
        payload = _topology_config(pipeline)
        payload["clients"][role] = value
        path = tmp_path / f"unused-{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        _assert_configuration_error(path)


def test_replaced_app_config_revalidates_before_constructing_any_client(tmp_path, monkeypatch):
    """Would fail if dataclasses.replace could bypass the exact topology boundary."""
    from clinical_trial_qa.config import ClientConfig, ConfigurationError, load_config

    path = tmp_path / "p5.json"
    path.write_text(json.dumps(_topology_config("p5")), encoding="utf-8")
    config = load_config(path)
    builds = []
    original_build = ClientConfig.build

    def tracked_build(client_config):
        builds.append(client_config.model)
        return original_build(client_config)

    monkeypatch.setattr(ClientConfig, "build", tracked_build)
    try:
        replace(config, pipeline="p1").build_runtime()
    except ConfigurationError:
        pass
    else:
        raise AssertionError("ConfigurationError was not raised")

    assert builds == []
    valid_runtime = replace(config, pipeline="p5").build_runtime()
    assert valid_runtime.pipeline is not None
    assert len(builds) == 5
