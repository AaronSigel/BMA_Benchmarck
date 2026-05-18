import json
from pathlib import Path

from benchmark.experiments.manifests import (
    build_manifest,
    sanitized_config_payload,
    stable_config_hash,
    write_manifest,
    write_manifest_for_matrix,
)
from benchmark.experiments.matrix import load_matrix
from benchmark.experiments.models import ExperimentMatrix, GeneratedExperimentManifest


def test_build_manifest_contains_reproducibility_fields() -> None:
    matrix = load_matrix("configs/matrices/smoke_matrix.yaml")

    manifest = build_manifest(matrix)

    assert manifest.matrix_id == "smoke_matrix"
    assert manifest.generated_at.endswith("Z")
    assert manifest.python_version
    assert manifest.platform
    assert manifest.task_ids == ["geometry_001_basic_primitives"]
    assert manifest.agent_ids == ["mock_agent"]
    assert manifest.mcp_profiles == ["minimal"]
    assert manifest.execution_modes == matrix.execution_modes
    assert manifest.repetitions == 1
    assert manifest.config_hash
    assert manifest.env_requirements


def test_config_hash_is_stable_for_same_matrix() -> None:
    matrix = load_matrix("configs/matrices/smoke_matrix.yaml")

    first = build_manifest(matrix)
    second = build_manifest(matrix)

    assert first.config_hash == second.config_hash


def test_stable_config_hash_ignores_dict_key_order() -> None:
    assert stable_config_hash({"b": 2, "a": 1}) == stable_config_hash({"a": 1, "b": 2})


def test_sanitized_payload_removes_secret_like_keys() -> None:
    matrix = ExperimentMatrix(
        matrix_id="secret_test",
        metadata={
            "api_key": "inline-secret",
            "nested": {
                "authorization": "Bearer secret",
                "safe": "value",
            },
            "api_key_env": "SHOULD_NOT_BE_IN_MANIFEST",
        },
    )

    payload = sanitized_config_payload(matrix)
    payload_json = json.dumps(payload)

    assert "inline-secret" not in payload_json
    assert "Bearer secret" not in payload_json
    assert "SHOULD_NOT_BE_IN_MANIFEST" not in payload_json
    assert payload["metadata"]["nested"]["safe"] == "value"


def test_manifest_does_not_include_secret_values(tmp_path: Path) -> None:
    matrix = ExperimentMatrix(
        matrix_id="secret_manifest",
        output_root=tmp_path,
        metadata={"api_key": "inline-secret"},
    )

    path = write_manifest_for_matrix(matrix)
    text = path.read_text(encoding="utf-8")

    assert path == tmp_path / "manifest.json"
    assert "inline-secret" not in text
    assert "secret_manifest" in text


def test_manifest_json_does_not_include_secret_like_env_values(tmp_path: Path) -> None:
    matrix = ExperimentMatrix(
        matrix_id="secret_env_manifest",
        output_root=tmp_path,
        metadata={
            "safe": "kept",
            "api_key_env": "SECRET_API_KEY_ENV",
            "token_value": "SECRET_TOKEN_VALUE",
        },
    )

    manifest = build_manifest(matrix)
    manifest_json = manifest.model_dump_json()

    assert "SECRET_API_KEY_ENV" not in manifest_json
    assert "SECRET_TOKEN_VALUE" not in manifest_json
    assert "secret_env_manifest" in manifest_json


def test_manifest_accepts_preflight_metadata_without_secrets(tmp_path: Path) -> None:
    matrix = ExperimentMatrix(
        matrix_id="preflight_manifest",
        output_root=tmp_path,
    )

    manifest = build_manifest(
        matrix,
        metadata={
            "runtime": {
                "mcp_profile": "no_python",
                "api_key_env": "SECRET_SHOULD_NOT_APPEAR",
            },
            "tool_contract_hash": "abc123",
        },
    )
    manifest_json = manifest.model_dump_json()

    assert "no_python" in manifest_json
    assert "abc123" in manifest_json
    assert "SECRET_SHOULD_NOT_APPEAR" not in manifest_json


def test_write_manifest_writes_json(tmp_path: Path) -> None:
    manifest = GeneratedExperimentManifest(
        matrix_id="manual",
        generated_at="2026-05-16T00:00:00Z",
        repetitions=1,
    )
    path = tmp_path / "nested" / "manifest.json"

    write_manifest(manifest, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["matrix_id"] == "manual"
