from pathlib import Path

import pytest

from benchmark.blender.models import SceneSnapshot
from benchmark.tasks.loader import load_task
from benchmark.validation.models import ValidationStatus
from benchmark.validation.scene_validator import SceneValidator
from benchmark.validation.validators.export_validator import ExportValidator
from benchmark.validation.validators.material_validator import MaterialValidator
from benchmark.validation.validators.object_validator import ObjectValidator

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "validation"


def load_snapshot_fixture(name: str) -> SceneSnapshot:
    return SceneSnapshot.model_validate_json((FIXTURES_DIR / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "fixture_name",
    [
        "valid_geometry_snapshot.json",
        "missing_object_snapshot.json",
        "wrong_material_snapshot.json",
        "valid_camera_light_snapshot.json",
    ],
)
def test_validation_snapshot_fixtures_are_valid_scene_snapshots(fixture_name: str) -> None:
    snapshot = load_snapshot_fixture(fixture_name)

    assert snapshot.scene_name
    assert snapshot.render_settings.resolution_x > 0


def test_valid_geometry_fixture_matches_geometry_001() -> None:
    task = load_task("tasks/geometry/geometry_001_basic_primitives.yaml")
    snapshot = load_snapshot_fixture("valid_geometry_snapshot.json")

    result = SceneValidator().validate(task, snapshot)

    assert result.overall_status is ValidationStatus.PASSED
    assert result.total_score == 1.0


def test_missing_object_fixture_exercises_object_validator() -> None:
    task = load_task("tasks/geometry/geometry_001_basic_primitives.yaml")
    snapshot = load_snapshot_fixture("missing_object_snapshot.json")

    result = ObjectValidator().validate(task, snapshot)

    assert result.status is ValidationStatus.FAILED
    assert any(issue.code == "object_missing" for issue in result.issues)


def test_wrong_material_fixture_exercises_material_validator() -> None:
    task = load_task("tasks/materials/materials_001_basic_colors.yaml")
    snapshot = load_snapshot_fixture("wrong_material_snapshot.json")

    result = MaterialValidator().validate(task, snapshot)

    assert result.status is ValidationStatus.FAILED
    assert {issue.code for issue in result.issues} >= {
        "material_color_mismatch",
        "object_material_missing",
    }


def test_valid_camera_light_fixture_matches_camera_and_light_tasks() -> None:
    snapshot = load_snapshot_fixture("valid_camera_light_snapshot.json")
    camera_task = load_task("tasks/camera/camera_001_front_view.yaml")
    light_task = load_task("tasks/lighting/lighting_001_area_light.yaml")

    camera_result = SceneValidator().validate(camera_task, snapshot)
    light_result = SceneValidator().validate(light_task, snapshot)

    assert camera_result.overall_status is ValidationStatus.PASSED
    assert light_result.overall_status is ValidationStatus.PASSED


def test_export_artifact_fixture_exercises_export_validator() -> None:
    task = load_task("tasks/export/export_001_blend_file.yaml")
    snapshot = load_snapshot_fixture("valid_geometry_snapshot.json")

    result = ExportValidator().validate(task, snapshot, FIXTURES_DIR / "export_artifacts")

    assert result.status is ValidationStatus.PASSED
    assert result.score == 1.0
    assert (FIXTURES_DIR / "export_artifacts" / "result.blend").stat().st_size > 0
    assert (FIXTURES_DIR / "export_artifacts" / "exports" / "result.glb").stat().st_size > 0
