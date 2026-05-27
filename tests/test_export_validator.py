from pathlib import Path

from benchmark.blender.models import RenderSettingsSnapshot, SceneSnapshot
from benchmark.tasks.models import (
    BenchmarkTask,
    DifficultyLevel,
    ExpectedExport,
    ExpectedScene,
    SuccessCriterion,
    TaskCategory,
)
from benchmark.validation.models import ValidationStatus
from benchmark.validation.validators.export_validator import ExportValidator


def scene_snapshot() -> SceneSnapshot:
    return SceneSnapshot(
        scene_name="Scene",
        objects=[],
        materials=[],
        lights=[],
        cameras=[],
        collections=["Collection"],
        render_settings=RenderSettingsSnapshot(
            engine="CYCLES",
            resolution_x=1920,
            resolution_y=1080,
            frame_start=1,
            frame_end=1,
            frame_current=1,
        ),
        frame_current=1,
        blender_version="4.0.0",
        created_at="2026-05-15T12:00:00Z",
    )


def task_with_exports(expected_exports: list[ExpectedExport]) -> BenchmarkTask:
    return BenchmarkTask(
        id="export_001_blend_file",
        title="Export scene",
        category=TaskCategory.EXPORT,
        difficulty=DifficultyLevel.EASY,
        prompt="Export the scene.",
        tags=["export"],
        allowed_tools=[],
        expected_scene=ExpectedScene(exports=expected_exports),
        success_criteria=[SuccessCriterion(metric="exports", weight=1.0)],
    )


def write_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_export_validator_skips_empty_expected_exports(tmp_path: Path) -> None:
    result = ExportValidator().validate(task_with_exports([]), scene_snapshot(), tmp_path)

    assert result.status is ValidationStatus.SKIPPED
    assert result.score == 0.0
    assert result.issues == []
    assert result.metrics == []


def test_export_validator_passes_existing_non_empty_standard_blend(tmp_path: Path) -> None:
    write_file(tmp_path / "result.blend", b"blend")
    task = task_with_exports([ExpectedExport(format="blend")])

    result = ExportValidator().validate(task, scene_snapshot(), tmp_path)

    assert result.status is ValidationStatus.PASSED
    assert result.score == 1.0
    assert result.issues == []


def test_export_validator_glb_missing_reports_not_found(tmp_path: Path) -> None:
    task = task_with_exports(
        [ExpectedExport(format="glb", filename="exports/result.glb", must_exist=True)]
    )
    result = ExportValidator().validate(task, scene_snapshot(), tmp_path)
    exists_row = next(row for row in result.check_table if row.check_name == "file exists")
    assert exists_row.entity_ref == "export"
    assert exists_row.actual == "not_found"
    assert result.status is ValidationStatus.FAILED


def test_export_validator_empty_glb_fails_non_empty_check(tmp_path: Path) -> None:
    write_file(tmp_path / "exports" / "result.glb", b"")
    task = task_with_exports(
        [ExpectedExport(format="glb", filename="exports/result.glb", must_exist=True)]
    )
    result = ExportValidator().validate(task, scene_snapshot(), tmp_path)
    size_row = next(row for row in result.check_table if row.check_name == "file size")
    assert size_row.passed is False
    assert result.issues


def test_export_validator_passes_existing_non_empty_standard_glb(tmp_path: Path) -> None:
    write_file(tmp_path / "exports" / "result.glb", b"glb")
    task = task_with_exports([ExpectedExport(format="glb")])

    result = ExportValidator().validate(task, scene_snapshot(), tmp_path)

    assert result.status is ValidationStatus.PASSED
    assert result.score == 1.0


def test_export_validator_uses_explicit_filename(tmp_path: Path) -> None:
    write_file(tmp_path / "custom" / "scene.fbx", b"fbx")
    task = task_with_exports([ExpectedExport(format="fbx", filename="custom/scene.fbx")])

    result = ExportValidator().validate(task, scene_snapshot(), tmp_path)

    assert result.status is ValidationStatus.PASSED
    assert result.score == 1.0


def test_export_validator_reports_missing_required_file(tmp_path: Path) -> None:
    task = task_with_exports([ExpectedExport(format="blend", must_exist=True)])

    result = ExportValidator().validate(task, scene_snapshot(), tmp_path)

    assert result.status is ValidationStatus.FAILED
    assert result.score == 0.0
    assert result.issues[0].code == "export_missing"
    assert result.issues[0].expected_path == "expected_scene.exports[0]"
    assert result.issues[0].message


def test_export_validator_allows_missing_optional_file(tmp_path: Path) -> None:
    task = task_with_exports([ExpectedExport(format="blend", must_exist=False)])

    result = ExportValidator().validate(task, scene_snapshot(), tmp_path)

    assert result.status is ValidationStatus.PASSED
    assert result.score == 1.0
    assert result.issues == []


def test_export_validator_reports_empty_file(tmp_path: Path) -> None:
    write_file(tmp_path / "exports" / "result.glb", b"")
    task = task_with_exports([ExpectedExport(format="glb")])

    result = ExportValidator().validate(task, scene_snapshot(), tmp_path)

    assert result.status is ValidationStatus.FAILED
    assert result.score == 0.0
    assert result.issues[0].code == "export_empty_file"
    assert result.issues[0].expected_path == "expected_scene.exports[0]"
    assert result.issues[0].actual_path == str(tmp_path / "exports" / "result.glb")


def test_export_validator_reports_unsupported_format(tmp_path: Path) -> None:
    data = {"format": "obj", "must_exist": True}
    expected = ExpectedExport.model_construct(**data)
    task = task_with_exports([expected])

    result = ExportValidator().validate(task, scene_snapshot(), tmp_path)

    assert result.status is ValidationStatus.FAILED
    assert result.score == 0.0
    assert result.issues[0].code == "export_format_unsupported"
    assert result.issues[0].expected_path == "expected_scene.exports[0].format"
