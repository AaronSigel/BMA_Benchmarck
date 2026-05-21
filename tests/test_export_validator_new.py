"""Targeted tests for export validator: issue codes and GLB import-back."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from benchmark.blender.models import SceneSnapshot
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
from benchmark.validation.validators.glb_import_back_validator import GlbImportBackValidator


def _task(exports: list[ExpectedExport]) -> BenchmarkTask:
    return BenchmarkTask(
        id="export_test",
        title="Export test",
        category=TaskCategory.EXPORT,
        difficulty=DifficultyLevel.EASY,
        prompt="Export the scene.",
        tags=["export"],
        allowed_tools=[],
        expected_scene=ExpectedScene(exports=exports),
        success_criteria=[SuccessCriterion(metric="export_validity", weight=1.0)],
    )


def _empty_snapshot() -> SceneSnapshot:
    from benchmark.blender.models import RenderSettingsSnapshot
    return SceneSnapshot(
        scene_name="Scene",
        objects=[],
        materials=[],
        lights=[],
        cameras=[],
        mesh_object_count=0,
        light_count=0,
        camera_count=0,
        all_object_count=0,
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


class TestExportMissingIssueCode:
    def test_missing_blend_file_emits_export_missing(self) -> None:
        task = _task([ExpectedExport(format="blend", must_exist=True)])
        validator = ExportValidator()
        with tempfile.TemporaryDirectory() as tmp:
            result = validator.validate(task, _empty_snapshot(), artifacts_dir=Path(tmp))
        assert result.status == ValidationStatus.FAILED
        codes = [i.code for i in result.issues]
        assert "export_missing" in codes

    def test_missing_file_with_must_exist_false_passes(self) -> None:
        task = _task([ExpectedExport(format="blend", must_exist=False)])
        validator = ExportValidator()
        with tempfile.TemporaryDirectory() as tmp:
            result = validator.validate(task, _empty_snapshot(), artifacts_dir=Path(tmp))
        assert result.status == ValidationStatus.PASSED
        assert result.issues == []

    def test_empty_file_emits_export_empty_file(self) -> None:
        task = _task([ExpectedExport(format="blend", filename="result.blend", must_exist=True)])
        validator = ExportValidator()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Create an empty (0-byte) file
            (tmp_path / "result.blend").write_bytes(b"")
            result = validator.validate(task, _empty_snapshot(), artifacts_dir=tmp_path)
        assert result.status == ValidationStatus.FAILED
        codes = [i.code for i in result.issues]
        assert "export_empty_file" in codes

    def test_valid_blend_file_passes(self) -> None:
        task = _task([ExpectedExport(format="blend", filename="result.blend", must_exist=True)])
        validator = ExportValidator()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "result.blend").write_bytes(b"BLENDER" + b"\x00" * 100)
            result = validator.validate(task, _empty_snapshot(), artifacts_dir=tmp_path)
        assert result.status == ValidationStatus.PASSED
        assert result.issues == []

    def test_no_exports_in_task_skips_validator(self) -> None:
        task = _task([])
        validator = ExportValidator()
        with tempfile.TemporaryDirectory() as tmp:
            result = validator.validate(task, _empty_snapshot(), artifacts_dir=Path(tmp))
        assert result.status == ValidationStatus.SKIPPED


class TestGlbImportBackValidator:
    def test_missing_glb_with_must_exist_emits_export_import_missing(self) -> None:
        task = _task([ExpectedExport(format="glb", must_exist=True)])
        validator = GlbImportBackValidator()
        with tempfile.TemporaryDirectory() as tmp:
            result = validator.validate(task, _empty_snapshot(), artifacts_dir=Path(tmp))
        assert result.status == ValidationStatus.FAILED
        codes = [i.code for i in result.issues]
        assert "export_import_missing" in codes

    def test_empty_glb_file_emits_file_too_small(self) -> None:
        task = _task([ExpectedExport(format="glb", filename="result.glb", must_exist=True)])
        validator = GlbImportBackValidator(min_file_size_bytes=16)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "result.glb").write_bytes(b"")  # 0 bytes < 16
            result = validator.validate(task, _empty_snapshot(), artifacts_dir=tmp_path)
        assert result.status == ValidationStatus.FAILED
        codes = [i.code for i in result.issues]
        assert "export_import_file_too_small" in codes

    def test_corrupt_glb_emits_export_import_failed(self) -> None:
        task = _task([ExpectedExport(format="glb", filename="result.glb", must_exist=True)])

        def failing_importer(path: Path) -> SceneSnapshot:
            raise RuntimeError("GLB parse failed: invalid magic bytes")

        validator = GlbImportBackValidator(importer=failing_importer)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "result.glb").write_bytes(b"NOT_A_VALID_GLB_FILE_AT_ALL")
            result = validator.validate(task, _empty_snapshot(), artifacts_dir=tmp_path)
        assert result.status == ValidationStatus.FAILED
        codes = [i.code for i in result.issues]
        assert "export_import_failed" in codes

    def test_no_glb_exports_in_task_skips_validator(self) -> None:
        task = _task([ExpectedExport(format="blend")])
        validator = GlbImportBackValidator()
        with tempfile.TemporaryDirectory() as tmp:
            result = validator.validate(task, _empty_snapshot(), artifacts_dir=Path(tmp))
        assert result.status == ValidationStatus.SKIPPED
