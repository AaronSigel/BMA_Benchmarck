"""Import-back validation for exported GLB files."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from benchmark.blender.models import SceneSnapshot
from benchmark.tasks.models import BenchmarkTask, ExpectedExport, ExpectedObject
from benchmark.validation.matcher import BLENDER_SUFFIX_RE, SceneMatcher, normalize_name
from benchmark.validation.models import (
    MetricScore,
    ValidationIssue,
    ValidationSeverity,
    ValidationStatus,
    ValidatorResult,
)
from benchmark.validation.scoring import vector_tolerance_score, weighted_average
from benchmark.validation.validators.export_validator import STANDARD_EXPORT_CANDIDATES

ImportBackCallable = Callable[[Path], SceneSnapshot]


class GlbImportBackValidator:
    name = "glb_import_back_validator"

    def __init__(
        self,
        importer: ImportBackCallable | None = None,
        *,
        min_file_size_bytes: int = 16,
    ) -> None:
        self.importer = importer or _import_glb_with_blender
        self.min_file_size_bytes = min_file_size_bytes
        self.matcher = SceneMatcher()

    def validate(
        self,
        task: BenchmarkTask,
        snapshot: SceneSnapshot,
        artifacts_dir: Path | None = None,
    ) -> ValidatorResult:
        del snapshot
        expected_exports = [
            export for export in task.expected_scene.exports if export.format.lower() == "glb"
        ]
        if not expected_exports:
            return ValidatorResult(
                name=self.name,
                status=ValidationStatus.SKIPPED,
                score=0.0,
                metrics=[],
            )

        root = Path(artifacts_dir) if artifacts_dir is not None else Path(".")
        issues: list[ValidationIssue] = []
        scores: list[float] = []

        for expected_index, expected in enumerate(expected_exports):
            expected_path = f"expected_scene.exports[{expected_index}]"
            scores.append(self._validate_one(task, expected, root, expected_path, issues))

        score = sum(scores) / len(scores) if scores else 0.0
        status = ValidationStatus.PASSED if not issues and score == 1.0 else ValidationStatus.FAILED
        return ValidatorResult(
            name=self.name,
            status=status,
            score=score,
            issues=issues,
            metrics=[
                MetricScore(
                    name="export_import_score",
                    score=score,
                    passed=score == 1.0,
                    issues=issues,
                )
            ],
        )

    def _validate_one(
        self,
        task: BenchmarkTask,
        expected: ExpectedExport,
        artifacts_dir: Path,
        expected_path: str,
        issues: list[ValidationIssue],
    ) -> float:
        candidates = _candidate_paths(expected, artifacts_dir)
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            if expected.must_exist:
                issues.append(_issue(
                    "export_import_missing",
                    "Expected GLB export file was not found for import-back validation.",
                    expected_path,
                    None,
                    expected.model_dump(mode="json", exclude_none=True),
                    [str(candidate) for candidate in candidates],
                ))
                return 0.0
            return 1.0

        file_size = path.stat().st_size
        if file_size < self.min_file_size_bytes:
            issues.append(_issue(
                "export_import_file_too_small",
                f"GLB export is too small to be a valid import target: {path}.",
                expected_path,
                str(path),
                {"min_file_size_bytes": self.min_file_size_bytes},
                {"file_size_bytes": file_size},
            ))
            return 0.0
        try:
            imported = self.importer(path)
        except Exception as error:
            issues.append(_issue(
                "export_import_failed",
                f"GLB import-back validation failed: {error}",
                expected_path,
                str(path),
                "importable GLB",
                str(error),
            ))
            return 0.0
        mesh_count_score = self._mesh_count_score(task, imported, issues)
        objects_score = self._expected_objects_score(task, imported, issues)
        material_score = self._material_score(task, imported, issues)
        duplicate_score = self._duplicate_score(imported, issues)
        bounds_score = self._bounds_score(task, imported, issues)
        sub_scores = [
            (mesh_count_score, 0.25),
            (objects_score, 0.35),
            (material_score, 0.2),
            (duplicate_score, 0.1),
            (bounds_score, 0.1),
        ]
        score = weighted_average(sub_scores)
        return score

    def _mesh_count_score(
        self,
        task: BenchmarkTask,
        imported: SceneSnapshot,
        issues: list[ValidationIssue],
    ) -> float:
        expected_count = len(task.expected_scene.objects)
        actual_count = len(imported.objects)
        if expected_count == actual_count:
            return 1.0
        issues.append(_issue(
            "export_import_mesh_count_mismatch",
            "Imported GLB mesh count does not match expected scene objects.",
            "expected_scene.objects",
            "imported_snapshot.objects",
            expected_count,
            actual_count,
        ))
        if expected_count == 0:
            return 0.0
        return max(0.0, 1.0 - abs(expected_count - actual_count) / expected_count)

    def _expected_objects_score(
        self,
        task: BenchmarkTask,
        imported: SceneSnapshot,
        issues: list[ValidationIssue],
    ) -> float:
        if not task.expected_scene.objects:
            return 1.0
        scores: list[float] = []
        available = list(imported.objects)
        for index, expected in enumerate(task.expected_scene.objects):
            actual = self.matcher.match_expected_object(expected, available)
            if actual is None:
                issues.append(_issue(
                    "export_import_object_missing",
                    f"Expected object was not found after GLB import: {expected.name or expected.type}.",
                    f"expected_scene.objects[{index}]",
                    "imported_snapshot.objects",
                    expected.model_dump(mode="json", exclude_none=True),
                    None,
                ))
                scores.append(0.0)
                continue
            available.remove(actual)
            scores.append(_object_transform_score(expected, actual))
            if scores[-1] < 1.0:
                issues.append(_issue(
                    "export_import_transform_mismatch",
                    "Imported object transform differs from expected scene.",
                    f"expected_scene.objects[{index}]",
                    f"imported_snapshot.objects[{imported.objects.index(actual)}]",
                    expected.model_dump(mode="json", exclude_none=True),
                    actual.model_dump(mode="json", exclude_none=True),
                ))
        return sum(scores) / len(scores)

    def _material_score(
        self,
        task: BenchmarkTask,
        imported: SceneSnapshot,
        issues: list[ValidationIssue],
    ) -> float:
        expected_materials = task.expected_scene.materials
        if not expected_materials:
            return 1.0
        matched = 0
        for index, expected in enumerate(expected_materials):
            actual = self.matcher.match_expected_material(expected, imported.materials)
            if actual is None:
                issues.append(_issue(
                    "export_import_material_missing",
                    f"Expected material was not found after GLB import: {expected.name}.",
                    f"expected_scene.materials[{index}]",
                    "imported_snapshot.materials",
                    expected.model_dump(mode="json", exclude_none=True),
                    None,
                ))
                continue
            if expected.base_color is not None and actual.base_color is not None:
                color_ok = all(
                    abs(getattr(expected.base_color, channel) - getattr(actual.base_color, channel))
                    <= expected.tolerance
                    for channel in ("r", "g", "b", "a")
                )
                if not color_ok:
                    issues.append(_issue(
                        "export_import_material_color_mismatch",
                        "Imported material base color differs from expected material.",
                        f"expected_scene.materials[{index}].base_color",
                        "imported_snapshot.materials.base_color",
                        expected.base_color.model_dump(mode="json"),
                        actual.base_color.model_dump(mode="json"),
                    ))
                    continue
            matched += 1
        return matched / len(expected_materials)

    def _duplicate_score(self, imported: SceneSnapshot, issues: list[ValidationIssue]) -> float:
        base_names = [BLENDER_SUFFIX_RE.sub("", obj.name) for obj in imported.objects]
        duplicate_count = len(base_names) - len(set(base_names))
        if duplicate_count == 0:
            return 1.0
        issues.append(_issue(
            "export_import_duplicate_names",
            "Imported GLB contains duplicate Blender base names.",
            None,
            "imported_snapshot.objects",
            0,
            duplicate_count,
        ))
        return 0.0

    def _bounds_score(
        self,
        task: BenchmarkTask,
        imported: SceneSnapshot,
        issues: list[ValidationIssue],
    ) -> float:
        expected_count = len(task.expected_scene.objects)
        actual_count = len(imported.objects)
        if expected_count == 0 or actual_count <= max(10, expected_count * 4):
            return 1.0
        issues.append(_issue(
            "export_import_suspicious_object_count",
            "Imported GLB contains a suspicious number of mesh objects.",
            "expected_scene.objects",
            "imported_snapshot.objects",
            expected_count,
            actual_count,
        ))
        return 0.0


def _candidate_paths(expected: ExpectedExport, artifacts_dir: Path) -> list[Path]:
    if expected.filename is not None:
        return [artifacts_dir / expected.filename]
    return [
        artifacts_dir / relative
        for relative in STANDARD_EXPORT_CANDIDATES.get(expected.format.lower(), [])
    ]


def _object_transform_score(expected: ExpectedObject, actual) -> float:
    scores: list[float] = []
    if expected.type is not None:
        scores.append(1.0 if str(actual.type).upper() == str(expected.type).upper() else 0.0)
    if expected.location is not None:
        scores.append(vector_tolerance_score(expected.location, actual.location, expected.tolerance))
    if expected.dimensions is not None:
        scores.append(vector_tolerance_score(expected.dimensions, actual.dimensions, expected.tolerance))
    if expected.scale is not None:
        scores.append(vector_tolerance_score(expected.scale, actual.scale, expected.tolerance))
    if expected.material is not None:
        normalized_slots = {normalize_name(name) for name in actual.material_slots}
        scores.append(1.0 if normalize_name(expected.material) in normalized_slots else 0.0)
    return sum(scores) / len(scores) if scores else 1.0


def _import_glb_with_blender(path: Path) -> SceneSnapshot:
    blender = os.environ.get("BLENDER_PATH") or shutil.which("blender")
    if not blender:
        raise RuntimeError("Blender executable not available for GLB import-back validation")
    with tempfile.TemporaryDirectory(prefix="bma_glb_import_") as tmp:
        tmp_path = Path(tmp)
        snapshot_path = tmp_path / "imported_scene_snapshot.json"
        script_path = tmp_path / "import_glb.py"
        script_path.write_text(_import_script(path, snapshot_path), encoding="utf-8")
        result = subprocess.run(
            [blender, "--background", "--factory-startup", "--python", str(script_path)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout or "blender import failed").strip())
        return SceneSnapshot.model_validate_json(snapshot_path.read_text(encoding="utf-8"))


def _import_script(glb_path: Path, snapshot_path: Path) -> str:
    import inspect as _inspect

    from benchmark.blender.scripts.collect_snapshot import collect_snapshot as _collect_snapshot

    module = _inspect.getmodule(_collect_snapshot)
    source = _inspect.getsource(module)
    return (
        "import bpy\n"
        "bpy.ops.object.select_all(action='SELECT')\n"
        "bpy.ops.object.delete()\n"
        f"bpy.ops.import_scene.gltf(filepath={str(glb_path)!r})\n"
        f"{source}\n"
        f"collect_snapshot({{'output_path': {str(snapshot_path)!r}}})\n"
    )


def _issue(
    code: str,
    message: str,
    expected_path: str | None,
    actual_path: str | None,
    expected_value,
    actual_value,
    severity: ValidationSeverity = ValidationSeverity.ERROR,
) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        message=message,
        severity=severity,
        expected_path=expected_path,
        actual_path=actual_path,
        expected_value=expected_value,
        actual_value=actual_value,
    )
