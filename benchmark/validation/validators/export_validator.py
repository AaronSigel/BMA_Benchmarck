"""Validation of expected exported artifact files."""

from pathlib import Path

from benchmark.blender.models import SceneSnapshot
from benchmark.tasks.models import BenchmarkTask, ExpectedExport
from benchmark.validation.checks import check_row
from benchmark.validation.models import (
    MetricScore,
    ValidationIssue,
    ValidationSeverity,
    ValidationStatus,
    ValidatorResult,
)

SUPPORTED_EXPORT_FORMATS = {"blend", "glb", "gltf", "fbx"}
STANDARD_EXPORT_CANDIDATES = {
    "blend": [Path("result.blend")],
    "glb": [Path("exports/result.glb"), Path("exports/result.gltf")],
    "gltf": [Path("exports/result.gltf"), Path("exports/result.glb")],
    "fbx": [Path("exports/result.fbx")],
}


class ExportValidator:
    name = "export_validator"

    def validate(
        self,
        task: BenchmarkTask,
        snapshot: SceneSnapshot,
        artifacts_dir: Path | None = None,
    ) -> ValidatorResult:
        del snapshot
        expected_exports = task.expected_scene.exports
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
        check_table = []

        for expected_index, expected in enumerate(expected_exports):
            expected_path = f"expected_scene.exports[{expected_index}]"
            export_score = self._validate_expected_export(expected, root, expected_path, issues, check_table)
            scores.append(export_score)

        score = sum(scores) / len(scores)
        status = ValidationStatus.PASSED if not issues and score == 1.0 else ValidationStatus.FAILED
        return ValidatorResult(
            name=self.name,
            status=status,
            score=score,
            issues=issues,
            metrics=[
                MetricScore(
                    name="export_file_score",
                    score=score,
                    passed=score == 1.0,
                    issues=issues,
                )
            ],
            check_table=check_table,
        )

    def _validate_expected_export(
        self,
        expected: ExpectedExport,
        artifacts_dir: Path,
        expected_path: str,
        issues: list[ValidationIssue],
        check_table: list,
    ) -> float:
        export_format = expected.format.lower()
        if export_format not in SUPPORTED_EXPORT_FORMATS:
            issue = self._unsupported_format_issue(expected, expected_path)
            issues.append(issue)
            check_table.append(check_row(
                validator_name=self.name,
                check_name="file exists",
                entity_ref="export",
                field="path",
                expected=expected.filename or f"result.{export_format}",
                actual="not_found",
                passed=False,
                score=0.0,
                issue=issue,
            ))
            return 0.0

        candidates = self._candidate_paths(expected, artifacts_dir)
        existing_candidates = [candidate for candidate in candidates if candidate.exists()]
        if not existing_candidates:
            if expected.must_exist:
                issue = self._missing_issue(expected, candidates, expected_path)
                issues.append(issue)
                check_table.append(check_row(
                    validator_name=self.name,
                    check_name="file exists",
                    entity_ref="export",
                    field="path",
                    expected=expected.filename or f"result.{export_format}",
                    actual="not_found",
                    passed=False,
                    score=0.0,
                    issue=issue,
                ))
                return 0.0
            check_table.append(check_row(
                validator_name=self.name,
                check_name="file exists",
                entity_ref="export",
                field="path",
                expected=expected.filename or f"result.{export_format}",
                actual="optional",
                passed=True,
                score=1.0,
            ))
            return 1.0

        best_candidate = existing_candidates[0]
        check_table.append(check_row(
            validator_name=self.name,
            check_name="file exists",
            entity_ref="export",
            field="path",
            expected=expected.filename or f"result.{export_format}",
            actual="found",
            passed=True,
            score=1.0,
        ))
        if best_candidate.stat().st_size <= 0:
            issue = self._empty_file_issue(expected, best_candidate, expected_path)
            issues.append(issue)
            check_table.append(check_row(
                validator_name=self.name,
                check_name="file size",
                entity_ref="export",
                field="file_size_bytes",
                expected="non_empty",
                actual=best_candidate.stat().st_size,
                passed=False,
                score=0.0,
                issue=issue,
            ))
            return 0.0
        check_table.append(check_row(
            validator_name=self.name,
            check_name="file size",
            entity_ref="export",
            field="file_size_bytes",
            expected="non_empty",
            actual=best_candidate.stat().st_size,
            passed=True,
            score=1.0,
        ))
        return 1.0

    def _candidate_paths(self, expected: ExpectedExport, artifacts_dir: Path) -> list[Path]:
        if expected.filename is not None:
            return [artifacts_dir / expected.filename]
        return [
            artifacts_dir / relative_path
            for relative_path in STANDARD_EXPORT_CANDIDATES.get(expected.format.lower(), [])
        ]

    def _unsupported_format_issue(
        self,
        expected: ExpectedExport,
        expected_path: str,
    ) -> ValidationIssue:
        return ValidationIssue(
            code="export_format_unsupported",
            message=f"Export format is not supported: {expected.format}.",
            severity=ValidationSeverity.ERROR,
            expected_path=f"{expected_path}.format",
            actual_path=None,
            expected_value=expected.format,
            actual_value=None,
        )

    def _missing_issue(
        self,
        expected: ExpectedExport,
        candidates: list[Path],
        expected_path: str,
    ) -> ValidationIssue:
        return ValidationIssue(
            code="export_missing",
            message=f"Expected export file was not found for format {expected.format}.",
            severity=ValidationSeverity.ERROR,
            expected_path=expected_path,
            actual_path=None,
            expected_value=expected.model_dump(mode="json", exclude_none=True),
            actual_value=[str(path) for path in candidates],
        )

    def _empty_file_issue(
        self,
        expected: ExpectedExport,
        path: Path,
        expected_path: str,
    ) -> ValidationIssue:
        return ValidationIssue(
            code="export_empty_file",
            message=f"Expected export file exists but is empty: {path}.",
            severity=ValidationSeverity.ERROR,
            expected_path=expected_path,
            actual_path=str(path),
            expected_value=expected.model_dump(mode="json", exclude_none=True),
            actual_value={"path": str(path), "file_size_bytes": path.stat().st_size},
        )
