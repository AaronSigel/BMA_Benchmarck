"""Tests for benchmark.analysis.validation_metrics."""
from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.analysis.validation_metrics import (
    ValidationMetricsSummary,
    compute_validation_summary,
    extract_issues,
    extract_score_and_status,
    extract_validation_metrics,
)
from benchmark.validation.models import (
    SceneValidationResult,
    ValidationIssue,
    ValidationSeverity,
    ValidationStatus,
    ValidatorResult,
)

FIXTURES = Path(__file__).parent / "fixtures" / "analysis"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _issue(severity: str = "error", code: str = "test_code") -> ValidationIssue:
    return ValidationIssue(code=code, message="test issue", severity=severity)


def _validator(
    name: str,
    status: str = "passed",
    score: float = 1.0,
    issues: list[ValidationIssue] | None = None,
) -> ValidatorResult:
    return ValidatorResult(
        name=name,
        status=ValidationStatus(status),
        score=score,
        max_score=1.0,
        issues=issues or [],
    )


def _scene_result(
    overall_status: str = "passed",
    total_score: float = 1.0,
    validators: list[ValidatorResult] | None = None,
    issues: list[ValidationIssue] | None = None,
) -> SceneValidationResult:
    return SceneValidationResult(
        task_id="t1",
        overall_status=ValidationStatus(overall_status),
        total_score=total_score,
        validators=validators or [],
        issues=issues or [],
        summary={},
    )


# ---------------------------------------------------------------------------
# Missing validation_result (None input)
# ---------------------------------------------------------------------------

class TestMissingValidationResult:
    def test_none_returns_unknown_status(self):
        s = compute_validation_summary(None)
        assert s.scene_overall_status == "unknown"

    def test_none_returns_zero_counts(self):
        s = compute_validation_summary(None)
        assert s.passed_validator_count == 0
        assert s.failed_validator_count == 0
        assert s.skipped_validator_count == 0
        assert s.validation_error_count == 0
        assert s.validation_warning_count == 0

    def test_none_returns_none_score(self):
        s = compute_validation_summary(None)
        assert s.scene_total_score is None

    def test_none_returns_none_validator_scores(self):
        s = compute_validation_summary(None)
        assert s.object_score is None
        assert s.transform_score is None
        assert s.material_score is None
        assert s.light_score is None
        assert s.camera_score is None
        assert s.export_score is None


# ---------------------------------------------------------------------------
# scene_total_score
# ---------------------------------------------------------------------------

class TestSceneTotalScore:
    def test_perfect_score(self):
        val = _scene_result(total_score=1.0)
        s = compute_validation_summary(val)
        assert s.scene_total_score == 1.0

    def test_zero_score(self):
        val = _scene_result(overall_status="failed", total_score=0.0)
        s = compute_validation_summary(val)
        assert s.scene_total_score == 0.0

    def test_partial_score(self):
        val = _scene_result(overall_status="warning", total_score=0.6)
        s = compute_validation_summary(val)
        assert s.scene_total_score == pytest.approx(0.6)

    def test_status_passed_propagated(self):
        val = _scene_result(overall_status="passed", total_score=1.0)
        s = compute_validation_summary(val)
        assert s.scene_overall_status == "passed"

    def test_status_failed_propagated(self):
        val = _scene_result(overall_status="failed", total_score=0.0)
        s = compute_validation_summary(val)
        assert s.scene_overall_status == "failed"

    def test_status_warning_propagated(self):
        val = _scene_result(overall_status="warning", total_score=0.7)
        s = compute_validation_summary(val)
        assert s.scene_overall_status == "warning"

    def test_fixture_success_score(self):
        from benchmark.analysis.trace_reader import read_validation_result
        val = read_validation_result(FIXTURES / "validation_result_success.json")
        s = compute_validation_summary(val)
        assert s.scene_total_score == 1.0
        assert s.scene_overall_status == "passed"

    def test_fixture_partial_score(self):
        from benchmark.analysis.trace_reader import read_validation_result
        val = read_validation_result(FIXTURES / "validation_result_partial.json")
        s = compute_validation_summary(val)
        assert s.scene_total_score == pytest.approx(0.6)
        assert s.scene_overall_status == "warning"


# ---------------------------------------------------------------------------
# Validator scores
# ---------------------------------------------------------------------------

class TestValidatorScores:
    def test_object_validator_score_extracted(self):
        val = _scene_result(validators=[_validator("object_validator", score=0.8)])
        s = compute_validation_summary(val)
        assert s.object_score == pytest.approx(0.8)

    def test_transform_validator_score_extracted(self):
        val = _scene_result(validators=[_validator("transform_validator", score=0.9)])
        s = compute_validation_summary(val)
        assert s.transform_score == pytest.approx(0.9)

    def test_material_validator_score_extracted(self):
        val = _scene_result(validators=[_validator("material_validator", score=0.5)])
        s = compute_validation_summary(val)
        assert s.material_score == pytest.approx(0.5)

    def test_light_validator_score_extracted(self):
        val = _scene_result(validators=[_validator("light_validator", score=0.7)])
        s = compute_validation_summary(val)
        assert s.light_score == pytest.approx(0.7)

    def test_camera_validator_score_extracted(self):
        val = _scene_result(validators=[_validator("camera_validator", score=1.0)])
        s = compute_validation_summary(val)
        assert s.camera_score == pytest.approx(1.0)

    def test_export_validator_score_extracted(self):
        val = _scene_result(validators=[_validator("export_validator", score=0.3)])
        s = compute_validation_summary(val)
        assert s.export_score == pytest.approx(0.3)

    def test_absent_validator_score_is_none(self):
        val = _scene_result(validators=[_validator("object_validator")])
        s = compute_validation_summary(val)
        assert s.transform_score is None
        assert s.material_score is None

    def test_unknown_validator_name_ignored(self):
        val = _scene_result(validators=[_validator("custom_validator", score=0.5)])
        s = compute_validation_summary(val)
        # no known field maps to "custom_validator", so all scores stay None
        assert s.object_score is None
        assert s.material_score is None

    def test_multiple_validators_all_extracted(self):
        val = _scene_result(validators=[
            _validator("object_validator", score=1.0),
            _validator("material_validator", score=0.4),
            _validator("light_validator", score=0.8),
        ])
        s = compute_validation_summary(val)
        assert s.object_score == pytest.approx(1.0)
        assert s.material_score == pytest.approx(0.4)
        assert s.light_score == pytest.approx(0.8)
        assert s.camera_score is None

    def test_fixture_success_has_object_and_transform_scores(self):
        from benchmark.analysis.trace_reader import read_validation_result
        val = read_validation_result(FIXTURES / "validation_result_success.json")
        s = compute_validation_summary(val)
        assert s.object_score == pytest.approx(1.0)
        assert s.transform_score == pytest.approx(1.0)

    def test_fixture_partial_has_object_and_material_scores(self):
        from benchmark.analysis.trace_reader import read_validation_result
        val = read_validation_result(FIXTURES / "validation_result_partial.json")
        s = compute_validation_summary(val)
        assert s.object_score == pytest.approx(1.0)
        assert s.material_score == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# error / warning counts
# ---------------------------------------------------------------------------

class TestIssueCount:
    def test_no_issues_zero_counts(self):
        val = _scene_result()
        s = compute_validation_summary(val)
        assert s.validation_error_count == 0
        assert s.validation_warning_count == 0

    def test_top_level_error_counted(self):
        val = _scene_result(issues=[_issue("error")])
        s = compute_validation_summary(val)
        assert s.validation_error_count == 1
        assert s.validation_warning_count == 0

    def test_top_level_warning_counted(self):
        val = _scene_result(issues=[_issue("warning")])
        s = compute_validation_summary(val)
        assert s.validation_error_count == 0
        assert s.validation_warning_count == 1

    def test_info_severity_not_counted(self):
        val = _scene_result(issues=[_issue("info")])
        s = compute_validation_summary(val)
        assert s.validation_error_count == 0
        assert s.validation_warning_count == 0

    def test_validator_level_issues_counted(self):
        v = _validator("object_validator", issues=[_issue("error"), _issue("warning")])
        val = _scene_result(validators=[v])
        s = compute_validation_summary(val)
        assert s.validation_error_count == 1
        assert s.validation_warning_count == 1

    def test_top_level_and_validator_issues_combined(self):
        v = _validator("material_validator", issues=[_issue("error")])
        val = _scene_result(
            issues=[_issue("warning"), _issue("error")],
            validators=[v],
        )
        s = compute_validation_summary(val)
        assert s.validation_error_count == 2
        assert s.validation_warning_count == 1

    def test_multiple_validators_issues_aggregated(self):
        v1 = _validator("object_validator", issues=[_issue("error")])
        v2 = _validator("material_validator", issues=[_issue("error"), _issue("warning")])
        val = _scene_result(validators=[v1, v2])
        s = compute_validation_summary(val)
        assert s.validation_error_count == 2
        assert s.validation_warning_count == 1

    def test_fixture_partial_has_errors(self):
        from benchmark.analysis.trace_reader import read_validation_result
        val = read_validation_result(FIXTURES / "validation_result_partial.json")
        s = compute_validation_summary(val)
        assert s.validation_error_count >= 1


# ---------------------------------------------------------------------------
# Passed / failed / skipped validators
# ---------------------------------------------------------------------------

class TestValidatorStatusCounts:
    def test_all_passed(self):
        val = _scene_result(validators=[
            _validator("object_validator"),
            _validator("transform_validator"),
        ])
        s = compute_validation_summary(val)
        assert s.passed_validator_count == 2
        assert s.failed_validator_count == 0
        assert s.skipped_validator_count == 0

    def test_one_failed(self):
        val = _scene_result(validators=[
            _validator("object_validator"),
            _validator("material_validator", status="failed", score=0.3),
        ])
        s = compute_validation_summary(val)
        assert s.passed_validator_count == 1
        assert s.failed_validator_count == 1

    def test_skipped_validators(self):
        val = _scene_result(validators=[
            _validator("object_validator"),
            _validator("camera_validator", status="skipped", score=0.0),
        ])
        s = compute_validation_summary(val)
        assert s.passed_validator_count == 1
        assert s.skipped_validator_count == 1
        assert s.failed_validator_count == 0

    def test_skipped_validator_score_is_none(self):
        val = _scene_result(validators=[
            _validator("camera_validator", status="skipped", score=0.0),
        ])
        s = compute_validation_summary(val)
        assert s.camera_score is None

    def test_fixture_success_all_passed(self):
        from benchmark.analysis.trace_reader import read_validation_result
        val = read_validation_result(FIXTURES / "validation_result_success.json")
        s = compute_validation_summary(val)
        assert s.passed_validator_count == 2
        assert s.failed_validator_count == 0
        assert s.skipped_validator_count == 0

    def test_fixture_partial_has_skipped(self):
        from benchmark.analysis.trace_reader import read_validation_result
        val = read_validation_result(FIXTURES / "validation_result_partial.json")
        s = compute_validation_summary(val)
        assert s.passed_validator_count == 1
        assert s.failed_validator_count == 1
        assert s.skipped_validator_count == 1


# ---------------------------------------------------------------------------
# extract_validation_metrics
# ---------------------------------------------------------------------------

class TestExtractValidationMetrics:
    def test_returns_one_per_validator(self):
        val = _scene_result(validators=[
            _validator("object_validator"),
            _validator("material_validator", score=0.5),
        ])
        metrics = extract_validation_metrics(val)
        assert len(metrics) == 2

    def test_metric_fields_populated(self):
        val = _scene_result(validators=[
            _validator("object_validator", score=0.8,
                       issues=[_issue("error")]),
        ])
        metrics = extract_validation_metrics(val)
        m = metrics[0]
        assert m.validator_name == "object_validator"
        assert m.score == pytest.approx(0.8)
        assert m.status == "passed"
        assert m.issue_count == 1

    def test_empty_validators(self):
        val = _scene_result()
        assert extract_validation_metrics(val) == []


# ---------------------------------------------------------------------------
# extract_score_and_status
# ---------------------------------------------------------------------------

class TestExtractScoreAndStatus:
    def test_returns_tuple(self):
        val = _scene_result(total_score=0.75, overall_status="warning")
        score, status = extract_score_and_status(val)
        assert score == pytest.approx(0.75)
        assert status == "warning"

    def test_perfect_score_passed(self):
        val = _scene_result(total_score=1.0, overall_status="passed")
        score, status = extract_score_and_status(val)
        assert score == 1.0
        assert status == "passed"


# ---------------------------------------------------------------------------
# extract_issues
# ---------------------------------------------------------------------------

class TestExtractIssues:
    def test_empty_when_no_issues(self):
        val = _scene_result()
        assert extract_issues(val) == []

    def test_top_level_issues_included(self):
        val = _scene_result(issues=[_issue("error", code="scene_issue")])
        issues = extract_issues(val)
        assert len(issues) == 1
        assert issues[0]["code"] == "scene_issue"

    def test_validator_issues_included_with_validator_name(self):
        v = _validator("material_validator",
                       issues=[_issue("error", code="mat_issue")])
        val = _scene_result(validators=[v])
        issues = extract_issues(val)
        assert len(issues) == 1
        assert issues[0]["code"] == "mat_issue"
        assert issues[0]["validator"] == "material_validator"

    def test_top_level_and_validator_issues_combined(self):
        v = _validator("object_validator",
                       issues=[_issue("warning", code="obj_warn")])
        val = _scene_result(
            issues=[_issue("error", code="top_err")],
            validators=[v],
        )
        issues = extract_issues(val)
        assert len(issues) == 2
        codes = {i["code"] for i in issues}
        assert "top_err" in codes
        assert "obj_warn" in codes

    def test_fixture_partial_has_issues(self):
        from benchmark.analysis.trace_reader import read_validation_result
        val = read_validation_result(FIXTURES / "validation_result_partial.json")
        issues = extract_issues(val)
        assert len(issues) >= 1
        assert any(i.get("code") == "material_roughness_mismatch" for i in issues)
