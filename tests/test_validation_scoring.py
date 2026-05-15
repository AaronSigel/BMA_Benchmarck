import pytest

from benchmark.blender.models import Vector3 as SnapshotVector3
from benchmark.tasks.models import Vector3 as ExpectedVector3
from benchmark.validation.scoring import (
    bool_score,
    clamp_score,
    tolerance_score,
    vector_distance,
    vector_tolerance_score,
    weighted_average,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-0.5, 0.0),
        (0.25, 0.25),
        (1.5, 1.0),
    ],
)
def test_clamp_score(value: float, expected: float) -> None:
    assert clamp_score(value) == expected


def test_bool_score() -> None:
    assert bool_score(True) == 1.0
    assert bool_score(False) == 0.0


def test_tolerance_score_is_one_inside_tolerance() -> None:
    assert tolerance_score(expected=10.0, actual=10.5, tolerance=0.5) == 1.0


def test_tolerance_score_smoothly_decreases_outside_tolerance() -> None:
    assert tolerance_score(expected=10.0, actual=10.75, tolerance=0.5) == pytest.approx(0.5)
    assert tolerance_score(expected=10.0, actual=11.0, tolerance=0.5) == 0.0


@pytest.mark.parametrize("actual", [-100.0, -1.0, 0.0, 0.5, 1.0, 100.0])
def test_tolerance_score_stays_in_score_range(actual: float) -> None:
    score = tolerance_score(expected=0.0, actual=actual, tolerance=0.25)

    assert 0.0 <= score <= 1.0


def test_tolerance_score_with_zero_tolerance() -> None:
    assert tolerance_score(expected=1.0, actual=1.0, tolerance=0.0) == 1.0
    assert tolerance_score(expected=1.0, actual=1.01, tolerance=0.0) == 0.0


def test_vector_distance_accepts_expected_and_snapshot_vectors() -> None:
    expected = ExpectedVector3(x=0.0, y=0.0, z=0.0)
    actual = SnapshotVector3(x=3.0, y=4.0, z=12.0)

    assert vector_distance(expected, actual) == 13.0


def test_vector_tolerance_score_is_one_for_matching_vectors() -> None:
    expected = ExpectedVector3(x=1.0, y=2.0, z=3.0)
    actual = SnapshotVector3(x=1.0, y=2.0, z=3.0)

    assert vector_tolerance_score(expected, actual, tolerance=0.05) == 1.0


def test_vector_tolerance_score_decreases_with_distance() -> None:
    expected = ExpectedVector3(x=0.0, y=0.0, z=0.0)
    actual = SnapshotVector3(x=0.75, y=0.0, z=0.0)

    assert vector_tolerance_score(expected, actual, tolerance=0.5) == pytest.approx(0.5)


def test_weighted_average_returns_zero_for_empty_scores() -> None:
    assert weighted_average([]) == 0.0


def test_weighted_average_uses_weights() -> None:
    assert weighted_average([(1.0, 0.25), (0.0, 0.75)]) == pytest.approx(0.25)


def test_weighted_average_ignores_non_positive_weights() -> None:
    assert weighted_average([(1.0, 0.0), (0.25, -1.0), (0.5, 2.0)]) == 0.5


def test_weighted_average_clamps_input_scores() -> None:
    assert weighted_average([(2.0, 1.0), (-1.0, 1.0)]) == 0.5
