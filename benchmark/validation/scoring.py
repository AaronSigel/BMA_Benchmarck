"""Scoring helpers for scene validation."""

from math import sqrt
from typing import Protocol


class Vector3Like(Protocol):
    x: float
    y: float
    z: float


def clamp_score(value: float) -> float:
    """Return *value* clamped to the inclusive 0..1 score range."""
    return max(0.0, min(1.0, float(value)))


def bool_score(condition: bool) -> float:
    return 1.0 if condition else 0.0


def tolerance_score(expected: float, actual: float, tolerance: float) -> float:
    """Score a numeric value using a tolerance band.

    Values inside tolerance receive 1.0.  Outside that band, the score falls
    linearly to 0.0 over the same tolerance distance.
    """
    distance = abs(expected - actual)
    if distance <= tolerance:
        return 1.0
    if tolerance <= 0.0:
        return 0.0
    return clamp_score(1.0 - ((distance - tolerance) / tolerance))


def vector_distance(expected: Vector3Like, actual: Vector3Like) -> float:
    return sqrt(
        (expected.x - actual.x) ** 2
        + (expected.y - actual.y) ** 2
        + (expected.z - actual.z) ** 2
    )


def vector_tolerance_score(expected: Vector3Like, actual: Vector3Like, tolerance: float) -> float:
    return tolerance_score(0.0, vector_distance(expected, actual), tolerance)


def weighted_average(scores: list[tuple[float, float]]) -> float:
    if not scores:
        return 0.0

    total_weight = sum(weight for _, weight in scores if weight > 0.0)
    if total_weight <= 0.0:
        return 0.0

    weighted_sum = sum(clamp_score(score) * weight for score, weight in scores if weight > 0.0)
    return clamp_score(weighted_sum / total_weight)
