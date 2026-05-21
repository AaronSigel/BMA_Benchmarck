"""Scene matching helpers."""

import re

from benchmark.blender.models import (
    CameraSnapshot,
    LightSnapshot,
    MaterialSnapshot,
    ObjectSnapshot,
)
from benchmark.tasks.models import (
    ExpectedCamera,
    ExpectedLight,
    ExpectedMaterial,
    ExpectedObject,
)

BLENDER_SUFFIX_RE = re.compile(r"\.\d{3}$")


def normalize_name(name: str) -> str:
    normalized = BLENDER_SUFFIX_RE.sub("", name.lower())
    return normalized.replace(" ", "").replace("_", "").replace("-", "")


def name_similarity(expected: str, actual: str) -> float:
    expected_normalized = normalize_name(expected)
    actual_normalized = normalize_name(actual)

    if expected_normalized == actual_normalized:
        return 1.0
    if expected_normalized in actual_normalized or actual_normalized in expected_normalized:
        return 0.8
    return 0.0


class SceneMatcher:
    def match_expected_object(
        self,
        expected: ExpectedObject,
        objects: list[ObjectSnapshot],
    ) -> ObjectSnapshot | None:
        candidates: list[tuple[float, int, ObjectSnapshot]] = []
        for index, actual in enumerate(objects):
            score = self._object_match_score(expected, actual)
            if score > 0.0:
                candidates.append((score, -index, actual))

        if not candidates:
            return None
        return max(candidates, key=lambda candidate: (candidate[0], candidate[1]))[2]

    def match_expected_material(
        self,
        expected: ExpectedMaterial,
        materials: list[MaterialSnapshot],
    ) -> MaterialSnapshot | None:
        return self._best_named_match(expected.name, materials)

    def match_expected_light(
        self,
        expected: ExpectedLight,
        lights: list[LightSnapshot],
    ) -> LightSnapshot | None:
        if expected.name is not None:
            named_match = self._best_named_match(expected.name, lights)
            if named_match is not None:
                return named_match

        expected_type = expected.type.upper()
        type_matches = [light for light in lights if light.type.upper() == expected_type]
        if not type_matches:
            return None
        if len(type_matches) == 1:
            return type_matches[0]
        if expected.location is not None:
            return min(
                type_matches,
                key=lambda light: self._location_distance(expected.location, light.location),
            )
        return type_matches[0]

    @staticmethod
    def _location_distance(expected_location, actual_location) -> float:
        dx = expected_location.x - actual_location.x
        dy = expected_location.y - actual_location.y
        dz = expected_location.z - actual_location.z
        return dx * dx + dy * dy + dz * dz

    def match_expected_camera(
        self,
        expected: ExpectedCamera,
        cameras: list[CameraSnapshot],
    ) -> CameraSnapshot | None:
        if expected.name is not None:
            named_match = self._best_named_match(expected.name, cameras)
            if named_match is not None:
                return named_match

        for camera in cameras:
            if camera.is_active:
                return camera
        return cameras[0] if cameras else None

    def _object_match_score(self, expected: ExpectedObject, actual: ObjectSnapshot) -> float:
        name_score = name_similarity(expected.name, actual.name) if expected.name else 0.0
        if name_score > 0.0:
            return 100.0 + name_score

        primitive_score = self._primitive_match_score(expected, actual)
        if primitive_score > 0.0:
            return 10.0 + primitive_score

        if expected.type.lower() == actual.type.lower():
            return 1.0
        return 0.0

    def _primitive_match_score(self, expected: ExpectedObject, actual: ObjectSnapshot) -> float:
        if expected.primitive is None or actual.primitive_hint is None:
            return 0.0
        return name_similarity(expected.primitive, actual.primitive_hint)

    def _best_named_match(self, expected_name: str, actual_items: list) -> object | None:
        candidates = [
            (name_similarity(expected_name, actual.name), -index, actual)
            for index, actual in enumerate(actual_items)
        ]
        candidates = [candidate for candidate in candidates if candidate[0] > 0.0]
        if not candidates:
            return None
        return max(candidates, key=lambda candidate: (candidate[0], candidate[1]))[2]
