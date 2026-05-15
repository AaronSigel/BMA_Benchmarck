from copy import deepcopy

from benchmark.blender.models import (
    CameraSnapshot,
    ColorRGBA,
    LightSnapshot,
    MaterialSnapshot,
    ObjectSnapshot,
    Vector3,
)
from benchmark.tasks.models import (
    ExpectedCamera,
    ExpectedLight,
    ExpectedMaterial,
    ExpectedObject,
)
from benchmark.validation.matcher import SceneMatcher, name_similarity, normalize_name


def vector(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Vector3:
    return Vector3(x=x, y=y, z=z)


def object_snapshot(
    name: str,
    type_: str = "MESH",
    primitive_hint: str | None = None,
) -> ObjectSnapshot:
    return ObjectSnapshot(
        name=name,
        type=type_,
        primitive_hint=primitive_hint,
        location=vector(),
        rotation_euler=vector(),
        scale=vector(1.0, 1.0, 1.0),
        dimensions=vector(2.0, 2.0, 2.0),
        material_slots=[],
        parent=None,
        collection_names=["Collection"],
        vertex_count=None,
        polygon_count=None,
    )


def material_snapshot(name: str) -> MaterialSnapshot:
    return MaterialSnapshot(
        name=name,
        base_color=ColorRGBA(r=1.0, g=0.0, b=0.0),
        roughness=0.5,
        metallic=0.0,
        use_nodes=True,
    )


def light_snapshot(name: str, type_: str = "AREA") -> LightSnapshot:
    return LightSnapshot(
        name=name,
        type=type_,
        location=vector(),
        rotation_euler=vector(),
        energy=500.0,
        color=ColorRGBA(r=1.0, g=1.0, b=1.0),
    )


def camera_snapshot(name: str, is_active: bool = False) -> CameraSnapshot:
    return CameraSnapshot(
        name=name,
        location=vector(),
        rotation_euler=vector(),
        lens=50.0,
        sensor_width=36.0,
        is_active=is_active,
    )


def test_normalize_name_removes_blender_suffix_and_separators() -> None:
    assert normalize_name("Red_Cube-01.001") == "redcube01"


def test_name_similarity_matches_blender_suffix() -> None:
    assert name_similarity("Cube", "Cube.001") == 1.0


def test_name_similarity_matches_spaces_and_underscores() -> None:
    assert name_similarity("red_cube", "Red Cube") == 1.0


def test_name_similarity_returns_partial_match_score() -> None:
    assert name_similarity("Cube", "Large Cube") == 0.8


def test_match_expected_object_prefers_name() -> None:
    matcher = SceneMatcher()
    objects = [
        object_snapshot("Sphere", primitive_hint="sphere"),
        object_snapshot("Cube.001", primitive_hint="cube"),
    ]

    actual = matcher.match_expected_object(ExpectedObject(name="Cube", type="MESH"), objects)

    assert actual is objects[1]


def test_match_expected_object_falls_back_to_primitive() -> None:
    matcher = SceneMatcher()
    objects = [
        object_snapshot("Generated Mesh", primitive_hint="sphere"),
        object_snapshot("Generated Mesh.001", primitive_hint="cube"),
    ]

    actual = matcher.match_expected_object(ExpectedObject(type="MESH", primitive="cube"), objects)

    assert actual is objects[1]


def test_match_expected_object_falls_back_to_type() -> None:
    matcher = SceneMatcher()
    objects = [object_snapshot("Key", type_="LIGHT"), object_snapshot("Any Mesh", type_="MESH")]

    actual = matcher.match_expected_object(ExpectedObject(type="mesh"), objects)

    assert actual is objects[1]


def test_match_expected_object_returns_none_without_match() -> None:
    matcher = SceneMatcher()

    actual = matcher.match_expected_object(
        ExpectedObject(name="Missing", type="CURVE"),
        [object_snapshot("Cube", type_="MESH", primitive_hint="cube")],
    )

    assert actual is None


def test_match_expected_material_by_name() -> None:
    matcher = SceneMatcher()
    materials = [material_snapshot("Blue"), material_snapshot("Red Material.001")]

    actual = matcher.match_expected_material(ExpectedMaterial(name="red_material"), materials)

    assert actual is materials[1]


def test_match_expected_light_by_name_then_type() -> None:
    matcher = SceneMatcher()
    lights = [light_snapshot("Fill", "POINT"), light_snapshot("Key Area", "AREA")]

    named = matcher.match_expected_light(ExpectedLight(name="key_area", type="AREA"), lights)
    typed = matcher.match_expected_light(ExpectedLight(type="POINT"), lights)

    assert named is lights[1]
    assert typed is lights[0]


def test_match_expected_camera_by_name_then_active_camera() -> None:
    matcher = SceneMatcher()
    cameras = [camera_snapshot("Camera A"), camera_snapshot("Render Camera", is_active=True)]

    named = matcher.match_expected_camera(ExpectedCamera(name="render_camera"), cameras)
    active = matcher.match_expected_camera(ExpectedCamera(), cameras)

    assert named is cameras[1]
    assert active is cameras[1]


def test_matcher_does_not_mutate_input_lists() -> None:
    matcher = SceneMatcher()
    objects = [object_snapshot("Cube.001", primitive_hint="cube")]
    materials = [material_snapshot("Red")]
    lights = [light_snapshot("Key")]
    cameras = [camera_snapshot("Camera", is_active=True)]
    original_objects = deepcopy(objects)
    original_materials = deepcopy(materials)
    original_lights = deepcopy(lights)
    original_cameras = deepcopy(cameras)

    matcher.match_expected_object(ExpectedObject(name="Cube", type="MESH"), objects)
    matcher.match_expected_material(ExpectedMaterial(name="Red"), materials)
    matcher.match_expected_light(ExpectedLight(type="AREA"), lights)
    matcher.match_expected_camera(ExpectedCamera(), cameras)

    assert objects == original_objects
    assert materials == original_materials
    assert lights == original_lights
    assert cameras == original_cameras
