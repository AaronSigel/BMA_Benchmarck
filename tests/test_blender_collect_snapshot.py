import json
import sys
from pathlib import Path
from types import SimpleNamespace

from benchmark.blender.models import SceneSnapshot
from benchmark.blender.scripts.collect_snapshot import collect_snapshot


class FakeCollection(list):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name


class FakeMeshData:
    def __init__(self, name: str, materials=None) -> None:
        self.name = name
        self.vertices = [object()] * 8
        self.polygons = [object()] * 6
        self.materials = materials or []


class FakeLightData:
    type = "AREA"
    energy = 500.0
    color = (1.0, 1.0, 1.0)


class FakeCameraData:
    lens = 50.0
    sensor_width = 36.0


def fake_object(
    name: str,
    object_type: str,
    data,
    location=(0.0, 0.0, 0.0),
    custom_props: dict | None = None,
):
    return SimpleNamespace(
        name=name,
        type=object_type,
        data=data,
        location=location,
        rotation_euler=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        dimensions=(2.0, 2.0, 2.0),
        material_slots=[],
        parent=None,
        users_collection=[],
        get=lambda key, default=None: (custom_props or {}).get(key, default),
    )


def make_fake_bpy(empty: bool = False):
    collection = FakeCollection("Collection")
    red = SimpleNamespace(
        name="RedMaterial",
        diffuse_color=(1.0, 0.0, 0.0, 1.0),
        roughness=0.5,
        metallic=0.0,
        use_nodes=False,
        node_tree=None,
    )

    objects = []
    if not empty:
        cube = fake_object("FixtureCube", "MESH", FakeMeshData("CubeMesh", [red]), (0.0, 0.0, 1.0))
        cube.users_collection = [collection]
        light = fake_object("FixtureAreaLight", "LIGHT", FakeLightData(), (0.0, -4.0, 6.0))
        camera = fake_object("FixtureCamera", "CAMERA", FakeCameraData(), (6.0, -6.0, 4.0))
        objects = [cube, light, camera]

    scene = SimpleNamespace(
        name="Fixture",
        objects=objects,
        camera=objects[-1] if objects else None,
        frame_start=1,
        frame_end=1,
        frame_current=1,
        render=SimpleNamespace(engine="CYCLES", resolution_x=512, resolution_y=512),
    )

    return SimpleNamespace(
        context=SimpleNamespace(scene=scene),
        data=SimpleNamespace(
            objects=objects,
            materials=[red] if not empty else [],
            collections=[collection],
        ),
        app=SimpleNamespace(version=(4, 0, 0)),
    )


def test_collect_snapshot_imports_without_bpy() -> None:
    assert "bpy" not in sys.modules


def test_collect_snapshot_returns_scene_snapshot_compatible_dict(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "bpy", make_fake_bpy())

    snapshot = collect_snapshot({})
    validated = SceneSnapshot.model_validate(snapshot)

    assert validated.scene_name == "Fixture"
    assert [obj.name for obj in validated.objects] == ["FixtureCube"]
    assert validated.mesh_object_count == 1
    assert validated.light_count == 1
    assert validated.camera_count == 1
    assert validated.all_object_count == 3
    assert validated.objects[0].primitive_hint == "cube"
    assert validated.objects[0].vertex_count == 8
    assert validated.objects[0].polygon_count == 6
    assert validated.objects[0].material_slots == ["RedMaterial"]
    assert validated.materials[0].base_color is not None
    assert validated.lights[0].name == "FixtureAreaLight"
    assert validated.cameras[0].is_active is True
    assert validated.collections == ["Collection"]
    assert validated.blender_version == "4.0.0"


def test_collect_snapshot_prefers_bma_primitive_hint_custom_property(monkeypatch) -> None:
    bpy = make_fake_bpy(empty=True)
    obj = fake_object(
        "Lowpoly_Roof",
        "MESH",
        FakeMeshData("GeneratedMesh"),
        custom_props={"bma_primitive_hint": "cone"},
    )
    bpy.context.scene.objects = [obj]
    bpy.data.objects = [obj]
    monkeypatch.setitem(sys.modules, "bpy", bpy)

    snapshot = SceneSnapshot.model_validate(collect_snapshot({}))

    assert snapshot.objects[0].primitive_hint == "cone"


def test_collect_snapshot_writes_scene_snapshot_json(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "bpy", make_fake_bpy())
    output_dir = tmp_path / "snapshot"

    snapshot = collect_snapshot({"output_dir": str(output_dir)})

    snapshot_path = output_dir / "scene_snapshot.json"
    assert snapshot_path.exists()
    restored = SceneSnapshot.model_validate_json(snapshot_path.read_text(encoding="utf-8"))
    assert restored == SceneSnapshot.model_validate(snapshot)


def test_collect_snapshot_handles_empty_scene(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "bpy", make_fake_bpy(empty=True))

    snapshot = SceneSnapshot.model_validate(collect_snapshot({}))

    assert snapshot.objects == []
    assert snapshot.materials == []
    assert snapshot.lights == []
    assert snapshot.cameras == []


def test_collect_snapshot_json_round_trip(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "bpy", make_fake_bpy())

    snapshot = collect_snapshot({})

    assert SceneSnapshot.model_validate_json(json.dumps(snapshot)) == SceneSnapshot.model_validate(snapshot)
