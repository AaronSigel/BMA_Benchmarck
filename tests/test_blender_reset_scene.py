import json
import sys
from types import SimpleNamespace

from benchmark.blender.scripts.reset_scene import reset_scene


class FakeCollection(list):
    def remove(self, datablock, do_unlink=False):
        super().remove(datablock)


class FakeScene:
    def __init__(self) -> None:
        self.name = "OldScene"
        self.frame_start = 10
        self.frame_end = 20
        self.frame_current = 12

    def frame_set(self, frame: int) -> None:
        self.frame_current = frame


def make_fake_bpy():
    return SimpleNamespace(
        context=SimpleNamespace(scene=FakeScene()),
        data=SimpleNamespace(
            objects=FakeCollection(
                [
                    SimpleNamespace(name="Cube"),
                    SimpleNamespace(name="Camera"),
                ]
            ),
            meshes=FakeCollection(
                [
                    SimpleNamespace(name="UsedMesh", users=1),
                    SimpleNamespace(name="UnusedMesh", users=0),
                ]
            ),
            materials=FakeCollection([SimpleNamespace(name="UnusedMaterial", users=0)]),
            lights=FakeCollection([SimpleNamespace(name="UnusedLight", users=0)]),
            cameras=FakeCollection([SimpleNamespace(name="UnusedCamera", users=0)]),
        ),
    )


def test_reset_scene_imports_without_bpy() -> None:
    assert "bpy" not in sys.modules


def test_reset_scene_removes_objects_and_resets_frame(
    monkeypatch,
) -> None:
    fake_bpy = make_fake_bpy()
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)

    result = reset_scene({"scene_name": "EmptyScene"})

    assert result == {
        "removed_objects": 2,
        "remaining_objects": 0,
        "scene_name": "EmptyScene",
    }
    assert list(fake_bpy.data.objects) == []
    assert [mesh.name for mesh in fake_bpy.data.meshes] == ["UsedMesh"]
    assert list(fake_bpy.data.materials) == []
    assert list(fake_bpy.data.lights) == []
    assert list(fake_bpy.data.cameras) == []
    assert fake_bpy.context.scene.frame_start == 1
    assert fake_bpy.context.scene.frame_end == 1
    assert fake_bpy.context.scene.frame_current == 1


def test_reset_scene_result_is_json_compatible(monkeypatch) -> None:
    fake_bpy = make_fake_bpy()
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)

    result = reset_scene({})

    assert json.loads(json.dumps(result)) == result
    assert result["scene_name"] == "OldScene"

