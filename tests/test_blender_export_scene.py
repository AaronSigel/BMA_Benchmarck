import sys
from pathlib import Path
from types import SimpleNamespace

from benchmark.blender.scripts.export_scene import export_scene


class FakeOperator:
    def __init__(self, content: bytes, available: bool = True) -> None:
        self.content = content
        self.available = available
        self.calls = []

    def __call__(self, **kwargs) -> None:
        self.calls.append(kwargs)
        Path(kwargs["filepath"]).write_bytes(self.content)

    def poll(self) -> bool:
        return self.available


class RaisingPollOperator(FakeOperator):
    def poll(self) -> bool:
        raise AttributeError("operator could not be found")


def make_fake_bpy(*, gltf_available: bool = True, fbx_available: bool = True):
    def save_as_mainfile(filepath: str) -> None:
        Path(filepath).write_bytes(b"BLEND")

    gltf = FakeOperator(b"GLB", available=gltf_available)
    fbx = FakeOperator(b"FBX", available=fbx_available)
    return SimpleNamespace(
        ops=SimpleNamespace(
            wm=SimpleNamespace(save_as_mainfile=save_as_mainfile),
            export_scene=SimpleNamespace(gltf=gltf, fbx=fbx),
        )
    )


def make_fake_bpy_with_broken_gltf():
    def save_as_mainfile(filepath: str) -> None:
        Path(filepath).write_bytes(b"BLEND")

    return SimpleNamespace(
        ops=SimpleNamespace(
            wm=SimpleNamespace(save_as_mainfile=save_as_mainfile),
            export_scene=SimpleNamespace(
                gltf=RaisingPollOperator(b"GLB"),
                fbx=FakeOperator(b"FBX"),
            ),
        )
    )


def test_export_scene_imports_without_bpy() -> None:
    assert "bpy" not in sys.modules


def test_export_scene_blend(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "bpy", make_fake_bpy())
    output_path = tmp_path / "nested" / "scene.blend"

    result = export_scene({"output_path": str(output_path), "format": "blend"})

    assert result == {
        "ok": True,
        "output_path": str(output_path),
        "format": "blend",
        "exists": True,
        "file_size_bytes": 5,
        "error": None,
    }


def test_export_scene_glb(monkeypatch, tmp_path: Path) -> None:
    fake_bpy = make_fake_bpy()
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    output_path = tmp_path / "scene.glb"

    result = export_scene({"output_path": str(output_path), "format": "glb"})

    assert result["ok"] is True
    assert result["format"] == "glb"
    assert result["file_size_bytes"] == 3
    assert fake_bpy.ops.export_scene.gltf.calls == [
        {"filepath": str(output_path), "export_format": "GLB"}
    ]


def test_export_scene_gltf(monkeypatch, tmp_path: Path) -> None:
    fake_bpy = make_fake_bpy()
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    output_path = tmp_path / "scene.gltf"

    result = export_scene({"output_path": str(output_path), "format": "gltf"})

    assert result["ok"] is True
    assert fake_bpy.ops.export_scene.gltf.calls == [
        {"filepath": str(output_path), "export_format": "GLTF_SEPARATE"}
    ]


def test_export_scene_returns_error_for_unsupported_format(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "bpy", make_fake_bpy())
    output_path = tmp_path / "scene.obj"

    result = export_scene({"output_path": str(output_path), "format": "obj"})

    assert result["ok"] is False
    assert result["output_path"] == str(output_path)
    assert result["format"] == "obj"
    assert result["exists"] is False
    assert "unsupported format" in result["error"]


def test_export_scene_handles_missing_fbx_operator(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "bpy", make_fake_bpy(fbx_available=False))
    output_path = tmp_path / "scene.fbx"

    result = export_scene({"output_path": str(output_path), "format": "fbx"})

    assert result["ok"] is False
    assert result["format"] == "fbx"
    assert result["exists"] is False
    assert "FBX" in result["error"]


def test_export_scene_handles_gltf_poll_error(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "bpy", make_fake_bpy_with_broken_gltf())
    output_path = tmp_path / "scene.glb"

    result = export_scene({"output_path": str(output_path), "format": "glb"})

    assert result["ok"] is False
    assert result["format"] == "glb"
    assert "glTF export operator" in result["error"]
    assert "operator could not be found" in result["error"]


def test_export_scene_requires_output_path(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "bpy", make_fake_bpy())

    result = export_scene({"format": "blend"})

    assert result["ok"] is False
    assert result["output_path"] is None
    assert "output_path" in result["error"]
