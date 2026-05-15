import sys
from pathlib import Path
from types import SimpleNamespace

from benchmark.blender.scripts.save_scene import save_scene


def make_fake_bpy() -> SimpleNamespace:
    def save_as_mainfile(filepath: str) -> None:
        Path(filepath).write_bytes(b"BLEND")

    return SimpleNamespace(
        ops=SimpleNamespace(wm=SimpleNamespace(save_as_mainfile=save_as_mainfile))
    )


def test_save_scene_imports_without_bpy() -> None:
    assert "bpy" not in sys.modules


def test_save_scene_creates_blend_file(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "bpy", make_fake_bpy())
    path = tmp_path / "nested" / "scene.blend"

    result = save_scene({"path": str(path)})

    assert result == {
        "ok": True,
        "path": str(path),
        "exists": True,
        "file_size_bytes": 5,
        "error": None,
    }
    assert path.read_bytes() == b"BLEND"


def test_save_scene_returns_error_for_non_blend_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "bpy", make_fake_bpy())
    path = tmp_path / "scene.txt"

    result = save_scene({"path": str(path)})

    assert result["ok"] is False
    assert result["path"] == str(path)
    assert result["exists"] is False
    assert result["file_size_bytes"] == 0
    assert ".blend" in result["error"]
    assert not path.exists()


def test_save_scene_returns_error_when_path_missing(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "bpy", make_fake_bpy())

    result = save_scene({})

    assert result["ok"] is False
    assert result["path"] is None
    assert result["exists"] is False
    assert result["file_size_bytes"] == 0
    assert "path" in result["error"]

