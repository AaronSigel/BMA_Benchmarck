import sys
from pathlib import Path
from types import SimpleNamespace

from benchmark.blender.scripts.render_scene import render_scene


class FakeRender:
    def __init__(self) -> None:
        self.resolution_x = 0
        self.resolution_y = 0
        self.filepath = ""
        self.image_settings = SimpleNamespace(file_format=None)
        self.film_transparent = False
        self._engine = "BLENDER_EEVEE"

    @property
    def engine(self) -> str:
        return self._engine

    @engine.setter
    def engine(self, value: str) -> None:
        self._engine = value


class FakeViewSettings:
    def __init__(self) -> None:
        self.view_transform = "Standard"
        self.exposure = 0.0
        self.gamma = 1.0
        self.look = "None"


def _make_types(available_engines: list[str]) -> SimpleNamespace:
    items = [SimpleNamespace(identifier=e) for e in available_engines]
    return SimpleNamespace(
        RenderSettings=SimpleNamespace(
            bl_rna=SimpleNamespace(
                properties={"engine": SimpleNamespace(enum_items=items)}
            )
        )
    )


_ALL_ENGINES = ["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES", "BLENDER_WORKBENCH"]


def make_fake_bpy(reject_engine: str | None = None, with_camera: bool = True):
    camera = SimpleNamespace(name="FixtureCamera", type="CAMERA") if with_camera else None
    render = FakeRender()
    scene = SimpleNamespace(render=render, camera=camera, view_settings=FakeViewSettings())

    available = [e for e in _ALL_ENGINES if e != reject_engine]

    def render_op(write_still: bool) -> None:
        assert write_still is True
        Path(render.filepath).write_bytes(b"\x89PNG\r\n")

    return SimpleNamespace(
        context=SimpleNamespace(scene=scene),
        data=SimpleNamespace(objects=[camera] if camera else []),
        ops=SimpleNamespace(render=SimpleNamespace(render=render_op)),
        types=_make_types(available),
    )


def test_render_scene_imports_without_bpy() -> None:
    assert "bpy" not in sys.modules


def test_render_scene_creates_png(monkeypatch, tmp_path: Path) -> None:
    fake_bpy = make_fake_bpy()
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    output_path = tmp_path / "renders" / "scene.png"

    result = render_scene(
        {
            "output_path": str(output_path),
            "resolution_x": 640,
            "resolution_y": 480,
            "camera_name": "FixtureCamera",
            "transparent": True,
        }
    )

    assert result["ok"] is True
    assert result["output_path"] == str(output_path)
    assert result["exists"] is True
    assert result["file_size_bytes"] > 0
    assert result["resolution_x"] == 640
    assert result["resolution_y"] == 480
    assert result["engine"] == "BLENDER_EEVEE_NEXT"
    assert fake_bpy.context.scene.render.image_settings.file_format == "PNG"
    assert fake_bpy.context.scene.render.film_transparent is True


def test_render_scene_falls_back_to_eevee(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "bpy", make_fake_bpy(reject_engine="BLENDER_EEVEE_NEXT"))
    output_path = tmp_path / "scene.png"

    result = render_scene({"output_path": str(output_path)})

    assert result["ok"] is True
    assert result["engine"] == "BLENDER_EEVEE"


def test_render_scene_returns_error_when_camera_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "bpy", make_fake_bpy(with_camera=False))

    result = render_scene(
        {
            "output_path": str(tmp_path / "scene.png"),
            "camera_name": "MissingCamera",
        }
    )

    assert result["ok"] is False
    assert result["exists"] is False
    assert "camera not found" in result["error"]


def test_render_scene_returns_error_without_active_camera(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "bpy", make_fake_bpy(with_camera=False))

    result = render_scene({"output_path": str(tmp_path / "scene.png")})

    assert result["ok"] is False
    assert "active camera" in result["error"]


def test_render_scene_requires_output_path(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "bpy", make_fake_bpy())

    result = render_scene({})

    assert result["ok"] is False
    assert result["output_path"] is None
    assert "output_path" in result["error"]
