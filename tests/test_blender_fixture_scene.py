import sys
from pathlib import Path
from types import SimpleNamespace

from benchmark.blender.scripts.create_fixture_scene import create_fixture_scene


# --- fake mathutils (replaces the Blender-only module) ---

class FakeVector:
    def __init__(self, data):
        self.x = float(data[0])
        self.y = float(data[1])
        self.z = float(data[2])

    def __sub__(self, other):
        return FakeVector((self.x - other.x, self.y - other.y, self.z - other.z))

    @property
    def length(self):
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5

    def to_track_quat(self, track, up):
        return _FakeQuat()


class _FakeQuat:
    def to_euler(self):
        return (0.0, 0.0, 0.0)


_fake_mathutils = SimpleNamespace(Vector=FakeVector)


# --- fake bpy infrastructure ---

_BOUND_BOX = [
    (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
    (-1, -1,  1), (1, -1,  1), (1, 1,  1), (-1, 1,  1),
]


class FakeLocation:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __sub__(self, other):
        return FakeVector((self.x - other.x, self.y - other.y, self.z - other.z))


class FakeMatrix:
    def __matmul__(self, vec):
        return vec  # identity transform


class FakeCollection(list):
    def remove(self, datablock, do_unlink=False):
        super().remove(datablock)


class FakeNodes:
    def get(self, name):
        return None

    def __iter__(self):
        return iter([])


class FakeNodeTree:
    def __init__(self):
        self.nodes = FakeNodes()


class FakeMaterials(FakeCollection):
    def new(self, name: str):
        material = SimpleNamespace(
            name=name,
            diffuse_color=None,
            users=1,
            node_tree=FakeNodeTree(),
        )
        self.append(material)
        return material


class FakeObjectData:
    def __init__(self) -> None:
        self.materials = []
        self.energy = None
        self.size = None


class FakeScene:
    def __init__(self) -> None:
        self.name = "Scene"
        self.frame_start = 1
        self.frame_end = 1
        self.frame_current = 1
        self.camera = None

    def frame_set(self, frame: int) -> None:
        self.frame_current = frame


class FakeBpy:
    def __init__(self) -> None:
        self.context = SimpleNamespace(
            scene=FakeScene(),
            object=None,
            view_layer=SimpleNamespace(update=lambda: None),
        )
        self.data = SimpleNamespace(
            objects=FakeCollection(),
            meshes=FakeCollection(),
            materials=FakeMaterials(),
            lights=FakeCollection(),
            cameras=FakeCollection(),
        )
        self.ops = SimpleNamespace(
            mesh=SimpleNamespace(
                primitive_cube_add=self.primitive_cube_add,
                primitive_uv_sphere_add=self.primitive_uv_sphere_add,
                primitive_cylinder_add=self.primitive_cylinder_add,
                primitive_plane_add=self.primitive_plane_add,
            ),
            object=SimpleNamespace(
                light_add=self.light_add,
                camera_add=self.camera_add,
            ),
            wm=SimpleNamespace(save_as_mainfile=self.save_as_mainfile),
        )

    def _add_object(self, name: str, location, object_type: str = "MESH"):
        obj = SimpleNamespace(
            name=name,
            type=object_type,
            location=FakeLocation(*location),
            rotation_euler=(0.0, 0.0, 0.0),
            data=FakeObjectData(),
            matrix_world=FakeMatrix(),
            bound_box=_BOUND_BOX,
        )
        self.data.objects.append(obj)
        self.context.object = obj
        return obj

    def primitive_cube_add(self, size, location):
        self._add_object("Cube", location)

    def primitive_uv_sphere_add(self, segments, ring_count, location, radius=1.0):
        self._add_object("Sphere", location)

    def primitive_cylinder_add(self, vertices, radius, depth, location):
        self._add_object("Cylinder", location)

    def primitive_plane_add(self, size, location):
        self._add_object("Plane", location)

    def light_add(self, type, location):
        light = self._add_object("Area", location, "LIGHT")
        self.data.lights.append(light.data)

    def camera_add(self, location):
        camera = self._add_object("Camera", location, "CAMERA")
        self.data.cameras.append(camera.data)

    def save_as_mainfile(self, filepath):
        Path(filepath).write_bytes(b"BLENDER")


def test_create_fixture_scene_imports_without_bpy() -> None:
    assert "bpy" not in sys.modules


def test_create_fixture_scene_creates_objects_materials_light_and_camera(monkeypatch) -> None:
    fake_bpy = FakeBpy()
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    monkeypatch.setitem(sys.modules, "mathutils", _fake_mathutils)

    result = create_fixture_scene({"scene_name": "Fixture"})

    assert result["scene_name"] == "Fixture"
    assert result["objects"] == [
        "FixtureCube",
        "FixtureSphere",
        "FixtureCylinder",
        "FixtureFloor",
    ]
    assert result["materials"] == ["RedMaterial", "BlueMaterial", "GrayMaterial"]
    assert result["light"] == "FixtureAreaLight"
    assert result["camera"] == "FixtureCamera"
    assert fake_bpy.context.scene.camera.name == "FixtureCamera"
    assert len(fake_bpy.data.objects) == 6


def test_create_fixture_scene_can_skip_camera_and_light(monkeypatch) -> None:
    fake_bpy = FakeBpy()
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    monkeypatch.setitem(sys.modules, "mathutils", _fake_mathutils)

    result = create_fixture_scene({"add_camera": False, "add_light": False})

    assert result["light"] is None
    assert result["camera"] is None
    assert len(fake_bpy.data.objects) == 4


def test_create_fixture_scene_saves_blend_file(monkeypatch, tmp_path: Path) -> None:
    fake_bpy = FakeBpy()
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    monkeypatch.setitem(sys.modules, "mathutils", _fake_mathutils)
    save_path = tmp_path / "nested" / "fixture.blend"

    result = create_fixture_scene({"save_path": str(save_path)})

    assert result["save_path"] == str(save_path)
    assert result["saved"] is True
    assert save_path.read_bytes() == b"BLENDER"
