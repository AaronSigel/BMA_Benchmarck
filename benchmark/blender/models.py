from pydantic import BaseModel, Field


class Vector3(BaseModel):
    x: float
    y: float
    z: float


class ColorRGBA(BaseModel):
    r: float = Field(ge=0.0, le=1.0)
    g: float = Field(ge=0.0, le=1.0)
    b: float = Field(ge=0.0, le=1.0)
    a: float = Field(default=1.0, ge=0.0, le=1.0)


class MaterialSnapshot(BaseModel):
    name: str
    base_color: ColorRGBA | None
    roughness: float | None
    metallic: float | None
    use_nodes: bool


class ObjectSnapshot(BaseModel):
    name: str
    type: str
    primitive_hint: str | None
    location: Vector3
    rotation_euler: Vector3
    scale: Vector3
    dimensions: Vector3
    material_slots: list[str]
    parent: str | None
    collection_names: list[str]
    vertex_count: int | None
    polygon_count: int | None


class LightSnapshot(BaseModel):
    name: str
    type: str
    location: Vector3
    rotation_euler: Vector3
    energy: float | None
    color: ColorRGBA | None


class CameraSnapshot(BaseModel):
    name: str
    location: Vector3
    rotation_euler: Vector3
    lens: float | None
    sensor_width: float | None
    is_active: bool


class RenderSettingsSnapshot(BaseModel):
    engine: str
    resolution_x: int = Field(gt=0)
    resolution_y: int = Field(gt=0)
    frame_start: int
    frame_end: int
    frame_current: int


class SceneSnapshot(BaseModel):
    scene_name: str
    objects: list[ObjectSnapshot]
    materials: list[MaterialSnapshot]
    lights: list[LightSnapshot]
    cameras: list[CameraSnapshot]
    collections: list[str]
    render_settings: RenderSettingsSnapshot
    frame_current: int
    blender_version: str
    created_at: str


class BlenderCommandResult(BaseModel):
    ok: bool
    command: str
    output_files: list[str]
    stdout: str | None
    stderr: str | None
    error: str | None
    duration_sec: float | None

