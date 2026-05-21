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
    """Blender scene render settings captured *before* render_scene runs.

    These reflect the scene state at snapshot time, not the final render
    output.  Use RenderResult for actual output dimensions and engine used.
    """

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
    mesh_object_count: int | None = None
    light_count: int | None = None
    camera_count: int | None = None
    all_object_count: int | None = None
    collections: list[str]
    render_settings: RenderSettingsSnapshot
    frame_current: int
    blender_version: str
    created_at: str


# ---------------------------------------------------------------------------
# Command result models — one per script function that smoke/run executes
# ---------------------------------------------------------------------------


class ResetResult(BaseModel):
    removed_objects: int
    remaining_objects: int
    scene_name: str


class FixtureResult(BaseModel):
    scene_name: str
    objects: list[str]
    materials: list[str]
    light: str | None
    camera: str | None
    floor_z: float
    clearance: float
    save_path: str | None
    saved: bool


class SaveResult(BaseModel):
    ok: bool
    path: str
    exists: bool
    file_size_bytes: int
    error: str | None


class RenderResult(BaseModel):
    """Actual output of render_scene — resolution and engine used for the render."""

    ok: bool
    output_path: str
    exists: bool
    file_size_bytes: int
    resolution_x: int
    resolution_y: int
    engine: str
    error: str | None


class ExportResult(BaseModel):
    ok: bool
    output_path: str
    format: str
    exists: bool
    file_size_bytes: int
    error: str | None


class SmokeResults(BaseModel):
    reset_scene: ResetResult
    create_fixture_scene: FixtureResult
    collect_snapshot: SceneSnapshot
    save_scene: SaveResult
    render_scene: RenderResult
    export_scene: ExportResult


class SmokeRunOutput(BaseModel):
    """Schema for smoke_output.json produced by smoke_scene.py."""

    ok: bool
    results: SmokeResults | None
    error: str | None


class BlenderCommandResult(BaseModel):
    ok: bool
    command: str
    output_files: list[str]
    stdout: str | None
    stderr: str | None
    error: str | None
    duration_sec: float | None
