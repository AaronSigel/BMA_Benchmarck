from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ToolCategory(str, Enum):
    INSPECTION = "inspection"
    OBJECT = "object"
    TRANSFORM = "transform"
    MATERIAL = "material"
    LIGHT = "light"
    CAMERA = "camera"
    EXPORT = "export"
    PYTHON = "python"
    ASSET = "asset"
    OTHER = "other"


class ToolParameter(BaseModel):
    name: str
    type: str
    description: str = ""
    required: bool = True
    default: Any = None


class ToolContract(BaseModel):
    """Describes a single MCP tool with benchmark-oriented metadata."""

    name: str
    description: str = ""
    category: ToolCategory = ToolCategory.OTHER
    parameters: list[ToolParameter] = Field(default_factory=list)
    returns: str = "str"
    profiles: list[str] = Field(default_factory=list)

    # Benchmark-oriented flags
    requires_python: bool = False           # tool executes arbitrary Python in Blender
    requires_external_network: bool = False  # tool calls external APIs (PolyHaven, Sketchfab…)
    benchmark_safe: bool = False            # tool is always safe to call in any benchmark run

    @property
    def required_params(self) -> list[ToolParameter]:
        return [p for p in self.parameters if p.required]

    @property
    def optional_params(self) -> list[ToolParameter]:
        return [p for p in self.parameters if not p.required]


# ---------------------------------------------------------------------------
# Canonical contracts for all upstream blender-mcp tools
# ---------------------------------------------------------------------------

def _p(name: str, typ: str, description: str = "", required: bool = True, default: Any = None) -> ToolParameter:
    return ToolParameter(name=name, type=typ, description=description, required=required, default=default)


TOOL_CONTRACTS: list[ToolContract] = [
    # --- Fork-only: benchmark meta ---
    ToolContract(
        name="get_bma_profile_info",
        description="Return the active BMA profile name and allowed tool list.",
        category=ToolCategory.INSPECTION,
        profiles=["minimal", "inspection_enabled", "no_python", "python_enabled", "full"],
        benchmark_safe=True,
    ),

    # --- Fork-only: bma_* benchmark-safe structured tools ---
    ToolContract(
        name="bma_get_scene_info",
        description="Return scene info as strict JSON without arbitrary Python. Safe in minimal/no_python.",
        category=ToolCategory.INSPECTION,
        profiles=["minimal", "inspection_enabled", "no_python", "python_enabled", "full"],
        benchmark_safe=True,
    ),
    ToolContract(
        name="bma_get_scene_snapshot",
        description="Return a structured scene snapshot as strict JSON without arbitrary Python.",
        category=ToolCategory.INSPECTION,
        profiles=["minimal", "inspection_enabled", "no_python", "python_enabled", "full"],
        benchmark_safe=True,
    ),
    ToolContract(
        name="bma_get_object_info",
        description="Return strict JSON details for one object without arbitrary Python.",
        category=ToolCategory.INSPECTION,
        parameters=[_p("object_name", "str", "Name of the object to inspect")],
        profiles=["minimal", "inspection_enabled", "no_python", "python_enabled", "full"],
        benchmark_safe=True,
    ),
    ToolContract(
        name="bma_create_object",
        description="Create a primitive object (MESH_CUBE|MESH_SPHERE|MESH_CYLINDER|MESH_PLANE|MESH_CONE|EMPTY).",
        category=ToolCategory.OBJECT,
        parameters=[
            _p("type", "str", "Object type: MESH_CUBE|MESH_SPHERE|MESH_CYLINDER|MESH_PLANE|MESH_CONE|EMPTY"),
            _p("name", "str", "Object name", required=False, default=""),
            _p("location", "list[float]", "XYZ location [x, y, z]", required=False, default=None),
            _p("rotation", "list[float]", "XYZ rotation radians [x, y, z]", required=False, default=None),
            _p("scale", "list[float]", "XYZ scale [x, y, z]", required=False, default=None),
            _p("dimensions", "list[float]", "Target object bounding-box dimensions [x, y, z]", required=False, default=None),
        ],
        profiles=["minimal", "inspection_enabled", "no_python", "python_enabled", "full"],
        benchmark_safe=True,
    ),
    ToolContract(
        name="bma_set_transform",
        description="Set location/rotation/scale/dimensions of a named object using strict JSON params.",
        category=ToolCategory.TRANSFORM,
        parameters=[
            _p("object_name", "str", "Target object name"),
            _p("location", "list[float]", "XYZ location", required=False, default=None),
            _p("rotation", "list[float]", "XYZ rotation radians", required=False, default=None),
            _p("scale", "list[float]", "XYZ scale", required=False, default=None),
            _p("dimensions", "list[float]", "Target object bounding-box dimensions", required=False, default=None),
        ],
        profiles=["minimal", "inspection_enabled", "no_python", "python_enabled", "full"],
        benchmark_safe=True,
    ),
    ToolContract(
        name="bma_set_material",
        description="Assign a simple BSDF material (RGBA color + metallic/roughness) to an object.",
        category=ToolCategory.MATERIAL,
        parameters=[
            _p("object_name", "str", "Target object name"),
            _p("color", "list[float]", "RGBA color [r, g, b, a]", required=False, default=None),
            _p("base_color", "list[float]", "RGBA color [r, g, b, a]", required=False, default=None),
            _p("material_name", "str", "Material datablock name to create/assign", required=False, default=None),
            _p("metallic", "float", "Metallic factor 0-1", required=False, default=0.0),
            _p("roughness", "float", "Roughness factor 0-1", required=False, default=0.5),
            _p("create_if_missing", "bool", "Create material when missing", required=False, default=True),
        ],
        profiles=["minimal", "inspection_enabled", "no_python", "python_enabled", "full"],
        benchmark_safe=True,
    ),
    ToolContract(
        name="bma_assign_material",
        description="Create/update and assign a simple BSDF material to an object.",
        category=ToolCategory.MATERIAL,
        parameters=[
            _p("object_name", "str", "Target object name"),
            _p("material_name", "str", "Material datablock name to create/assign", required=False, default=None),
            _p("base_color", "list[float]", "RGBA color [r, g, b, a]", required=False, default=None),
            _p("color", "list[float]", "RGBA color [r, g, b, a]", required=False, default=None),
            _p("metallic", "float", "Metallic factor 0-1", required=False, default=0.0),
            _p("roughness", "float", "Roughness factor 0-1", required=False, default=0.5),
            _p("create_if_missing", "bool", "Create material when missing", required=False, default=True),
        ],
        profiles=["minimal", "inspection_enabled", "no_python", "python_enabled", "full"],
        benchmark_safe=True,
    ),
    ToolContract(
        name="bma_create_light",
        description="Create a light (POINT|SUN|SPOT|AREA) with optional location and energy.",
        category=ToolCategory.LIGHT,
        parameters=[
            _p("type", "str", "Light type: POINT|SUN|SPOT|AREA"),
            _p("name", "str", "Light name", required=False, default=""),
            _p("location", "list[float]", "XYZ location", required=False, default=None),
            _p("rotation", "list[float]", "XYZ rotation radians", required=False, default=None),
            _p("energy", "float", "Light energy in Watts", required=False, default=1000.0),
            _p("color", "list[float]", "RGB color [r, g, b]", required=False, default=None),
        ],
        profiles=["minimal", "inspection_enabled", "no_python", "python_enabled", "full"],
        benchmark_safe=True,
    ),
    ToolContract(
        name="bma_create_camera",
        description="Create a camera object. If target is provided, orient camera with look-at and make it active by default.",
        category=ToolCategory.CAMERA,
        parameters=[
            _p("name", "str", "Camera name", required=False, default="Camera"),
            _p("location", "list[float]", "XYZ location", required=False, default=None),
            _p("rotation", "list[float]", "XYZ rotation radians", required=False, default=None),
            _p("lens", "float", "Focal length in mm", required=False, default=50.0),
            _p("focal_length", "float", "Focal length in mm", required=False, default=None),
            _p("target", "list[float]", "XYZ look-at target point", required=False, default=None),
            _p("make_active", "bool", "Set scene.camera to the created camera", required=False, default=True),
        ],
        profiles=["minimal", "inspection_enabled", "no_python", "python_enabled", "full"],
        benchmark_safe=True,
    ),
    ToolContract(
        name="bma_create_camera_look_at",
        description="Create a camera at a location and orient it toward a target point.",
        category=ToolCategory.CAMERA,
        parameters=[
            _p("name", "str", "Camera name"),
            _p("location", "list[float]", "XYZ camera location [x, y, z]"),
            _p("target", "list[float]", "XYZ target point [x, y, z]"),
            _p("focal_length", "float", "Focal length in mm", required=False, default=35.0),
            _p("make_active", "bool", "Set scene.camera to the created camera", required=False, default=True),
            _p("sensor_width", "float", "Camera sensor width in mm", required=False, default=None),
            _p("clip_start", "float", "Near clipping plane", required=False, default=None),
            _p("clip_end", "float", "Far clipping plane", required=False, default=None),
        ],
        profiles=["minimal", "inspection_enabled", "no_python", "python_enabled", "full"],
        benchmark_safe=True,
    ),
    ToolContract(
        name="bma_export_scene",
        description="Export the current scene. format: BLEND|GLB|GLTF|FBX. Use format=blend with filename=result.blend for .blend tasks.",
        category=ToolCategory.EXPORT,
        parameters=[
            _p("filepath", "str", "Output file path; optional because the benchmark runner injects it from format/filename", required=False, default=None),
            _p("format", "str", "Export format: BLEND|GLB|GLTF|FBX", required=False, default="GLB"),
            _p("filename", "str", "Expected benchmark filename, e.g. result.blend or exports/result.glb", required=False, default=None),
        ],
        profiles=["minimal", "inspection_enabled", "no_python", "python_enabled", "full"],
        benchmark_safe=True,
    ),

    # --- Upstream: core scene inspection ---
    ToolContract(
        name="get_scene_info",
        description="Return current Blender scene state (objects, materials, lights, cameras).",
        category=ToolCategory.INSPECTION,
        profiles=["minimal", "inspection_enabled", "no_python", "python_enabled", "full"],
        benchmark_safe=True,
    ),
    ToolContract(
        name="get_object_info",
        description="Return details about a named object (transforms, mesh stats, materials).",
        category=ToolCategory.OBJECT,
        parameters=[_p("object_name", "str", "Name of the object to inspect")],
        profiles=["minimal", "inspection_enabled", "no_python", "python_enabled", "full"],
        benchmark_safe=True,
    ),
    ToolContract(
        name="get_viewport_screenshot",
        description="Capture a screenshot of the current 3D viewport.",
        category=ToolCategory.INSPECTION,
        parameters=[_p("max_size", "int", "Maximum pixel size for the largest dimension", required=False, default=800)],
        returns="Image",
        profiles=["inspection_enabled", "no_python", "python_enabled", "full"],
    ),

    # --- Upstream: Python execution ---
    ToolContract(
        name="execute_blender_code",
        description="Execute arbitrary Python code inside Blender.",
        category=ToolCategory.PYTHON,
        parameters=[_p("code", "str", "Python code to execute in Blender's context")],
        requires_python=True,
        profiles=["python_enabled", "full"],
    ),

    # --- Upstream: Poly Haven (external asset) ---
    ToolContract(
        name="get_polyhaven_status",
        description="Check whether the PolyHaven integration is enabled in the Blender add-on.",
        category=ToolCategory.ASSET,
        requires_external_network=True,
        profiles=["full"],
    ),
    ToolContract(
        name="get_polyhaven_categories",
        description="List PolyHaven asset categories for a given asset type.",
        category=ToolCategory.ASSET,
        parameters=[_p("asset_type", "str", "hdris | textures | models | all", required=False, default="hdris")],
        requires_external_network=True,
        profiles=["full"],
    ),
    ToolContract(
        name="search_polyhaven_assets",
        description="Search PolyHaven assets by type and category.",
        category=ToolCategory.ASSET,
        parameters=[
            _p("asset_type", "str", "hdris | textures | models | all"),
            _p("categories", "str", "Comma-separated category filter", required=False, default=""),
        ],
        requires_external_network=True,
        profiles=["full"],
    ),
    ToolContract(
        name="download_polyhaven_asset",
        description="Download and import a PolyHaven asset into the current Blender scene.",
        category=ToolCategory.ASSET,
        parameters=[
            _p("asset_name", "str", "PolyHaven asset slug"),
            _p("asset_type", "str", "hdris | textures | models"),
            _p("resolution", "str", "e.g. 1k | 2k | 4k", required=False, default="1k"),
            _p("file_format", "str", "e.g. hdr | exr | png | blend", required=False, default=""),
        ],
        requires_external_network=True,
        profiles=["full"],
    ),
    ToolContract(
        name="set_texture",
        description="Apply a downloaded texture to a named object.",
        category=ToolCategory.MATERIAL,
        parameters=[
            _p("object_name", "str", "Target object name"),
            _p("texture_path", "str", "Path to texture file"),
        ],
        requires_external_network=False,
        profiles=["no_python", "python_enabled", "full"],
    ),

    # --- Upstream: Sketchfab (external asset) ---
    ToolContract(
        name="get_sketchfab_status",
        description="Check whether the Sketchfab integration is enabled.",
        category=ToolCategory.ASSET,
        requires_external_network=True,
        profiles=["full"],
    ),
    ToolContract(
        name="search_sketchfab_models",
        description="Search Sketchfab for 3D models.",
        category=ToolCategory.ASSET,
        parameters=[
            _p("query", "str", "Search keywords"),
            _p("max_results", "int", "Maximum number of results", required=False, default=24),
        ],
        requires_external_network=True,
        profiles=["full"],
    ),
    ToolContract(
        name="download_sketchfab_model",
        description="Download and import a Sketchfab model by UID.",
        category=ToolCategory.ASSET,
        parameters=[
            _p("uid", "str", "Sketchfab model UID"),
            _p("target_size", "float", "Target model size in Blender units", required=False, default=1.0),
        ],
        requires_external_network=True,
        profiles=["full"],
    ),

    # --- Upstream: Hyper3D Rodin (external asset) ---
    ToolContract(
        name="get_hyper3d_status",
        description="Check whether the Hyper3D Rodin integration is enabled.",
        category=ToolCategory.ASSET,
        requires_external_network=True,
        profiles=["full"],
    ),
    ToolContract(
        name="generate_hyper3d_model_via_text",
        description="Generate a 3D model from a text prompt using Hyper3D Rodin.",
        category=ToolCategory.ASSET,
        parameters=[
            _p("prompt", "str", "Text description of the desired 3D asset"),
            _p("tier", "str", "Quality tier (Sketch | Regular)", required=False, default="Regular"),
        ],
        requires_external_network=True,
        profiles=["full"],
    ),
    ToolContract(
        name="generate_hyper3d_model_via_images",
        description="Generate a 3D model from reference images using Hyper3D Rodin.",
        category=ToolCategory.ASSET,
        parameters=[
            _p("image_paths", "list[str]", "Local paths or URLs of reference images"),
            _p("tier", "str", "Quality tier", required=False, default="Regular"),
        ],
        requires_external_network=True,
        profiles=["full"],
    ),
    ToolContract(
        name="poll_rodin_job_status",
        description="Poll the status of a Hyper3D Rodin generation job.",
        category=ToolCategory.ASSET,
        parameters=[_p("job_id", "str", "Job identifier returned by generate_hyper3d_model_*")],
        requires_external_network=True,
        profiles=["full"],
    ),
    ToolContract(
        name="import_generated_asset",
        description="Import a completed Hyper3D Rodin asset into Blender.",
        category=ToolCategory.ASSET,
        parameters=[_p("job_id", "str", "Completed Rodin job identifier")],
        requires_external_network=True,
        profiles=["full"],
    ),

    # --- Upstream: Hunyuan3D (external asset) ---
    ToolContract(
        name="get_hunyuan3d_status",
        description="Check whether the Hunyuan3D integration is enabled.",
        category=ToolCategory.ASSET,
        requires_external_network=True,
        profiles=["full"],
    ),
    ToolContract(
        name="generate_hunyuan3d_model",
        description="Generate a 3D model from text or images using Hunyuan3D.",
        category=ToolCategory.ASSET,
        parameters=[
            _p("prompt", "str", "Text prompt or image description", required=False, default=""),
            _p("image_path", "str", "Path to reference image", required=False, default=""),
        ],
        requires_external_network=True,
        profiles=["full"],
    ),
    ToolContract(
        name="poll_hunyuan_job_status",
        description="Poll the status of a Hunyuan3D generation job.",
        category=ToolCategory.ASSET,
        parameters=[_p("job_id", "str", "Job identifier returned by generate_hunyuan3d_model")],
        requires_external_network=True,
        profiles=["full"],
    ),
    ToolContract(
        name="import_generated_asset_hunyuan",
        description="Import a completed Hunyuan3D asset into Blender.",
        category=ToolCategory.ASSET,
        parameters=[_p("job_id", "str", "Completed Hunyuan3D job identifier")],
        requires_external_network=True,
        profiles=["full"],
    ),
]

# Fast lookup by tool name
TOOL_CONTRACT_MAP: dict[str, ToolContract] = {c.name: c for c in TOOL_CONTRACTS}
