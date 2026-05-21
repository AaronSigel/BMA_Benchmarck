from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class TaskCategory(str, Enum):
    GEOMETRY = "geometry"
    MATERIALS = "materials"
    LIGHTING = "lighting"
    CAMERA = "camera"
    EXPORT = "export"


class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Vector3(BaseModel):
    x: float
    y: float
    z: float


class ColorRGBA(BaseModel):
    r: float = Field(ge=0.0, le=1.0)
    g: float = Field(ge=0.0, le=1.0)
    b: float = Field(ge=0.0, le=1.0)
    a: float = Field(default=1.0, ge=0.0, le=1.0)


class ExpectedObject(BaseModel):
    name: str | None = None
    type: str
    primitive: Literal["cube", "sphere", "cylinder", "cone", "plane"] | None = None
    location: Vector3 | None = None
    rotation: Vector3 | None = None
    scale: Vector3 | None = None
    dimensions: Vector3 | None = None
    material: str | None = None
    tolerance: float = Field(default=0.05, gt=0.0)


class ExpectedMaterial(BaseModel):
    name: str
    base_color: ColorRGBA | None = None
    roughness: float | None = Field(default=None, ge=0.0, le=1.0)
    metallic: float | None = Field(default=None, ge=0.0, le=1.0)
    tolerance: float = Field(default=0.05, gt=0.0)


class ExpectedLight(BaseModel):
    name: str | None = None
    type: Literal["AREA", "SUN", "POINT", "SPOT"]
    location: Vector3 | None = None
    rotation: Vector3 | None = None
    target: Vector3 | None = None
    direction_tolerance_deg: float = Field(default=10.0, gt=0.0)
    energy: float | None = None
    tolerance: float = Field(default=0.05, gt=0.0)


class ExpectedCamera(BaseModel):
    name: str | None = None
    location: Vector3 | None = None
    rotation: Vector3 | None = None
    focal_length: float | None = None
    target: Vector3 | str | None = None
    require_active: bool | None = None
    direction_tolerance_deg: float = Field(default=5.0, gt=0.0)
    tolerance: float = Field(default=0.05, gt=0.0)


class ExpectedExport(BaseModel):
    format: Literal["blend", "glb", "fbx"]
    filename: str | None = None
    must_exist: bool = True


class ExpectedScene(BaseModel):
    objects: list[ExpectedObject] = Field(default_factory=list)
    materials: list[ExpectedMaterial] = Field(default_factory=list)
    lights: list[ExpectedLight] = Field(default_factory=list)
    cameras: list[ExpectedCamera] = Field(default_factory=list)
    exports: list[ExpectedExport] = Field(default_factory=list)


class SuccessCriterion(BaseModel):
    metric: str
    weight: float = Field(ge=0.0, le=1.0)
    required: bool = True


class TaskMetadata(BaseModel):
    author: str | None = None
    version: str = "1.0"
    description: str | None = None


class BenchmarkTask(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    id: str
    title: str
    category: TaskCategory
    difficulty: DifficultyLevel
    prompt: str
    tags: list[str]
    allowed_tools: list[str]
    expected_scene: ExpectedScene
    success_criteria: list[SuccessCriterion]
    metadata: TaskMetadata | None = None

    @field_validator("id", "prompt")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be empty")
        return value

    @model_validator(mode="after")
    def validate_success_criteria_weight_sum(self) -> "BenchmarkTask":
        total_weight = sum(criterion.weight for criterion in self.success_criteria)
        if total_weight > 1.0:
            raise ValueError("success_criteria weight sum must be no greater than 1.0")
        return self
