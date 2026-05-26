from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class RunArtifactRef(BaseModel):
    run_id: str
    run_dir: Path
    task_id: str | None = None
    category: str | None = None
    model: str | None = None
    strategy: str | None = None
    mcp_profile: str | None = None
    pass_type: str | None = None
    scene_score: float | None = None
    strict_success: bool | None = None
    snapshot_path: Path | None = None
    validation_result_path: Path | None = None
    render_path: Path | None = None
    viewport_path: Path | None = None
    blend_path: Path | None = None
    glb_path: Path | None = None
    artifact_manifest: dict[str, Any] = Field(default_factory=dict)
    run_result: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    validation_result: dict[str, Any] = Field(default_factory=dict)
    render_missing_reason: str | None = None

    model_config = {"arbitrary_types_allowed": True}


class SceneExample(BaseModel):
    run_id: str
    task_id: str
    category: str | None = None
    model: str | None = None
    strategy: str | None = None
    mcp_profile: str | None = None
    pass_type: str
    scene_score: float | None = None
    strict_success: bool | None = None
    run_dir: Path
    snapshot_path: Path | None = None
    validation_result_path: Path | None = None
    render_path: Path | None = None
    viewport_path: Path | None = None
    blend_path: Path | None = None
    glb_path: Path | None = None
    top_issues: list[str] = Field(default_factory=list)
    check_table_excerpt: list[dict[str, Any]] = Field(default_factory=list)
    selection_reason: str
    render_missing_reason: str | None = None
    card_path: Path | None = None
    thumbnail_path: Path | None = None

    model_config = {"arbitrary_types_allowed": True}


class SceneExampleSelectionConfig(BaseModel):
    examples_per_status: int = Field(default=4, ge=1)
    min_clean_pass_score: float = Field(default=0.95, ge=0.0, le=1.0)
    failed_prefer_strategies: list[str] = Field(default_factory=lambda: ["direct_tool_calling"])
    soft_pass_prefer_tasks: list[str] = Field(default_factory=lambda: ["export_002_glb_file"])
    soft_pass_prefer_strategies: list[str] = Field(
        default_factory=lambda: ["react", "plan_execute_react_repair"]
    )
    diversify_issue_codes: bool = True
    priority_tasks: list[str] = Field(default_factory=lambda: [
        "geometry_002_positions",
        "materials_004_multiple_objects",
        "lighting_003_three_point_lighting",
        "camera_003_composition_view",
        "export_002_glb_file",
    ])
    pass_types: list[str] = Field(default_factory=lambda: [
        "clean_pass",
        "soft_pass",
        "failed_validation",
    ])


class SceneExampleBundle(BaseModel):
    examples: list[SceneExample] = Field(default_factory=list)
    config: SceneExampleSelectionConfig
    warnings: list[str] = Field(default_factory=list)
