from pathlib import Path

import yaml

from benchmark.blender.models import (
    ObjectSnapshot,
    RenderSettingsSnapshot,
    SceneSnapshot,
    Vector3 as SnapshotVector3,
)
from benchmark.tasks.models import (
    BenchmarkTask,
    DifficultyLevel,
    ExpectedObject,
    ExpectedScene,
    SuccessCriterion,
    TaskCategory,
    Vector3,
)
from benchmark.validation import cli
from benchmark.validation.models import SceneValidationResult, ValidationStatus


def vector(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Vector3:
    return Vector3(x=x, y=y, z=z)


def snapshot_vector(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> SnapshotVector3:
    return SnapshotVector3(x=x, y=y, z=z)


def object_snapshot(name: str = "Cube") -> ObjectSnapshot:
    return ObjectSnapshot(
        name=name,
        type="MESH",
        primitive_hint="cube",
        location=snapshot_vector(),
        rotation_euler=snapshot_vector(),
        scale=snapshot_vector(1.0, 1.0, 1.0),
        dimensions=snapshot_vector(2.0, 2.0, 2.0),
        material_slots=[],
        parent=None,
        collection_names=["Collection"],
        vertex_count=None,
        polygon_count=None,
    )


def scene_snapshot(objects: list[ObjectSnapshot] | None = None) -> SceneSnapshot:
    return SceneSnapshot(
        scene_name="Scene",
        objects=objects or [],
        materials=[],
        lights=[],
        cameras=[],
        collections=["Collection"],
        render_settings=RenderSettingsSnapshot(
            engine="CYCLES",
            resolution_x=1920,
            resolution_y=1080,
            frame_start=1,
            frame_end=1,
            frame_current=1,
        ),
        frame_current=1,
        blender_version="4.0.0",
        created_at="2026-05-15T12:00:00Z",
    )


def benchmark_task() -> BenchmarkTask:
    return BenchmarkTask(
        id="geometry_001_basic_primitives",
        title="Create cube",
        category=TaskCategory.GEOMETRY,
        difficulty=DifficultyLevel.EASY,
        prompt="Create a cube at the origin.",
        tags=["geometry"],
        allowed_tools=[],
        expected_scene=ExpectedScene(
            objects=[
                ExpectedObject(
                    name="Cube",
                    type="MESH",
                    primitive="cube",
                    location=vector(0.0, 0.0, 0.0),
                )
            ]
        ),
        success_criteria=[
            SuccessCriterion(metric="object_existence", weight=0.4),
            SuccessCriterion(metric="geometry_accuracy", weight=0.3),
            SuccessCriterion(metric="object_placement", weight=0.3),
        ],
    )


def write_task(path: Path, task: BenchmarkTask) -> None:
    path.write_text(
        yaml.safe_dump(task.model_dump(mode="json", exclude_none=True), sort_keys=False),
        encoding="utf-8",
    )


def test_validation_cli_validate_creates_result_file(tmp_path: Path, capsys) -> None:
    task_path = tmp_path / "task.yaml"
    snapshot_path = tmp_path / "scene_snapshot.json"
    output_path = tmp_path / "validation_result.json"
    write_task(task_path, benchmark_task())
    snapshot_path.write_text(scene_snapshot([object_snapshot("Cube.001")]).model_dump_json(), encoding="utf-8")

    exit_code = cli.main(
        [
            "validate",
            "--task",
            str(task_path),
            "--snapshot",
            str(snapshot_path),
            "--artifacts-dir",
            str(tmp_path),
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    result = SceneValidationResult.model_validate_json(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert output_path.exists()
    assert result.task_id == "geometry_001_basic_primitives"
    assert result.overall_status is ValidationStatus.PASSED
    assert "task_id: geometry_001_basic_primitives" in captured.out
    assert "overall_status: passed" in captured.out


def test_validation_cli_summary_reads_result_file(tmp_path: Path, capsys) -> None:
    task_path = tmp_path / "task.yaml"
    snapshot_path = tmp_path / "scene_snapshot.json"
    output_path = tmp_path / "validation_result.json"
    write_task(task_path, benchmark_task())
    snapshot_path.write_text(scene_snapshot([]).model_dump_json(), encoding="utf-8")
    assert cli.main(
        [
            "validate",
            "--task",
            str(task_path),
            "--snapshot",
            str(snapshot_path),
            "--output",
            str(output_path),
        ]
    ) == 0
    capsys.readouterr()

    exit_code = cli.main(["summary", "--result", str(output_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "task_id: geometry_001_basic_primitives" in captured.out
    assert "overall_status: failed" in captured.out
    assert "object_missing" in captured.out


def test_validation_cli_reports_missing_file(capsys, tmp_path: Path) -> None:
    exit_code = cli.main(["summary", "--result", str(tmp_path / "missing.json")])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "ERROR:" in captured.out
    assert "Failed to read validation result" in captured.out


def test_validation_cli_validate_reports_missing_snapshot(capsys, tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path, benchmark_task())

    exit_code = cli.main(
        [
            "validate",
            "--task",
            str(task_path),
            "--snapshot",
            str(tmp_path / "missing_snapshot.json"),
            "--output",
            str(tmp_path / "validation_result.json"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "ERROR:" in captured.out
    assert "Failed to read scene snapshot" in captured.out
