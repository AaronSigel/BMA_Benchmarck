from pathlib import Path

from bma_benchmark.reporting.scene_examples.models import RunArtifactRef, SceneExampleSelectionConfig
from bma_benchmark.reporting.scene_examples.selection import select_scene_examples


def _run(run_id: str, task_id: str, pass_type: str, score: float) -> RunArtifactRef:
    return RunArtifactRef(
        run_id=run_id,
        run_dir=Path("/tmp") / run_id,
        task_id=task_id,
        category=task_id.split("_", 1)[0],
        pass_type=pass_type,
        scene_score=score,
        validation_result_path=Path("/tmp") / run_id / "validation_result.json",
    )


def test_selection_includes_pass_types_priority_and_is_deterministic() -> None:
    runs = [
        _run("b", "materials_004_multiple_objects", "clean_pass", 0.9),
        _run("a", "camera_003_composition_view", "clean_pass", 0.8),
        _run("c", "geometry_002_positions", "soft_pass", 0.7),
        _run("d", "export_002_glb_file", "failed_validation", 0.2),
    ]
    config = SceneExampleSelectionConfig(examples_per_status=2)

    first = select_scene_examples(runs, config)
    second = select_scene_examples(list(reversed(runs)), config)

    assert {example.pass_type for example in first.examples} == {"clean_pass", "soft_pass", "failed_validation"}
    assert first.examples[0].task_id == "materials_004_multiple_objects"
    assert [e.run_id for e in first.examples] == [e.run_id for e in second.examples]
