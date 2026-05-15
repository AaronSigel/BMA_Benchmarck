import pytest
from pydantic import ValidationError

from benchmark.tasks.models import (
    BenchmarkTask,
    ColorRGBA,
    DifficultyLevel,
    ExpectedExport,
    ExpectedLight,
    ExpectedMaterial,
    ExpectedObject,
    ExpectedScene,
    SuccessCriterion,
    TaskCategory,
    TaskMetadata,
    Vector3,
)


def make_valid_task() -> BenchmarkTask:
    return BenchmarkTask(
        id="geometry_001_basic_cube",
        title="Create a basic cube scene",
        category=TaskCategory.GEOMETRY,
        difficulty=DifficultyLevel.EASY,
        prompt="Create a cube at the origin with a red material.",
        tags=["geometry", "cube"],
        allowed_tools=["create_object", "assign_material", "inspect_scene"],
        expected_scene=ExpectedScene(
            objects=[
                ExpectedObject(
                    name="Cube",
                    type="mesh",
                    primitive="cube",
                    location=Vector3(x=0.0, y=0.0, z=0.0),
                    material="Red",
                )
            ],
            materials=[
                ExpectedMaterial(
                    name="Red",
                    base_color=ColorRGBA(r=1.0, g=0.0, b=0.0),
                    roughness=0.5,
                    metallic=0.0,
                )
            ],
        ),
        success_criteria=[
            SuccessCriterion(metric="object_existence", weight=0.5),
            SuccessCriterion(metric="geometry_accuracy", weight=0.5),
        ],
        metadata=TaskMetadata(author="benchmark", description="Basic geometry task"),
    )


def test_valid_benchmark_task_can_be_created() -> None:
    task = make_valid_task()

    assert task.id == "geometry_001_basic_cube"
    assert task.category is TaskCategory.GEOMETRY
    assert task.expected_scene.objects[0].location == Vector3(x=0.0, y=0.0, z=0.0)


@pytest.mark.parametrize("weight", [-0.1, 1.1])
def test_invalid_success_criterion_weight_raises_validation_error(weight: float) -> None:
    with pytest.raises(ValidationError):
        SuccessCriterion(metric="invalid_weight", weight=weight)


def test_success_criteria_weight_sum_must_not_exceed_one() -> None:
    with pytest.raises(ValidationError):
        BenchmarkTask(
            id="invalid-weight-sum",
            title="Invalid weight sum",
            category="geometry",
            difficulty="easy",
            prompt="Create a cube.",
            tags=[],
            allowed_tools=[],
            expected_scene=ExpectedScene(),
            success_criteria=[
                SuccessCriterion(metric="first", weight=0.6),
                SuccessCriterion(metric="second", weight=0.5),
            ],
        )


def test_invalid_color_channel_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        ColorRGBA(r=1.2, g=0.0, b=0.0)


def test_empty_prompt_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        BenchmarkTask(
            id="empty-prompt",
            title="Empty prompt",
            category="geometry",
            difficulty="easy",
            prompt="   ",
            tags=[],
            allowed_tools=[],
            expected_scene=ExpectedScene(),
            success_criteria=[],
        )


def test_empty_id_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        BenchmarkTask(
            id="",
            title="Empty id",
            category="geometry",
            difficulty="easy",
            prompt="Create a cube.",
            tags=[],
            allowed_tools=[],
            expected_scene=ExpectedScene(),
            success_criteria=[],
        )


@pytest.mark.parametrize(
    "model_factory",
    [
        lambda: ExpectedObject(type="mesh", tolerance=0.0),
        lambda: ExpectedMaterial(name="Mat", tolerance=-0.01),
    ],
)
def test_tolerance_must_be_positive(model_factory) -> None:
    with pytest.raises(ValidationError):
        model_factory()


def test_invalid_primitive_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        ExpectedObject(type="mesh", primitive="torus")


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_material_numeric_parameters_must_be_between_zero_and_one(value: float) -> None:
    with pytest.raises(ValidationError):
        ExpectedMaterial(name="Invalid", roughness=value)

    with pytest.raises(ValidationError):
        ExpectedMaterial(name="Invalid", metallic=value)


def test_invalid_light_type_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        ExpectedLight(type="HEMI")


def test_invalid_export_format_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        ExpectedExport(format="obj")
