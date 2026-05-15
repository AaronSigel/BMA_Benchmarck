import copy
import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "benchmark" / "schemas" / "task.schema.json"

VALID_TASK_YAML = """
id: geometry-cube-schema
title: Create a red cube
category: geometry
difficulty: easy
prompt: Create a red cube at the origin.
tags:
  - geometry
  - cube
allowed_tools:
  - mesh.create_cube
  - material.create
expected_scene:
  objects:
    - name: Cube
      type: mesh
      primitive: cube
      location:
        x: 0
        y: 0
        z: 0
      material: Red
      tolerance: 0.05
  materials:
    - name: Red
      base_color:
        r: 1
        g: 0
        b: 0
        a: 1
      roughness: 0.5
      metallic: 0
  lights: []
  cameras: []
  exports: []
success_criteria:
  - metric: object_exists
    weight: 0.5
    required: true
  - metric: material_matches
    weight: 0.5
metadata:
  author: benchmark
  version: "1.0"
  description: Schema validation fixture
"""


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


@pytest.fixture()
def valid_task_data() -> dict:
    data = yaml.safe_load(VALID_TASK_YAML)
    assert isinstance(data, dict)
    return data


def test_valid_yaml_task_passes_jsonschema_validation(
    validator: Draft202012Validator, valid_task_data: dict
) -> None:
    validator.validate(valid_task_data)


def test_task_without_id_fails_jsonschema_validation(
    validator: Draft202012Validator, valid_task_data: dict
) -> None:
    invalid_task = copy.deepcopy(valid_task_data)
    invalid_task.pop("id")

    with pytest.raises(ValidationError):
        validator.validate(invalid_task)


def test_unknown_category_fails_jsonschema_validation(
    validator: Draft202012Validator, valid_task_data: dict
) -> None:
    invalid_task = copy.deepcopy(valid_task_data)
    invalid_task["category"] = "animation"

    with pytest.raises(ValidationError):
        validator.validate(invalid_task)


def test_weight_greater_than_one_fails_jsonschema_validation(
    validator: Draft202012Validator, valid_task_data: dict
) -> None:
    invalid_task = copy.deepcopy(valid_task_data)
    invalid_task["success_criteria"][0]["weight"] = 1.1

    with pytest.raises(ValidationError):
        validator.validate(invalid_task)

