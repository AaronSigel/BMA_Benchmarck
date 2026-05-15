import sys

import pytest
from pydantic import ValidationError

from benchmark.blender.config import (
    BlenderConfig,
    find_blender_executable,
    get_blender_config,
)
from benchmark.blender.errors import (
    BlenderError,
    BlenderNotFoundError,
    BlenderProcessError,
    BlenderTimeoutError,
)


def test_find_blender_executable_uses_env_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BMA_BLENDER_BIN", "/opt/blender/blender")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/blender")

    assert find_blender_executable() == "/opt/blender/blender"


def test_find_blender_executable_uses_path_when_env_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BMA_BLENDER_BIN", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/blender")

    assert find_blender_executable() == "/usr/bin/blender"


def test_find_blender_executable_returns_none_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BMA_BLENDER_BIN", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)

    assert find_blender_executable() is None


def test_get_blender_config_returns_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BMA_BLENDER_BIN", "/custom/blender")

    config = get_blender_config()

    assert config == BlenderConfig(blender_bin="/custom/blender")
    assert config.default_timeout_sec == 120
    assert config.headless is True


def test_get_blender_config_raises_when_blender_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BMA_BLENDER_BIN", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)

    with pytest.raises(BlenderNotFoundError):
        get_blender_config()


def test_blender_config_timeout_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        BlenderConfig(blender_bin="/usr/bin/blender", default_timeout_sec=0)


def test_blender_errors_share_base_class() -> None:
    assert issubclass(BlenderNotFoundError, BlenderError)
    assert issubclass(BlenderProcessError, BlenderError)
    assert issubclass(BlenderTimeoutError, BlenderError)


def test_blender_config_imports_without_bpy() -> None:
    assert "bpy" not in sys.modules

