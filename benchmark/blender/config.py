import os
import shutil

from pydantic import BaseModel, Field

from benchmark.blender.errors import BlenderNotFoundError


class BlenderConfig(BaseModel):
    blender_bin: str
    default_timeout_sec: int = Field(default=120, gt=0)
    headless: bool = True


def find_blender_executable() -> str | None:
    env_blender_bin = os.environ.get("BMA_BLENDER_BIN")
    if env_blender_bin:
        return env_blender_bin

    return shutil.which("blender")


def get_blender_config() -> BlenderConfig:
    blender_bin = find_blender_executable()
    if blender_bin is None:
        raise BlenderNotFoundError(
            "Blender executable not found. Set BMA_BLENDER_BIN or add blender to PATH."
        )

    return BlenderConfig(blender_bin=blender_bin)

