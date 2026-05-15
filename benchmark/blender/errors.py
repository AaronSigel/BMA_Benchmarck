class BlenderError(Exception):
    """Base error for Blender automation failures."""


class BlenderNotFoundError(BlenderError):
    """Raised when the Blender executable cannot be found."""


class BlenderProcessError(BlenderError):
    """Raised when a Blender subprocess exits unsuccessfully."""


class BlenderTimeoutError(BlenderError):
    """Raised when a Blender subprocess exceeds its timeout."""

