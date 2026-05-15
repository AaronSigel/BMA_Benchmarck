"""Errors raised by scene validation components."""


class ValidationError(Exception):
    """Base error for scene validation failures."""


class ValidationInputError(ValidationError):
    """Raised when task or snapshot input cannot be loaded or parsed."""

