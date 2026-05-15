class RunnerError(Exception):
    """Base exception for experiment runner errors."""


class RunnerConfigError(RunnerError):
    """Raised when a runner configuration cannot be loaded or validated."""


class ExecutionBackendError(RunnerError):
    """Raised when an execution backend cannot produce a scene snapshot."""


class ExperimentRunError(RunnerError):
    """Raised when a benchmark run fails before producing a run result."""
