from __future__ import annotations

import os
from pathlib import Path


def load_project_dotenv(start: Path | None = None) -> Path | None:
    """Load a project-root .env file into os.environ if it exists.

    Existing environment variables win over .env values. This keeps shell/CI
    overrides authoritative while making local CLI runs work without an
    explicit `source .env`.
    """
    if os.environ.get("BMA_DISABLE_DOTENV", "").lower() in {"1", "true", "yes"}:
        return None

    root = find_project_root(start or Path.cwd())
    dotenv_path = root / ".env"
    if not dotenv_path.is_file():
        return None
    load_dotenv_file(dotenv_path)
    return dotenv_path


def find_project_root(start: Path) -> Path:
    path = start.resolve()
    if path.is_file():
        path = path.parent
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "benchmark").is_dir():
            return candidate
    return path


def load_dotenv_file(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_dotenv_line(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        os.environ.setdefault(key, value)


def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].lstrip()
    if "=" not in stripped:
        return None

    key, value = stripped.split("=", 1)
    key = key.strip()
    if not key:
        return None

    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    else:
        value = value.split("#", 1)[0].strip()
    return key, value
