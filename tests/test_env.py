import os
from pathlib import Path

from benchmark.env import find_project_root, load_project_dotenv


def test_load_project_dotenv_reads_project_root_file(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    nested = project / "configs" / "matrices"
    nested.mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
    (project / "benchmark").mkdir()
    (project / ".env").write_text(
        "\n".join(
            [
                "# local secrets",
                "OPENROUTER_API_KEY=sk-local",
                "QUOTED_VALUE=\"hello world\"",
                "export EXPORTED_VALUE=enabled",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("QUOTED_VALUE", raising=False)
    monkeypatch.delenv("EXPORTED_VALUE", raising=False)

    loaded = load_project_dotenv(nested)

    assert loaded == project / ".env"
    assert os.environ["OPENROUTER_API_KEY"] == "sk-local"
    assert os.environ["QUOTED_VALUE"] == "hello world"
    assert os.environ["EXPORTED_VALUE"] == "enabled"


def test_load_project_dotenv_does_not_override_existing_values(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
    (project / "benchmark").mkdir()
    (project / ".env").write_text("OPENROUTER_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-shell")

    load_project_dotenv(project)

    assert os.environ["OPENROUTER_API_KEY"] == "from-shell"


def test_load_project_dotenv_can_be_disabled(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
    (project / "benchmark").mkdir()
    (project / ".env").write_text("OPENROUTER_API_KEY=sk-local\n", encoding="utf-8")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("BMA_DISABLE_DOTENV", "1")

    loaded = load_project_dotenv(project)

    assert loaded is None
    assert "OPENROUTER_API_KEY" not in os.environ


def test_find_project_root_falls_back_to_start_path(tmp_path: Path) -> None:
    assert find_project_root(tmp_path) == tmp_path.resolve()
