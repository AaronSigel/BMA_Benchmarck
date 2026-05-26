from pathlib import Path

from bma_benchmark.validation_audit.collector import collect_validator_audit
from bma_benchmark.validation_audit.writers import write_validator_audit


def test_validation_audit_creates_rows_for_all_tasks(tmp_path: Path) -> None:
    report = collect_validator_audit(Path("tasks"))
    task_ids = {path.stem for path in Path("tasks").glob("*/*.yaml")}

    assert task_ids
    assert task_ids <= {row.task_id for row in report.rows}
    assert all(row.validator_name and row.checked_field for row in report.rows)
    assert any(row.weight is not None for row in report.rows)
    assert any(row.tolerance is not None for row in report.rows)

    write_validator_audit(report, tmp_path)
    assert (tmp_path / "validator_inventory.csv").is_file()
    assert (tmp_path / "validator_inventory.json").is_file()
    assert (tmp_path / "validator_audit.md").is_file()
    assert (tmp_path / "validator_limitations.md").is_file()
