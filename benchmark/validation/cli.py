import argparse
from pathlib import Path

from pydantic import ValidationError

from benchmark.blender.models import SceneSnapshot
from benchmark.tasks.loader import TaskLoadError, load_task
from benchmark.validation.models import SceneValidationResult
from benchmark.validation.scene_validator import SceneValidator


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate":
        return _validate(args.task, args.snapshot, args.artifacts_dir, args.output)
    if args.command == "summary":
        return _summary(args.result)

    parser.print_help()
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate benchmark scene snapshots.")
    subparsers = parser.add_subparsers(dest="command")

    validate_parser = subparsers.add_parser("validate", help="Validate one task and snapshot.")
    validate_parser.add_argument("--task", type=Path, required=True)
    validate_parser.add_argument("--snapshot", type=Path, required=True)
    validate_parser.add_argument("--artifacts-dir", type=Path, default=None)
    validate_parser.add_argument("--output", type=Path, required=True)

    summary_parser = subparsers.add_parser("summary", help="Print a validation result summary.")
    summary_parser.add_argument("--result", type=Path, required=True)

    return parser


def _validate(
    task_path: Path,
    snapshot_path: Path,
    artifacts_dir: Path | None,
    output_path: Path,
) -> int:
    try:
        task = load_task(task_path)
        snapshot = _load_snapshot(snapshot_path)
        result = SceneValidator().validate(task, snapshot, artifacts_dir=artifacts_dir)
        _write_result(result, output_path)
    except (TaskLoadError, ValidationError, OSError) as error:
        print(f"ERROR: {error}")
        return 1

    print(_format_summary(result))
    print(f"Saved validation result: {output_path}")
    return 0


def _summary(result_path: Path) -> int:
    try:
        result = _load_result(result_path)
    except (ValidationError, OSError) as error:
        print(f"ERROR: {error}")
        return 1

    print(_format_summary(result))
    return 0


def _load_snapshot(path: Path) -> SceneSnapshot:
    try:
        return SceneSnapshot.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise OSError(f"Failed to read scene snapshot {path}: {error}") from error
    except ValidationError as error:
        raise ValidationError.from_exception_data(
            title="SceneSnapshot",
            line_errors=error.errors(),
        ) from error


def _load_result(path: Path) -> SceneValidationResult:
    try:
        return SceneValidationResult.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise OSError(f"Failed to read validation result {path}: {error}") from error


def _write_result(result: SceneValidationResult, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    except OSError as error:
        raise OSError(f"Failed to write validation result {path}: {error}") from error


def _format_summary(result: SceneValidationResult) -> str:
    lines = [
        f"task_id: {result.task_id}",
        f"overall_status: {result.overall_status.value}",
        f"total_score: {result.total_score:.3f}",
        f"issues: {len(result.issues)}",
    ]
    for issue in result.issues:
        lines.append(f"- {issue.severity.value} {issue.code}: {issue.message}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
