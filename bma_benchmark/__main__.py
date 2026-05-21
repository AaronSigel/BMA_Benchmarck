from __future__ import annotations

import sys

from benchmark.runner.cli import main as runner_main


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "run-matrix":
        argv = ["matrix", *argv[1:]]
    return runner_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
