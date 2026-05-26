from __future__ import annotations

import unittest


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite, pattern: str | None) -> unittest.TestSuite:
    return loader.discover("tests", pattern="test_evidence_pack*.py")


if __name__ == "__main__":
    raise SystemExit(unittest.main(module="__main__", verbosity=2))
