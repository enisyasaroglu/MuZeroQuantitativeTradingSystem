import sys
import pytest


def run_regression_suite():
    """Executes the complete regression test suite and exits with the result code."""
    args = [
        "tests/",
        "-v",
        "--tb=short",
        "--disable-warnings",
    ]
    
    exit_code = pytest.main(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    run_regression_suite()