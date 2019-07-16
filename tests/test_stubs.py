# From numpy-stubs
import os

import pytest
from mypy import api

TESTS_DIR = os.path.dirname(__file__)
PASS_DIR = os.path.join(TESTS_DIR, "pass")
FAIL_DIR = os.path.join(TESTS_DIR, "fail")
REVEAL_DIR = os.path.join(TESTS_DIR, "reveal")


def get_test_cases(directory: str):
    for root, __, files in os.walk(directory):
        for fname in files:
            if os.path.splitext(fname)[-1] == ".py":
                fullpath = os.path.join(root, fname)
                # Use relative path for nice py.test name
                relpath = os.path.relpath(fullpath, start=directory)
                skip_py2 = fname.endswith("_py3.py")

                for py_version_number in (2, 3):
                    if py_version_number == 2 and skip_py2:
                        continue
                    py2_arg = ['--py2'] if py_version_number == 2 else []

                    yield pytest.param(
                        fullpath,
                        py2_arg,
                        # Manually specify a name for the test
                        id="{} - python{}".format(relpath, py_version_number),
                    )


@pytest.mark.parametrize("path,py2_arg", get_test_cases(PASS_DIR))
def test_success(path, py2_arg):
    stdout, stderr, exitcode = api.run([path] + py2_arg)
    assert stdout == ''
    assert exitcode == 0


@pytest.mark.parametrize("path,py2_arg", get_test_cases(FAIL_DIR))
def test_fail(path, py2_arg):
    stdout, stderr, exitcode = api.run([path] + py2_arg)

    assert exitcode != 0

    with open(path) as fin:
        lines = fin.readlines()

    errors = {}
    for error_line in stdout.split("\n"):
        error_line = error_line.strip()
        if not error_line:
            continue

        lineno = int(error_line.split(":")[1])
        errors[lineno] = errors.get(lineno, "") + error_line

    for i, line in enumerate(lines):
        lineno = i + 1
        if " E:" not in line and lineno not in errors:
            continue

        target_line = lines[lineno - 1]
        if "# E:" in target_line:
            marker = target_line.split("# E:")[-1].strip()
            assert lineno in errors, f'Extra error "{marker}"'
            assert marker in errors[lineno]
        else:
            pytest.fail(f'Error {repr(errors[lineno])} not found')


@pytest.mark.parametrize("path,py2_arg", get_test_cases(REVEAL_DIR))
def test_reveal(path, py2_arg):
    stdout, stderr, exitcode = api.run([path] + py2_arg)

    with open(path) as fin:
        lines = fin.readlines()

    for error_line in stdout.split("\n"):
        error_line = error_line.strip()
        if not error_line:
            continue

        lineno = int(error_line.split(":")[1])
        assert "Revealed type is" in error_line
        marker = lines[lineno - 1].split("# E:")[-1].strip()
        assert marker in error_line
