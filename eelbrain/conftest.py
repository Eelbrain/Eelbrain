# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Pytest configuration

Skipping slow tests: https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
"""
import pytest


SKIP_MAYAVI = None


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")
    parser.addoption("--no-gui", action="store_true", default=False, help="skip tests that use GUI functionality")
    parser.addoption("--no-mayavi", action="store_true", default=False, help="skip tests that use mayavi")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "framework_build: requires framework build")
    config.addinivalue_line("markers", "mayavi: requires mayavi")
    config.addinivalue_line("markers", "file_test: requires repo test file(s)")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="slow test (set --runslow to run)")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    if config.getoption("--no-gui"):
        skip = pytest.mark.skip(reason="requires framework build")
        for item in items:
            if "framework_build" in item.keywords:
                item.add_marker(skip)
    if config.getoption("--no-mayavi"):
        global SKIP_MAYAVI
        SKIP_MAYAVI = True
        skip = pytest.mark.skip(reason="requires mayavi")
        for item in items:
            if "mayavi" in item.keywords:
                item.add_marker(skip)
