"""Pytest configuration."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--e2e",
        action="store_true",
        default=False,
        help="Run tests marked as e2e",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if config.getoption("--e2e"):
        return

    skip_e2e = pytest.mark.skip(reason="requires --e2e")
    for item in items:
        if item.get_closest_marker("e2e") is not None:
            item.add_marker(skip_e2e)
