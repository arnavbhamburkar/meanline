"""Shared pytest fixtures."""

import pytest

from meanline.units import ureg


@pytest.fixture(scope="session")
def units():
    """Return the shared pint unit registry."""
    return ureg
