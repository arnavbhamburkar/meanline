import importlib

import pytest

import meanline

MODULES = [
    "thermo",
    "velocity_triangles",
    "annulus",
    "cascade",
    "losses",
    "blade_row",
    "stage",
    "compressor",
    "maps",
    "optimize",
    "units",
]


def test_version():
    assert isinstance(meanline.__version__, str)


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(f"meanline.{name}")


def test_registry_is_shared(units):
    from meanline.units import ureg

    assert units is ureg
    assert (1 * units.km).to("m").magnitude == pytest.approx(1000.0)
