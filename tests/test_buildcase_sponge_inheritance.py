"""BuildCase retains FVCOM sponge functionality after mixin extraction."""

from fvtools.grid.fvcom._components.sponge import Sponge
from fvtools.setup.BuildCase import BuildCase


def test_buildcase_inherits_sponge():
    assert issubclass(BuildCase, Sponge)
    assert hasattr(BuildCase, "sponge_nodes")
    assert hasattr(BuildCase, "write_sponge")
