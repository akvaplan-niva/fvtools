"""OBC integration must not pass deprecated reflected Python lists into Numba."""
from types import SimpleNamespace
import warnings

import numpy as np
import pytest
from numba.core.errors import NumbaPendingDeprecationWarning

from fvtools.grid import tge


def test_only_zero_index_is_still_an_open_boundary():
    """An OBC list [0] is valid; truth-testing its entries loses node zero."""
    nodes = tge.get_obc(SimpleNamespace(obc_nodes=[0], filepath='fixture.2dm'))
    assert nodes.dtype == np.int64
    np.testing.assert_array_equal(nodes, [0])


@pytest.mark.parametrize('obc_nodes', [[], [0], [0, 1]])
def test_tge_uses_array_or_none_at_jit_boundary(monkeypatch, small_grid_file, obc_nodes):
    from fvtools.grid.fvcom_grd import FVCOM_grid

    grid = FVCOM_grid(str(small_grid_file))
    grid.obc_nodes = list(obc_nodes)  # The public API also accepts Python lists.
    compiled = tge.set_boundary
    seen = []

    def checked(isonb, isbce, nv, nbe, obc, nt):
        # Verify the actual Python -> Numba boundary, not just a helper's output.
        assert obc is None or (isinstance(obc, np.ndarray) and obc.dtype == np.int64)
        seen.append(obc)
        return compiled(isonb, isbce, nv, nbe, obc, nt)

    monkeypatch.setattr(tge, 'set_boundary', checked)
    with warnings.catch_warnings():
        warnings.simplefilter('error', NumbaPendingDeprecationWarning)
        result = tge.calculate_nbe_and_boundaries(grid)

    assert len(seen) == 1
    if obc_nodes:
        np.testing.assert_array_equal(seen[0], obc_nodes)
        np.testing.assert_array_equal(result.ISONB[obc_nodes], [2] * len(obc_nodes))
    else:
        assert seen[0] is None
    assert compiled.nopython_signatures
