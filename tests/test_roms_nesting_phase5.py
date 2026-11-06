"""Numerical regression and mask tests for bounded-memory ROMS interpolation.

The reference methods below reproduce the phase-4 NumPy expressions explicitly.
No THREDDS endpoint, NetCDF file or network access is used by this suite.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from fvtools.interpolators.roms_interpolators import LinearInterpolation


def _same(actual, expected):
    assert type(actual) is type(expected)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    mask1 = np.ma.getmaskarray(actual)
    mask2 = np.ma.getmaskarray(expected)
    np.testing.assert_array_equal(mask1, mask2)
    np.testing.assert_array_equal(np.ma.getdata(actual)[~mask1],
                                  np.ma.getdata(expected)[~mask2])


def _old_horizontal(n4, timestep, variables):
    # Frozen expressions from roms_interpolators.py before phase 5.
    zlen = timestep.salt.shape[-1]
    if 'u' in variables:
        timestep.u = np.sum(timestep.u[n4.u_index, :] *
                            np.repeat(n4.u_coef[:, :, None], zlen, axis=2), axis=1)
    if 'v' in variables:
        timestep.v = np.sum(timestep.v[n4.v_index, :] *
                            np.repeat(n4.v_coef[:, :, None], zlen, axis=2), axis=1)
    if 'ua' in variables:
        timestep.ua = np.sum(timestep.ua[n4.u_index] * n4.u_coef, axis=1)
    if 'va' in variables:
        timestep.va = np.sum(timestep.va[n4.v_index] * n4.v_coef, axis=1)
    if 'zeta' in variables:
        timestep.zeta = np.sum(timestep.zeta[n4.rho_index] * n4.rho_coef, axis=1)
    if 'temp' in variables:
        timestep.temp = np.sum(timestep.temp[n4.rho_index, :] *
                               np.repeat(n4.rho_coef[:, :, None], zlen, axis=2), axis=1)
    if 'salt' in variables:
        timestep.salt = np.sum(timestep.salt[n4.rho_index, :] *
                               np.repeat(n4.rho_coef[:, :, None], zlen, axis=2), axis=1)
    return timestep


def _old_vertical(n4, timestep, variables):
    for variable, coord in [('salt', 'rho'), ('temp', 'rho'), ('u', 'u'), ('v', 'v')]:
        if variable not in variables:
            continue
        array = np.flip(getattr(timestep, variable), axis=1).T
        ind1 = getattr(n4, f'vi_ind1_{coord}')
        ind2 = getattr(n4, f'vi_ind2_{coord}')
        w1 = getattr(n4, f'vi_weigths1_{coord}')
        w2 = getattr(n4, f'vi_weigths2_{coord}')
        setattr(timestep, variable,
                array[ind1, range(array.shape[1])] * w1 +
                array[ind2, range(array.shape[1])] * w2)
    return timestep


def _fixture(npoints, dtype, masked):
    rng = np.random.default_rng(1328)
    nsource, nlevels, nfvlayers = 210, 9, 6
    fields = {}
    for coord in ('rho', 'u', 'v'):
        fields[f'{coord}_index'] = rng.integers(0, nsource, (npoints, 4))
        fields[f'{coord}_coef'] = rng.uniform(-.1, 1.0, (npoints, 4))
        fields[f'vi_ind1_{coord}'] = rng.integers(0, nlevels, (nfvlayers, npoints))
        fields[f'vi_ind2_{coord}'] = rng.integers(0, nlevels, (nfvlayers, npoints))
        fields[f'vi_weigths1_{coord}'] = rng.uniform(0, 1, (nfvlayers, npoints))
        fields[f'vi_weigths2_{coord}'] = 1.0-fields[f'vi_weigths1_{coord}']
    n4 = SimpleNamespace(**fields)
    arrays = {}
    for name in ('salt', 'temp', 'u', 'v', 'ua', 'va', 'zeta'):
        shape = (nsource, nlevels) if name in ('salt', 'temp', 'u', 'v') else (nsource,)
        arr = rng.normal(size=shape).astype(dtype)
        if arr.size:
            arr.flat[10] = np.nan
        if masked:
            mask = np.zeros(shape, dtype=bool)
            mask.flat[18] = True
            mask.flat[30] = True
            arr = np.ma.array(arr, mask=mask)
        arrays[name] = arr
    return n4, SimpleNamespace(**arrays)


def _copy_timestep(timestep):
    return SimpleNamespace(**{n: getattr(timestep, n).copy() for n in
                              ('salt', 'temp', 'u', 'v', 'ua', 'va', 'zeta')})


@pytest.mark.parametrize('npoints', [0, 7, 4096, 4103, 8203])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('masked', [False, True])
def test_original_numerical_values_shapes_and_masks(npoints, dtype, masked):
    n4, original = _fixture(npoints, dtype, masked)
    old = _copy_timestep(original)
    new = _copy_timestep(original)
    variables = ['salt', 'temp', 'zeta', 'u', 'v', 'ua', 'va']
    interpolation = LinearInterpolation()
    interpolation.N4 = n4

    _old_horizontal(n4, old, variables)
    interpolation.horizontal_interpolation(new, variables)
    for name in variables:
        _same(getattr(new, name), getattr(old, name))

    _old_vertical(n4, old, variables)
    interpolation.vertical_interpolation(new, variables)
    for name in variables:
        _same(getattr(new, name), getattr(old, name))


def test_optional_variables_do_not_change_untouched_fields():
    n4, original = _fixture(4103, np.float64, True)
    old, new = _copy_timestep(original), _copy_timestep(original)
    interpolation = LinearInterpolation()
    interpolation.N4 = n4
    variables = ['zeta', 'temp', 'v']
    _old_horizontal(n4, old, variables)
    interpolation.horizontal_interpolation(new, variables)
    for name in ('salt', 'temp', 'u', 'v', 'ua', 'va', 'zeta'):
        _same(getattr(new, name), getattr(old, name))
    _old_vertical(n4, old, variables)
    interpolation.vertical_interpolation(new, variables)
    for name in ('salt', 'temp', 'u', 'v', 'ua', 'va', 'zeta'):
        _same(getattr(new, name), getattr(old, name))
