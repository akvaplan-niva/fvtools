"""CPU selection in Slurm/Jupyter must be explicit, bounded and testable."""
import os
from types import SimpleNamespace
import numpy as np
import pytest
from fvtools.prepro._buildcase import tiled_bathymetry as bath


def _machine(monkeypatch, logical=256, affinity=5, slurm='1'):
    monkeypatch.setattr(bath.os, 'cpu_count', lambda: logical)
    if hasattr(bath.os, 'sched_getaffinity'):
        monkeypatch.setattr(bath.os, 'sched_getaffinity', lambda pid: set(range(affinity)))
    else:
        monkeypatch.setattr(bath.os, 'sched_getaffinity', lambda pid: set(range(affinity)), raising=False)
    if slurm is None:
        monkeypatch.delenv('SLURM_CPUS_PER_TASK', raising=False)
    else:
        monkeypatch.setenv('SLURM_CPUS_PER_TASK', slurm)
    monkeypatch.setenv('FVTOOLS_BATHY_WORKERS', '5')
    monkeypatch.delenv('FVTOOLS_BATHY_CPU_POLICY', raising=False)


def test_default_respects_slurm_even_if_affinity_is_larger(monkeypatch):
    _machine(monkeypatch)
    assert bath._cpu_slots() == 1
    assert bath._num_workers(bath.TilingOptions(), 256) == 1
    details = bath._cpu_resources()
    assert details['affinity'] == 5
    assert details['slurm'] == 1
    assert details['policy'] == 'slurm'


def test_explicit_affinity_opt_in_allows_five_but_not_six(monkeypatch):
    _machine(monkeypatch)
    monkeypatch.setenv('FVTOOLS_BATHY_CPU_POLICY', 'affinity')
    assert bath._cpu_slots() == 5
    assert bath._num_workers(bath.TilingOptions(), 256) == 5
    monkeypatch.setenv('FVTOOLS_BATHY_WORKERS', '50')
    assert bath._num_workers(bath.TilingOptions(), 256) == 5
    assert bath._num_workers(bath.TilingOptions(workers=3), 256) == 3
    assert bath._num_workers(bath.TilingOptions(workers=50), 2) == 2


def test_default_handles_absent_or_malformed_slurm(monkeypatch):
    _machine(monkeypatch, slurm=None)
    assert bath._cpu_slots() == 5
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', 'not-a-number')
    assert bath._cpu_slots() == 5
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '0')
    assert bath._cpu_slots() == 5


def test_affinity_never_exceeds_host_cpu_count(monkeypatch):
    _machine(monkeypatch, logical=4, affinity=5, slurm='12')
    monkeypatch.setenv('FVTOOLS_BATHY_CPU_POLICY', 'affinity')
    assert bath._cpu_slots() == 4
    assert bath._num_workers(bath.TilingOptions(), 99) == 4


def test_invalid_policy_does_not_silently_oversubscribe(monkeypatch):
    _machine(monkeypatch)
    monkeypatch.setenv('FVTOOLS_BATHY_CPU_POLICY', 'ignore-all')
    with pytest.raises(ValueError, match='FVTOOLS_BATHY_CPU_POLICY'):
        bath._num_workers(bath.TilingOptions(), 99)


def test_affinity_api_unavailable(monkeypatch):
    _machine(monkeypatch, logical=8, affinity=5, slurm=None)
    def unsupported(_):
        raise OSError('not supported')
    monkeypatch.setattr(bath.os, 'sched_getaffinity', unsupported)
    monkeypatch.setenv('FVTOOLS_BATHY_CPU_POLICY', 'affinity')
    assert bath._cpu_slots() == 8


def test_cpu_diagnostics_reports_mismatch(monkeypatch, tmp_path, capsys):
    # Verify the message is emitted in the real entry point with a cheap data set.
    _machine(monkeypatch)
    pts = np.array([[0.,0.,4.], [5.,0.,5.], [0.,5.,6.], [5.,5.,7.]])
    path=tmp_path/'tiny.npy'
    np.save(path,pts)
    xy=np.array([[1.,1.], [2.,2.]])
    bath.interpolate_unstructured_npy(
        path, xy,
        options=bath.TilingOptions(nodes_per_tile=1, bins_per_axis=2))
    output=capsys.readouterr().out
    assert 'CPU selection: affinity=5, SLURM_CPUS_PER_TASK=1, policy=slurm, limit=1' in output
    assert 'Use FVTOOLS_BATHY_CPU_POLICY=affinity only after' in output


def test_opt_in_uses_thread_pool_for_fast_kdtree(monkeypatch, tmp_path, capsys):
    _machine(monkeypatch, logical=8, affinity=2, slurm='1')
    monkeypatch.setenv('FVTOOLS_BATHY_CPU_POLICY', 'affinity')
    monkeypatch.setenv('FVTOOLS_BATHY_WORKERS','2')
    monkeypatch.setenv('FVTOOLS_BATHY_FAST_INDEX_MIN_POINTS','10')
    monkeypatch.setenv('FVTOOLS_BATHY_FAST_INDEX_MB','64')
    from scipy.interpolate import LinearNDInterpolator
    rng=np.random.default_rng(915)
    pts = rng.uniform(-20, 20, size=(400,2))
    src=np.column_stack((pts, pts[:,0]**2/100 + pts[:,1]/15))
    path=tmp_path/'bathy.npy'; np.save(path,src)
    xy=rng.uniform(-10,10,size=(28,2))
    got=bath.interpolate_unstructured_npy(
        path, xy,
        options=bath.TilingOptions(nodes_per_tile=7, bins_per_axis=5))
    expected=LinearNDInterpolator(pts,src[:,2])(xy)
    np.testing.assert_allclose(got,expected,atol=1e-9,rtol=1e-9)
    out=capsys.readouterr().out
    assert 'Fast global kNN / local Delaunay with 2 CPU worker(s)' in out
