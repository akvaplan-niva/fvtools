"""Offline checks of the final TPXO copy-on-write publication."""
import errno

import pytest

from fvtools.setup.nesting import tpxo


def test_replace_recovers_from_transient_windows_lock(tmp_path, monkeypatch):
    dest = tmp_path / 'restart.nc'
    stage = tmp_path / '.restart.nc.partial'
    dest.write_bytes(b'old')
    stage.write_bytes(b'new')
    original_replace = tpxo.os.replace
    calls = []
    sleeps = []

    def busy_once(src, dst):
        calls.append((src, dst))
        if len(calls) == 1:
            raise PermissionError(errno.EACCES, 'file temporarily busy')
        return original_replace(src, dst)

    monkeypatch.setattr(tpxo.os, 'replace', busy_once)
    monkeypatch.setattr(tpxo.time, 'sleep', sleeps.append)
    tpxo._replace_staged_tpxo_file(stage, dest, attempts=3, delay=0.01)
    assert dest.read_bytes() == b'new'
    assert not stage.exists()
    assert len(calls) == 2
    assert sleeps == [0.01]


def test_replace_persistent_lock_does_not_alter_original(tmp_path, monkeypatch):
    dest = tmp_path / 'restart.nc'
    stage = tmp_path / '.restart.nc.partial'
    dest.write_bytes(b'old')
    stage.write_bytes(b'new')
    sleeps = []
    attempts = []

    def always_locked(src, dst):
        attempts.append((src, dst))
        raise PermissionError(errno.EACCES, 'file remains busy')

    monkeypatch.setattr(tpxo.os, 'replace', always_locked)
    monkeypatch.setattr(tpxo.time, 'sleep', sleeps.append)
    with pytest.raises(PermissionError, match='file remains busy'):
        tpxo._replace_staged_tpxo_file(stage, dest, attempts=3, delay=0.01)
    assert dest.read_bytes() == b'old'
    assert stage.read_bytes() == b'new'
    assert len(attempts) == 3
    assert sleeps == [0.01, 0.02]
