"""Windows-safe atomic publication of FVCOM restart files."""

import errno
import pytest

from fvtools.setup.initial._helpers import restart_io


def test_publish_retry_recovers_from_transient_access_denial(tmp_path, monkeypatch):
    destination = tmp_path / "restart.nc"
    staged = tmp_path / "staged.nc"
    destination.write_bytes(b"original")
    staged.write_bytes(b"updated")

    replace = restart_io.os.replace
    sleeps = []
    calls = []

    def busy_once(src, dst):
        calls.append((src, dst))
        if len(calls) == 1:
            raise PermissionError(errno.EACCES, "temporarily locked")
        return replace(src, dst)

    monkeypatch.setattr(restart_io.os, "replace", busy_once)
    monkeypatch.setattr(restart_io.time, "sleep", sleeps.append)
    restart_io.publish_staged_restart(staged, destination, attempts=3, delay=0.01)

    assert destination.read_bytes() == b"updated"
    assert not staged.exists()
    assert len(calls) == 2
    assert sleeps == [0.01]


def test_publish_retry_persistent_lock_preserves_original(tmp_path, monkeypatch):
    destination = tmp_path / "restart.nc"
    staged = tmp_path / "staged.nc"
    destination.write_bytes(b"original")
    staged.write_bytes(b"updated")
    sleeps = []

    def always_locked(src, dst):
        raise PermissionError(errno.EACCES, "file remains locked")

    monkeypatch.setattr(restart_io.os, "replace", always_locked)
    monkeypatch.setattr(restart_io.time, "sleep", sleeps.append)
    with pytest.raises(PermissionError, match="file remains locked"):
        restart_io.publish_staged_restart(staged, destination, attempts=3, delay=0.01)

    assert destination.read_bytes() == b"original"
    assert staged.read_bytes() == b"updated"
    assert sleeps == [0.01, 0.02]
