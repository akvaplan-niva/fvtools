#!/usr/bin/env python3
"""Fix Windows memmap test fixture file collision, without changing production code.

Run from the fvtools repository root:
    uv run python fix_buildcase_dense_test_windows.py
"""
from pathlib import Path
import hashlib
import os
import tempfile

TARGET = Path('tests/test_buildcase_dense_fallback.py')
OLD_SHA256 = "a77005ab0bfc79fd4b056f8ba0b1298499e329e978d3bcb0d8d8c494172eb648"
NEW_SHA256 = "bd2bc82410f99550182e6b72f88337b3887b90159a003ff66306b5df34040d23"
OLD = "    path = tmp_path/'dense.npy'\n"
NEW = "    # Windows does not allow overwriting a .npy file while a read-only\n    # NumPy memmap still holds it open (possibly via worker processes).\n    # Each invocation must therefore create a distinct source file.\n    path = tmp_path / f'dense_workers_{workers}_cap_{max_points}_seed_{seed}.npy'\n"


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    project = Path.cwd()
    if not (project / 'pyproject.toml').is_file():
        raise SystemExit('Kjør skriptet fra fvtools-prosjektroten (pyproject.toml mangler).')
    path = project / TARGET
    if not path.is_file():
        raise SystemExit(f'Finner ikke {TARGET}. Installer dense_halo_fix først.')
    original = path.read_bytes()
    digest = _digest(original)
    if digest == NEW_SHA256:
        print('Testen er allerede rettet. Ingen filer endret.')
        return
    if digest != OLD_SHA256:
        raise SystemExit(f'Ukjent eller lokalt endret testfil ({digest}); ingen filer endret.')
    changed = original.replace(OLD.encode('utf-8'), NEW.encode('utf-8'))
    if _digest(changed) != NEW_SHA256:
        raise SystemExit('Endringen stemmer ikke med forventet testfil; ingen filer endret.')
    fd, tmp = tempfile.mkstemp(prefix='.dense_test_', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(changed)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    print(f'Oppdatert {TARGET}; produksjonskoden er uendret.')


if __name__ == '__main__':
    main()
