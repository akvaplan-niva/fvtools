"""Tests for the phase 6B default-call compatibility fix.

These tests compile only the publisher function from the installed source, so
imports of unrelated scientific dependencies are not required to test dispatch.
"""
import ast
import os
from pathlib import Path
from uuid import uuid4

import pytest


SOURCE = Path('fvtools/nesting/roms_nesting_fg.py')


def publisher(monkeypatch, kind):
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    method = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_publish_nesting_file')
    events = []
    def create(path, *args):
        Path(path).write_bytes(b'COMPLETE')
    class Runner:
        def __init__(self, outfile, *args):
            self.outfile = outfile
        def dump_single(self, *args, **kwargs):
            if kind == 'old' and (args or kwargs):
                raise AssertionError('Default dispatch passed new kwargs to the old API')
            events.append((args, kwargs))
            if kind == 'error':
                raise OSError('synthetic failure')
            Path(self.outfile).write_bytes(b'PRODUCT')
        def dump(self, nprocs=None):
            raise AssertionError('Unexpected parallel execution')
    # For the old-signature case, Runner actually has no keyword arguments.
    if kind in ('old', 'error'):
        def old_dump(self):
            events.append(((), {}))
            if kind == 'error':
                raise OSError('synthetic failure')
            Path(self.outfile).write_bytes(b'PRODUCT')
        Runner.dump_single = old_dump
    namespace = {'os': os, 'Path': Path, 'uuid4': uuid4,
                 'create_nc_forcing_file': create, 'Roms2FVCOMNest': Runner}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace['_publish_nesting_file'], events


def call(fn, filename, **kwargs):
    fn(str(filename), object(), 'NKv3', [61000.], False, 'epsg:32633',
       ['roms.nc'], [0], object(), 1, **kwargs)


def test_default_works_with_old_serial_signature(tmp_path, monkeypatch):
    fn, events = publisher(monkeypatch, 'old')
    target = tmp_path / 'nest.nc'
    call(fn, target)
    assert target.read_bytes() == b'PRODUCT'
    assert events == [((), {})]
    assert not list(tmp_path.glob('*.partial'))


def test_explicit_block_options_are_forwarded(tmp_path, monkeypatch):
    fn, events = publisher(monkeypatch, 'new')
    target = tmp_path / 'nest.nc'
    call(fn, target, time_block_size=2, max_block_mb=32)
    assert target.read_bytes() == b'PRODUCT'
    assert events == [((), {'time_block_size': 2, 'max_block_mb': 32})]


def test_default_failure_preserves_existing_file(tmp_path, monkeypatch):
    fn, events = publisher(monkeypatch, 'error')
    target = tmp_path / 'nest.nc'
    target.write_bytes(b'ORIGINAL')
    with pytest.raises(OSError, match='synthetic failure'):
        call(fn, target)
    assert target.read_bytes() == b'ORIGINAL'
    assert not list(tmp_path.glob('*.partial'))
    assert events == [((), {})]
