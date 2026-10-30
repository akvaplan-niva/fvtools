"""Guard against reintroducing the removed fvgrid module."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_no_remaining_fvgrid_imports():
    offenders = []
    for folder in (ROOT / "fvtools", ROOT / "tests"):
        for path in folder.rglob("*.py"):
            # Avoid false positives from comments and test descriptions.
            source = path.read_text(encoding="utf-8-sig")
            if "fvgrid" not in source:
                continue
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    if any(part.name == "fvgrid" or part.name.endswith(".fvgrid")
                           for part in node.names):
                        offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
                elif isinstance(node, ast.ImportFrom):
                    if (node.module or "").endswith("fvgrid") or (
                        node.module in {"fvtools.grid", "grid", None} and
                        any(part.name == "fvgrid" for part in node.names)
                    ):
                        offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not offenders, "Remaining fvgrid imports: " + ", ".join(offenders)


def test_legacy_fvgrid_module_has_been_removed():
    assert not (ROOT / "fvtools/grid/fvgrid.py").exists()
