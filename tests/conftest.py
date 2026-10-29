import sys
from pathlib import Path

# Run the tests against this clone, also when fvtools is not installed
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
