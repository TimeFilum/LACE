"""Run LACE experiments from the repository root."""
from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    runpy.run_path(str(Path(__file__).resolve().parent / "src" / "run.py"), run_name="__main__")
