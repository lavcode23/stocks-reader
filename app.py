import sys
from pathlib import Path
import runpy

ROOT = Path(__file__).parent.resolve()

# Make src importable
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Debug (optional – remove later)
print("PYTHONPATH:", sys.path)

# Run real Streamlit app
runpy.run_path(str(SRC / "ui" / "app.py"), run_name="__main__")
