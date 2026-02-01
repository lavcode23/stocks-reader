from pathlib import Path
import sys
import runpy

ROOT = Path(__file__).parent
sys.path.append(str(ROOT / "src"))

runpy.run_path(str(ROOT / "src" / "ui" / "app.py"), run_name="__main__")
