import runpy
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
runpy.run_path(str(ROOT / "src" / "ui" / "app.py"), run_name="__main__")
