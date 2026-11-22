"""Refresh the downloadable source and the branch-based GitHub Pages directory."""
from pathlib import Path
import runpy
import shutil

ROOT = Path(__file__).resolve().parent
runpy.run_path(str(ROOT / "package_source.py"), run_name="__main__")
publication = ROOT / "docs"
publication.mkdir(exist_ok=True)
for name in ["index.html", "style.css", "app.js", "research-data.js",
             "research.json", "capacity-research.zip"]:
    shutil.copyfile(ROOT / "dist" / name, publication / name)
(publication / ".nojekyll").write_text("")
print("GitHub Pages files refreshed in docs/.")
