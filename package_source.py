"""Build a deterministic, portable source bundle without host or account metadata."""
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'dist' / 'capacity-research.zip'
TOP_LEVEL = ['README.md', 'engine.py', 'research.py', 'package_source.py',
             'capacity_research.ipynb', 'prepare_pages.py', '.gitignore']
files = [ROOT / name for name in TOP_LEVEL]
for folder, pattern in [('data', '*.csv'), ('tests', '*.py'), ('tests', '*.cjs'),
                        ('dist', '*.html'), ('dist', '*.css'), ('dist', '*.js'),
                        ('dist', '*.json')]:
    files.extend((ROOT / folder).glob(pattern))
with ZipFile(OUTPUT, 'w', compression=ZIP_DEFLATED, compresslevel=9) as archive:
    for path in sorted(set(files)):
        name = 'finite-edge/' + path.relative_to(ROOT).as_posix()
        info = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
        info.compress_type = ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, path.read_bytes())
print(f'Packaged {len(set(files))} files into {OUTPUT.name} ({OUTPUT.stat().st_size:,} bytes).')
