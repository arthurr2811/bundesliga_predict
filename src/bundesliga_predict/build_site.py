"""Baut die statische Seite fuer GitHub Pages in einen einzigen Ordner.

Lokal liest das Frontend `../data/output/` (Ordner `frontend/` neben `data/`).
Pages liefert nur einen Ordner aus, deshalb landen hier Frontend und Daten
nebeneinander und der Datenpfad in `app.js` wird umgeschrieben. Kopiert werden
nur die JSONs und das Archiv, keine Tuning- oder Backtest-CSVs.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = _PROJECT_ROOT / "frontend"
OUTPUT_DIR = _PROJECT_ROOT / "data" / "output"
SITE_DIR = _PROJECT_ROOT / "_site"

_LOCAL_DATA = 'const DATA = "../data/output/";'
_SITE_DATA = 'const DATA = "data/";'


def build_site(
    site: Path = SITE_DIR,
    frontend: Path = FRONTEND_DIR,
    output: Path = OUTPUT_DIR,
) -> Path:
    if not (output / "meta.json").exists():
        raise SystemExit(f"{output / 'meta.json'} fehlt -- erst `cli update` laufen lassen")
    if site.exists():
        shutil.rmtree(site)
    shutil.copytree(frontend, site)

    app = site / "app.js"
    source = app.read_text(encoding="utf-8")
    # Bricht ab statt still mit falschem Pfad zu deployen.
    if source.count(_LOCAL_DATA) != 1:
        raise SystemExit(f"{_LOCAL_DATA!r} nicht genau einmal in app.js gefunden")
    app.write_text(source.replace(_LOCAL_DATA, _SITE_DATA), encoding="utf-8")

    data = site / "data"
    data.mkdir()
    for path in sorted(output.glob("*.json")):
        shutil.copy2(path, data / path.name)
    shutil.copytree(output / "archive", data / "archive")
    return site


def main() -> None:
    site = build_site(Path(sys.argv[1]) if len(sys.argv) > 1 else SITE_DIR)
    print(f"Seite gebaut: {site}")


if __name__ == "__main__":
    main()
