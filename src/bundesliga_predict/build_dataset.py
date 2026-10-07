"""Führt historische und laufende Saison zu data/processed/matches.csv zusammen."""

import sys
from datetime import date
from pathlib import Path

import pandas as pd

from bundesliga_predict.historic_source import load_historic
from bundesliga_predict.live_source import fetch_live, parse_live
from bundesliga_predict.matchday_source import (
    attach_matchdays,
    ensure_matchdays,
    season_start_year,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RAW_HISTORIC_DIR = _PROJECT_ROOT / "data" / "raw" / "historic_data"
_MATCHDAY_CACHE = _PROJECT_ROOT / "data" / "raw" / "matchdays.csv"
_PROCESSED_PATH = _PROJECT_ROOT / "data" / "processed" / "matches.csv"
# Abgeschlossene Saisons ohne Quoten, committet: damit laeuft `update` auch
# dort, wo die Roh-CSVs fehlen (GitHub Actions).
HISTORIC_PATH = _PROJECT_ROOT / "data" / "processed" / "historic.csv"

COLUMNS = [
    "season",
    "date",
    "matchday",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "finished",
]


def current_season(today: date | None = None) -> int:
    """Startjahr der laufenden Saison; ab Juli zaehlt die neue."""
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def build_historic(raw_dir: Path = _RAW_HISTORIC_DIR) -> pd.DataFrame:
    """Abgeschlossene Saisons aus den Roh-CSVs, inklusive Spieltagsnummer.

    Die Saisons stammen von football-data.co.uk und bekommen ihre
    Spieltagsnummer aus dem OpenLigaDB-Cache.
    """
    historic = load_historic(raw_dir)
    matchdays = ensure_matchdays(_MATCHDAY_CACHE, sorted(historic["season"].unique()))
    return attach_matchdays(historic, matchdays)[COLUMNS]


def read_historic(path: Path = HISTORIC_PATH) -> pd.DataFrame:
    """Liest `historic.csv` mit denselben Typen, die `build_historic` liefert."""
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    frame["home_goals"] = frame["home_goals"].astype("Int64")
    frame["away_goals"] = frame["away_goals"].astype("Int64")
    frame["finished"] = frame["finished"].astype(bool)
    return frame[COLUMNS]


def load_historic_dataset() -> pd.DataFrame:
    """Roh-CSVs, wenn vorhanden, sonst die committete `historic.csv`."""
    if any(_RAW_HISTORIC_DIR.glob("D1_*.csv")):
        return build_historic(_RAW_HISTORIC_DIR)
    if HISTORIC_PATH.exists():
        return read_historic(HISTORIC_PATH)
    raise SystemExit(
        f"Weder Roh-CSVs in {_RAW_HISTORIC_DIR} noch {HISTORIC_PATH} vorhanden."
    )


def build_dataset(season: int | None = None) -> pd.DataFrame:
    """Einheitlicher Datensatz aus beiden Quellen, inklusive Spieltagsnummer.

    Alle Saisons nach der letzten historischen bis zur laufenden kommen von
    OpenLigaDB. Normalerweise ist das nur die laufende; nach einem
    Saisonwechsel schliesst das die Luecke, bis die Historie nachgezogen ist.
    """
    season = season if season is not None else current_season()
    historic = load_historic_dataset()

    frames = [historic]
    last_historic = season_start_year(historic["season"].max())
    for start_year in range(last_historic + 1, season + 1):
        raw = fetch_live(start_year)
        # Spielplan der neuen Saison ist im Sommer evtl. noch nicht veroeffentlicht.
        if raw:
            frames.append(parse_live(raw))

    combined = pd.concat(frames, ignore_index=True)[COLUMNS]
    combined = combined.sort_values(["date", "home_team"]).reset_index(drop=True)
    return combined


def write_historic() -> None:
    """Erzeugt `historic.csv` aus den Roh-CSVs (lokal, nach neuer Saison)."""
    historic = build_historic().sort_values(["date", "home_team"])
    HISTORIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    historic.to_csv(HISTORIC_PATH, index=False)
    print(f"{len(historic)} Spiele geschrieben nach {HISTORIC_PATH}")


def main() -> None:
    if "--historic" in sys.argv[1:]:
        write_historic()
        return
    dataset = build_dataset()
    _PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(_PROCESSED_PATH, index=False)
    print(f"{len(dataset)} Spiele geschrieben nach {_PROCESSED_PATH}")


if __name__ == "__main__":
    main()
