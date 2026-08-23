"""Führt historische und laufende Saison zu data/processed/matches.csv zusammen."""

from pathlib import Path

import pandas as pd

from bundesliga_predict.historic_source import load_historic
from bundesliga_predict.live_source import fetch_live, parse_live
from bundesliga_predict.matchday_source import attach_matchdays, ensure_matchdays

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RAW_HISTORIC_DIR = _PROJECT_ROOT / "data" / "raw" / "historic_data"
_MATCHDAY_CACHE = _PROJECT_ROOT / "data" / "raw" / "matchdays.csv"
_PROCESSED_PATH = _PROJECT_ROOT / "data" / "processed" / "matches.csv"

CURRENT_SEASON = 2026

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


def build_dataset(current_season: int = CURRENT_SEASON) -> pd.DataFrame:
    """Einheitlicher Datensatz aus beiden Quellen, inklusive Spieltagsnummer.

    Die historischen Saisons stammen von football-data.co.uk und bekommen ihre
    Spieltagsnummer aus dem OpenLigaDB-Cache; die laufende Saison bringt sie
    direkt mit.
    """
    historic = load_historic(_RAW_HISTORIC_DIR)
    matchdays = ensure_matchdays(_MATCHDAY_CACHE, sorted(historic["season"].unique()))
    historic = attach_matchdays(historic, matchdays)

    live = parse_live(fetch_live(current_season))

    combined = pd.concat([historic, live], ignore_index=True)[COLUMNS]
    combined = combined.sort_values(["date", "home_team"]).reset_index(drop=True)
    return combined


def main() -> None:
    dataset = build_dataset()
    _PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(_PROCESSED_PATH, index=False)
    print(f"{len(dataset)} Spiele geschrieben nach {_PROCESSED_PATH}")


if __name__ == "__main__":
    main()
