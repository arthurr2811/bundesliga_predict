"""Führt historische und laufende Saison zu data/processed/matches.csv zusammen."""

from pathlib import Path

import pandas as pd

from bundesliga_predict.historic_source import load_historic
from bundesliga_predict.live_source import fetch_live, parse_live

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RAW_HISTORIC_DIR = _PROJECT_ROOT / "data" / "raw" / "historic_data"
_PROCESSED_PATH = _PROJECT_ROOT / "data" / "processed" / "matches.csv"

CURRENT_SEASON = 2026


def build_dataset(current_season: int = CURRENT_SEASON) -> pd.DataFrame:
    historic = load_historic(_RAW_HISTORIC_DIR)
    live = parse_live(fetch_live(current_season))

    combined = pd.concat([historic, live], ignore_index=True)
    combined = combined.sort_values(["date", "home_team"]).reset_index(drop=True)
    return combined


def main() -> None:
    dataset = build_dataset()
    _PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(_PROCESSED_PATH, index=False)
    print(f"{len(dataset)} Spiele geschrieben nach {_PROCESSED_PATH}")


if __name__ == "__main__":
    main()
