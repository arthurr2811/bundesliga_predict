"""Laden der historischen Saisons von football-data.co.uk."""

from pathlib import Path

import pandas as pd

_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]


def _season_from_filename(path: Path) -> str:
    # z.B. "D1_2016-17.csv" -> "2016/17"
    start, end = path.stem.removeprefix("D1_").split("-")
    return f"{start}/{end}"


def load_historic(raw_dir: Path) -> pd.DataFrame:
    """Liest alle D1_<saison>.csv-Dateien in raw_dir und vereinheitlicht sie.

    football-data.co.uk führt keine Spieltag-Nummer -> matchday bleibt leer.
    Alle Spiele in diesen Dateien sind abgeschlossen (finished=True).
    """
    frames = []
    for path in sorted(raw_dir.glob("D1_*.csv")):
        df = pd.read_csv(path, usecols=_COLUMNS)
        df["season"] = _season_from_filename(path)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    # Jahr in "Date" ist je nach Saison 2- oder 4-stellig -> format="mixed"
    combined["date"] = pd.to_datetime(
        combined["Date"], dayfirst=True, format="mixed"
    ).dt.date

    result = pd.DataFrame(
        {
            "season": combined["season"],
            "date": combined["date"],
            "matchday": pd.NA,
            "home_team": combined["HomeTeam"],
            "away_team": combined["AwayTeam"],
            "home_goals": combined["FTHG"].astype("Int64"),
            "away_goals": combined["FTAG"].astype("Int64"),
            "finished": True,
        }
    )
    return result
