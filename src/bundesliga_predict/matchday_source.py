"""Spieltagsnummern für die abgeschlossenen Saisons.

football-data.co.uk liefert keine Spieltagsnummer, OpenLigaDB dagegen auch
für vergangene Saisons. Abgeschlossene Saisons ändern sich nicht mehr, deshalb
werden sie einmal abgerufen und in `data/raw/matchdays.csv` abgelegt; danach
geht nur noch die laufende Saison über das Netz.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bundesliga_predict.live_source import fetch_live, parse_live

COLUMNS = ["season", "home_team", "away_team", "matchday"]


def season_label(start_year: int) -> str:
    """2016 -> '2016/17'."""
    return f"{start_year}/{str(start_year + 1)[-2:]}"


def season_start_year(season: str) -> int:
    """'2016/17' -> 2016."""
    return int(str(season).split("/")[0])


def fetch_matchdays(start_year: int) -> pd.DataFrame:
    """Spieltagsnummern einer Saison von OpenLigaDB."""
    return parse_live(fetch_live(start_year))[COLUMNS]


def load_matchdays(path: Path) -> pd.DataFrame:
    """Liest den Cache; leeres Ergebnis, wenn er noch nicht existiert."""
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path)


def ensure_matchdays(path: Path, seasons: list[str]) -> pd.DataFrame:
    """Stellt sicher, dass der Cache alle `seasons` enthält, und gibt ihn zurück.

    Es werden ausschliesslich fehlende Saisons abgerufen. Der Cache ist damit
    nach dem ersten Lauf offline nutzbar.
    """
    cached = load_matchdays(path)
    fehlend = sorted(set(seasons) - set(cached["season"]))
    if not fehlend:
        return cached

    frames = [cached] if not cached.empty else []
    for season in fehlend:
        frames.append(fetch_matchdays(season_start_year(season)))

    complete = pd.concat(frames, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    complete.to_csv(path, index=False)
    return complete


def attach_matchdays(matches: pd.DataFrame, matchdays: pd.DataFrame) -> pd.DataFrame:
    """Ergänzt die Spieltagsnummer je Partie.

    Verknüpft wird über Saison und Paarung, nicht über das Datum: eine Paarung
    kommt pro Saison genau einmal vor, und ein verlegtes Spiel soll auch dann
    zugeordnet werden, wenn die Quellen sich beim Datum unterscheiden.

    Fehlt für eine Partie die Nummer, bricht die Funktion ab. Ein stiller
    Ausfall würde die Blockbildung im Backtest unbemerkt verfälschen.
    """
    key = ["season", "home_team", "away_team"]
    joined = matches.drop(columns=["matchday"], errors="ignore").merge(
        matchdays[COLUMNS], on=key, how="left"
    )

    if len(joined) != len(matches):
        raise ValueError("Spieltags-Join hat Zeilen vervielfacht - Paarung nicht eindeutig.")

    fehlend = joined[joined["matchday"].isna()]
    if not fehlend.empty:
        beispiel = fehlend[key].head(5).to_dict("records")
        raise ValueError(
            f"{len(fehlend)} Partien ohne Spieltagsnummer, z. B. {beispiel}"
        )

    joined["matchday"] = joined["matchday"].astype(int)
    return joined
