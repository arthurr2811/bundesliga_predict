"""Abrufen und Vereinheitlichen der laufenden Saison von OpenLigaDB."""

import pandas as pd
import requests

from bundesliga_predict.team_mapping import normalize_openligadb_team

_API_URL = "https://api.openligadb.de/getmatchdata/bl1/{season}"


def fetch_live(season: int) -> list[dict]:
    """Ruft den kompletten Saison-Spielplan (alle Spieltage) ab.

    Bewusst immer die ganze Saison statt nur des aktuellen Spieltags, damit
    nachtraegliche Korrekturen/verschobene Spiele automatisch mitkommen.
    """
    response = requests.get(_API_URL.format(season=season), timeout=30)
    response.raise_for_status()
    return response.json()


def _final_score(match: dict) -> tuple[int | None, int | None]:
    for result in match["matchResults"]:
        if result["resultTypeKind"] == "After90Minutes":
            return result["pointsTeam1"], result["pointsTeam2"]
    return None, None


def parse_live(matches: list[dict]) -> pd.DataFrame:
    """Wandelt die rohe OpenLigaDB-Antwort in unser einheitliches Format um."""
    rows = []
    for match in matches:
        home_goals, away_goals = _final_score(match)
        league_season = match["leagueSeason"]
        rows.append(
            {
                "season": f"{league_season}/{str(league_season + 1)[-2:]}",
                "date": pd.to_datetime(match["matchDateTime"]).date(),
                "matchday": match["group"]["groupOrderID"],
                "home_team": normalize_openligadb_team(match["team1"]["teamName"]),
                "away_team": normalize_openligadb_team(match["team2"]["teamName"]),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "finished": match["matchIsFinished"],
            }
        )

    df = pd.DataFrame(rows)
    df["home_goals"] = df["home_goals"].astype("Int64")
    df["away_goals"] = df["away_goals"].astype("Int64")
    return df
