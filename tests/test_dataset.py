"""Strukturprüfung des gebauten Datensatzes.

Läuft gegen `data/processed/matches.csv`, falls vorhanden. Der Datensatz
entsteht aus zwei Quellen mit unterschiedlichen Team-Schreibweisen, die über
`team_mapping` zusammengeführt werden
"""

from pathlib import Path

import pandas as pd
import pytest

MATCHES_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "matches.csv"

TEAMS_PER_LEAGUE = 18
MATCHDAYS_PER_SEASON = 2 * (TEAMS_PER_LEAGUE - 1)
MATCHES_PER_MATCHDAY = TEAMS_PER_LEAGUE // 2


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    if not MATCHES_PATH.exists():
        pytest.skip("matches.csv nicht gebaut")
    return pd.read_csv(MATCHES_PATH)


def test_jede_saison_ist_vollstaendig(matches):
    spiele = matches.groupby("season").size()
    erwartet = MATCHDAYS_PER_SEASON * MATCHES_PER_MATCHDAY
    abweichend = spiele[spiele != erwartet]
    assert abweichend.empty, f"Saisons mit falscher Spielzahl: {abweichend.to_dict()}"


def test_jede_partie_hat_eine_spieltagsnummer(matches):
    assert matches["matchday"].notna().all()


def test_jeder_spieltag_hat_neun_partien_und_jedes_team_einmal(matches):
    for (season, matchday), spieltag in matches.groupby(["season", "matchday"]):
        teams = list(spieltag["home_team"]) + list(spieltag["away_team"])
        assert len(spieltag) == MATCHES_PER_MATCHDAY, f"{season} ST {matchday}"
        assert len(teams) == len(set(teams)), f"{season} ST {matchday}: Team doppelt"


def test_jede_paarung_kommt_je_saison_genau_einmal_vor(matches):
    doppelt = matches.duplicated(["season", "home_team", "away_team"])
    assert not doppelt.any()


def test_teamnamen_sind_kanonisch(matches):
    from bundesliga_predict.team_mapping import CANONICAL_TEAMS

    verwendet = set(matches["home_team"]) | set(matches["away_team"])
    assert verwendet <= CANONICAL_TEAMS, f"unbekannt: {sorted(verwendet - CANONICAL_TEAMS)}"


def test_gespielte_partien_haben_ein_ergebnis(matches):
    gespielt = matches[matches["finished"].astype(bool)]
    assert gespielt[["home_goals", "away_goals"]].notna().all().all()
