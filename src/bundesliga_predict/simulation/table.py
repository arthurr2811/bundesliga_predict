"""Von Ergebnissen zur Tabelle.

Zwei Ebenen, die dieselbe Sortierregel benutzen: `standings` fuer eine einzelne
Tabelle (aktueller Stand, Abschlusstabelle) und `positions` fuer die
Simulation, die zehntausend Tabellen auf einmal sortiert.

Sortiert wird nach Punkten, Tordifferenz, erzielten Toren. Der direkte
Vergleich, den die DFL-Regeln danach vorsehen, fehlt bewusst
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from bundesliga_predict.config import POINTS_DRAW, POINTS_WIN
from bundesliga_predict.model.fit import finished_matches

STANDINGS_COLUMNS = [
    "position",
    "team",
    "played",
    "won",
    "drawn",
    "lost",
    "goals_for",
    "goals_against",
    "goal_difference",
    "points",
]


def positions(
    points: np.ndarray,
    goal_difference: np.ndarray,
    goals_for: np.ndarray,
    tiebreak: np.ndarray | None = None,
) -> np.ndarray:
    """Platzierungen 1..n aus den drei Sortierkriterien.

    Arbeitet auf der letzten Achse, beliebig viele davor: `(n_teams,)` fuer
    eine Tabelle, `(n_simulationen, n_teams)` fuer die Simulation.

    `tiebreak` entscheidet, wenn alle drei Kriterien gleich sind -- in der
    Simulation ein Zufallswert
    """
    # np.lexsort sortiert aufsteigend und nimmt den *letzten* Schluessel als
    # wichtigsten. Negiert ergibt das absteigend, und weil die Sortierung
    # stabil ist, behalten Gleichstaende die Eingabe-Reihenfolge.
    keys = [-np.asarray(goals_for), -np.asarray(goal_difference), -np.asarray(points)]
    if tiebreak is not None:
        keys.insert(0, tiebreak)

    order = np.lexsort(keys, axis=-1)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(1, order.shape[-1] + 1), axis=-1)
    return ranks


def team_records(matches: pd.DataFrame, teams: Iterable[str]) -> pd.DataFrame:
    """Spiele, Siege, Tore und Punkte je Team -- unsortiert, in `teams`-Reihenfolge.

    Gezaehlt werden nur ausgetragene Partien; ein Team ohne Spiel steht mit
    Nullen da (Saisonstart).
    """
    teams = list(teams)
    index = {team: number for number, team in enumerate(teams)}

    played = finished_matches(matches)
    unknown = (set(played["home_team"]) | set(played["away_team"])) - index.keys()
    if unknown:
        raise ValueError(f"Spiele von Teams ausserhalb der Liga: {sorted(unknown)}")

    home = played["home_team"].map(index).to_numpy()
    away = played["away_team"].map(index).to_numpy()
    home_goals = played["home_goals"].to_numpy(dtype=int)
    away_goals = played["away_goals"].to_numpy(dtype=int)

    counters = {name: np.zeros(len(teams), dtype=int) for name in
                ("won", "drawn", "lost", "goals_for", "goals_against")}
    for side, own, other in ((home, home_goals, away_goals), (away, away_goals, home_goals)):
        np.add.at(counters["won"], side, own > other)
        np.add.at(counters["drawn"], side, own == other)
        np.add.at(counters["lost"], side, own < other)
        np.add.at(counters["goals_for"], side, own)
        np.add.at(counters["goals_against"], side, other)

    record = pd.DataFrame({"team": teams, **counters})
    record["played"] = record["won"] + record["drawn"] + record["lost"]
    record["goal_difference"] = record["goals_for"] - record["goals_against"]
    record["points"] = record["won"] * POINTS_WIN + record["drawn"] * POINTS_DRAW
    return record


def standings(
    matches: pd.DataFrame, teams: Iterable[str] | None = None
) -> pd.DataFrame:
    """Tabelle aus gespielten Partien, sortiert und mit Platznummer.

    `teams` bestimmt, wer in der Tabelle steht -- noetig vor dem ersten
    Spieltag, wenn noch niemand gespielt hat. Ohne Angabe alle Teams, die
    vorkommen.
    """
    if teams is None:
        teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))

    record = team_records(matches, teams)
    record["position"] = positions(
        record["points"].to_numpy(),
        record["goal_difference"].to_numpy(),
        record["goals_for"].to_numpy(),
    )
    return (
        record.sort_values("position")[STANDINGS_COLUMNS]
        .reset_index(drop=True)
    )
