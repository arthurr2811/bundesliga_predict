"""Monte-Carlo der Restsaison.

Gezogen wird aus der Torematrix jedes offenen Spiels, nicht aus zwei
unabhaengigen Poissons -- sonst geht die Dixon-Coles-Korrektur verloren.

Die Modellparameter sind ueber alle Laeufe hinweg dieselben.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bundesliga_predict.config import (
    N_SIMULATIONS,
    POINTS_DRAW,
    POINTS_WIN,
    SIMULATION_SEED,
)
from bundesliga_predict.model.params import DixonColesParams
from bundesliga_predict.predict.matrix import score_matrix
from bundesliga_predict.simulation.table import positions, team_records


@dataclass(frozen=True)
class SimulationConfig:
    n_simulations: int = N_SIMULATIONS
    seed: int = SIMULATION_SEED


@dataclass(frozen=True)
class SeasonForecast:
    """Rohergebnis der Simulation: je Lauf und Team Punkte, Tore, Platz.

    Alles Weitere sind Sichten darauf, damit nichts doppelt gerechnet wird.
    """

    teams: tuple[str, ...]
    points: np.ndarray = field(repr=False)  # (n_simulationen, n_teams)
    goal_difference: np.ndarray = field(repr=False)
    goals_for: np.ndarray = field(repr=False)
    position: np.ndarray = field(repr=False)

    @property
    def n_simulations(self) -> int:
        return self.points.shape[0]

    @property
    def position_probabilities(self) -> np.ndarray:
        """(n_teams, n_teams): Zeile Team, Spalte Platz 1..n."""
        n_teams = len(self.teams)
        counts = np.stack(
            [
                np.bincount(self.position[:, team] - 1, minlength=n_teams)
                for team in range(n_teams)
            ]
        )
        return counts / self.n_simulations

    def probability_of_places(self, first: int, last: int) -> np.ndarray:
        """Wahrscheinlichkeit je Team, auf Platz `first`..`last` zu landen."""
        return ((self.position >= first) & (self.position <= last)).mean(axis=0)

    def summary(self) -> pd.DataFrame:
        """Je Team erwartete Punkte, mittlerer Platz und Punkte-Bandbreite."""
        frame = pd.DataFrame(
            {
                "team": self.teams,
                "expected_points": self.points.mean(axis=0),
                "expected_position": self.position.mean(axis=0),
                "points_p05": np.percentile(self.points, 5, axis=0),
                "points_p95": np.percentile(self.points, 95, axis=0),
                "expected_goal_difference": self.goal_difference.mean(axis=0),
            }
        )
        return frame.sort_values("expected_points", ascending=False).reset_index(drop=True)

    def position_table(self) -> pd.DataFrame:
        """Platzverteilung als Tabelle, Zeilen nach erwartetem Platz sortiert."""
        frame = pd.DataFrame(
            self.position_probabilities,
            index=pd.Index(self.teams, name="team"),
            columns=range(1, len(self.teams) + 1),
        )
        return frame.iloc[np.argsort(self.position.mean(axis=0))]


def open_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Die noch nicht ausgetragenen Partien."""
    return matches[~matches["finished"].astype(bool)]


def sample_scores(
    params: DixonColesParams,
    fixtures: pd.DataFrame,
    rng: np.random.Generator,
    n_simulations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Zieht je Lauf und Partie ein Ergebnis. Beide Rueckgaben (n_simulationen, n_partien).

    Die Torematrix wird flach gelegt und aufsummiert; ein `searchsorted` je
    Partie zieht daraus alle Laeufe auf einmal.
    """
    home_goals = np.empty((n_simulations, len(fixtures)), dtype=np.int16)
    away_goals = np.empty_like(home_goals)

    pairings = zip(fixtures["home_team"], fixtures["away_team"])
    for number, (home_team, away_team) in enumerate(pairings):
        matrix = score_matrix(params, home_team, away_team)
        cumulative = np.cumsum(matrix.ravel())
        drawn = np.searchsorted(cumulative, rng.random(n_simulations) * cumulative[-1])
        home_goals[:, number], away_goals[:, number] = np.divmod(drawn, matrix.shape[1])

    return home_goals, away_goals


def simulate_season(
    params: DixonColesParams,
    matches: pd.DataFrame,
    teams: Iterable[str] | None = None,
    config: SimulationConfig | None = None,
) -> SeasonForecast:
    """Spielt die offenen Partien in `matches` `n_simulations` mal durch.

    `matches` ist eine einzelne Saison: gespielte Partien liefern den
    Startstand, offene werden simuliert. Ist nichts mehr offen, kommt die
    Abschlusstabelle heraus, in jedem Lauf dieselbe.
    """
    config = config or SimulationConfig()
    if teams is None:
        teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    teams = tuple(teams)
    index = {team: number for number, team in enumerate(teams)}

    start = team_records(matches, teams)
    fixtures = open_matches(matches)
    unknown = (set(fixtures["home_team"]) | set(fixtures["away_team"])) - index.keys()
    if unknown:
        raise ValueError(f"Offene Spiele von Teams ausserhalb der Liga: {sorted(unknown)}")

    rng = np.random.default_rng(config.seed)
    home_goals, away_goals = sample_scores(params, fixtures, rng, config.n_simulations)

    shape = (config.n_simulations, len(teams))
    points = np.tile(start["points"].to_numpy(), (config.n_simulations, 1))
    scored = np.tile(start["goals_for"].to_numpy(), (config.n_simulations, 1))
    conceded = np.tile(start["goals_against"].to_numpy(), (config.n_simulations, 1))

    home_index = fixtures["home_team"].map(index).to_numpy()
    away_index = fixtures["away_team"].map(index).to_numpy()
    for number, (home, away) in enumerate(zip(home_index, away_index)):
        scored_home = home_goals[:, number]
        scored_away = away_goals[:, number]
        home_points = np.where(
            scored_home > scored_away, POINTS_WIN, np.where(scored_home == scored_away, POINTS_DRAW, 0)
        )
        away_points = np.where(
            scored_away > scored_home, POINTS_WIN, np.where(scored_home == scored_away, POINTS_DRAW, 0)
        )
        points[:, home] += home_points
        points[:, away] += away_points
        scored[:, home] += scored_home
        scored[:, away] += scored_away
        conceded[:, home] += scored_away
        conceded[:, away] += scored_home

    goal_difference = scored - conceded
    # Voelliger Gleichstand wird ausgelost
    position = positions(points, goal_difference, scored, tiebreak=rng.random(shape))

    return SeasonForecast(
        teams=teams,
        points=points,
        goal_difference=goal_difference,
        goals_for=scored,
        position=position,
    )
