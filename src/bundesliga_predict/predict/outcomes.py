"""Von der Torematrix zu den Grössen, die am Ende jemand liest.

Die Matrix aus `predict.matrix` ist die vollständige Information über ein
Spiel; alles hier ist nur noch Aufsummieren einzelner Zellen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bundesliga_predict.model.params import DixonColesParams
from bundesliga_predict.predict.matrix import score_matrix

# Reihenfolge der 1X2-Wahrscheinlichkeiten. Überall gleich, damit Modell,
# Baselines und Metriken denselben Vektor meinen.
OUTCOMES = ("home", "draw", "away")


@dataclass(frozen=True)
class MatchPrediction:
    """Vorhersage für eine Paarung."""

    home_team: str
    away_team: str
    home_win: float
    draw: float
    away_win: float
    expected_home_goals: float
    expected_away_goals: float

    @property
    def probabilities(self) -> np.ndarray:
        """1X2 in der Reihenfolge von `OUTCOMES`."""
        return np.array([self.home_win, self.draw, self.away_win])


def outcome_probabilities(matrix: np.ndarray) -> tuple[float, float, float]:
    """Heimsieg / Unentschieden / Auswärtssieg aus der Torematrix."""
    home_win = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, 1).sum())
    return home_win, draw, away_win


def exact_score_probability(matrix: np.ndarray, home_goals: int, away_goals: int) -> float:
    """Wahrscheinlichkeit eines konkreten Ergebnisses."""
    return float(matrix[home_goals, away_goals])


def most_likely_score(matrix: np.ndarray) -> tuple[int, int]:
    """Das wahrscheinlichste Einzelergebnis (nicht der wahrscheinlichste Ausgang)."""
    home_goals, away_goals = np.unravel_index(int(np.argmax(matrix)), matrix.shape)
    return int(home_goals), int(away_goals)


def over_under(matrix: np.ndarray, line: float = 2.5) -> tuple[float, float]:
    """Wahrscheinlichkeit für mehr / weniger Tore als `line`."""
    goals = np.arange(matrix.shape[0])
    total = goals[:, None] + goals[None, :]
    over = float(matrix[total > line].sum())
    return over, 1.0 - over


def predict_match(
    params: DixonColesParams, home_team: str, away_team: str
) -> MatchPrediction:
    """Komplette Vorhersage einer Paarung aus den gefitteten Parametern."""
    matrix = score_matrix(params, home_team, away_team)
    home_win, draw, away_win = outcome_probabilities(matrix)
    lambda_home, lambda_away = params.expected_goals(home_team, away_team)
    return MatchPrediction(
        home_team=home_team,
        away_team=away_team,
        home_win=home_win,
        draw=draw,
        away_win=away_win,
        expected_home_goals=lambda_home,
        expected_away_goals=lambda_away,
    )
