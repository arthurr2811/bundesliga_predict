"""Von Modellparametern zur Wahrscheinlichkeitsmatrix eines Spiels."""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from bundesliga_predict.config import MAX_GOALS
from bundesliga_predict.model.likelihood import tau_correction
from bundesliga_predict.model.params import DixonColesParams


def score_matrix(
    params: DixonColesParams, home_team: str, away_team: str, max_goals: int = MAX_GOALS
) -> np.ndarray:
    """Matrix P[i, j] = Wahrscheinlichkeit für i Heimtore und j Auswärtstore.

    Wird auf Summe 1 normiert: die tau-Korrektur verschiebt Masse zwischen den
    vier torarmen Zellen, und bei `max_goals` wird die Verteilung abgeschnitten.
    """
    lambda_home, lambda_away = params.expected_goals(home_team, away_team)

    goals = np.arange(max_goals + 1)
    matrix = np.outer(poisson.pmf(goals, lambda_home), poisson.pmf(goals, lambda_away))

    home_grid, away_grid = np.meshgrid(goals, goals, indexing="ij")
    matrix = matrix * tau_correction(
        home_grid, away_grid, lambda_home, lambda_away, params.rho
    )

    return matrix / matrix.sum()
