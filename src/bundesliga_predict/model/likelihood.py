"""Gewichtete Log-Likelihood des Dixon-Coles-Modells.

Alles vektorisiert über numpy für Performance
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import gammaln

from bundesliga_predict.config import DEFAULT_PRIOR_SD, PRIOR_MATCH_WEIGHT
from bundesliga_predict.model.params import split_vector

# Untergrenze für tau. Für extreme rho kann die Korrektur rechnerisch
# negativ werden; dann ist der Log undefiniert. Statt harter Nebenbedingung
# clippen wir und machen die Stelle damit für den Optimierer unattraktiv.
_TAU_FLOOR = 1e-10


def tau_correction(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
    rho: float,
) -> np.ndarray:
    """Dixon-Coles-Korrekturfaktor für torarme Ergebnisse.

    Zwei unabhängige Poisson-Verteilungen unterschätzen 0:0 und 1:1 und
    überschätzen 1:0/0:1. tau korrigiert genau diese vier Zellen; alle
    anderen Ergebnisse bleiben unverändert (Faktor 1).
    """
    tau = np.ones(np.broadcast(home_goals, away_goals, lambda_home, lambda_away).shape)
    tau = np.where(
        (home_goals == 0) & (away_goals == 0), 1.0 - lambda_home * lambda_away * rho, tau
    )
    tau = np.where((home_goals == 0) & (away_goals == 1), 1.0 + lambda_home * rho, tau)
    tau = np.where((home_goals == 1) & (away_goals == 0), 1.0 + lambda_away * rho, tau)
    tau = np.where((home_goals == 1) & (away_goals == 1), 1.0 - rho, tau)
    return tau


@dataclass(frozen=True)
class LikelihoodData:
    """Vorbereitete Arrays für den Fit."""

    teams: tuple[str, ...]
    home_idx: np.ndarray
    away_idx: np.ndarray
    home_goals: np.ndarray
    away_goals: np.ndarray
    weights: np.ndarray
    log_factorials: np.ndarray  # konstanter Anteil der Poisson-Dichte
    shrinkage: np.ndarray  # je Team: wie stark zieht der Prior?

    @property
    def n_teams(self) -> int:
        return len(self.teams)


def prepare(matches: pd.DataFrame, weights: np.ndarray) -> LikelihoodData:
    """Baut aus gespielten Partien plus Gewichten die Arrays für den Fit."""
    teams = tuple(sorted(set(matches["home_team"]) | set(matches["away_team"])))
    index = {team: i for i, team in enumerate(teams)}

    home_idx = matches["home_team"].map(index).to_numpy()
    away_idx = matches["away_team"].map(index).to_numpy()
    home_goals = matches["home_goals"].to_numpy(dtype=float)
    away_goals = matches["away_goals"].to_numpy(dtype=float)

    # Gewichtete Spielmasse(verfügbare Daten) je Team: Basis für die Shrinkage. Ein Aufsteiger
    # mit drei Spielen wird stark zum Ligadurchschnitt gezogen, Bayern mit
    # zehn Saisons praktisch gar nicht.
    team_weight = np.zeros(len(teams))
    np.add.at(team_weight, home_idx, weights)
    np.add.at(team_weight, away_idx, weights)
    shrinkage = PRIOR_MATCH_WEIGHT / (PRIOR_MATCH_WEIGHT + team_weight)

    return LikelihoodData(
        teams=teams,
        home_idx=home_idx,
        away_idx=away_idx,
        home_goals=home_goals,
        away_goals=away_goals,
        weights=weights,
        log_factorials=gammaln(home_goals + 1.0) + gammaln(away_goals + 1.0),
        shrinkage=shrinkage,
    )


def negative_log_likelihood(
    vector: np.ndarray, data: LikelihoodData, prior_sd: float = DEFAULT_PRIOR_SD
) -> float:
    """Zielfunktion des Fits: negative gewichtete Log-Likelihood plus Prior."""
    intercept, home_advantage, rho, attack, defense = split_vector(vector, data.n_teams)

    log_lambda_home = (
        intercept + attack[data.home_idx] - defense[data.away_idx] + home_advantage
    )
    log_lambda_away = intercept + attack[data.away_idx] - defense[data.home_idx]
    lambda_home = np.exp(log_lambda_home)
    lambda_away = np.exp(log_lambda_away)

    tau = tau_correction(data.home_goals, data.away_goals, lambda_home, lambda_away, rho)

    log_prob = (
        np.log(np.clip(tau, _TAU_FLOOR, None))
        + data.home_goals * log_lambda_home
        - lambda_home
        + data.away_goals * log_lambda_away
        - lambda_away
        - data.log_factorials
    )

    penalty = 0.0
    if np.isfinite(prior_sd) and prior_sd > 0:
        penalty = float(
            np.sum(data.shrinkage * (attack**2 + defense**2)) / (2.0 * prior_sd**2)
        )

    return -float(np.sum(data.weights * log_prob)) + penalty
