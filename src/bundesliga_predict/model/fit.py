"""Maximum-Likelihood-Fit der Dixon-Coles-Parameter."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from bundesliga_predict.config import RHO_BOUNDS
from bundesliga_predict.model import likelihood
from bundesliga_predict.model.params import DixonColesParams, n_free_params, to_params
from bundesliga_predict.model.prior import PriorConfig
from bundesliga_predict.model.weights import WeightConfig, match_weights

# Startwerte: grob plausible Bundesliga-Grössen
_INITIAL_HOME_ADVANTAGE = 0.25
_INITIAL_RHO = -0.05


def finished_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Nur ausgetragene Spiele mit vollständigem Ergebnis."""
    played = matches[matches["finished"].astype(bool)]
    return played.dropna(subset=["home_goals", "away_goals"])


def fit(
    matches: pd.DataFrame,
    reference_date: pd.Timestamp | None = None,
    weight_config: WeightConfig | None = None,
    prior: PriorConfig | None = None,
    reference_season: str | None = None,
    bootstrap_rng: np.random.Generator | None = None,
) -> DixonColesParams:
    prior = prior or PriorConfig()
    played = finished_matches(matches).copy()
    played["date"] = pd.to_datetime(played["date"])

    if reference_date is None:
        reference_date = played["date"].max()
    reference_date = pd.Timestamp(reference_date)

    played = played[played["date"] <= reference_date]
    if played.empty:
        raise ValueError(f"Keine gespielten Partien bis {reference_date.date()}.")

    if reference_season is None:
        reference_season = played.loc[played["date"].idxmax(), "season"]
    weights = match_weights(
        played["date"], played["season"], reference_date, reference_season, weight_config
    )
    if bootstrap_rng is not None:
        weights = weights * bootstrap_rng.exponential(1.0, size=len(weights))
    data = likelihood.prepare(played, weights, prior)

    goals_per_team = (played["home_goals"].sum() + played["away_goals"].sum()) / (
        2 * len(played)
    )
    initial = np.zeros(n_free_params(data.n_teams))
    initial[0] = np.log(goals_per_team)
    initial[1] = _INITIAL_HOME_ADVANTAGE
    initial[2] = _INITIAL_RHO

    bounds = [(None, None), (None, None), RHO_BOUNDS] + [(None, None)] * (
        len(initial) - 3
    )

    result = minimize(
        likelihood.negative_log_likelihood,
        initial,
        args=(data, prior),
        method="L-BFGS-B",
        bounds=bounds,
    )
    if not result.success:
        raise RuntimeError(f"Fit nicht konvergiert: {result.message}")

    return to_params(result.x, data.teams, fitted_through=reference_date.date())
