"""Maximum-Likelihood-Fit der Dixon-Coles-Parameter."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from bundesliga_predict.config import DEFAULT_PRIOR_SD, RHO_BOUNDS
from bundesliga_predict.model import likelihood
from bundesliga_predict.model.params import DixonColesParams, n_free_params, to_params
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
    prior_sd: float = DEFAULT_PRIOR_SD,
) -> DixonColesParams:
    """Schätzt die Modellparameter auf allen Spielen bis `reference_date`.

    Der Stichtag ist zugleich Filter und Bezugspunkt der Zeitgewichtung. Ohne
    Angabe wird das letzte gespielte Spiel im Datensatz verwendet. Der Schnitt
    läuft über das Datum, nicht über den Spieltag.
    """
    played = finished_matches(matches).copy()
    played["date"] = pd.to_datetime(played["date"])

    if reference_date is None:
        reference_date = played["date"].max()
    reference_date = pd.Timestamp(reference_date)

    played = played[played["date"] <= reference_date]
    if played.empty:
        raise ValueError(f"Keine gespielten Partien bis {reference_date.date()}.")

    reference_season = played.loc[played["date"].idxmax(), "season"]
    weights = match_weights(
        played["date"], played["season"], reference_date, reference_season, weight_config
    )
    data = likelihood.prepare(played, weights)

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
        args=(data, prior_sd),
        method="L-BFGS-B",
        bounds=bounds,
    )
    if not result.success:
        raise RuntimeError(f"Fit nicht konvergiert: {result.message}")

    return to_params(result.x, data.teams, fitted_through=reference_date.date())
