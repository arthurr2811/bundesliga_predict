"""Auswertung der Backtest-Vorhersagen gegen die Baselines.

Getrennt vom Backtest selbst, weil derselbe Lauf gegen mehrere Massstäbe
gehalten wird: Ligadurchschnitt (Untergrenze) und Markt (praktische
Obergrenze). Der Marktvergleich läuft nur auf Spielen, für die eine Quote
vorliegt -- damit die Zahlen vergleichbar bleiben, wird dort auch das Modell
auf genau dieser Teilmenge gemessen.
"""

from __future__ import annotations

import pandas as pd

from bundesliga_predict.evaluation import metrics
from bundesliga_predict.evaluation.backtest import (
    baseline_probabilities,
    model_probabilities,
)
from bundesliga_predict.evaluation.baselines import market_probabilities


def _scores_row(name: str, probabilities, outcomes) -> dict:
    return {"predictor": name, **metrics.score(probabilities, outcomes).as_dict()}


def compare(predictions: pd.DataFrame, odds: pd.DataFrame | None = None) -> pd.DataFrame:
    """Vergleichstabelle Modell vs. Baselines über alle Spiele des Backtests."""
    outcomes = metrics.outcome_index(
        predictions["home_goals"], predictions["away_goals"]
    )

    rows = [
        _scores_row("modell", model_probabilities(predictions), outcomes),
        _scores_row("ligadurchschnitt", baseline_probabilities(predictions), outcomes),
    ]

    if odds is not None:
        market, mask = market_probabilities(predictions, odds)
        if mask.any():
            rows.append(_scores_row("markt", market[mask], outcomes[mask]))
            # Nur zum fairen Vergleich mit dem Markt: dieselben Spiele.
            rows.append(
                _scores_row(
                    "modell (nur Spiele mit Quote)",
                    model_probabilities(predictions)[mask],
                    outcomes[mask],
                )
            )

    return pd.DataFrame(rows)


def by_season(predictions: pd.DataFrame) -> pd.DataFrame:
    """Modell und Baseline je Saison -- zeigt, ob eine Saison alles trägt."""
    outcomes = metrics.outcome_index(
        predictions["home_goals"], predictions["away_goals"]
    )
    rows = []
    for season, group in predictions.groupby("season", sort=True):
        index = outcomes[predictions["season"].to_numpy() == season]
        model = metrics.score(model_probabilities(group), index)
        baseline = metrics.score(baseline_probabilities(group), index)
        rows.append(
            {
                "season": season,
                "n_matches": model.n_matches,
                "rps": model.rps,
                "log_loss": model.log_loss,
                "brier": model.brier,
                "rps_baseline": baseline.rps,
            }
        )
    return pd.DataFrame(rows)


def format_table(table: pd.DataFrame) -> str:
    """Kompakte Konsolenausgabe mit vier Nachkommastellen."""
    return table.to_string(index=False, float_format=lambda value: f"{value:.4f}")
