"""Zeitgewichtung der historischen Spiele.

Zwei Effekte, bewusst getrennt:

1. Stetiger Zerfall über die Zeit (Halbwertszeit in Tagen) -- Form aus
   Dixon/Coles (1997).
2. Ein zusätzlicher Abschlag je Saisonwechsel (also geringerer Einfluss von Spielen der
   letzten Saison auf die neue). Zwischen zwei Saisons wechseln Spieler, Trainer und
   drei Vereine komplett; dieser Bruch ist sprunghaft und lässt sich mit einem stetigen
   Zerfall nicht abbilden.

Beides sind Hyperparameter (Defaults in `config`), keine geschätzten
Modellparameter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bundesliga_predict.config import DEFAULT_HALF_LIFE_DAYS, DEFAULT_SEASON_PENALTY


@dataclass(frozen=True)
class WeightConfig:
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS
    season_penalty: float = DEFAULT_SEASON_PENALTY


def season_start_year(season: str) -> int:
    """'2016/17' -> 2016."""
    return int(str(season).split("/")[0])


def match_weights(
    dates: pd.Series,
    seasons: pd.Series,
    reference_date: pd.Timestamp,
    reference_season: str,
    config: WeightConfig | None = None,
) -> np.ndarray:
    """Gewicht je Spiel bezogen auf einen Stichtag.

    Spiele nach dem Stichtag bekommen kein negatives Alter zugewiesen (Gewicht
    bleibt 1)
    """
    config = config or WeightConfig()

    age_days = (pd.Timestamp(reference_date) - pd.to_datetime(dates)).dt.days.to_numpy()
    age_days = np.clip(age_days, 0, None)

    reference_start = season_start_year(reference_season)
    season_gap = reference_start - np.array([season_start_year(s) for s in seasons])
    season_gap = np.clip(season_gap, 0, None)

    if np.isinf(config.half_life_days):
        decay = np.ones_like(age_days, dtype=float)
    else:
        decay = 0.5 ** (age_days / config.half_life_days)

    return decay * config.season_penalty**season_gap
