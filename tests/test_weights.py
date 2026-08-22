import numpy as np
import pandas as pd

from bundesliga_predict.model.weights import WeightConfig, match_weights, season_start_year


def test_season_start_year():
    assert season_start_year("2016/17") == 2016


def test_halflife_and_season_penalty():
    dates = pd.Series(["2026-08-01", "2026-02-01", "2025-08-01"])
    seasons = pd.Series(["2026/27", "2025/26", "2025/26"])
    config = WeightConfig(half_life_days=180.0, season_penalty=0.5)

    weights = match_weights(
        dates, seasons, pd.Timestamp("2026-08-01"), "2026/27", config
    )

    # Frisches Spiel derselben Saison: volles Gewicht.
    assert weights[0] == 1.0
    # 181 Tage alt und eine Saison zurueck: rund die Haelfte, nochmal halbiert.
    assert np.isclose(weights[1], 0.5 ** (181 / 180) * 0.5, atol=1e-6)
    # Ein Jahr alt, ebenfalls eine Saison zurueck.
    assert np.isclose(weights[2], 0.5 ** (365 / 180) * 0.5, atol=1e-6)


def test_no_decay_when_halflife_infinite():
    dates = pd.Series(["2016-08-01", "2026-08-01"])
    seasons = pd.Series(["2016/17", "2026/27"])
    config = WeightConfig(half_life_days=np.inf, season_penalty=1.0)

    weights = match_weights(
        dates, seasons, pd.Timestamp("2026-08-01"), "2026/27", config
    )

    assert np.allclose(weights, 1.0)
