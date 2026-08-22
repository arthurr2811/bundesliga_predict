"""Parameter-Recovery: der schaerfste Test fuer den Modellkern.

Wir erzeugen synthetische Saisons aus bekannten Parametern, fitten diese
zurueck und pruefen, ob die Schaetzung die Wahrheit trifft. Faengt praktisch
jeden Vorzeichen-, Index- oder Normierungsfehler in Likelihood, Parameter-
Packing und Torematrix -- anders als ein Test gegen fest verdrahtete Zahlen,
der nur zementiert, was der Code heute tut.
"""

import numpy as np
import pandas as pd
import pytest

from bundesliga_predict.model.fit import fit
from bundesliga_predict.model.params import DixonColesParams
from bundesliga_predict.model.weights import WeightConfig
from bundesliga_predict.predict.matrix import score_matrix

N_TEAMS = 18
N_SEASONS = 6
TRUE_HOME_ADVANTAGE = 0.25
TRUE_INTERCEPT = np.log(1.4)
TRUE_RHO = -0.08


def make_true_params(rng: np.random.Generator) -> DixonColesParams:
    teams = tuple(f"Team {i:02d}" for i in range(N_TEAMS))

    attack = rng.normal(0.0, 0.25, N_TEAMS)
    defense = rng.normal(0.0, 0.20, N_TEAMS)
    attack -= attack.mean()  # Nebenbedingung des Modells
    defense -= defense.mean()

    return DixonColesParams(
        teams=teams,
        attack=dict(zip(teams, attack)),
        defense=dict(zip(teams, defense)),
        home_advantage=TRUE_HOME_ADVANTAGE,
        intercept=TRUE_INTERCEPT,
        rho=TRUE_RHO,
    )


def simulate_seasons(params: DixonColesParams, rng: np.random.Generator) -> pd.DataFrame:
    """Doppelte Hin-/Rueckrunde je Saison, Ergebnisse aus der echten
    Modellverteilung gezogen (inkl. tau-Korrektur)."""
    rows = []
    for season_offset in range(N_SEASONS):
        start_year = 2020 + season_offset
        season = f"{start_year}/{str(start_year + 1)[-2:]}"
        match_day = pd.Timestamp(f"{start_year}-08-15")

        for home in params.teams:
            for away in params.teams:
                if home == away:
                    continue
                matrix = score_matrix(params, home, away)
                flat = rng.choice(matrix.size, p=matrix.ravel())
                home_goals, away_goals = np.unravel_index(flat, matrix.shape)
                rows.append(
                    {
                        "season": season,
                        "date": match_day,
                        "home_team": home,
                        "away_team": away,
                        "home_goals": int(home_goals),
                        "away_goals": int(away_goals),
                        "finished": True,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def recovered() -> tuple[DixonColesParams, DixonColesParams]:
    rng = np.random.default_rng(20260822)
    truth = make_true_params(rng)
    matches = simulate_seasons(truth, rng)

    # Ohne Zeitgewichtung und ohne Prior fitten: hier soll ausschliesslich der
    # Modellkern geprueft werden, nicht die Hyperparameter.
    estimate = fit(
        matches,
        weight_config=WeightConfig(half_life_days=np.inf, season_penalty=1.0),
        prior_sd=np.inf,
    )
    return truth, estimate


def test_recovers_team_strengths(recovered):
    truth, estimate = recovered

    for name in ("attack", "defense"):
        true_values = np.array([getattr(truth, name)[t] for t in truth.teams])
        estimated = np.array([getattr(estimate, name)[t] for t in truth.teams])

        error = true_values - estimated
        # Bewusst RMSE statt Maximum: der Fit ist erwartungstreu, aber einzelne
        # Teams streuen (Stichprobenrauschen). Der RMSE halbiert sich sauber,
        # wenn man die simulierten Saisons vervielfacht.
        assert np.abs(error.mean()) < 0.02, name
        assert np.sqrt((error**2).mean()) < 0.09, name
        assert np.corrcoef(true_values, estimated)[0, 1] > 0.94, name


def test_recovers_global_params(recovered):
    truth, estimate = recovered

    assert estimate.home_advantage == pytest.approx(truth.home_advantage, abs=0.06)
    assert estimate.intercept == pytest.approx(truth.intercept, abs=0.05)
    # rho ist der am schwaechsten identifizierte Parameter: es haengen nur
    # vier Ergebniszellen daran, entsprechend gross ist die Toleranz.
    assert estimate.rho == pytest.approx(truth.rho, abs=0.08)


def test_score_matrix_is_a_distribution(recovered):
    _, estimate = recovered
    matrix = score_matrix(estimate, estimate.teams[0], estimate.teams[1])

    assert matrix.sum() == pytest.approx(1.0)
    assert (matrix >= 0).all()
