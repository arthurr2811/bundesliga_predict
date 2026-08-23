"""Der Prior soll genau dort wirken, wo Daten fehlen -- und sonst nirgends.

Gebaut wird eine synthetische Liga aus lauter exakt gleich starken Teams. Was
der Fit danach an Unterschieden zwischen den Teams ausweist, kann also nur vom
Prior kommen. Ein Team spielt nur zwei Partien, die uebrigen eine volle
Doppelrunde.
"""

import numpy as np
import pandas as pd
import pytest

from bundesliga_predict.model.fit import fit
from bundesliga_predict.model.params import DixonColesParams
from bundesliga_predict.model.prior import PriorConfig, with_unknown_teams
from bundesliga_predict.model.weights import WeightConfig

NEWCOMER = "Team 00"
ESTABLISHED = "Team 01"
_TEAMS = tuple(f"Team {i:02d}" for i in range(8))
_FLAT_WEIGHTS = WeightConfig(half_life_days=np.inf, season_penalty=1.0)


# So oft wird die Doppelrunde der etablierten Teams wiederholt. Sie sollen
# deutlich mehr Spielmasse haben als `PRIOR_MATCH_WEIGHT`, damit die Shrinkage
# bei ihnen praktisch verschwindet.
_ROUNDS = 8


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    """Alle Teams gleich stark, aber `NEWCOMER` mit nur zwei Partien."""
    rng = np.random.default_rng(20260823)

    def match(home: str, away: str) -> dict:
        return {
            "season": "2025/26",
            "date": pd.Timestamp("2025-08-15"),
            "home_team": home,
            "away_team": away,
            # Gleiche Verteilung fuer alle: keine echten Staerkeunterschiede.
            "home_goals": int(rng.poisson(1.5)),
            "away_goals": int(rng.poisson(1.2)),
            "finished": True,
        }

    established = [team for team in _TEAMS if team != NEWCOMER]
    rows = [
        match(home, away)
        for _ in range(_ROUNDS)
        for home in established
        for away in established
        if home != away
    ]
    rows += [match(NEWCOMER, ESTABLISHED), match(ESTABLISHED, NEWCOMER)]
    return pd.DataFrame(rows)


def _fit(matches: pd.DataFrame, prior: PriorConfig) -> DixonColesParams:
    return fit(matches, weight_config=_FLAT_WEIGHTS, prior=prior)


def test_pulls_data_poor_team_towards_prior_mean(matches):
    """Der Mittelwert verschiebt das datenarme Team spuerbar, aber nicht ganz.
    """
    neutral = _fit(matches, PriorConfig(attack_mean=0.0, defense_mean=0.0))
    promoted = _fit(matches, PriorConfig(attack_mean=-0.25, defense_mean=-0.14))

    for name, mean in (("attack", -0.25), ("defense", -0.14)):
        shift = getattr(promoted, name)[NEWCOMER] - getattr(neutral, name)[NEWCOMER]
        assert 0.4 < shift / mean < 1.0, name


def test_leaves_established_teams_alone(matches):
    """Der Prior-Mittelwert verschiebt datenreiche Teams praktisch nicht."""
    neutral = _fit(matches, PriorConfig(attack_mean=0.0, defense_mean=0.0))
    promoted = _fit(matches, PriorConfig(attack_mean=-0.25, defense_mean=-0.14))

    moved_newcomer = abs(promoted.attack[NEWCOMER] - neutral.attack[NEWCOMER])
    moved_established = abs(promoted.attack[ESTABLISHED] - neutral.attack[ESTABLISHED])
    assert moved_established < moved_newcomer / 5


def test_neutral_mean_keeps_data_poor_team_average(matches):
    """Mittelwert 0 ist die alte Regularisierung: das datenarme Team landet
    beim Ligadurchschnitt -- genau der Punkt, der zu optimistisch war."""
    params = _fit(matches, PriorConfig(attack_mean=0.0, defense_mean=0.0))
    others = [params.attack[team] for team in _TEAMS if team != NEWCOMER]
    assert min(others) < params.attack[NEWCOMER] < max(others)


def test_strong_prior_reaches_its_mean(matches):
    """Je haerter der Prior, desto naeher liegt das datenarme Team am Ziel."""
    params = _fit(matches, PriorConfig(sd=0.1, attack_mean=-0.25, defense_mean=-0.14))

    for name, mean in (("attack", -0.25), ("defense", -0.14)):
        value = getattr(params, name)[NEWCOMER]
        assert abs(value - mean) < abs(value), name  # naeher am Prior als an 0


def test_unknown_teams_get_prior_mean():
    """Teams ohne jede Historie tauchen im Fit nicht auf und werden ergaenzt."""
    params = DixonColesParams(
        teams=("A",), attack={"A": 0.1}, defense={"A": 0.0},
        home_advantage=0.2, intercept=0.3, rho=-0.05,
    )
    prior = PriorConfig(attack_mean=-0.25, defense_mean=-0.14)

    extended = with_unknown_teams(params, {"A", "B"}, prior)
    assert extended.teams == ("A", "B")
    assert extended.attack == {"A": 0.1, "B": -0.25}
    assert extended.defense == {"A": 0.0, "B": -0.14}
    # Bereits bekannte Teams bleiben unangetastet.
    assert with_unknown_teams(params, {"A"}, prior) is params


def test_unknown_teams_stay_neutral_without_prior():
    """Ohne Regularisierung gibt es keinen Mittelwert -- dann Ligadurchschnitt."""
    params = DixonColesParams(
        teams=("A",), attack={"A": 0.1}, defense={"A": 0.0},
        home_advantage=0.2, intercept=0.3, rho=-0.05,
    )
    extended = with_unknown_teams(params, {"A", "B"}, PriorConfig(sd=np.inf))
    assert extended.attack["B"] == 0.0
    assert extended.defense["B"] == 0.0
