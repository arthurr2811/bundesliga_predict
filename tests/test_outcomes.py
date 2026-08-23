"""Torematrix -> 1X2 und Symmetrie der tau-Korrektur."""

import numpy as np
import pytest

from bundesliga_predict.model.params import DixonColesParams
from bundesliga_predict.predict import outcomes
from bundesliga_predict.predict.matrix import score_matrix


def _params(rho: float = -0.05, home_advantage: float = 0.0) -> DixonColesParams:
    return DixonColesParams(
        teams=("A", "B"),
        attack={"A": 0.2, "B": -0.2},
        defense={"A": 0.1, "B": -0.1},
        home_advantage=home_advantage,
        intercept=np.log(1.4),
        rho=rho,
    )


def test_matrix_summiert_auf_eins():
    matrix = score_matrix(_params(), "A", "B")
    assert matrix.sum() == pytest.approx(1.0)


def test_1x2_summiert_auf_eins():
    matrix = score_matrix(_params(), "A", "B")
    assert sum(outcomes.outcome_probabilities(matrix)) == pytest.approx(1.0)


def test_ohne_heimvorteil_ist_die_paarung_symmetrisch():
    """A gegen B ohne Heimvorteil = gespiegeltes B gegen A."""
    params = _params(home_advantage=0.0)
    hin = outcomes.outcome_probabilities(score_matrix(params, "A", "B"))
    rueck = outcomes.outcome_probabilities(score_matrix(params, "B", "A"))
    assert hin[0] == pytest.approx(rueck[2])
    assert hin[1] == pytest.approx(rueck[1])
    assert hin[2] == pytest.approx(rueck[0])


def test_heimvorteil_erhoeht_die_heimsiegwahrscheinlichkeit():
    ohne = outcomes.outcome_probabilities(score_matrix(_params(home_advantage=0.0), "A", "B"))
    mit = outcomes.outcome_probabilities(score_matrix(_params(home_advantage=0.3), "A", "B"))
    assert mit[0] > ohne[0]


def test_negatives_rho_erhoeht_die_unentschieden_masse_in_den_tau_zellen():
    """Genau das ist der Zweck der Dixon-Coles-Korrektur: 0:0 und 1:1 hoch."""
    neutral = score_matrix(_params(rho=0.0), "A", "B")
    korrigiert = score_matrix(_params(rho=-0.15), "A", "B")
    assert korrigiert[0, 0] > neutral[0, 0]
    assert korrigiert[1, 1] > neutral[1, 1]
    assert korrigiert[1, 0] < neutral[1, 0]
    assert korrigiert[0, 1] < neutral[0, 1]


def test_over_under_ergaenzt_sich_zu_eins():
    matrix = score_matrix(_params(), "A", "B")
    over, under = outcomes.over_under(matrix, 2.5)
    assert over + under == pytest.approx(1.0)


def test_most_likely_scores_sind_absteigend_und_beginnen_beim_maximum():
    matrix = score_matrix(_params(home_advantage=0.25), "A", "B")
    top = outcomes.most_likely_scores(matrix, 3)
    assert len(top) == 3
    assert (top[0][0], top[0][1]) == outcomes.most_likely_score(matrix)
    assert top[0][2] >= top[1][2] >= top[2][2]
    for home, away, probability in top:
        assert probability == pytest.approx(matrix[home, away])


def test_predict_match_liefert_konsistente_erwartungswerte():
    params = _params(home_advantage=0.25)
    prediction = outcomes.predict_match(params, "A", "B")
    lambda_home, lambda_away = params.expected_goals("A", "B")
    assert prediction.expected_home_goals == pytest.approx(lambda_home)
    assert prediction.expected_away_goals == pytest.approx(lambda_away)
    assert prediction.probabilities.sum() == pytest.approx(1.0)
