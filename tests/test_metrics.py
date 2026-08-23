"""Testet auf korrekte Metriken. Die Metriken sind die Messlatte für alles Weitere"""

import numpy as np
import pytest

from bundesliga_predict.evaluation import metrics


def test_perfekte_vorhersage_ist_null():
    probabilities = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    scores = metrics.score(probabilities, np.array([0, 2]))
    assert scores.rps == pytest.approx(0.0)
    assert scores.log_loss == pytest.approx(0.0)
    assert scores.brier == pytest.approx(0.0)


def test_rps_bestraft_den_ferneren_fehler_haerter():
    """Heimsieg tritt ein: 'Remis' vorherzusagen ist besser als 'Auswärtssieg'."""
    outcome = np.array([0])
    nah = metrics.ranked_probability_score(np.array([[0.0, 1.0, 0.0]]), outcome)
    fern = metrics.ranked_probability_score(np.array([[0.0, 0.0, 1.0]]), outcome)
    assert nah[0] < fern[0]


def test_rps_bekanntes_beispiel():
    """Nachgerechnet: p = (0.5, 0.3, 0.2), Heimsieg.

    Kumuliert: (0.5, 0.8, 1.0) gegen (1, 1, 1)
    -> ((0.5-1)^2 + (0.8-1)^2) / 2 = (0.25 + 0.04) / 2 = 0.145
    """
    value = metrics.ranked_probability_score(np.array([[0.5, 0.3, 0.2]]), np.array([0]))
    assert value[0] == pytest.approx(0.145)


def test_log_loss_entspricht_der_getroffenen_wahrscheinlichkeit():
    value = metrics.log_loss(np.array([[0.2, 0.3, 0.5]]), np.array([1]))
    assert value[0] == pytest.approx(-np.log(0.3))


def test_brier_summiert_ueber_alle_klassen():
    # (0.2-0)^2 + (0.3-1)^2 + (0.5-0)^2 = 0.04 + 0.49 + 0.25
    value = metrics.brier_score(np.array([[0.2, 0.3, 0.5]]), np.array([1]))
    assert value[0] == pytest.approx(0.78)


def test_log_loss_bleibt_endlich_bei_wahrscheinlichkeit_null():
    value = metrics.log_loss(np.array([[1.0, 0.0, 0.0]]), np.array([1]))
    assert np.isfinite(value[0])


def test_outcome_index_ordnet_richtig_zu():
    index = metrics.outcome_index(np.array([2, 1, 0]), np.array([1, 1, 3]))
    assert list(index) == [0, 1, 2]
