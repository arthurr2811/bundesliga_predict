"""Monte-Carlo der Restsaison."""

import numpy as np
import pandas as pd
import pytest

from bundesliga_predict.model.params import DixonColesParams
from bundesliga_predict.predict.matrix import score_matrix
from bundesliga_predict.simulation.season import (
    SimulationConfig,
    sample_scores,
    simulate_season,
)
from bundesliga_predict.simulation.table import standings

TEAMS = ["A", "B", "C", "D"]


def _params(attack: dict[str, float] | None = None, rho: float = -0.05) -> DixonColesParams:
    attack = attack or dict.fromkeys(TEAMS, 0.0)
    return DixonColesParams(
        teams=tuple(TEAMS),
        attack=attack,
        defense=dict.fromkeys(TEAMS, 0.0),
        home_advantage=0.25,
        intercept=np.log(1.4),
        rho=rho,
    )


def _fixtures(*pairs: tuple[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "home_team": home,
                "away_team": away,
                "home_goals": np.nan,
                "away_goals": np.nan,
                "finished": False,
            }
            for home, away in pairs
        ]
    )


def _saison(offen: bool = True) -> pd.DataFrame:
    """Kleine Liga: jeder gegen jeden, Hinrunde gespielt, Rueckrunde offen."""
    hin = []
    for i, home in enumerate(TEAMS):
        for away in TEAMS[i + 1 :]:
            hin.append(
                {
                    "home_team": home,
                    "away_team": away,
                    "home_goals": 1.0,
                    "away_goals": 0.0,
                    "finished": True,
                }
            )
    rueck = _fixtures(*[(row["away_team"], row["home_team"]) for row in hin])
    if not offen:
        return pd.DataFrame(hin)
    return pd.concat([pd.DataFrame(hin), rueck], ignore_index=True)


def test_gleicher_seed_gleiches_ergebnis():
    config = SimulationConfig(n_simulations=500, seed=42)
    erste = simulate_season(_params(), _saison(), config=config)
    zweite = simulate_season(_params(), _saison(), config=config)

    assert np.array_equal(erste.points, zweite.points)
    assert np.array_equal(erste.position, zweite.position)


def test_anderer_seed_aendert_das_ergebnis():
    a = simulate_season(_params(), _saison(), config=SimulationConfig(n_simulations=500, seed=1))
    b = simulate_season(_params(), _saison(), config=SimulationConfig(n_simulations=500, seed=2))
    assert not np.array_equal(a.points, b.points)


def test_platzverteilung_summiert_in_beide_richtungen_auf_eins():
    """Jedes Team belegt genau einen Platz, jeder Platz genau ein Team."""
    forecast = simulate_season(
        _params(), _saison(), config=SimulationConfig(n_simulations=500)
    )
    verteilung = forecast.position_probabilities

    assert np.allclose(verteilung.sum(axis=1), 1.0)
    assert np.allclose(verteilung.sum(axis=0), 1.0)


def test_uebermaechtiges_team_wird_immer_erster():
    forecast = simulate_season(
        _params(attack={"A": 3.0, "B": 0.0, "C": 0.0, "D": 0.0}),
        _saison(),
        config=SimulationConfig(n_simulations=500),
    )
    meister = dict(zip(forecast.teams, forecast.probability_of_places(1, 1)))
    assert meister["A"] == 1.0


def test_ohne_offene_spiele_kommt_die_abschlusstabelle_heraus():
    """Degenerierter Fall: nichts mehr zu simulieren, jeder Lauf identisch."""
    saison = _saison(offen=False)
    forecast = simulate_season(
        _params(), saison, teams=TEAMS, config=SimulationConfig(n_simulations=20)
    )
    tabelle = standings(saison, teams=TEAMS)

    assert (forecast.points == forecast.points[0]).all()
    erwartet = tabelle.set_index("team")["points"].reindex(list(forecast.teams))
    assert list(forecast.points[0]) == list(erwartet)


def test_gespielte_partien_gehen_als_startstand_ein():
    """Ein Vorsprung aus der Hinrunde darf nicht verlorengehen."""
    saison = _saison()
    forecast = simulate_season(
        _params(), saison, config=SimulationConfig(n_simulations=200)
    )
    start = standings(saison, teams=TEAMS).set_index("team")["points"]

    minimum = forecast.points.min(axis=0)
    for number, team in enumerate(forecast.teams):
        assert minimum[number] >= start[team]


def test_gezogene_ergebnisse_folgen_der_torematrix():
    """Die Haeufigkeit einzelner Ergebnisse muss zur Matrix passen -- inklusive
    der Dixon-Coles-Korrektur, die 0:0 und 1:1 anhebt."""
    params = _params(rho=-0.15)
    fixtures = _fixtures(("A", "B"))
    n = 40_000

    home, away = sample_scores(params, fixtures, np.random.default_rng(3), n)
    matrix = score_matrix(params, "A", "B")

    for tore_heim, tore_gast in ((0, 0), (1, 1), (1, 0), (2, 1)):
        gezogen = np.mean((home[:, 0] == tore_heim) & (away[:, 0] == tore_gast))
        erwartet = matrix[tore_heim, tore_gast]
        # drei Standardfehler des Anteils
        assert abs(gezogen - erwartet) < 3 * np.sqrt(erwartet * (1 - erwartet) / n)


def test_monte_carlo_fehler_liegt_im_erwarteten_rahmen():
    """Zwei Seeds duerfen sich nur im Rahmen des Stichprobenfehlers unterscheiden."""
    n = 4000
    laeufe = [
        simulate_season(
            _params(attack={"A": 0.4, "B": 0.1, "C": 0.0, "D": -0.3}),
            _saison(),
            config=SimulationConfig(n_simulations=n, seed=seed),
        ).probability_of_places(1, 1)
        for seed in (11, 12)
    ]

    unterschied = np.abs(laeufe[0] - laeufe[1])
    # Differenz zweier unabhaengiger Anteile: sqrt(2) * 0.5/sqrt(n) als Obergrenze
    assert (unterschied < 4 * np.sqrt(2) * 0.5 / np.sqrt(n)).all()


def test_offenes_spiel_mit_unbekanntem_team_faellt_auf():
    saison = pd.concat([_saison(), _fixtures(("A", "Aufsteiger"))], ignore_index=True)
    with pytest.raises(ValueError, match="ausserhalb der Liga"):
        simulate_season(_params(), saison, teams=TEAMS)


def test_summary_und_platztabelle_sind_vollstaendig():
    forecast = simulate_season(
        _params(), _saison(), config=SimulationConfig(n_simulations=200)
    )
    summary = forecast.summary()
    assert list(summary["team"].sort_values()) == sorted(TEAMS)
    assert (summary["points_p05"] <= summary["expected_points"]).all()
    assert (summary["expected_points"] <= summary["points_p95"]).all()

    tabelle = forecast.position_table()
    assert tabelle.shape == (len(TEAMS), len(TEAMS))
    assert np.allclose(tabelle.to_numpy().sum(), len(TEAMS))
