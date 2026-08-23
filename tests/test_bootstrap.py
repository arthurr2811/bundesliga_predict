"""Parameter-Unsicherheit"""

import numpy as np
import pandas as pd
import pytest

from dataclasses import replace

from bundesliga_predict.model.bootstrap import (
    BootstrapConfig,
    bootstrap_params,
    jitter_unknown_teams,
    spread,
)
from bundesliga_predict.model.fit import fit
from bundesliga_predict.model.prior import PriorConfig, with_unknown_teams
from bundesliga_predict.simulation.season import (
    SimulationConfig,
    simulate_season,
    split_runs,
)
from tests.test_backtest import _synthetic_matches

WENIGE = BootstrapConfig(n_replicates=8, seed=3)


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    return _synthetic_matches(n_seasons=3, n_teams=6)


def _teams(matches: pd.DataFrame) -> set[str]:
    return set(matches["home_team"]) | set(matches["away_team"])


def test_ziehungen_streuen_um_die_punktschaetzung(matches):
    stichtag = pd.to_datetime(matches["date"]).max()
    saison = matches["season"].max()
    replicates = bootstrap_params(
        matches,
        teams=_teams(matches),
        reference_date=stichtag,
        reference_season=saison,
        config=WENIGE,
    )

    assert len(replicates) == WENIGE.n_replicates
    streuung = spread(replicates)
    assert (streuung["attack_sd"] > 0).all()
    assert (streuung["defense_sd"] > 0).all()

    # Der Mittelwert der Ziehungen darf nicht neben der Punktschaetzung liegen:
    # der Exp(1)-Faktor hat Erwartungswert 1, er verschiebt nichts.
    punkt = fit(matches, reference_date=stichtag, reference_season=saison)
    abweichung = max(
        abs(streuung.set_index("team").loc[team, "attack_mean"] - punkt.attack[team])
        for team in punkt.teams
    )
    assert abweichung < 0.25


def test_gleicher_seed_gleiche_ziehungen(matches):
    argumente = dict(
        teams=_teams(matches),
        reference_date=pd.to_datetime(matches["date"]).max(),
        reference_season=matches["season"].max(),
        config=WENIGE,
    )
    erste = bootstrap_params(matches, **argumente)
    zweite = bootstrap_params(matches, **argumente)

    assert [p.attack for p in erste] == [p.attack for p in zweite]


def test_zusatzunsicherheit_haengt_an_der_zahl_offener_spiele(matches):
    saison = matches["season"].max()
    teams = sorted(_teams(matches))
    letzter = int(matches["matchday"].max())
    config = SimulationConfig(n_simulations=2000, seed=6)

    punkt = with_unknown_teams(
        fit(matches, reference_date=pd.to_datetime(matches["date"]).max()),
        set(teams),
        PriorConfig(),
    )
    # Zehn Ligen, in denen die erste Haelfte der Teams mal deutlich staerker
    # und mal deutlich schwaecher ist als geschaetzt.
    gespreizt = [
        replace(
            punkt,
            attack={
                team: value + (versatz if number % 2 else -versatz)
                for number, (team, value) in enumerate(punkt.attack.items())
            },
        )
        for versatz in np.linspace(-0.3, 0.3, 10)
    ]

    def zusatzstreuung(gespielte_spieltage: int) -> float:
        stand = matches[matches["season"] == saison].copy()
        stand.loc[stand["matchday"] > gespielte_spieltage, "finished"] = False
        ohne = simulate_season(punkt, stand, teams=teams, config=config)
        mit = simulate_season(gespreizt, stand, teams=teams, config=config)
        return mit.points.std(axis=0).mean() - ohne.points.std(axis=0).mean()

    frueh = zusatzstreuung(0)
    spaet = zusatzstreuung(letzter - 1)

    # Vor Saisonstart klar messbar, am vorletzten Spieltag praktisch weg.
    assert frueh > 0.3
    assert spaet < 0.15 * frueh


def test_teams_ohne_historie_bekommen_eigene_streuung(matches):
    """Aufsteiger kommen im Fit nicht vor -- ohne Jitter haetten sie keine."""
    stichtag = pd.to_datetime(matches["date"]).max()
    saison = matches["season"].max()
    mit_aufsteiger = _teams(matches) | {"Aufsteiger"}

    replicates = bootstrap_params(
        matches,
        teams=mit_aufsteiger,
        reference_date=stichtag,
        reference_season=saison,
        config=WENIGE,
    )
    streuung = spread(replicates).set_index("team")

    assert "Aufsteiger" in streuung.index
    assert streuung.loc["Aufsteiger", "attack_sd"] > 0


def test_jitter_laesst_bekannte_teams_unberuehrt(matches):
    params = fit(matches, reference_date=pd.to_datetime(matches["date"]).max())
    ergaenzt = with_unknown_teams(params, set(params.teams) | {"Neu"}, PriorConfig())
    gestreut = jitter_unknown_teams(
        ergaenzt, {"Neu"}, np.random.default_rng(0), WENIGE
    )

    for team in params.teams:
        assert gestreut.attack[team] == ergaenzt.attack[team]
    assert gestreut.attack["Neu"] != ergaenzt.attack["Neu"]


def test_abgeschalteter_bootstrap_ist_inaktiv():
    assert not BootstrapConfig(n_replicates=0).active
    assert not BootstrapConfig(n_replicates=1).active
    assert BootstrapConfig(n_replicates=2).active


def test_laeufe_werden_restlos_verteilt():
    assert sum(split_runs(10_000, 100)) == 10_000
    assert sum(split_runs(1001, 100)) == 1001
    # Der Rest geht an die ersten Saetze, der Unterschied bleibt bei eins.
    verteilung = split_runs(1001, 100)
    assert max(verteilung) - min(verteilung) == 1

    with pytest.raises(ValueError, match="Parametersaetze"):
        split_runs(5, 10)


def test_bootstrap_verbreitert_die_punkteverteilung(matches):
    """Die eigentliche Wirkung: mehr Unsicherheit, nicht bloss andere Zahlen."""
    saison = matches["season"].max()
    stand = matches[matches["season"] == saison].copy()
    offen = stand["matchday"] > 3
    stand.loc[offen, "finished"] = False
    teams = sorted(_teams(matches))

    stichtag = pd.to_datetime(stand.loc[~offen, "date"]).max()
    punkt = with_unknown_teams(
        fit(matches, reference_date=stichtag, reference_season=saison),
        set(teams),
        PriorConfig(),
    )
    replicates = bootstrap_params(
        matches,
        teams=set(teams),
        reference_date=stichtag,
        reference_season=saison,
        config=BootstrapConfig(n_replicates=20, seed=11),
    )

    config = SimulationConfig(n_simulations=2000, seed=4)
    ohne = simulate_season(punkt, stand, teams=teams, config=config)
    mit = simulate_season(replicates, stand, teams=teams, config=config)

    assert ohne.points.shape == mit.points.shape
    assert mit.points.std(axis=0).mean() > ohne.points.std(axis=0).mean()
