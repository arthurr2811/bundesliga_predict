"""Kalibrierungs-Check: Stichtage, Zuverlaessigkeit, Intervall-Abdeckung."""

import numpy as np
import pandas as pd
import pytest

from bundesliga_predict.evaluation import calibration
from bundesliga_predict.model.bootstrap import BootstrapConfig
from bundesliga_predict.simulation.season import SimulationConfig
from tests.test_backtest import _synthetic_matches

KLEIN = SimulationConfig(n_simulations=200, seed=5)
OHNE_BOOTSTRAP = BootstrapConfig(n_replicates=0)


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    return _synthetic_matches(n_seasons=3, n_teams=6)


def test_stichtage_beginnen_vor_dem_ersten_spiel_und_lassen_den_letzten_weg(matches):
    saison = sorted(matches["season"].unique())[0]
    frame = matches[matches["season"] == saison]

    punkte = calibration.checkpoint_dates(frame)
    spieltage = [matchday for matchday, _ in punkte]
    letzter = int(frame["matchday"].max())

    assert spieltage[0] == 0
    assert punkte[0][1] < pd.to_datetime(frame["date"]).min()
    assert spieltage == list(range(0, letzter))
    # Am letzten Stichtag muss noch etwas offen sein, sonst ist nichts zu pruefen.
    assert punkte[-1][1] < pd.to_datetime(frame["date"]).max()


def test_stichtag_null_prognostiziert_die_ganze_saison(matches):
    """Vor dem ersten Spiel steht jedes Team bei null Punkten Startguthaben."""
    saison = sorted(matches["season"].unique())[-1]
    config = calibration.CalibrationConfig(
        start_season=saison, matchdays=(0,), simulation=KLEIN, bootstrap=OHNE_BOOTSTRAP
    )
    checkpoints = calibration.run_checkpoints(matches, config)

    assert len(checkpoints) == matches["home_team"].nunique()
    assert set(checkpoints["matchday"]) == {0}
    assert checkpoints["final_points"].sum() > 0


def test_nur_abgeschlossene_saisons_werden_geprueft(matches):
    """Eine laufende Saison hat keinen Ausgang, gegen den sich pruefen liesse."""
    laufend = matches.copy()
    letzte = laufend["season"] == laufend["season"].max()
    laufend.loc[letzte & (laufend["matchday"] > 5), "finished"] = False

    assert laufend["season"].max() not in calibration.complete_seasons(laufend)


def test_ereignisse_summieren_sich_je_stichtag_auf_die_platzzahl(matches):
    """Ueber alle Teams eines Stichtags muss genau ein Meister prognostiziert sein."""
    saison = sorted(matches["season"].unique())[-1]
    checkpoints = calibration.run_checkpoints(
        matches,
        calibration.CalibrationConfig(
            start_season=saison, matchdays=(0, 3), simulation=KLEIN, bootstrap=OHNE_BOOTSTRAP
        ),
    )
    je_stichtag = checkpoints.groupby("matchday")["champion"].sum()

    assert je_stichtag.round(6).eq(1.0).all()


def _outcomes(predicted: list[float], occurred: list[bool]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": "2020/21",
            "matchday": 1,
            "team": [f"Team {i}" for i in range(len(predicted))],
            "event": "champion",
            "predicted": predicted,
            "occurred": occurred,
        }
    )


def test_reliability_zeigt_ueberheblichkeit_als_positive_luecke():
    """Zehnmal 90 % versprochen, fuenfmal eingetreten -- das muss auffallen."""
    outcomes = _outcomes([0.9] * 10, [True] * 5 + [False] * 5)
    table = calibration.reliability(outcomes)

    assert len(table) == 1
    zeile = table.iloc[0]
    assert zeile["n"] == 10
    assert zeile["predicted"] == pytest.approx(0.9)
    assert zeile["observed"] == pytest.approx(0.5)
    assert zeile["gap"] == pytest.approx(0.4)


def test_perfekte_prognosen_haben_keine_luecke():
    outcomes = _outcomes([1.0] * 3 + [0.0] * 3, [True] * 3 + [False] * 3)
    table = calibration.reliability(outcomes)

    assert table["gap"].abs().max() == pytest.approx(0.0)
    assert calibration.brier(outcomes) == pytest.approx(0.0)


def _checkpoints(final_points: list[int], p05: int = 30, p95: int = 50) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": "2020/21",
            "matchday": 0,
            "team": [f"Team {i}" for i in range(len(final_points))],
            "points_p05": p05,
            "points_p95": p95,
            "expected_points": (p05 + p95) / 2,
            "final_points": final_points,
        }
    )


def test_abdeckung_zaehlt_die_treffer_im_intervall():
    # Drei drin (Raender zaehlen mit), einer darunter, einer darueber.
    checkpoints = _checkpoints([30, 40, 50, 29, 51])
    table = calibration.coverage(checkpoints)

    assert table.loc[0, "n"] == 5
    assert table.loc[0, "coverage"] == pytest.approx(0.6)
    assert table.loc[0, "width"] == pytest.approx(20.0)
    assert table.loc[0, "gap"] == pytest.approx(0.6 - calibration.INTERVAL_COVERAGE)


def test_abdeckung_laesst_sich_gruppieren():
    checkpoints = pd.concat(
        [
            _checkpoints([40, 40]).assign(season="2019/20"),
            _checkpoints([10, 90]).assign(season="2020/21"),
        ],
        ignore_index=True,
    )
    table = calibration.coverage(checkpoints, by="season").set_index("season")

    assert table.loc["2019/20", "coverage"] == pytest.approx(1.0)
    assert table.loc["2020/21", "coverage"] == pytest.approx(0.0)


def test_phase_trennt_vor_der_saison_von_spieltag_eins():
    checkpoints = pd.DataFrame({"matchday": [0, 1, 12, 34]})
    labels = calibration.phase(checkpoints)

    assert str(labels.iloc[0]) == "vor Saison"
    assert str(labels.iloc[1]) == "ST 1-8"
    assert labels.notna().all()


def test_kalibrierungslauf_haelt_den_stichtag_ein(matches):
    """Kein Blick nach vorn: der Fit endet am Stichtag."""
    saison = sorted(matches["season"].unique())[-1]
    config = calibration.CalibrationConfig(
        start_season=saison, matchdays=(4,), simulation=KLEIN, bootstrap=OHNE_BOOTSTRAP
    )
    checkpoints = calibration.run_checkpoints(matches, config)
    as_of = checkpoints["as_of"].iloc[0]

    gespielt = matches[
        (matches["season"] == saison) & (pd.to_datetime(matches["date"]) <= as_of)
    ]
    assert set(gespielt["matchday"]) == {1, 2, 3, 4}
    assert np.isfinite(checkpoints["expected_points"]).all()
