"""Die Kette fit -> predict -> simulate -> JSON."""

import json

import numpy as np
import pandas as pd
import pytest

from bundesliga_predict import pipeline
from bundesliga_predict.simulation.season import SimulationConfig
from tests.test_backtest import _synthetic_matches

KLEIN = SimulationConfig(n_simulations=200, seed=5)


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    """Drei Kunst-Saisons, die letzte ab der Haelfte noch offen."""
    frame = _synthetic_matches(n_seasons=3, n_teams=6)
    letzte = frame["season"] == frame["season"].max()
    offen = letzte & (frame["matchday"] > 5)
    frame.loc[offen, ["home_goals", "away_goals"]] = np.nan
    frame.loc[offen, "finished"] = False
    return frame


def test_zielsaison_ist_die_naechste_in_der_gespielt_wird(matches):
    saisons = sorted(matches["season"].unique())
    mitten_drin = pd.Timestamp(matches.loc[matches["season"] == saisons[1], "date"].iloc[0])

    assert pipeline.target_season(matches, mitten_drin) == saisons[1]
    # Nach dem letzten Spiel ueberhaupt bleibt nur die letzte Saison uebrig.
    assert pipeline.target_season(matches, pd.Timestamp("2100-01-01")) == saisons[-1]


def test_zwischen_zwei_saisons_zeigt_der_stichtag_auf_die_kommende(matches):
    saisons = sorted(matches["season"].unique())
    ende = pd.to_datetime(matches.loc[matches["season"] == saisons[0], "date"]).max()

    assert pipeline.target_season(matches, ende + pd.Timedelta(days=1)) == saisons[1]


def test_stichtag_macht_gespielte_partien_wieder_offen(matches):
    """Der Kern von --as-of: was nach dem Stichtag liegt, gilt als ungespielt."""
    saison = sorted(matches["season"].unique())[0]
    termine = pd.to_datetime(matches.loc[matches["season"] == saison, "date"])
    stichtag = termine.min() + pd.Timedelta(days=30)

    stand = pipeline.season_state(matches, saison, stichtag)
    assert stand["finished"].sum() == (termine <= stichtag).sum()
    assert not stand.loc[stand["date"] > stichtag, "finished"].any()


def test_lauf_liefert_vorhersagen_fuer_jedes_offene_spiel(matches):
    run = pipeline.run_forecast(matches, simulation=KLEIN)

    offen = (~run.matches["finished"]).sum()
    assert len(run.predictions) == offen
    wahrscheinlichkeiten = run.predictions[["p_home", "p_draw", "p_away"]].to_numpy()
    assert np.allclose(wahrscheinlichkeiten.sum(axis=1), 1.0)


def test_vor_dem_ersten_spieltag_ist_die_ganze_saison_offen(matches):
    saison = sorted(matches["season"].unique())[-1]
    anpfiff = pd.to_datetime(matches.loc[matches["season"] == saison, "date"]).min()
    run = pipeline.run_forecast(
        matches, as_of=anpfiff - pd.Timedelta(days=1), simulation=KLEIN
    )

    assert run.season == saison
    assert (~run.matches["finished"]).all()
    assert (run.forecast.points.min(axis=0) >= 0).all()


def test_spiel_am_stichtag_zaehlt_als_gespielt(matches):
    """Der Lauf am Sonntagabend muss den Spieltag schon kennen."""
    saison = sorted(matches["season"].unique())[-1]
    anpfiff = pd.to_datetime(matches.loc[matches["season"] == saison, "date"]).min()
    run = pipeline.run_forecast(matches, as_of=anpfiff, simulation=KLEIN)

    gespielt = run.matches[run.matches["finished"]]
    assert len(gespielt) == 1
    assert gespielt["date"].max() == anpfiff


def test_gleicher_stichtag_gleiche_prognose(matches):
    laeufe = [
        pipeline.run_forecast(matches, as_of="2018-10-01", simulation=KLEIN)
        for _ in range(2)
    ]
    assert np.array_equal(laeufe[0].forecast.position, laeufe[1].forecast.position)


def test_payload_enthaelt_alle_vier_dokumente(matches):
    payload = pipeline.to_payload(pipeline.run_forecast(matches, simulation=KLEIN))
    assert set(payload) == {"meta", "matches", "table", "probabilities"}

    meta = payload["meta"]
    assert meta["matches_played"] + meta["matches_open"] == len(payload["matches"])
    assert meta["n_simulations"] == KLEIN.n_simulations


def test_gespielte_partien_tragen_das_ergebnis_offene_die_vorhersage(matches):
    payload = pipeline.to_payload(pipeline.run_forecast(matches, simulation=KLEIN))

    gespielt = [m for m in payload["matches"] if m["finished"]]
    offen = [m for m in payload["matches"] if not m["finished"]]
    assert gespielt and offen
    assert all("home_goals" in m and "p_home" not in m for m in gespielt)
    assert all("likely_score" in m and "home_goals" not in m for m in offen)


def test_platzverteilung_im_payload_summiert_je_team_auf_eins(matches):
    payload = pipeline.to_payload(pipeline.run_forecast(matches, simulation=KLEIN))

    for team in payload["probabilities"]:
        assert sum(team["positions"]) == pytest.approx(1.0, abs=1e-6)
        assert team["champion"] == pytest.approx(team["positions"][0], abs=1e-6)


def test_write_payload_schreibt_lesbares_json(matches, tmp_path):
    payload = pipeline.to_payload(pipeline.run_forecast(matches, simulation=KLEIN))
    geschrieben = pipeline.write_payload(payload, tmp_path)

    assert {pfad.name for pfad in geschrieben} == {
        "meta.json",
        "matches.json",
        "table.json",
        "probabilities.json",
    }
    for pfad in geschrieben:
        assert json.loads(pfad.read_text(encoding="utf-8"))
