"""Blockbildung und Walk-forward.
"""

import numpy as np
import pandas as pd
import pytest

from bundesliga_predict.evaluation.backtest import (
    BacktestConfig,
    assign_blocks,
    run_backtest,
)


def _round_robin(teams: list[str]) -> list[list[tuple[str, str]]]:
    """Spielplan einer Hinrunde nach dem Kreisverfahren."""
    if len(teams) % 2:
        raise ValueError("Kreisverfahren braucht eine gerade Teamzahl.")

    rotation = list(teams)
    half = len(teams) // 2
    matchdays = []
    for round_index in range(len(teams) - 1):
        pairs = [(rotation[i], rotation[-1 - i]) for i in range(half)]
        # Das erste Paar enthält immer das feste Team -- Heimrecht abwechseln,
        # sonst spielt es die ganze Hinrunde zu Hause.
        if round_index % 2:
            pairs[0] = (pairs[0][1], pairs[0][0])
        matchdays.append(pairs)
        # Erstes Team bleibt stehen, der Rest rotiert um eine Position.
        rotation = [rotation[0], rotation[-1], *rotation[1:-1]]
    return matchdays


def _synthetic_matches(n_seasons: int = 3, n_teams: int = 6, seed: int = 7) -> pd.DataFrame:
    """Kleine Kunst-Liga: jede Saison Hin- und Rückrunde, ein Spieltag pro Woche.

    Die Spieltage laufen über zwei aufeinanderfolgende Tage, damit die
    Blockbildung an derselben Struktur getestet wird wie auf echten Daten.
    """
    rng = np.random.default_rng(seed)
    teams = [f"Team {i}" for i in range(n_teams)]
    first_half = _round_robin(teams)
    schedule = first_half + [[(away, home) for home, away in day] for day in first_half]

    rows = []
    for season_index in range(n_seasons):
        start_year = 2016 + season_index
        season = f"{start_year}/{str(start_year + 1)[-2:]}"
        kickoff = pd.Timestamp(f"{start_year}-08-15")
        for matchday, pairs in enumerate(schedule, start=1):
            saturday = kickoff + pd.Timedelta(days=7 * (matchday - 1))
            for position, (home, away) in enumerate(pairs):
                rows.append(
                    {
                        "season": season,
                        # Erste Partie am Freitag, der Rest am Samstag.
                        "date": saturday - pd.Timedelta(days=1 if position == 0 else 0),
                        "matchday": matchday,
                        "home_team": home,
                        "away_team": away,
                        "home_goals": int(rng.poisson(1.6)),
                        "away_goals": int(rng.poisson(1.2)),
                        "finished": True,
                    }
                )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _calendar(rows: list[tuple[str, int, str]]) -> pd.DataFrame:
    """Hilfskonstruktor: (Saison, Spieltag, Datum)."""
    frame = pd.DataFrame(rows, columns=["season", "matchday", "date"])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def test_gestreckter_spieltag_bleibt_ein_block():
    """Freitag bis Montag ist ein Spieltag, also ein Vorhersage-Zeitpunkt."""
    matches = _calendar(
        [
            ("2016/17", 1, "2016-08-26"),
            ("2016/17", 1, "2016-08-27"),
            ("2016/17", 1, "2016-08-29"),
        ]
    )
    assert list(assign_blocks(matches)) == [0, 0, 0]


def test_englische_woche_ist_kein_sonderfall_mehr():
    """Sonntag -> Dienstag, nur zwei Tage Abstand, aber zwei Spieltage.
    """
    matches = _calendar(
        [
            ("2016/17", 5, "2016-09-18"),
            ("2016/17", 6, "2016-09-20"),
        ]
    )
    assert list(assign_blocks(matches)) == [0, 1]


def test_saisonwechsel_trennt():
    matches = _calendar(
        [
            ("2016/17", 34, "2017-05-20"),
            ("2017/18", 1, "2017-05-21"),
        ]
    )
    assert list(assign_blocks(matches)) == [0, 1]


def test_verlegte_partie_wird_abgetrennt():
    """Wochen später nachgeholt heisst: eigener Zeitpunkt, eigener Fit."""
    matches = _calendar(
        [
            ("2016/17", 13, "2016-12-02"),
            ("2016/17", 13, "2016-12-03"),
            ("2016/17", 13, "2017-01-24"),  # nachgeholt
        ]
    )
    assert list(assign_blocks(matches)) == [0, 0, 1]


def test_bloecke_sind_chronologisch_nummeriert():
    """Die nachgeholte Partie liegt hinter dem folgenden Spieltag.

    Sonst würde der Walk-forward sie zu früh abarbeiten und mit veralteten
    Parametern vorhersagen.
    """
    matches = _calendar(
        [
            ("2016/17", 13, "2016-12-02"),
            ("2016/17", 14, "2016-12-09"),
            ("2016/17", 13, "2017-01-24"),  # nachgeholt, aber zuletzt gespielt
        ]
    )
    assert list(assign_blocks(matches)) == [0, 1, 2]


def test_fehlende_spieltagsnummer_faellt_auf():
    matches = _calendar([("2016/17", 1, "2016-08-26")])
    matches["matchday"] = pd.NA
    with pytest.raises(ValueError, match="Spieltagsnummer fehlt"):
        assign_blocks(matches)


def test_kein_block_mischt_spieltage():
    matches = _synthetic_matches()
    matches["block"] = assign_blocks(matches)
    for _, block in matches.groupby("block"):
        assert block["matchday"].nunique() == 1
        assert block["season"].nunique() == 1


def test_backtest_liefert_gueltige_wahrscheinlichkeiten():
    predictions = run_backtest(
        _synthetic_matches(), BacktestConfig(start_season="2018/19", min_history=30)
    )

    assert not predictions.empty
    probabilities = predictions[["p_home", "p_draw", "p_away"]].to_numpy()
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert (probabilities > 0).all()
    assert np.allclose(
        predictions[["base_home", "base_draw", "base_away"]].to_numpy().sum(axis=1), 1.0
    )


def test_backtest_bewertet_jedes_spiel_der_zielsaisons_genau_einmal():
    matches = _synthetic_matches()
    predictions = run_backtest(
        matches, BacktestConfig(start_season="2018/19", min_history=30)
    )

    erwartet = matches[matches["season"] >= "2018/19"]
    assert len(predictions) == len(erwartet)
    paare = set(zip(predictions["home_team"], predictions["away_team"], predictions["date"]))
    assert len(paare) == len(predictions)


def test_kein_blick_in_die_zukunft(monkeypatch):
    """Der Fit darf nie Spiele ab dem Vorhersageblock zu sehen bekommen."""
    from bundesliga_predict.evaluation import backtest as backtest_module

    echter_fit = backtest_module.fit
    gesehen = []

    def spion(matches, reference_date=None, **kwargs):
        gesehen.append((pd.to_datetime(matches["date"]).max(), pd.Timestamp(reference_date)))
        return echter_fit(matches, reference_date=reference_date, **kwargs)

    monkeypatch.setattr(backtest_module, "fit", spion)
    run_backtest(_synthetic_matches(), BacktestConfig(start_season="2018/19", min_history=30))

    assert gesehen
    for letztes_spiel, stichtag in gesehen:
        assert letztes_spiel <= stichtag


def test_team_ohne_historie_wird_als_durchschnitt_behandelt():
    """Ein Aufsteiger ohne einzige Partie darf den Lauf nicht abbrechen."""
    matches = _synthetic_matches()
    letzte_saison = matches["season"].max()
    neu = matches[matches["season"] == letzte_saison].replace({"Team 0": "Aufsteiger"})
    matches = pd.concat(
        [matches[matches["season"] != letzte_saison], neu], ignore_index=True
    )

    predictions = run_backtest(
        matches, BacktestConfig(start_season=letzte_saison, min_history=30)
    )
    aufsteiger = predictions[
        (predictions["home_team"] == "Aufsteiger")
        | (predictions["away_team"] == "Aufsteiger")
    ]
    assert not aufsteiger.empty
    assert aufsteiger[["p_home", "p_draw", "p_away"]].notna().all().all()


def test_end_season_schneidet_nur_die_auswertung():
    """Der Holdout-Schnitt darf die Historie nicht anfassen.

    Sonst waeren die Tuning-Ergebnisse nicht mit dem Volllauf vergleichbar:
    ein Block muss auf allem gefittet werden, was vor ihm liegt, auch wenn
    spaetere Saisons von der Bewertung ausgenommen sind.
    """
    matches = _synthetic_matches(n_seasons=3)
    saisons = sorted(matches["season"].unique())
    voll = run_backtest(matches, BacktestConfig(start_season=saisons[0], min_history=30))
    geschnitten = run_backtest(
        matches,
        BacktestConfig(start_season=saisons[0], end_season=saisons[-2], min_history=30),
    )

    assert (geschnitten["season"] <= saisons[-2]).all()
    assert len(geschnitten) < len(voll)

    # Die verbleibenden Vorhersagen muessen Zahl fuer Zahl dieselben sein.
    spalten = ["p_home", "p_draw", "p_away"]
    erwartet = voll[voll["season"] <= saisons[-2]].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        geschnitten[spalten].reset_index(drop=True), erwartet[spalten]
    )


def test_saison_ohne_daten_faellt_auf():
    with pytest.raises(ValueError):
        run_backtest(_synthetic_matches(n_seasons=1), BacktestConfig(start_season="2030/31"))
