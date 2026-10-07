"""Datenbeschaffung ohne Roh-CSVs, wie in GitHub Actions."""

from datetime import date

import pandas as pd
import pytest

from bundesliga_predict import build_dataset


@pytest.mark.parametrize(
    "heute, erwartet",
    [
        (date(2026, 10, 7), 2026),
        (date(2027, 5, 20), 2026),
        (date(2027, 6, 30), 2026),
        (date(2027, 7, 1), 2027),
    ],
)
def test_laufende_saison_wechselt_im_juli(heute, erwartet):
    assert build_dataset.current_season(heute) == erwartet


def _historic(path):
    pd.DataFrame(
        {
            "season": ["2025/26", "2025/26"],
            "date": ["2026-05-16", "2025-08-22"],
            "matchday": [34, 1],
            "home_team": ["Bayern Munich", "Dortmund"],
            "away_team": ["Dortmund", "Bayern Munich"],
            "home_goals": [2, 1],
            "away_goals": [1, 1],
            "finished": [True, True],
        }
    ).to_csv(path, index=False)


def _live_match(season):
    return {
        "leagueSeason": season,
        "matchDateTime": f"{season}-08-28T20:30:00",
        "group": {"groupOrderID": 1},
        "team1": {"teamName": "FC Bayern München"},
        "team2": {"teamName": "Borussia Dortmund"},
        "matchResults": [],
        "matchIsFinished": False,
    }


@pytest.fixture
def ohne_rohdaten(tmp_path, monkeypatch):
    """Leeres Rohdaten-Verzeichnis, Historie nur aus historic.csv, kein Netz."""
    raw = tmp_path / "raw"
    raw.mkdir()
    historic = tmp_path / "historic.csv"
    _historic(historic)
    monkeypatch.setattr(build_dataset, "_RAW_HISTORIC_DIR", raw)
    monkeypatch.setattr(build_dataset, "HISTORIC_PATH", historic)

    abgerufen = []

    def fetch_live(season):
        abgerufen.append(season)
        # Spielplan 2027/28 noch nicht veroeffentlicht.
        return [] if season == 2027 else [_live_match(season)]

    monkeypatch.setattr(build_dataset, "fetch_live", fetch_live)
    return abgerufen


def test_ohne_rohdaten_kommt_die_historie_aus_historic_csv(ohne_rohdaten):
    dataset = build_dataset.build_dataset(season=2026)

    assert ohne_rohdaten == [2026]
    assert list(dataset.columns) == build_dataset.COLUMNS
    assert list(dataset["season"]) == ["2025/26", "2025/26", "2026/27"]
    # Datum als echtes Datum, sonst sortiert die Mischung mit OpenLigaDB falsch.
    assert dataset["date"].is_monotonic_increasing
    assert dataset["home_goals"].dtype == "Int64"


def test_nach_saisonwechsel_wird_die_luecke_live_geholt(ohne_rohdaten):
    dataset = build_dataset.build_dataset(season=2027)

    assert ohne_rohdaten == [2026, 2027]
    assert set(dataset["season"]) == {"2025/26", "2026/27"}


def test_ohne_jede_historie_bricht_der_lauf_ab(ohne_rohdaten, tmp_path, monkeypatch):
    monkeypatch.setattr(build_dataset, "HISTORIC_PATH", tmp_path / "fehlt.csv")
    with pytest.raises(SystemExit):
        build_dataset.build_dataset(season=2026)
