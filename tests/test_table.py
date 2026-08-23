"""Tabellenberechnung.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bundesliga_predict.simulation.table import positions, standings, team_records

_ROOT = Path(__file__).resolve().parents[1]
MATCHES_PATH = _ROOT / "data" / "processed" / "matches.csv"
FINAL_TABLES_PATH = Path(__file__).resolve().parent / "data" / "final_tables.csv"

VERGLEICHSSPALTEN = [
    "position",
    "team",
    "played",
    "won",
    "drawn",
    "lost",
    "goals_for",
    "goals_against",
    "goal_difference",
    "points",
]


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    if not MATCHES_PATH.exists():
        pytest.skip("matches.csv nicht gebaut")
    return pd.read_csv(MATCHES_PATH)


@pytest.fixture(scope="module")
def echte_tabellen() -> pd.DataFrame:
    return pd.read_csv(FINAL_TABLES_PATH)


def _spiele(*ergebnisse: tuple[str, str, int, int]) -> pd.DataFrame:
    """Hilfskonstruktor: (Heim, Gast, Heimtore, Gasttore)."""
    return pd.DataFrame(
        [
            {
                "home_team": home,
                "away_team": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "finished": True,
            }
            for home, away, home_goals, away_goals in ergebnisse
        ],
        columns=["home_team", "away_team", "home_goals", "away_goals", "finished"],
    )


def test_abschlusstabellen_stimmen_mit_openligadb_ueberein(matches, echte_tabellen):
    for season, echt in echte_tabellen.groupby("season"):
        spiele = matches[matches["season"] == season]
        assert len(spiele) == 306, f"{season}: unvollstaendige Saison"

        berechnet = standings(spiele)
        pd.testing.assert_frame_equal(
            berechnet[VERGLEICHSSPALTEN],
            echt.sort_values("position")[VERGLEICHSSPALTEN].reset_index(drop=True),
            check_dtype=False,
            obj=f"Abschlusstabelle {season}",
        )


def test_punkte_und_tore_werden_richtig_gezaehlt():
    tabelle = standings(_spiele(("A", "B", 3, 1), ("B", "C", 2, 2), ("C", "A", 0, 1)))

    a = tabelle.set_index("team").loc["A"]
    assert (a["points"], a["won"], a["goals_for"], a["goals_against"]) == (6, 2, 4, 1)
    b = tabelle.set_index("team").loc["B"]
    assert (b["points"], b["drawn"], b["lost"], b["goal_difference"]) == (1, 1, 1, -2)
    # B und C haben je einen Punkt, C die bessere Differenz (-1 gegen -2).
    assert list(tabelle["team"]) == ["A", "C", "B"]


def test_tordifferenz_schlaegt_erzielte_tore():
    """Gleiche Punkte: erst Differenz, dann erzielte Tore."""
    tabelle = standings(
        _spiele(("A", "X", 1, 0), ("B", "X", 5, 4), ("C", "X", 4, 3), ("X", "A", 0, 0))
    )
    assert list(tabelle["team"][:3]) == ["A", "B", "C"]


def test_offene_spiele_zaehlen_nicht():
    spiele = _spiele(("A", "B", 3, 1))
    offen = pd.DataFrame(
        [{"home_team": "B", "away_team": "A", "home_goals": np.nan,
          "away_goals": np.nan, "finished": False}]
    )
    tabelle = standings(pd.concat([spiele, offen], ignore_index=True))
    assert list(tabelle["played"]) == [1, 1]


def test_team_ohne_spiel_steht_mit_nullen_drin():
    """Vor dem ersten Spieltag hat niemand gespielt -- die Tabelle gibt es trotzdem."""
    tabelle = standings(_spiele(), teams=["A", "B", "C"])

    assert len(tabelle) == 3
    assert (tabelle["played"] == 0).all()
    assert (tabelle["points"] == 0).all()
    assert list(tabelle["position"]) == [1, 2, 3]


def test_unbekanntes_team_faellt_auf():
    with pytest.raises(ValueError, match="ausserhalb der Liga"):
        team_records(_spiele(("A", "B", 1, 0)), teams=["A"])


def test_positions_sortiert_viele_tabellen_auf_einmal():
    """Dieselbe Regel, aber auf (n_simulationen, n_teams) -- so nutzt sie die Simulation."""
    punkte = np.array([[10, 20, 15], [30, 5, 5]])
    differenz = np.array([[0, 0, 0], [0, 3, 1]])
    tore = np.array([[0, 0, 0], [0, 0, 0]])

    assert positions(punkte, differenz, tore).tolist() == [[3, 1, 2], [1, 2, 3]]


def test_gleichstand_wird_ohne_tiebreak_deterministisch_aufgeloest():
    gleich = np.zeros((4, 3), dtype=int)
    ergebnis = positions(gleich, gleich, gleich)
    assert ergebnis.tolist() == [[1, 2, 3]] * 4


def test_tiebreak_entscheidet_bei_voelligem_gleichstand():
    gleich = np.zeros(3, dtype=int)
    assert positions(gleich, gleich, gleich, tiebreak=np.array([0.9, 0.1, 0.5])).tolist() == [
        3,
        1,
        2,
    ]
