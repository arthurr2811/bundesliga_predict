"""Grid-Search Test
"""

import pandas as pd

from bundesliga_predict.evaluation import tuning
from bundesliga_predict.evaluation.tuning import Combination


def _grid(**overrides) -> dict[str, tuple]:
    grid = {
        "half_life_days": (90.0, 180.0),
        "season_penalty": (0.8,),
        "prior_sd": (0.35,),
        "prior_scale": (0.0, 1.0, 2.0),
    }
    grid.update(overrides)
    return grid


def test_expand_grid_liefert_das_kartesische_produkt():
    combinations = tuning.expand_grid(_grid())
    assert len(combinations) == 2 * 1 * 1 * 3
    assert len(set(combinations)) == len(combinations)
    assert Combination(90.0, 0.8, 0.35, 1.0) in combinations


def test_prior_scale_skaliert_beide_mittelwerte():
    """Eine Achse, zwei Werte -- das Verhaeltnis bleibt das gemessene."""
    from bundesliga_predict.config import DEFAULT_PRIOR_ATTACK, DEFAULT_PRIOR_DEFENSE

    doppelt = Combination(180.0, 0.8, 0.35, 2.0).to_backtest_config(None).prior
    assert doppelt.attack_mean == 2.0 * DEFAULT_PRIOR_ATTACK
    assert doppelt.defense_mean == 2.0 * DEFAULT_PRIOR_DEFENSE

    # Skalierung 0 ist der alte Zustand: Ziel ist der Ligadurchschnitt.
    neutral = Combination(180.0, 0.8, 0.35, 0.0).to_backtest_config(None).prior
    assert neutral.attack_mean == 0.0
    assert neutral.defense_mean == 0.0


def test_end_season_landet_in_der_backtest_config():
    config = Combination(180.0, 0.8, 0.35, 1.0).to_backtest_config("2023/24")
    assert config.end_season == "2023/24"
    assert config.weight_config.half_life_days == 180.0
    assert config.prior.sd == 0.35


def test_fertige_kombinationen_werden_uebersprungen(tmp_path):
    """Wiederaufnahme: was in der CSV steht, wird nicht neu gerechnet."""
    path = tmp_path / "results.csv"
    fields = list(Combination.__dataclass_fields__)
    schon_da = Combination(90.0, 0.8, 0.35, 1.0)
    pd.DataFrame([{**{f: getattr(schon_da, f) for f in fields}, "rps": 0.2}]).to_csv(
        path, index=False
    )

    done = tuning._completed_combinations(path, fields)
    assert tuple(float(getattr(schon_da, f)) for f in fields) in done
    assert len(done) == 1


def test_leere_oder_fehlende_datei_blockiert_nicht(tmp_path):
    fields = list(Combination.__dataclass_fields__)
    assert tuning._completed_combinations(tmp_path / "fehlt.csv", fields) == set()

    leer = tmp_path / "leer.csv"
    leer.write_text("", encoding="utf-8")
    assert tuning._completed_combinations(leer, fields) == set()


def test_best_liefert_das_plateau_nicht_nur_das_minimum():
    results = pd.DataFrame(
        {
            "half_life_days": [90.0, 180.0, 365.0, 730.0],
            "rps": [0.2000, 0.2003, 0.2004, 0.2100],
        }
    )
    plateau = tuning.best(results, within=0.0005)

    # Drei Werte liegen innerhalb der Rauschbreite, der vierte nicht.
    assert list(plateau["rps"]) == [0.2000, 0.2003, 0.2004]
    assert plateau.iloc[0]["rps"] == 0.2000  # bestes zuerst
