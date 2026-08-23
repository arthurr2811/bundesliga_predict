"""Grid-Search der Hyperparameter ueber den Backtest.

Halbwertszeit, Saisonwechsel-Malus und die beiden Prior-Groessen

1. **Zielgroesse ist RPS.** Log-Loss und Brier laufen mit, sind aber Kontrolle
   und kein Auswahlkriterium.
2. **Getunt wird nur auf einem Teil der Saisons.** Die juengsten bleiben als
   Holdout unangetastet (`end_season`).
3. **Gewaehlt wird aus dem Plateau, nicht das Argmin.** Deshalb steht in jeder
   Ergebniszeile auch die RPS je Saison: ein Sieger, der nur eine Saison
   traegt, ist keiner.
"""

from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from bundesliga_predict.config import (
    DEFAULT_PRIOR_ATTACK,
    DEFAULT_PRIOR_DEFENSE,
    PRIOR_MATCH_WEIGHT,
)
from bundesliga_predict.evaluation import metrics
from bundesliga_predict.evaluation.backtest import (
    BacktestConfig,
    baseline_probabilities,
    model_probabilities,
    run_backtest,
)
from bundesliga_predict.model.prior import PriorConfig
from bundesliga_predict.model.weights import WeightConfig

# Saisons, auf denen getunt werden darf. Alles danach ist Holdout.
TUNING_END_SEASON = "2023/24"

# Stufe A: bewusst breit statt fein. Verfeinert wird spaeter um den Sieger.
GRID_STAGE_A: dict[str, tuple] = {
    # `inf` heisst "gar kein Zeitzerfall" -- ein Ankerpunkt, kein Kandidat:
    # zeigt, wie viel die Zeitgewichtung ueberhaupt beitraegt.
    "half_life_days": (60.0, 120.0, 240.0, 480.0, float("inf")),
    "season_penalty": (0.6, 0.8, 1.0),
    "prior_sd": (0.15, 0.25, 0.40),
    # Faktor auf die gemessenen Aufsteiger-Mittelwerte. Eine Achse statt zwei:
    # das Verhaeltnis Angriff/Abwehr kommt aus der Messung, offen ist nur die
    # Staerke.
    "prior_scale": (0.0, 1.0, 2.0, 3.0),
}

# Stufe B: eng um den Sieger aus Stufe A, plus `prior_match_weight` als
# fuenfte Achse.
GRID_STAGE_B: dict[str, tuple] = {
    "half_life_days": (240.0, 360.0, 480.0, 720.0),
    "season_penalty": (0.5, 0.65, 0.8),
    "prior_sd": (0.08, 0.15, 0.25),
    "prior_scale": (2.0, 3.0, 4.0),
    "prior_match_weight": (8.0, 17.0, 34.0),
}


@dataclass(frozen=True)
class Combination:
    """Ein Punkt im Grid. Die Feldnamen sind zugleich die CSV-Spalten."""

    half_life_days: float
    season_penalty: float
    prior_sd: float
    prior_scale: float
    # Default = der bisherige Festwert, damit Grids ohne diese Achse (Stufe A)
    # unveraendert funktionieren.
    prior_match_weight: float = PRIOR_MATCH_WEIGHT

    def to_backtest_config(self, end_season: str | None) -> BacktestConfig:
        return BacktestConfig(
            end_season=end_season,
            weight_config=WeightConfig(
                half_life_days=self.half_life_days, season_penalty=self.season_penalty
            ),
            prior=PriorConfig(
                sd=self.prior_sd,
                attack_mean=DEFAULT_PRIOR_ATTACK * self.prior_scale,
                defense_mean=DEFAULT_PRIOR_DEFENSE * self.prior_scale,
                match_weight=self.prior_match_weight,
            ),
        )


def expand_grid(grid: dict[str, tuple]) -> list[Combination]:
    """Kartesisches Produkt der Achsen, in stabiler Reihenfolge."""
    names = list(grid)
    return [
        Combination(**dict(zip(names, values)))
        for values in itertools.product(*(grid[name] for name in names))
    ]


def evaluate(
    combination: Combination, matches: pd.DataFrame, end_season: str | None
) -> dict:
    """Ein Backtest-Lauf, zu einer Ergebniszeile verdichtet."""
    predictions = run_backtest(matches, combination.to_backtest_config(end_season))
    outcomes = metrics.outcome_index(
        predictions["home_goals"], predictions["away_goals"]
    )
    model = metrics.score(model_probabilities(predictions), outcomes)
    baseline = metrics.score(baseline_probabilities(predictions), outcomes)

    row = {
        **asdict(combination),
        "n_matches": model.n_matches,
        "rps": model.rps,
        "log_loss": model.log_loss,
        "brier": model.brier,
        "rps_baseline": baseline.rps,
    }

    # RPS je Saison: zeigt, ob ein Sieger nur von einer Saison getragen wird.
    seasons = predictions["season"].to_numpy()
    for season in sorted(set(seasons)):
        selection = seasons == season
        row[f"rps_{season.replace('/', '_')}"] = metrics.score(
            model_probabilities(predictions)[selection], outcomes[selection]
        ).rps
    return row


# Jeder Worker-Prozess laedt den Datensatz einmal und behaelt ihn.
_MATCHES: pd.DataFrame | None = None


def _worker_init(matches_path: str) -> None:
    global _MATCHES
    _MATCHES = pd.read_csv(matches_path)


def _worker_evaluate(combination: Combination, end_season: str | None) -> dict:
    assert _MATCHES is not None, "Worker ohne Datensatz gestartet."
    return evaluate(combination, _MATCHES, end_season)


def _completed_combinations(path: Path, fields: list[str]) -> set[tuple]:
    """Bereits gerechnete Kombinationen aus einer vorhandenen Ergebnis-CSV."""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    done = pd.read_csv(path)
    if not set(fields) <= set(done.columns):
        return set()
    return {tuple(float(row[field]) for field in fields) for _, row in done.iterrows()}


def run_grid(
    matches_path: Path,
    output_path: Path,
    grid: dict[str, tuple] | None = None,
    end_season: str | None = TUNING_END_SEASON,
    workers: int = 6,
    limit: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Faehrt den Grid ab und schreibt jede fertige Zeile sofort weg.

    Zeilenweises Anhaengen ist Absicht: ein Abbruch nach vierzig Minuten soll
    nichts kosten. Ein erneuter Aufruf ueberspringt, was schon in der CSV
    steht, der Lauf ist also beliebig oft fortsetzbar.
    """
    grid = grid or GRID_STAGE_A
    fields = list(Combination.__dataclass_fields__)

    combinations = expand_grid(grid)
    done = _completed_combinations(output_path, fields)
    todo = [
        combination
        for combination in combinations
        if tuple(float(getattr(combination, field)) for field in fields) not in done
    ]

    open_count = len(todo)
    if limit is not None:
        todo = todo[:limit]

    if verbose:
        portion = f", davon {len(todo)} in diesem Aufruf" if limit is not None else ""
        print(
            f"{len(combinations)} Kombinationen, davon {len(done)} schon gerechnet, "
            f"{open_count} offen{portion}. Auswertung bis einschliesslich "
            f"{end_season or 'Ende'}, {workers} Worker."
        )
    if not todo:
        return pd.read_csv(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_worker_init, initargs=(str(matches_path),)
    ) as pool:
        futures = {
            pool.submit(_worker_evaluate, combination, end_season): combination
            for combination in todo
        }
        for finished, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            frame = pd.DataFrame([row])
            header = not output_path.exists() or output_path.stat().st_size == 0
            frame.to_csv(output_path, mode="a", header=header, index=False)
            if verbose:
                print(
                    f"[{finished}/{len(todo)}] rps={row['rps']:.4f}  "
                    + "  ".join(f"{field}={row[field]}" for field in fields)
                )

    return pd.read_csv(output_path)


def best(results: pd.DataFrame, within: float = 0.0002) -> pd.DataFrame:
    """Alle Kombinationen im Plateau um das Minimum, bestes zuerst.

    `within` ist die Breite, ab der ein Unterschied als Rauschen gilt. Das
    Argmin allein ist bei hundert Kombinationen keine belastbare Auswahl.
    """
    ranked = results.sort_values("rps")
    return ranked[ranked["rps"] <= ranked["rps"].min() + within]
