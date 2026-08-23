"""Kalibrierungs-Check der Simulationsschicht.

Der Backtest misst die Spielvorhersagen (RPS je Partie). Er sagt nichts
darueber, ob die *Saison*-Aussagen stimmen.

1. **Ereignisse.** Treten Dinge mit 90-%-Prognose auch in ~90 % der Faelle
   ein? Gemessen ueber alle Ereignisse aus `PLACE_RULES`, gebinnt nach
   vorhergesagter Wahrscheinlichkeit.
2. **Intervalle.** Liegt die echte Endpunktzahl in ~90 % der Faelle im
   90-%-Intervall (`points_p05`..`points_p95`)?
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bundesliga_predict.config import PLACE_RULES
from bundesliga_predict.model.bootstrap import BootstrapConfig
from bundesliga_predict.model.prior import PriorConfig
from bundesliga_predict.model.weights import WeightConfig
from bundesliga_predict.pipeline import event_probabilities, run_forecast
from bundesliga_predict.simulation.season import SimulationConfig
from bundesliga_predict.simulation.table import standings

# Grenzen der Zuverlaessigkeits-Bins. Aussen fein, in der Mitte grob: dort
# liegen wenige Faelle, und die Aussage "90 % heisst 90 %" haengt an den
# Raendern.
RELIABILITY_BINS = (0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 1.0)

# Nennweite des Punkte-Intervalls; deckt sich mit points_p05/points_p95.
INTERVAL_COVERAGE = 0.90


@dataclass(frozen=True)
class CalibrationConfig:
    start_season: str = "2018/19"
    end_season: str | None = None
    # Nur diese Spieltags-Stichtage rechnen (0 = vor dem ersten Spieltag).
    # None heisst: alle.
    matchdays: tuple[int, ...] | None = None
    weight_config: WeightConfig = field(default_factory=WeightConfig)
    prior: PriorConfig = field(default_factory=PriorConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)


def complete_seasons(matches: pd.DataFrame) -> list[str]:
    """Saisons, die vollstaendig ausgespielt sind -- nur die haben eine Wahrheit."""
    finished = matches.groupby("season")["finished"].all()
    return sorted(finished[finished].index)


def checkpoint_dates(season_matches: pd.DataFrame) -> list[tuple[int, pd.Timestamp]]:
    """Stichtage einer Saison als (gespielte Spieltage, Datum).

    Spieltag 0 ist der Tag vor dem ersten Spiel
    """
    dates = pd.to_datetime(season_matches["date"])
    last = season_matches.assign(date=dates).groupby("matchday")["date"].max()
    last = last.sort_index()

    points = [(0, dates.min() - pd.Timedelta(days=1))]
    points += [(int(matchday), date) for matchday, date in last.items()]
    return points[:-1]


def _final_points(season_matches: pd.DataFrame, teams: tuple[str, ...]) -> pd.DataFrame:
    """Abschlusstabelle der Saison, auf Punkte und Platz reduziert."""
    table = standings(season_matches, teams=teams)
    return table[["team", "points", "position"]].rename(
        columns={"points": "final_points", "position": "final_position"}
    )


def run_checkpoints(
    matches: pd.DataFrame,
    config: CalibrationConfig | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Prognose an jedem Stichtag, je Team eine Zeile, mit dem echten Ausgang.
    """
    config = config or CalibrationConfig()
    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"])

    seasons = [s for s in complete_seasons(matches) if s >= config.start_season]
    if config.end_season is not None:
        seasons = [s for s in seasons if s <= config.end_season]
    if not seasons:
        raise ValueError(
            f"Keine abgeschlossene Saison in {config.start_season}.."
            f"{config.end_season or 'Ende'}."
        )

    rows = []
    for season in seasons:
        season_matches = matches[matches["season"] == season]
        for matchday, as_of in checkpoint_dates(season_matches):
            if config.matchdays is not None and matchday not in config.matchdays:
                continue

            run = run_forecast(
                matches,
                as_of=as_of,
                weight_config=config.weight_config,
                prior=config.prior,
                simulation=config.simulation,
                bootstrap=config.bootstrap,
            )
            if run.season != season:
                # Kann nur passieren, wenn der Stichtag ausserhalb der Saison
                # liegt -- dann waere die Zuordnung zur Wahrheit falsch.
                raise RuntimeError(
                    f"Stichtag {as_of.date()} trifft {run.season}, erwartet {season}."
                )

            summary = run.forecast.summary()
            events = event_probabilities(run.forecast)
            truth = _final_points(season_matches, run.forecast.teams)

            frame = (
                summary.merge(events, on="team")
                .merge(truth, on="team")
                .assign(season=season, matchday=matchday, as_of=as_of)
            )
            rows.append(frame)

            if verbose:
                offen = int((~run.matches["finished"]).sum())
                print(f"{season} ST {matchday:>2} ({as_of.date()}): {offen} offen")

    result = pd.concat(rows, ignore_index=True)
    columns = ["season", "matchday", "as_of", "team"]
    return result[columns + [c for c in result.columns if c not in columns]]


# --- Auswertung 1: Ereignisse ---------------------------------------------


def event_outcomes(checkpoints: pd.DataFrame) -> pd.DataFrame:
    """Lange Form: je Zeile eine Prognose mit `predicted` und `occurred`.
    """
    rows = []
    for event, (first, last) in PLACE_RULES.items():
        rows.append(
            pd.DataFrame(
                {
                    "season": checkpoints["season"],
                    "matchday": checkpoints["matchday"],
                    "team": checkpoints["team"],
                    "event": event,
                    "predicted": checkpoints[event].astype(float),
                    "occurred": checkpoints["final_position"].between(first, last),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def reliability(
    outcomes: pd.DataFrame, bins: tuple[float, ...] = RELIABILITY_BINS
) -> pd.DataFrame:
    """Zuverlaessigkeits-Tabelle: prognostiziert gegen tatsaechlich eingetreten.
    """
    binned = pd.cut(outcomes["predicted"], bins=list(bins), include_lowest=True)
    grouped = outcomes.groupby(binned, observed=True)
    table = grouped.agg(
        n=("predicted", "size"),
        predicted=("predicted", "mean"),
        observed=("occurred", "mean"),
    ).reset_index(names="bin")
    table["gap"] = table["predicted"] - table["observed"]
    return table


def event_summary(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Je Ereignis die Summe der Prognosen gegen die Zahl der Eintritte.
    """
    table = outcomes.groupby("event").agg(
        n=("predicted", "size"),
        predicted=("predicted", "sum"),
        observed=("occurred", "sum"),
    )
    table["gap"] = table["predicted"] - table["observed"]
    return table.reset_index()


def brier(outcomes: pd.DataFrame) -> float:
    """Brier-Score ueber alle Ereignis-Prognosen -- ein Wert fuer den Vergleich."""
    error = outcomes["predicted"] - outcomes["occurred"].astype(float)
    return float(np.mean(error**2))


# --- Auswertung 2: Punkte-Intervalle --------------------------------------


def interval_hits(checkpoints: pd.DataFrame) -> pd.Series:
    """Liegt die echte Endpunktzahl im 90-%-Intervall?"""
    return (checkpoints["final_points"] >= checkpoints["points_p05"]) & (
        checkpoints["final_points"] <= checkpoints["points_p95"]
    )


def coverage(checkpoints: pd.DataFrame, by: str | list[str] | None = None) -> pd.DataFrame:
    """Abdeckung des 90-%-Intervalls, gesamt oder gruppiert.
    """
    frame = checkpoints.assign(
        hit=interval_hits(checkpoints),
        width=checkpoints["points_p95"] - checkpoints["points_p05"],
        error=checkpoints["final_points"] - checkpoints["expected_points"],
    )
    grouped = frame.groupby(by) if by is not None else frame.groupby(lambda _: "gesamt")
    table = grouped.agg(
        n=("hit", "size"),
        coverage=("hit", "mean"),
        width=("width", "mean"),
        mean_error=("error", "mean"),
        mae=("error", lambda values: values.abs().mean()),
    )
    table["gap"] = table["coverage"] - INTERVAL_COVERAGE
    return table.reset_index(names=by if by is not None else "gruppe")


def phase(checkpoints: pd.DataFrame) -> pd.Series:
    """Grobe Saisonphase -- die Unsicherheit ist am Anfang am groessten."""
    return pd.cut(
        checkpoints["matchday"],
        bins=[-1, 0, 8, 17, 25, 34],
        labels=["vor Saison", "ST 1-8", "ST 9-17", "ST 18-25", "ST 26+"],
    )
