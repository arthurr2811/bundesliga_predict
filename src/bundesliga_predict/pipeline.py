"""Vom Datensatz zur Prognose: fit -> predict -> simulate -> JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from bundesliga_predict.config import PLACE_RULES
from bundesliga_predict.model.fit import fit
from bundesliga_predict.model.params import DixonColesParams
from bundesliga_predict.model.prior import PriorConfig, with_unknown_teams
from bundesliga_predict.model.weights import WeightConfig
from bundesliga_predict.predict.outcomes import predict_matches
from bundesliga_predict.simulation.season import (
    SeasonForecast,
    SimulationConfig,
    open_matches,
    simulate_season,
)
from bundesliga_predict.simulation.table import standings

# Wahrscheinlichkeiten auf vier Stellen -- darunter ist bei 10.000 Laeufen
# ohnehin nur Monte-Carlo-Rauschen.
_DIGITS = 4


@dataclass(frozen=True)
class ForecastRun:
    """Alles, was ein Lauf hervorbringt. Reine Daten, kein Datei-Bezug."""

    season: str
    as_of: pd.Timestamp
    matches: pd.DataFrame
    params: DixonColesParams
    predictions: pd.DataFrame
    forecast: SeasonForecast
    simulation: SimulationConfig


def target_season(matches: pd.DataFrame, as_of: pd.Timestamp) -> str:
    """Die Saison, um die es an diesem Stichtag geht.

    Die frueheste, in der noch gespielt wird. Zwischen zwei Saisons ist das
    die kommende -- richtig so, denn dann ist die alte entschieden.
    """
    dates = pd.to_datetime(matches["date"])
    kommend = matches.loc[dates > as_of, "season"]
    return kommend.min() if not kommend.empty else matches["season"].max()


def season_state(matches: pd.DataFrame, season: str, as_of: pd.Timestamp) -> pd.DataFrame:
    """Die Saison, wie sie am Stichtag aussah.

    Was danach liegt, gilt als offen -- auch wenn es im Datensatz laengst ein
    Ergebnis hat. Nur so rekonstruiert ein Lauf mit vergangenem Stichtag den
    damaligen Wissensstand.
    """
    frame = matches[matches["season"] == season].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["finished"] = frame["finished"].astype(bool) & (frame["date"] <= as_of)
    return frame.sort_values(["date", "home_team"]).reset_index(drop=True)


def run_forecast(
    matches: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
    weight_config: WeightConfig | None = None,
    prior: PriorConfig | None = None,
    simulation: SimulationConfig | None = None,
) -> ForecastRun:
    """Fit, Spielvorhersagen und Saison-Simulation zum Stichtag."""
    as_of = pd.Timestamp(as_of or pd.Timestamp.today().normalize())
    simulation = simulation or SimulationConfig()
    prior = prior or PriorConfig()

    season = target_season(matches, as_of)
    state = season_state(matches, season, as_of)
    teams = sorted(set(state["home_team"]) | set(state["away_team"]))

    params = fit(
        matches,
        reference_date=as_of,
        weight_config=weight_config,
        prior=prior,
        reference_season=season,
    )
    # Aufsteiger ohne jede Bundesliga-Historie kommen im Fit nicht vor.
    params = with_unknown_teams(params, set(teams), prior)

    predictions = predict_matches(params, open_matches(state))
    forecast = simulate_season(params, state, teams=teams, config=simulation)

    return ForecastRun(
        season=season,
        as_of=as_of,
        matches=state,
        params=params,
        predictions=predictions,
        forecast=forecast,
        simulation=simulation,
    )


def event_probabilities(forecast: SeasonForecast) -> pd.DataFrame:
    """Meister, Europapokal, Relegation, Abstieg -- je Team, aus `PLACE_RULES`."""
    return pd.DataFrame(
        {
            "team": forecast.teams,
            **{
                name: forecast.probability_of_places(first, last)
                for name, (first, last) in PLACE_RULES.items()
            },
        }
    )


def _matches_payload(run: ForecastRun) -> list[dict]:
    """Alle Partien der Saison: gespielte mit Ergebnis, offene mit Vorhersage."""
    vorhersage = run.predictions.set_index(["date", "home_team"])
    rows = []
    for match in run.matches.itertuples():
        row = {
            "matchday": int(match.matchday),
            "date": match.date.date().isoformat(),
            "home_team": match.home_team,
            "away_team": match.away_team,
            "finished": bool(match.finished),
        }
        if match.finished:
            row |= {
                "home_goals": int(match.home_goals),
                "away_goals": int(match.away_goals),
            }
        else:
            prediction = vorhersage.loc[(match.date, match.home_team)]
            row |= {
                "p_home": round(float(prediction["p_home"]), _DIGITS),
                "p_draw": round(float(prediction["p_draw"]), _DIGITS),
                "p_away": round(float(prediction["p_away"]), _DIGITS),
                "expected_home_goals": round(float(prediction["expected_home_goals"]), 2),
                "expected_away_goals": round(float(prediction["expected_away_goals"]), 2),
                "likely_score": [
                    int(prediction["likely_home_goals"]),
                    int(prediction["likely_away_goals"]),
                ],
                # Die drei wahrscheinlichsten Ergebnisse als [heim, gast, wkt.].
                "likely_scores": [
                    [home, away, round(probability, _DIGITS)]
                    for home, away, probability in prediction["likely_scores"]
                ],
            }
        rows.append(row)
    return rows


def _table_payload(run: ForecastRun) -> dict:
    aktuell = standings(run.matches, teams=run.forecast.teams)
    erwartet = run.forecast.summary().round(
        {
            "expected_points": 1,
            "expected_position": 2,
            "expected_goal_difference": 1,
            "points_p05": 0,
            "points_p95": 0,
        }
    )
    return {
        "current": aktuell.to_dict(orient="records"),
        "expected": erwartet.to_dict(orient="records"),
    }


def _probabilities_payload(run: ForecastRun) -> list[dict]:
    events = event_probabilities(run.forecast).round(_DIGITS).set_index("team")
    verteilung = run.forecast.position_probabilities.round(_DIGITS)
    return [
        {
            "team": team,
            "positions": verteilung[number].tolist(),
            **events.loc[team].to_dict(),
        }
        for number, team in enumerate(run.forecast.teams)
    ]


def _meta_payload(run: ForecastRun) -> dict:
    offen = int((~run.matches["finished"]).sum())
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of": run.as_of.date().isoformat(),
        "season": run.season,
        "matches_played": len(run.matches) - offen,
        "matches_open": offen,
        "n_simulations": run.simulation.n_simulations,
        "seed": run.simulation.seed,
        "model": {
            "home_advantage": round(run.params.home_advantage, 4),
            "intercept": round(run.params.intercept, 4),
            "rho": round(run.params.rho, 4),
            "fitted_through": run.params.fitted_through.isoformat(),
        },
        "place_rules": {name: list(bereich) for name, bereich in PLACE_RULES.items()},
    }


def to_payload(run: ForecastRun) -> dict[str, object]:
    """Die vier JSON-Dokumente, die das Frontend liest."""
    return {
        "meta": _meta_payload(run),
        "matches": _matches_payload(run),
        "table": _table_payload(run),
        "probabilities": _probabilities_payload(run),
    }


def write_payload(payload: dict[str, object], directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, document in payload.items():
        path = directory / f"{name}.json"
        path.write_text(
            json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        written.append(path)
    return written
