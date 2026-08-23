"""Walk-forward-Backtest: die Messgrundlage für alles Weitere.

Ablauf: die Spiele werden in Spieltags-Blöcke zerlegt. Vor jedem Block wird
das Modell ausschliesslich auf Spielen *vor* diesem Block neu gefittet und
sagt dann dessen Partien vorher. Kein Spiel wird also mit Wissen über sich
selbst oder spätere Partien bewertet.

Warum Blöcke und nicht einzelne Spiele: ein Spieltag ist die Einheit, in der
später auch die Pipeline läuft (nach jedem Spieltag neu rechnen). Und der Fit
kostet ~0,3 s -- pro Spiel neu zu fitten wäre neunmal so teuer, ohne dass sich
die Datenlage zwischen zwei Partien desselben Spieltags nennenswert ändert.

Die Ligadurchschnitts-Baseline wird gleich mitgerechnet: sie muss denselben
Datenschnitt sehen wie das Modell, sonst ist der Vergleich unfair.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bundesliga_predict.evaluation.baselines import league_average_probabilities
from bundesliga_predict.model.fit import finished_matches, fit
from bundesliga_predict.model.prior import PriorConfig, with_unknown_teams
from bundesliga_predict.model.weights import WeightConfig
from bundesliga_predict.predict.outcomes import predict_match

# Ab so vielen Tagen Abstand gilt eine Partie innerhalb eines Spieltags als verlegt
# Sie wird dann als eigener Block betrachtet, da sie Zeitlich zu weit entfernt liegt.
_POSTPONED_AFTER_DAYS = 3


def assign_blocks(matches: pd.DataFrame) -> pd.Series:
    """Ordnet jedem Spiel eine fortlaufende Block-Nummer zu, chronologisch.

    Ein Block ist ein Spieltag. Verlegte Partien werden davon abgetrennt und
    bilden einen eigenen Block: sie finden Wochen später statt und sind damit
    ein eigener Vorhersage-Zeitpunkt, an dem das Modell mehr weiß.
    """
    if matches["matchday"].isna().any():
        raise ValueError("Spieltagsnummer fehlt - Datensatz mit build_dataset neu bauen.")

    ordered = matches[["season", "matchday", "date"]].sort_values(
        ["season", "matchday", "date"]
    )
    gap = ordered.groupby(["season", "matchday"])["date"].diff().dt.days
    run = (gap.isna() | (gap > _POSTPONED_AFTER_DAYS)).cumsum()

    # Nach Startdatum durchnummerieren, damit die Blöcke in der Reihenfolge
    # laufen, in der sie tatsächlich gespielt wurden.
    start = ordered["date"].groupby(run).min().sort_values()
    renumbered = {run_id: number for number, run_id in enumerate(start.index)}
    return run.map(renumbered).reindex(matches.index)


@dataclass(frozen=True)
class BacktestConfig:
    """Backtest Hyperparameter

    `weight_config` und `prior`
    """

    start_season: str = "2018/19"
    weight_config: WeightConfig = field(default_factory=WeightConfig)
    prior: PriorConfig = field(default_factory=PriorConfig)
    # Mindestanzahl gespielter Partien, bevor überhaupt gefittet wird.
    min_history: int = 200


def run_backtest(
    matches: pd.DataFrame,
    config: BacktestConfig | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Führt den Walk-forward durch und gibt die Vorhersagen je Spiel zurück.

    Spalten: Spiel-Identifikation (`season`, `date`, Teams, Tore), die
    Modellwahrscheinlichkeiten `p_home`/`p_draw`/`p_away`, die erwarteten Tore
    und die Baseline `base_home`/`base_draw`/`base_away`. Bewertet wird erst
    danach (`metrics`), damit dieselben Vorhersagen gegen verschiedene
    Baselines gehalten werden können.
    """
    config = config or BacktestConfig()

    played = finished_matches(matches).copy()
    played["date"] = pd.to_datetime(played["date"])
    played = played.sort_values("date").reset_index(drop=True)

    evaluated = played[played["season"] >= config.start_season].copy()
    if evaluated.empty:
        raise ValueError(f"Keine Spiele ab Saison {config.start_season}.")
    evaluated["block"] = assign_blocks(evaluated)

    rows = []
    for _, block in evaluated.groupby("block", sort=True):
        block_start = block["date"].min()
        history = played[played["date"] < block_start]
        if len(history) < config.min_history:
            continue

        # Der Malus für den Saisonwechsel muss sich auf die vorherzusagende
        # Saison beziehen, nicht auf die des letzten gespielten Spiels.
        target_season = block["season"].iloc[0]
        params = fit(
            history,
            reference_date=block_start - pd.Timedelta(days=1),
            weight_config=config.weight_config,
            prior=config.prior,
            reference_season=target_season,
        )
        # Aufsteiger ohne jede BL-Historie kommen im Fit nicht vor; sie
        # bekommen den Prior-Mittelwert.
        params = with_unknown_teams(
            params, set(block["home_team"]) | set(block["away_team"]), config.prior
        )
        baseline = league_average_probabilities(history, len(block))

        if verbose:
            print(
                f"{block_start.date()} ({target_season}): "
                f"{len(block)} Spiele, {len(history)} Spiele Historie"
            )

        for position, match in enumerate(block.itertuples()):
            prediction = predict_match(params, match.home_team, match.away_team)
            rows.append(
                {
                    "season": match.season,
                    "date": match.date,
                    "home_team": match.home_team,
                    "away_team": match.away_team,
                    "home_goals": match.home_goals,
                    "away_goals": match.away_goals,
                    "p_home": prediction.home_win,
                    "p_draw": prediction.draw,
                    "p_away": prediction.away_win,
                    "expected_home_goals": prediction.expected_home_goals,
                    "expected_away_goals": prediction.expected_away_goals,
                    "base_home": baseline[position, 0],
                    "base_draw": baseline[position, 1],
                    "base_away": baseline[position, 2],
                }
            )

    return pd.DataFrame(rows)


def model_probabilities(predictions: pd.DataFrame) -> np.ndarray:
    """Die 1X2-Spalten des Modells als Array in der Reihenfolge von `OUTCOMES`."""
    return predictions[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)


def baseline_probabilities(predictions: pd.DataFrame) -> np.ndarray:
    """Dasselbe für die Ligadurchschnitts-Baseline."""
    return predictions[["base_home", "base_draw", "base_away"]].to_numpy(dtype=float)
