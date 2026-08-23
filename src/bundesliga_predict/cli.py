"""Kommandozeile des Projekts.
   Datei-I/O
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bundesliga_predict.evaluation import report, tuning
from bundesliga_predict.evaluation.backtest import BacktestConfig, run_backtest
from bundesliga_predict.evaluation.baselines import load_odds
from bundesliga_predict.model.prior import PriorConfig
from bundesliga_predict.model.weights import WeightConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MATCHES_PATH = PROJECT_ROOT / "data" / "processed" / "matches.csv"
RAW_HISTORIC_DIR = PROJECT_ROOT / "data" / "raw" / "historic_data"
BACKTEST_OUTPUT = PROJECT_ROOT / "data" / "output" / "backtest_predictions.csv"


def _load_matches() -> pd.DataFrame:
    if not MATCHES_PATH.exists():
        raise SystemExit(
            f"{MATCHES_PATH} fehlt -- erst `python -m bundesliga_predict.build_dataset`."
        )
    return pd.read_csv(MATCHES_PATH)


def command_tune(args: argparse.Namespace) -> None:
    grid = dict(tuning.GRID_STAGE_A)
    if args.smoke:
        # Nur die aktuellen Defaults, ein einziger Lauf: prueft, dass der
        # Grid-Search denselben Backtest faehrt wie `backtest`.
        grid = {
            "half_life_days": (WeightConfig().half_life_days,),
            "season_penalty": (WeightConfig().season_penalty,),
            "prior_sd": (PriorConfig().sd,),
            "prior_scale": (1.0,),
        }

    results = tuning.run_grid(
        MATCHES_PATH,
        Path(args.output),
        grid=grid,
        # Leerstring heisst "kein Schnitt"
        end_season=args.end_season or None,
        workers=args.workers,
        limit=args.limit,
    )
    print(f"\n{len(results)} Kombinationen ausgewertet\n")
    print(report.format_table(tuning.best(results).head(15)))


def command_backtest(args: argparse.Namespace) -> None:
    config = BacktestConfig(
        start_season=args.start_season,
        end_season=args.end_season or None,
        weight_config=WeightConfig(
            half_life_days=args.half_life, season_penalty=args.season_penalty
        ),
        prior=PriorConfig(
            sd=args.prior_sd,
            attack_mean=args.prior_attack,
            defense_mean=args.prior_defense,
        ),
    )
    predictions = run_backtest(_load_matches(), config, verbose=args.verbose)

    odds = load_odds(RAW_HISTORIC_DIR)
    print(f"\n{len(predictions)} Spiele vorhergesagt ab Saison {args.start_season}\n")
    print(report.format_table(report.compare(predictions, odds)))
    print("\nJe Saison:\n")
    print(report.format_table(report.by_season(predictions)))

    if args.save:
        destination = Path(args.save)
        destination.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(destination, index=False)
        print(f"\nVorhersagen gespeichert: {destination}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bundesliga-predict")
    subparsers = parser.add_subparsers(dest="command", required=True)

    backtest = subparsers.add_parser(
        "backtest", help="Walk-forward-Backtest gegen Baselines"
    )
    backtest.add_argument("--start-season", default="2018/19")
    backtest.add_argument(
        "--end-season",
        default=None,
        help="Letzte bewertete Saison, einschliesslich (Historie bleibt unberuehrt)",
    )
    backtest.add_argument("--half-life", type=float, default=WeightConfig().half_life_days)
    backtest.add_argument(
        "--season-penalty", type=float, default=WeightConfig().season_penalty
    )
    backtest.add_argument("--prior-sd", type=float, default=PriorConfig().sd)
    backtest.add_argument(
        "--prior-attack",
        type=float,
        default=PriorConfig().attack_mean,
        help="Prior-Mittelwert Angriff (0 = Ligadurchschnitt, negativ = Aufsteiger-Prior)",
    )
    backtest.add_argument(
        "--prior-defense", type=float, default=PriorConfig().defense_mean,
        help="Prior-Mittelwert Abwehr",
    )
    backtest.add_argument(
        "--save",
        nargs="?",
        const=str(BACKTEST_OUTPUT),
        metavar="PFAD",
        help="Vorhersagen als CSV ablegen (ohne Pfad: data/output/backtest_predictions.csv)",
    )
    backtest.add_argument("-v", "--verbose", action="store_true")
    backtest.set_defaults(func=command_backtest)

    tune = subparsers.add_parser(
        "tune", help="Grid-Search der Hyperparameter ueber den Backtest"
    )
    tune.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "output" / "tuning_stage_a.csv"),
        help="Ergebnis-CSV; vorhandene Kombinationen werden uebersprungen",
    )
    tune.add_argument(
        "--end-season",
        default=tuning.TUNING_END_SEASON,
        help="Letzte Tuning-Saison; alles danach bleibt Holdout",
    )
    tune.add_argument("--workers", type=int, default=6)
    tune.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Nur die naechsten n offenen Kombinationen rechnen",
    )
    tune.add_argument(
        "--smoke",
        action="store_true",
        help="Nur die aktuellen Defaults rechnen (Kontrolllauf statt Grid)",
    )
    tune.set_defaults(func=command_tune)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
