"""Kommandozeile des Projekts.
   Datei-I/O
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bundesliga_predict.evaluation import report
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


def command_backtest(args: argparse.Namespace) -> None:
    config = BacktestConfig(
        start_season=args.start_season,
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

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
