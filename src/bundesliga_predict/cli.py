"""Kommandozeile des Projekts.
   Datei-I/O
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bundesliga_predict.config import DEFAULT_PRIOR_SD
from bundesliga_predict.evaluation import report
from bundesliga_predict.evaluation.backtest import BacktestConfig, run_backtest
from bundesliga_predict.evaluation.baselines import load_odds
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
        prior_sd=args.prior_sd,
    )
    predictions = run_backtest(_load_matches(), config, verbose=args.verbose)

    odds = load_odds(RAW_HISTORIC_DIR)
    print(f"\n{len(predictions)} Spiele vorhergesagt ab Saison {args.start_season}\n")
    print(report.format_table(report.compare(predictions, odds)))
    print("\nJe Saison:\n")
    print(report.format_table(report.by_season(predictions)))

    if args.save:
        BACKTEST_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(BACKTEST_OUTPUT, index=False)
        print(f"\nVorhersagen gespeichert: {BACKTEST_OUTPUT}")


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
    backtest.add_argument("--prior-sd", type=float, default=DEFAULT_PRIOR_SD)
    backtest.add_argument("--save", action="store_true", help="Vorhersagen als CSV ablegen")
    backtest.add_argument("-v", "--verbose", action="store_true")
    backtest.set_defaults(func=command_backtest)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
