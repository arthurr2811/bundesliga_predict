"""Kommandozeile des Projekts.
   Datei-I/O
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bundesliga_predict import pipeline
from bundesliga_predict.evaluation import calibration, report, tuning
from bundesliga_predict.evaluation.backtest import BacktestConfig, run_backtest
from bundesliga_predict.evaluation.baselines import load_odds
from bundesliga_predict.model.bootstrap import BootstrapConfig
from bundesliga_predict.model.prior import PriorConfig
from bundesliga_predict.model.weights import WeightConfig
from bundesliga_predict.simulation.season import SimulationConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MATCHES_PATH = PROJECT_ROOT / "data" / "processed" / "matches.csv"
RAW_HISTORIC_DIR = PROJECT_ROOT / "data" / "raw" / "historic_data"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
BACKTEST_OUTPUT = OUTPUT_DIR / "backtest_predictions.csv"
CALIBRATION_OUTPUT = OUTPUT_DIR / "calibration_checkpoints.csv"


def _bootstrap(args: argparse.Namespace) -> BootstrapConfig:
    """`--bootstrap 0` schaltet die Parameter-Unsicherheit ab."""
    return BootstrapConfig(n_replicates=args.bootstrap)


def _load_matches() -> pd.DataFrame:
    if not MATCHES_PATH.exists():
        raise SystemExit(
            f"{MATCHES_PATH} fehlt -- erst `python -m bundesliga_predict.build_dataset`."
        )
    return pd.read_csv(MATCHES_PATH)


_GRIDS = {"a": tuning.GRID_STAGE_A, "b": tuning.GRID_STAGE_B}


def command_tune(args: argparse.Namespace) -> None:
    grid = dict(_GRIDS[args.stage])
    output = args.output or PROJECT_ROOT / "data" / "output" / f"tuning_stage_{args.stage}.csv"
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
        Path(output),
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


def command_calibrate(args: argparse.Namespace) -> None:
    """Prognosen vergangener Spieltage gegen den tatsaechlichen Saisonausgang."""
    config = calibration.CalibrationConfig(
        start_season=args.start_season,
        end_season=args.end_season or None,
        matchdays=tuple(args.matchdays) if args.matchdays else None,
        simulation=SimulationConfig(n_simulations=args.simulations, seed=args.seed),
        bootstrap=_bootstrap(args),
    )
    checkpoints = calibration.run_checkpoints(
        _load_matches(), config, verbose=args.verbose
    )
    outcomes = calibration.event_outcomes(checkpoints)

    saisons = checkpoints["season"].nunique()
    stichtage = len(checkpoints.groupby(["season", "matchday"]))
    print(
        f"\n{stichtage} Stichtage aus {saisons} Saisons, "
        f"{len(checkpoints)} Team-Prognosen, "
        f"{config.simulation.n_simulations} Simulationen je Stichtag, "
        f"{config.bootstrap.n_replicates if config.bootstrap.active else 0} "
        f"Parameter-Ziehungen\n"
    )

    print("Ereignisse: Prognose gegen Eintritt\n")
    print(report.format_table(calibration.event_summary(outcomes)))
    print(f"\nZuverlaessigkeit (Brier {calibration.brier(outcomes):.4f}):\n")
    print(report.format_table(calibration.reliability(outcomes)))

    print("\n\n90-%-Intervall der Endpunktzahl\n")
    print(report.format_table(calibration.coverage(checkpoints)))
    print("\nJe Saisonphase:\n")
    print(
        report.format_table(
            calibration.coverage(checkpoints.assign(phase=calibration.phase), by="phase")
        )
    )
    print("\nJe Saison:\n")
    print(report.format_table(calibration.coverage(checkpoints, by="season")))

    if args.save:
        destination = Path(args.save)
        destination.parent.mkdir(parents=True, exist_ok=True)
        checkpoints.to_csv(destination, index=False)
        print(f"\nStichtags-Prognosen gespeichert: {destination}")


def command_simulate(args: argparse.Namespace) -> None:
    run = pipeline.run_forecast(
        _load_matches(),
        as_of=args.as_of,
        simulation=SimulationConfig(
            n_simulations=args.simulations, seed=args.seed
        ),
        bootstrap=_bootstrap(args),
    )
    geschrieben = pipeline.write_payload(pipeline.to_payload(run), Path(args.output))

    offen = int((~run.matches["finished"]).sum())
    print(
        f"\nSaison {run.season}, Stand {run.as_of.date()}: "
        f"{len(run.matches) - offen} gespielt, {offen} offen, "
        f"{run.simulation.n_simulations} Simulationen"
        + (
            f" auf {run.n_replicates} Parameter-Ziehungen\n"
            if run.n_replicates
            else " auf einem festen Parametersatz\n"
        ),
    )
    tabelle = run.forecast.summary().merge(
        pipeline.event_probabilities(run.forecast), on="team"
    )
    spalten = [
        "team",
        "expected_points",
        "champion",
        "champions_league",
        "relegation_playoff",
        "relegated",
    ]
    print(report.format_table(tabelle[spalten].round(3)))
    print("\n" + "\n".join(f"geschrieben: {pfad}" for pfad in geschrieben))


def command_update(args: argparse.Namespace) -> None:
    """Datensatz neu bauen und danach simulieren -- der Lauf nach jedem Spieltag."""
    from bundesliga_predict import build_dataset

    dataset = build_dataset.build_dataset()
    MATCHES_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(MATCHES_PATH, index=False)
    print(f"{len(dataset)} Spiele geschrieben nach {MATCHES_PATH}")

    command_simulate(args)


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
        "--stage",
        choices=sorted(_GRIDS),
        default="a",
        help="a = breite Suche, b = Verfeinerung um den Sieger plus match_weight",
    )
    tune.add_argument(
        "--output",
        default=None,
        help="Ergebnis-CSV (Standard: data/output/tuning_stage_<stage>.csv); "
        "vorhandene Kombinationen werden uebersprungen",
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

    calibrate = subparsers.add_parser(
        "calibrate",
        help="Kalibrierung der Saison-Prognosen gegen den echten Ausgang",
    )
    calibrate.add_argument("--start-season", default=calibration.CalibrationConfig().start_season)
    calibrate.add_argument(
        "--end-season", default=None, help="Letzte gepruefte Saison, einschliesslich"
    )
    calibrate.add_argument(
        "--matchdays",
        type=int,
        nargs="+",
        default=None,
        metavar="ST",
        help="Nur diese Stichtage rechnen (0 = vor Saisonstart); Standard: alle",
    )
    calibrate.add_argument(
        "--simulations", type=int, default=SimulationConfig().n_simulations
    )
    calibrate.add_argument("--seed", type=int, default=SimulationConfig().seed)
    calibrate.add_argument(
        "--bootstrap",
        type=int,
        default=BootstrapConfig().n_replicates,
        metavar="N",
        help="Parameter-Ziehungen je Stichtag (0 = ohne, alter Stand)",
    )
    calibrate.add_argument(
        "--save",
        nargs="?",
        const=str(CALIBRATION_OUTPUT),
        metavar="PFAD",
        help="Stichtags-Prognosen als CSV ablegen "
        "(ohne Pfad: data/output/calibration_checkpoints.csv)",
    )
    calibrate.add_argument("-v", "--verbose", action="store_true")
    calibrate.set_defaults(func=command_calibrate)

    for name, hilfe in (
        ("simulate", "Restsaison simulieren und JSON schreiben"),
        ("update", "Datensatz aktualisieren, dann simulieren"),
    ):
        befehl = subparsers.add_parser(name, help=hilfe)
        befehl.add_argument(
            "--as-of",
            default=None,
            metavar="DATUM",
            help="Stichtag (Standard: heute); ein vergangener rekonstruiert "
            "die Prognose von damals",
        )
        befehl.add_argument(
            "--simulations", type=int, default=SimulationConfig().n_simulations
        )
        befehl.add_argument("--seed", type=int, default=SimulationConfig().seed)
        befehl.add_argument(
            "--bootstrap",
            type=int,
            default=BootstrapConfig().n_replicates,
            metavar="N",
            help="Parameter-Ziehungen fuer die Simulation "
            "(0 = ohne; kostet je Ziehung einen Fit)",
        )
        befehl.add_argument("--output", default=str(OUTPUT_DIR), metavar="VERZEICHNIS")
        befehl.set_defaults(func=command_simulate if name == "simulate" else command_update)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
