# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Bundesliga season forecast (Dixon-Coles Poisson model + Monte-Carlo simulation),
published as a static site on GitHub Pages and updated daily by GitHub Actions.
Docs are in German: `README.md` (setup, commands), `documentation.md` (model,
backtest, calibration, log), `deploymentplan.md` (Pages/Actions setup).

## Common Commands

    pip install -r requirements.txt && pip install -e .
    pytest -q
    python -m bundesliga_predict.cli update            # fetch + fit + simulate + write JSON
    python -m bundesliga_predict.cli simulate --as-of YYYY-MM-DD
    python -m bundesliga_predict.cli backtest|calibrate|tune   # need raw CSVs with odds
    python -m bundesliga_predict.build_dataset --historic      # rewrite data/processed/historic.csv
    python -m bundesliga_predict.build_site _site              # assemble Pages site

## Architecture

- `src/bundesliga_predict/`: `build_dataset.py` (historic + OpenLigaDB live →
  `data/processed/matches.csv`), `model/` (fit, prior, weights, bootstrap),
  `predict/`, `simulation/`, `pipeline.py` (run → JSON payload, matchday
  archive), `evaluation/` (backtest, calibration, tuning), `cli.py`.
- `frontend/`: vanilla JS, reads `../data/output/*.json` locally;
  `build_site.py` rewrites that path to `data/` for Pages.
- Data: raw football-data.co.uk CSVs (`data/raw/`) are git-ignored; CI uses the
  committed `data/processed/historic.csv` instead. Current forecast JSONs in
  `data/output/` are git-ignored (built and deployed by CI); only
  `data/output/archive/` is committed, by the workflow bot.
