"""Vergleichsmassstäbe für den Backtest.

Zwei Baselines um zu überprüfen wie brauchbar unser Modell ist:

- **Ligadurchschnitt** als Untergrenze: predicted immer das durchschnittliche
  Bundesligaergebnis (~ etwa 44 % Heim, 24 % Remis, 32 % Auswärts) völlig unabhängig
  davon, wer gegen wen spielt. Wer die nicht schlägt, hat aus Teamstärken nichts gelernt.
- **Buchmacherquoten** als praktische Obergrenze: der Markt kennt Aufstellungen,
  Verletzungen und Transfers. Er ist Vergleichsmassstab, keine Zielgrösse --
  bewertet wird er wie das Modell gegen die tatsächlichen Ergebnisse.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Buchmacher in Reihenfolge der Präferenz. Bet365 ist in allen Saisons
# vorhanden, Pinnacle (PS) und der Marktdurchschnitt (Avg) springen ein, wo
# einzelne Zeilen fehlen.
_ODDS_SOURCES = (("B365H", "B365D", "B365A"), ("PSH", "PSD", "PSA"), ("AvgH", "AvgD", "AvgA"))


def _season_from_filename(path: Path) -> str:
    start, end = path.stem.removeprefix("D1_").split("-")
    return f"{start}/{end}"


def load_odds(raw_dir: Path) -> pd.DataFrame:
    """Liest die 1X2-Quoten aus den football-data-Rohdateien.

    Rückgabe: eine Zeile je Spiel mit `season`, `date`, `home_team`,
    `away_team` und den margenbereinigten Wahrscheinlichkeiten
    `market_home`, `market_draw`, `market_away`.
    """
    frames = []
    for path in sorted(raw_dir.glob("D1_*.csv")):
        raw = pd.read_csv(path)
        available = [cols for cols in _ODDS_SOURCES if set(cols) <= set(raw.columns)]
        if not available:
            continue

        # Erste Quelle, die für die jeweilige Zeile eine Quote hat.
        odds = pd.DataFrame(index=raw.index, columns=["H", "D", "A"], dtype=float)
        for cols in available:
            candidate = raw[list(cols)].apply(pd.to_numeric, errors="coerce")
            candidate.columns = ["H", "D", "A"]
            odds = odds.fillna(candidate.where(candidate.notna().all(axis=1)))

        frames.append(
            pd.DataFrame(
                {
                    "season": _season_from_filename(path),
                    "date": pd.to_datetime(raw["Date"], dayfirst=True, format="mixed"),
                    "home_team": raw["HomeTeam"],
                    "away_team": raw["AwayTeam"],
                    "odds_home": odds["H"],
                    "odds_draw": odds["D"],
                    "odds_away": odds["A"],
                }
            )
        )

    combined = pd.concat(frames, ignore_index=True).dropna(
        subset=["odds_home", "odds_draw", "odds_away", "home_team", "away_team"]
    )
    implied = 1.0 / combined[["odds_home", "odds_draw", "odds_away"]].to_numpy()
    normalised = implied / implied.sum(axis=1, keepdims=True)

    combined["date"] = combined["date"].dt.date
    combined[["market_home", "market_draw", "market_away"]] = normalised
    return combined.drop(columns=["odds_home", "odds_draw", "odds_away"])


def market_probabilities(
    predictions: pd.DataFrame, odds: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Ordnet die Marktwahrscheinlichkeiten den Backtest-Zeilen zu.

    Rückgabe: (Wahrscheinlichkeiten, Maske). Die Maske markiert die Zeilen, für
    die überhaupt eine Quote vorlag -- der Marktvergleich läuft nur auf denen,
    das Modell wird für diesen Vergleich auf derselben Teilmenge gemessen.
    """
    keys = ["date", "home_team", "away_team"]
    left = predictions[keys].copy()
    left["date"] = pd.to_datetime(left["date"]).dt.date

    merged = left.merge(
        odds[keys + ["market_home", "market_draw", "market_away"]], on=keys, how="left"
    )
    probabilities = merged[["market_home", "market_draw", "market_away"]].to_numpy(
        dtype=float
    )
    mask = ~np.isnan(probabilities).any(axis=1)
    return probabilities, mask


def league_average_probabilities(
    history: pd.DataFrame, n_matches: int
) -> np.ndarray:
    """Konstante Wahrscheinlichkeiten aus den Ausgängen vor dem Stichtag.

    `history` sind die zum Vorhersagezeitpunkt bekannten Spiele; die Baseline
    darf also genauso wenig in die Zukunft schauen wie das Modell.
    """
    from bundesliga_predict.evaluation.metrics import outcome_index

    index = outcome_index(history["home_goals"], history["away_goals"])
    frequencies = np.bincount(index, minlength=3) / len(index)
    return np.tile(frequencies, (n_matches, 1))
