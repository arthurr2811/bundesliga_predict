"""Bewertungsmasse für 1X2-Vorhersagen.

Alle drei Maße sind negativ orientiert: kleiner ist besser. Sie messen
Unterschiedliches und werden deshalb nebeneinander berichtet:

- **RPS** (Ranked Probability Score) berücksichtigt, dass die drei Ausgänge
  geordnet sind (Heimsieg - Remis - Auswärtssieg). Ein Modell, das statt des
  Heimsiegs ein Remis erwartet, wird milder bestraft als eines, das auf
  Auswärtssieg tippt. Das ist das Standardmass für Fussballvorhersagen.
- **Log-Loss** bestraft selbstsichere Fehlprognosen sehr hart (unbeschränkt).
  Es ist zugleich genau die Grösse, die der Fit maximiert -- der ehrlichste
  Blick darauf, ob das Modell out-of-sample dasselbe tut wie in-sample.
- **Brier** ist der quadratische Fehler über alle drei Klassen, beschränkt
  und robust gegen einzelne Ausreisser.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Untergrenze für Wahrscheinlichkeiten im Log-Loss. Ohne sie macht ein
# einziges Spiel mit p = 0 die gesamte Auswertung unendlich.
_PROBABILITY_FLOOR = 1e-15


@dataclass(frozen=True)
class Scores:
    """Mittelwerte der drei Masse über eine Menge von Spielen."""

    n_matches: int
    rps: float
    log_loss: float
    brier: float

    def as_dict(self) -> dict[str, float]:
        return {
            "n_matches": self.n_matches,
            "rps": self.rps,
            "log_loss": self.log_loss,
            "brier": self.brier,
        }


def one_hot(outcome_index: np.ndarray, n_outcomes: int = 3) -> np.ndarray:
    """Beobachtete Ausgänge als 0/1-Matrix (n_matches x n_outcomes)."""
    observed = np.zeros((len(outcome_index), n_outcomes))
    observed[np.arange(len(outcome_index)), outcome_index] = 1.0
    return observed


def ranked_probability_score(
    probabilities: np.ndarray, outcome_index: np.ndarray
) -> np.ndarray:
    """RPS je Spiel: mittlere quadrierte Abweichung der Summenverteilungen."""
    observed = one_hot(outcome_index, probabilities.shape[1])
    difference = np.cumsum(probabilities, axis=1) - np.cumsum(observed, axis=1)
    # Nur die ersten k-1 Schwellen tragen bei; die letzte ist per Definition 0.
    return np.sum(difference[:, :-1] ** 2, axis=1) / (probabilities.shape[1] - 1)


def log_loss(probabilities: np.ndarray, outcome_index: np.ndarray) -> np.ndarray:
    """Negativer Log der Wahrscheinlichkeit, die dem Eingetretenen gegeben wurde."""
    hit = probabilities[np.arange(len(outcome_index)), outcome_index]
    return -np.log(np.clip(hit, _PROBABILITY_FLOOR, None))


def brier_score(probabilities: np.ndarray, outcome_index: np.ndarray) -> np.ndarray:
    """Brier je Spiel: quadrierter Fehler summiert über alle Ausgänge."""
    observed = one_hot(outcome_index, probabilities.shape[1])
    return np.sum((probabilities - observed) ** 2, axis=1)


def score(probabilities: np.ndarray, outcome_index: np.ndarray) -> Scores:
    """Alle drei Masse, gemittelt über die übergebenen Spiele."""
    probabilities = np.asarray(probabilities, dtype=float)
    outcome_index = np.asarray(outcome_index, dtype=int)
    if len(probabilities) == 0:
        raise ValueError("Keine Spiele zu bewerten.")
    return Scores(
        n_matches=len(probabilities),
        rps=float(np.mean(ranked_probability_score(probabilities, outcome_index))),
        log_loss=float(np.mean(log_loss(probabilities, outcome_index))),
        brier=float(np.mean(brier_score(probabilities, outcome_index))),
    )


def outcome_index(home_goals: np.ndarray, away_goals: np.ndarray) -> np.ndarray:
    """Tatsächlicher Ausgang als Index in `predict.outcomes.OUTCOMES`."""
    home_goals = np.asarray(home_goals, dtype=float)
    away_goals = np.asarray(away_goals, dtype=float)
    return np.where(home_goals > away_goals, 0, np.where(home_goals == away_goals, 1, 2))
