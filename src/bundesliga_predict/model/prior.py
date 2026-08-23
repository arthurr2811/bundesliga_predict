"""Prior auf die Team-Stärken: wohin Teams mit wenig Bundesligaspielen gezogen werden.

Die Shrinkage selbst steckt in `likelihood.prepare` und hängt an der
gewichteten Spielmasse je Team -- ein Team mit zehn Saisons wird praktisch
nicht gezogen, ein Aufsteiger stark. Wer wenig Bundesliga-Historie hat, ist fast
immer gerade aufgestiegen, und Aufsteiger sind im Mittel schwächer als die
Liga.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from bundesliga_predict.config import (
    DEFAULT_PRIOR_ATTACK,
    DEFAULT_PRIOR_DEFENSE,
    DEFAULT_PRIOR_SD,
    PRIOR_MATCH_WEIGHT,
)
from bundesliga_predict.model.params import DixonColesParams


@dataclass(frozen=True)
class PriorConfig:
    """Normal-Prior auf Angriff und Abwehr.

    `sd` steuert, wie hart gezogen wird, `attack_mean`/`defense_mean` wohin,
    `match_weight`, wie schnell der Prior mit wachsender Datenmenge verblasst.
    `sd = inf` schaltet die Regularisierung ganz ab (Tests, Messungen).
    """

    sd: float = DEFAULT_PRIOR_SD
    attack_mean: float = DEFAULT_PRIOR_ATTACK
    defense_mean: float = DEFAULT_PRIOR_DEFENSE
    match_weight: float = PRIOR_MATCH_WEIGHT

    @property
    def active(self) -> bool:
        return bool(np.isfinite(self.sd) and self.sd > 0)

    def shrinkage(self, team_weight: np.ndarray) -> np.ndarray:
        """Wie stark zieht der Prior je Team, gegeben dessen Spielmasse?

        Geht mit wachsender Datenmenge gegen 0
        """
        return self.match_weight / (self.match_weight + team_weight)


def with_unknown_teams(
    params: DixonColesParams, teams: set[str], prior: PriorConfig | None = None
) -> DixonColesParams:
    """Ergänzt Teams ohne jede Historie mit den Prior-Mittelwerten.

    Betrifft Aufsteiger, die noch nie erstklassig gespielt haben: sie tauchen
    im Fit gar nicht auf, weil es keine Partie von ihnen gibt.
    """
    prior = prior or PriorConfig()
    missing = teams - set(params.teams)
    if not missing:
        return params

    attack, defense = (prior.attack_mean, prior.defense_mean) if prior.active else (0.0, 0.0)
    return replace(
        params,
        teams=params.teams + tuple(sorted(missing)),
        attack={**params.attack, **{team: attack for team in missing}},
        defense={**params.defense, **{team: defense for team in missing}},
    )
