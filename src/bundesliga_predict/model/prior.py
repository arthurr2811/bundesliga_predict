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
)
from bundesliga_predict.model.params import DixonColesParams


@dataclass(frozen=True)
class PriorConfig:
    """Normal-Prior auf Angriff und Abwehr.

    `sd` steuert, wie hart gezogen wird, `attack_mean`/`defense_mean` wohin.
    `sd = inf` schaltet die Regularisierung ganz ab (Tests, Messungen).
    """

    sd: float = DEFAULT_PRIOR_SD
    attack_mean: float = DEFAULT_PRIOR_ATTACK
    defense_mean: float = DEFAULT_PRIOR_DEFENSE

    @property
    def active(self) -> bool:
        return bool(np.isfinite(self.sd) and self.sd > 0)


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
