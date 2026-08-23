"""Parameter-Unsicherheit per Bayesian Bootstrap.


Der Kalibrierungs-Check hat gezeigt: vor Saisonstart deckt das
90-%-Punkteintervall nur 75 % der Faelle ab, ab Spieltag 26 dagegen 94 %.

Der Bayesian Bootstrap behebt das ohne neue Annahme: statt eines Fits werden
`n_replicates` Fits gerechnet, jeder mit einem zusaetzlichen Exp(1)-Faktor auf
jedem Spielgewicht. Wie weit die Ergebnisse streuen, sagt die Datenlage selbst
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from bundesliga_predict.config import (
    BOOTSTRAP_SEED,
    DEFAULT_UNKNOWN_ATTACK_SD,
    DEFAULT_UNKNOWN_DEFENSE_SD,
    N_BOOTSTRAP,
)
from bundesliga_predict.model.fit import fit
from bundesliga_predict.model.params import DixonColesParams
from bundesliga_predict.model.prior import PriorConfig, with_unknown_teams
from bundesliga_predict.model.weights import WeightConfig


@dataclass(frozen=True)
class BootstrapConfig:
    """Wie viele Parameter-Ziehungen, mit welchem Seed, mit welcher
    Zusatzstreuung fuer Teams ohne Historie."""

    n_replicates: int = N_BOOTSTRAP
    seed: int = BOOTSTRAP_SEED
    unknown_attack_sd: float = DEFAULT_UNKNOWN_ATTACK_SD
    unknown_defense_sd: float = DEFAULT_UNKNOWN_DEFENSE_SD

    @property
    def active(self) -> bool:
        """Unter zwei Ziehungen gibt es nichts zu streuen."""
        return self.n_replicates > 1


def jitter_unknown_teams(
    params: DixonColesParams,
    unknown: set[str],
    rng: np.random.Generator,
    config: BootstrapConfig | None = None,
) -> DixonColesParams:
    """Streut die Staerken der Teams ohne Historie um ihren Prior-Mittelwert.
    """
    config = config or BootstrapConfig()
    if not unknown:
        return params

    return replace(
        params,
        attack={
            **params.attack,
            **{
                team: params.attack[team] + rng.normal(0.0, config.unknown_attack_sd)
                for team in unknown
            },
        },
        defense={
            **params.defense,
            **{
                team: params.defense[team] + rng.normal(0.0, config.unknown_defense_sd)
                for team in unknown
            },
        },
    )


def bootstrap_params(
    matches: pd.DataFrame,
    teams: set[str],
    reference_date: pd.Timestamp,
    reference_season: str,
    weight_config: WeightConfig | None = None,
    prior: PriorConfig | None = None,
    config: BootstrapConfig | None = None,
) -> list[DixonColesParams]:
    """`n_replicates` Parametersaetze, jeder ein eigener Fit auf umgewichteten Daten.
    """
    config = config or BootstrapConfig()
    prior = prior or PriorConfig()

    streams = np.random.SeedSequence(config.seed).spawn(config.n_replicates)
    replicates = []
    for stream in streams:
        rng = np.random.default_rng(stream)
        params = fit(
            matches,
            reference_date=reference_date,
            weight_config=weight_config,
            prior=prior,
            reference_season=reference_season,
            bootstrap_rng=rng,
        )
        unknown = set(teams) - set(params.teams)
        params = with_unknown_teams(params, set(teams), prior)
        replicates.append(jitter_unknown_teams(params, unknown, rng, config))

    return replicates


def spread(replicates: list[DixonColesParams]) -> pd.DataFrame:
    """Streuung der Ziehungen je Team -- die Groesse, um die es hier geht.
    """
    return pd.DataFrame(
        [
            {
                "team": team,
                "attack_mean": np.mean([p.attack[team] for p in replicates]),
                "attack_sd": np.std([p.attack[team] for p in replicates], ddof=1),
                "defense_mean": np.mean([p.defense[team] for p in replicates]),
                "defense_sd": np.std([p.defense[team] for p in replicates], ddof=1),
            }
            for team in sorted(replicates[0].teams)
        ]
    )
