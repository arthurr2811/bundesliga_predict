"""Parameter des Dixon-Coles-Modells: Container plus Umwandlung in/aus dem
flachen Vektor, mit dem der Optimierer arbeitet.

Modellgleichungen:

    log lambda_home = intercept + attack[home] - defense[away] + home_advantage
    log lambda_away = intercept + attack[away] - defense[home]

`intercept` fängt das Torniveau der Liga ab. Damit Angriffs- und
Abwehrwerte überhaupt eindeutig bestimmt sind, gilt die Nebenbedingung
`sum(attack) == 0` und `sum(defense) == 0`. Ein Team mit attack = defense = 0
ist damit exakt Ligadurchschnitt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

# Vektor-Layout: [intercept, home_advantage, rho, attack_frei, defense_frei]
# "frei" heisst: die letzten Werte fehlen, sie folgen aus der Summe-Null-Bedingung.
_N_GLOBAL = 3


@dataclass(frozen=True)
class DixonColesParams:
    """Gefittete Modellparameter."""

    teams: tuple[str, ...]
    attack: dict[str, float]
    defense: dict[str, float]
    home_advantage: float
    intercept: float
    rho: float
    fitted_through: date | None = None

    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Erwartete Tore beider Teams für eine Paarung."""
        base = self.intercept
        lambda_home = np.exp(
            base + self.attack[home_team] - self.defense[away_team] + self.home_advantage
        )
        lambda_away = np.exp(base + self.attack[away_team] - self.defense[home_team])
        return float(lambda_home), float(lambda_away)

    def to_dict(self) -> dict:
        return {
            "teams": list(self.teams),
            "attack": self.attack,
            "defense": self.defense,
            "home_advantage": self.home_advantage,
            "intercept": self.intercept,
            "rho": self.rho,
            "fitted_through": self.fitted_through.isoformat() if self.fitted_through else None,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> DixonColesParams:
        fitted_through = raw.get("fitted_through")
        return cls(
            teams=tuple(raw["teams"]),
            attack=dict(raw["attack"]),
            defense=dict(raw["defense"]),
            home_advantage=raw["home_advantage"],
            intercept=raw["intercept"],
            rho=raw["rho"],
            fitted_through=date.fromisoformat(fitted_through) if fitted_through else None,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> DixonColesParams:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def n_free_params(n_teams: int) -> int:
    """Länge des Optimierungsvektors für n Teams."""
    return _N_GLOBAL + 2 * (n_teams - 1)


def split_vector(vector: np.ndarray, n_teams: int) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Zerlegt den Optimierungsvektor in intercept, home_advantage, rho,
    attack und defense -- letztere bereits um den abhängigen letzten Wert
    ergänzt (Summe Null)."""
    intercept, home_advantage, rho = vector[:_N_GLOBAL]
    free = n_teams - 1
    attack_free = vector[_N_GLOBAL : _N_GLOBAL + free]
    defense_free = vector[_N_GLOBAL + free : _N_GLOBAL + 2 * free]

    attack = np.append(attack_free, -attack_free.sum())
    defense = np.append(defense_free, -defense_free.sum())
    return float(intercept), float(home_advantage), float(rho), attack, defense


def to_params(
    vector: np.ndarray, teams: tuple[str, ...], fitted_through: date | None = None
) -> DixonColesParams:
    intercept, home_advantage, rho, attack, defense = split_vector(vector, len(teams))
    return DixonColesParams(
        teams=teams,
        attack={team: float(value) for team, value in zip(teams, attack)},
        defense={team: float(value) for team, value in zip(teams, defense)},
        home_advantage=home_advantage,
        intercept=intercept,
        rho=rho,
        fitted_through=fitted_through,
    )


def to_vector(params: DixonColesParams) -> np.ndarray:
    """Umkehrung von `to_params`; vor allem für Tests und Warmstarts."""
    attack = np.array([params.attack[team] for team in params.teams[:-1]])
    defense = np.array([params.defense[team] for team in params.teams[:-1]])
    return np.concatenate(
        [[params.intercept, params.home_advantage, params.rho], attack, defense]
    )
