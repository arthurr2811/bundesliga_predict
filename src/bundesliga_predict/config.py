"""Zentrale Konstanten und Default-Hyperparameter.

Hyperparameter sind bewusst hier und nicht im Modellcode: sie werden nicht
mitgeschätzt, sondern später per Grid-Search über den Backtest bestimmt.
"""

# Grösste im Modell berücksichtigte Torzahl je Team. 10 deckt praktisch
# alle Bundesliga-Ergebnisse ab; die Restwahrscheinlichkeit ist verschwindend.
MAX_GOALS = 10

# Zeitgewichtung: nach so vielen Tagen zählt ein Spiel nur noch halb.
DEFAULT_HALF_LIFE_DAYS = 180.0

# Zusätzlicher multiplikativer Abschlag je Saisonwechsel zwischen Spiel und
# Stichtag. Kaderumbruch wirkt sprunghaft, nicht als stetiger Zerfall.
DEFAULT_SEASON_PENALTY = 0.8

# Regularisierung Richtung Ligadurchschnitt (Angriff/Abwehr = 0), formuliert
# als Standardabweichung eines Normal-Priors auf die Team-Stärken. Kleiner
# Wert = stärkeres Ziehen zum Durchschnitt.
DEFAULT_PRIOR_SD = 0.35

# Referenz-Datenmenge für die Shrinkage: bei so viel gewichteter Spielmasse
# wirkt die Regularisierung nur noch halb so stark. Grob eine halbe Saison.
PRIOR_MATCH_WEIGHT = 17.0

# Zulässiger Bereich für rho. Weiter aussen wird die tau-Korrektur für
# realistische Torerwartungen negativ.
RHO_BOUNDS = (-0.9, 0.9)
