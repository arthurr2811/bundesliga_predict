"""Zentrale Konstanten und Default-Hyperparameter.

Die Werte stammen aus dem Grid-Search über den Backtest (504 Läufe in
zwei Stufen, getunt auf 2018/19-2023/24).
"""

# Grösste im Modell berücksichtigte Torzahl je Team. 10 deckt praktisch
# alle Bundesliga-Ergebnisse ab; die Restwahrscheinlichkeit ist verschwindend.
MAX_GOALS = 10

# Zeitgewichtung: nach so vielen Tagen zählt ein Spiel nur noch halb.
DEFAULT_HALF_LIFE_DAYS = 480.0

# Zusätzlicher multiplikativer Abschlag je Saisonwechsel zwischen Spiel und
# Stichtag. Kaderumbruch wirkt sprunghaft, nicht als stetiger Zerfall.
DEFAULT_SEASON_PENALTY = 0.65

# Regularisierung der Team-Stärken, formuliert als Normal-Prior.
DEFAULT_PRIOR_SD = 0.25

# Wohin datenarme Teams gezogen werden. Nicht auf den Ligadurchschnitt (0),
# denn wer wenig Bundesliga-Historie hat, ist fast immer ein Aufsteiger -- und
# Aufsteiger sind im Schnitt schwächer als die Liga.
DEFAULT_PRIOR_ATTACK = -0.50
DEFAULT_PRIOR_DEFENSE = -0.28

# Referenz-Datenmenge für die Shrinkage: bei so viel gewichteter Spielmasse
# wirkt die Regularisierung nur noch halb so stark.
#
# wie viel stärker datenarme Teams gezogen werden als
# datenreiche.
PRIOR_MATCH_WEIGHT = 17.0

# Punkte je Ausgang
POINTS_WIN = 3
POINTS_DRAW = 1

# Läufe der Saison-Simulation
N_SIMULATIONS = 10_000

# Fester Seed: dieselben Daten müssen dieselbe Prognose ergeben.
SIMULATION_SEED = 20262027

# Bootstrap-Ziehungen der Modellparameter
N_BOOTSTRAP = 100
BOOTSTRAP_SEED = 8112026

# Streuung, mit der Teams ohne jede Bundesliga-Historie je Ziehung um den
# Prior-Mittelwert gestreut werden
DEFAULT_UNKNOWN_ATTACK_SD = 0.40
DEFAULT_UNKNOWN_DEFENSE_SD = 0.32

# Was welcher Tabellenplatz bedeutet, als Platzbereich (von, bis).
PLACE_RULES = {
    "champion": (1, 1),
    "champions_league": (1, 4),
    "europa_league": (5, 5),
    "conference_league": (6, 6),
    "relegation_playoff": (16, 16),
    "relegated": (17, 18),
}

# Zulässiger Bereich für rho. Weiter aussen wird die tau-Korrektur für
# realistische Torerwartungen negativ.
RHO_BOUNDS = (-0.9, 0.9)
