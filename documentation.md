# Bundesliga-Vorhersage – Projektdokumentation

Kleines Übungs- und Portfolio-Projekt: Vorhersage der Bundesliga-Saison
(Spielergebnisse, Tabelle, Meister-/Europapokal-/Abstiegswahrscheinlichkeiten)
mit einem Dixon-Coles-Poisson-Modell. Diese Datei hält fest, was gebaut wurde
und warum – als Grundlage für einen späteren Blogpost, nicht als vollständige
technische Doku.

## Idee & Ansatz

Jedes Team bekommt eine Angriffs- und eine Abwehrstärke sowie einen
Heimvorteil-Faktor (Dixon-Coles-Modell, eine Erweiterung des klassischen
Poisson-Tormodells). Daraus lassen sich Wahrscheinlichkeiten für jedes
Spielergebnis berechnen. Die restliche Saison wird per Monte-Carlo-Simulation
tausendfach durchgespielt, um Wahrscheinlichkeiten für Tabellenplätze,
Meisterschaft, internationale Plätze und Abstieg zu bekommen.

Der Ansatz orientiert sich an etablierten, öffentlich dokumentierten Modellen
(siehe Quellen unten).

## Log

_(Wird pro Schritt ergänzt: was wurde gebaut, welche Entscheidungen wurden
getroffen und warum, welche Probleme kamen auf.)_

## Quellen / Inspiration

- Dixon, Coles (1997): Modelling Association Football Scores and Inefficiencies in the Betting Market
- https://pena.lt/y/2021/06/24/predicting-football-results-using-python-and-dixon-and-coles/
- https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling-dixon-coles-and-time-weighting/
- https://github.com/opisthokonta/goalmodel
