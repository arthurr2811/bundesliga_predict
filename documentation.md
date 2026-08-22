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

## Datenquellen

- **football-data.co.uk** liefert die historischen Saisons 2016/17 bis 2025/26
  als CSV (~150 Spalten, v. a. Wettquoten verschiedener Buchmacher), lokal
  abgelegt unter `data/raw/historic_data/D1_<saison>.csv`.
- **OpenLigaDB** (`api.openligadb.de`) liefert den Spielplan und die
  Ergebnisse der laufenden Saison 2026/27 als JSON (`getmatchdata/bl1/2026`),
  inkl. noch nicht gespielter Spiele (`matchIsFinished: false`) sowie
  Torschützen/Torfolge für bereits ausgetragene Spiele.

Die beiden Quellen unterscheiden sich in Datumsformat, Teamnamen-Schreibweise
(z. B. `Bayern Munich` vs. `FC Bayern München`) und Detailtiefe – dafür sorgt
die Vereinheitlichungs-Pipeline (`src/bundesliga_predict/`).

## Einheitliches Datenformat

Nach der Vereinheitlichung landet jedes Spiel als eine Zeile in
`data/processed/matches.csv` mit den Spalten `season`, `date`, `matchday`,
`home_team`, `away_team`, `home_goals`, `away_goals`, `finished`.

Begründung für die Auswahl:
- **Gebraucht:** `home_goals`/`away_goals` sind die einzige Zielgröße, die das
  Dixon-Coles-Poisson-Modell braucht; `date`/`season` werden für die
  Zeit-Gewichtung und die Saisonwechsel-Behandlung benötigt; `matchday` zum
  Gruppieren (Neufitten nach Spieltag, offene Spiele für die Simulation
  bestimmen); `finished` unterscheidet gespielte von noch ausstehenden
  Spielen (letztere braucht die Saison-Simulation als Startpunkt).
  Teamnamen werden auf eine feste, kanonische Schreibweise gemappt (Basis:
  football-data.co.uk-Namen, da sie über alle Saisons konsistent sind),
  damit ein Team über beide Quellen hinweg als dasselbe Team erkannt wird.
- **Weggelassen:** Halbzeitstände, Schüsse, Ecken, Karten, Schiedsrichter,
  Zuschauerzahlen, Torschützen/Torminuten und sämtliche Wettquoten – all das
  fließt in ein reines Tor-basiertes Dixon-Coles-Modell nicht ein und bläht
  nur den Datensatz auf. (Wettquoten könnten später optional für einen
  Modell-Vergleich/Backtest gegen den Wettmarkt interessant sein, werden
  dafür aber aus den Rohdaten neu gelesen statt dauerhaft mitgeführt.)

## Log

_(Wird pro Schritt ergänzt: was wurde gebaut, welche Entscheidungen wurden
getroffen und warum, welche Probleme kamen auf.)_

## Quellen / Inspiration

- Dixon, Coles (1997): Modelling Association Football Scores and Inefficiencies in the Betting Market
- https://pena.lt/y/2021/06/24/predicting-football-results-using-python-and-dixon-and-coles/
- https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling-dixon-coles-and-time-weighting/
- https://github.com/opisthokonta/goalmodel
