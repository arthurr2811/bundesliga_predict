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

## Modellkern

Das Modell beschreibt die Tore beider Teams als Poisson-verteilt:

    log lambda_heim = intercept + attack[heim] - defense[gast] + heimvorteil
    log lambda_gast = intercept + attack[gast] - defense[heim]

`intercept` ist das Torniveau der Liga. Damit Angriffs- und Abwehrwerte
eindeutig sind, gilt `sum(attack) = sum(defense) = 0`. Angenehmer Nebeneffekt:
ein Team mit 0/0 ist exakt Ligadurchschnitt -- genau der Punkt, auf den
Aufsteiger gezogen werden.

Dazu die **Dixon-Coles-Korrektur** `tau`: zwei unabhaengige Poisson-
Verteilungen unterschaetzen systematisch 0:0 und 1:1 und ueberschaetzen
1:0/0:1. Ein einzelner Parameter `rho` korrigiert diese vier Zellen. Auf den
echten Daten faellt er negativ aus (rund -0.04), also genau in die von Dixon
und Coles beschriebene Richtung.

Drei Entscheidungen, die nicht aus dem Paper folgen:

- **Zeitgewichtung mit zwei Effekten.** Ein stetiger Zerfall ueber die Tage
  (Halbwertszeit) *plus* ein zusaetzlicher Abschlag je Saisonwechsel. Zwischen
  zwei Saisons wechseln Spieler, Trainer und drei Vereine -- dieser Bruch ist
  sprunghaft und laesst sich mit einem stetigen Zerfall nicht abbilden.
- **Aufsteiger per Shrinkage.** Ein Normal-Prior zieht Angriff und Abwehr zum
  Ligadurchschnitt, und zwar umso staerker, je weniger gewichtete Spielmasse
  ein Team hat. Ohne das wuerde ein Aufsteiger nach drei guten Spielen als
  Topteam geschaetzt. Die Alternative -- 2.-Liga-Daten mitfitten -- ist
  bewusst zurueckgestellt, bis der Backtest zeigt, ob sich der Aufwand lohnt.
- **Hyperparameter werden nicht mitgeschaetzt.** Halbwertszeit, Saison-Abschlag
  und Prior-Staerke stehen in `config.py` und werden spaeter per Grid-Search
  ueber den Backtest bestimmt. Das trennt sauber, was die Daten sagen, von dem,
  was man einstellt.

Geschaetzt wird per Maximum-Likelihood (`scipy`, L-BFGS-B) auf der
gewichteten Log-Likelihood. Bei 30 Teams und 3060 Spielen dauert ein Fit
etwa 0,3 Sekunden -- schnell genug, um im Backtest tausende Fits zu rechnen.

### Wie man so etwas testet

Bei einem statistischen Modell ist der naheliegende Test -- Ergebnis gegen
fest verdrahtete Zahlen -- wertlos: er zementiert nur, was der Code heute
tut. Stattdessen **Parameter-Recovery**: aus bekannten Parametern werden
synthetische Saisons gezogen, daraus wird zurueckgefittet, und geprueft wird,
ob die Wahrheit wieder herauskommt. Der Fit erweist sich als erwartungstreu,
und der Fehler halbiert sich sauber, wenn man die Datenmenge vervierfacht.
Ein Vorzeichenfehler, ein vertauschter Index oder eine falsch normierte
Torematrix faellt damit sofort auf.

## Log

**Datenpipeline.** Historische Saisons und laufende Saison vereinheitlicht.
Dabei aufgefallen: football-data.co.uk liefert keine Spieltagsnummer. Statt
sie kuenstlich zu rekonstruieren, laufen alle Schnitte in Modell und Backtest
ueber das Datum; `matchday` bleibt reine Anzeige-Information der laufenden
Saison.

**Modellkern.** Parameter-Container, Zeitgewichtung, Likelihood und Fit
gebaut, abgesichert durch den Recovery-Test. Erster Fit auf echten Daten:
Heimvorteil 0.19 (rund 20 % mehr Tore zu Hause), Ligaschnitt 1.39 Tore je
Team, Bayern mit Abstand vorn. Noch offen: Backtest und Hyperparameter-Tuning
-- bis dahin sind die Default-Hyperparameter geraten, nicht belegt.

## Quellen / Inspiration

- Dixon, Coles (1997): Modelling Association Football Scores and Inefficiencies in the Betting Market
- https://pena.lt/y/2021/06/24/predicting-football-results-using-python-and-dixon-and-coles/
- https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling-dixon-coles-and-time-weighting/
- https://github.com/opisthokonta/goalmodel
