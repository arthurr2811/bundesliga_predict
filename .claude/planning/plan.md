# Projektplan: Bundesliga-Vorhersage

Ziel: ML-Modell (Dixon-Coles/Poisson), das Spielergebnisse, Tabelle und
Meister-/Abstiegs-/Europapokal-Wahrscheinlichkeiten für die laufende Saison
vorhersagt, samt kleinem Frontend zur Anzeige. Läuft nach jedem Spieltag neu.

Zusätzlich Portfolio-/Übungsprojekt: `documentation.md` im Projekt-Root wird
laufend gepflegt (was gebaut wurde, welche Entscheidungen
warum getroffen wurden) als Basis für einen späteren Blogpost.

## Aktueller Stand
- Projektsetup und Datenpipeline abgeschlossen (`data/processed/matches.csv`,
  3367 Spiele, Saisons 2016/17–2026/27).
- Architektur festgelegt (siehe unten).
- Modellkern gebaut: `config`, `model/params`, `model/weights`,
  `model/likelihood`, `model/fit`, `predict/matrix`, abgesichert durch
  Parameter-Recovery-Test. Fit auf echten Daten laeuft (0,3 s, 30 Teams).
- Weiter mit Schritt 3: Backtest und Hyperparameter-Tuning. Erst danach sind
  Halbwertszeit, Saison-Abschlag und Prior-Staerke belegt statt geraten.

### Naechster Schritt (Stand 22.08.2026)
1. Backtest bauen (walk-forward, RPS/Log-Loss/Brier gegen die beiden
   Baselines). Der muss zuerst stehen, weil er die Messgrundlage fuer alles
   Weitere ist.
2. Danach als erste Verbesserung den Aufsteiger-Prior gegen den Backtest
   testen (siehe Befund unten).
3. Erst dann Hyperparameter-Grid-Search.

### Befund: der Aufsteiger-Prior sitzt an der falschen Stelle
Die Shrinkage zieht datenarme Teams aktuell Richtung 0, und 0 ist wegen
`sum(attack) = 0` exakt der Ligadurchschnitt. Das ist ein neutraler Prior
("wir wissen nichts ueber das Team") -- wir wissen aber mehr: intuitiv und gemessen ueber
19 Aufsteiger-Saisons in unseren Daten liegt ein Aufsteiger im Schnitt bei
attack -0.31 und defense -0.13, und 95 % der Aufsteiger sind im Angriff unter
Ligaschnitt. Die Shrinkage macht Aufsteiger also systematisch zu stark.

Am haertesten trifft es Teams ganz ohne Bundesliga-Historie (z. B.
Elversberg): die starten bei exakt 0, also als Durchschnittsteam. Teams wie
Hamburg oder Nuernberg haben wenigstens alte, abgewertete BL-Spiele.

Geplante Korrektur: Shrinkage nicht Richtung 0, sondern Richtung eines
Prior-Mittelwerts, der fuer datenarme Teams negativ ist. Das Shrinkage-Gewicht
dafuer existiert bereits (es haengt an der gewichteten Spielmasse je Team),
greift also automatisch nur dort, wo ein Team neu ist. Wichtig: die Streuung
ist gross (sigma ~ 0.25 im Angriff; Stuttgart 2020/21 kam mit +0.08 hoch,
Nuernberg 2018/19 mit -0.73) -- der Prior darf die Erwartung verschieben, aber
nicht so eng sein, dass ein starker Aufsteiger zu lange braucht, um im Modell
anzukommen. Ob es die Vorhersage wirklich verbessert, entscheidet der
Backtest, nicht die Plausibilitaet.

## Architektur

Vier entkoppelte Schichten, jede mit reinen Funktionen (DataFrame/Dataclass
rein und raus). Datei-I/O ausschliesslich in `pipeline.py`/`cli.py`, damit der
Backtest dieselben Funktionen mit abgeschnittenen Daten aufrufen kann.

    data/processed/matches.csv
          |
     [1] model/       Parameter schaetzen  -> data/models/params_<stand>.json
          |
     [2] predict/     Params -> Spiel-Wahrscheinlichkeiten
          |
     [3] simulation/  Monte-Carlo Restsaison -> Platz-/Titel-/Abstiegs-Wkt.
          |
     [4] data/output/*.json  <- Frontend liest nur das

### Modul-Layout

    src/bundesliga_predict/
      data/           historic_source, live_source, team_mapping, build_dataset
      model/
        params.py     Dataclass DixonColesParams (attack, defense, home_adv, rho)
        weights.py    Zeitgewichtung (Tages-Decay + Saisonwechsel-Malus)
        likelihood.py vektorisierte neg. gewichtete Log-Likelihood + tau
        fit.py        scipy.optimize.minimize (L-BFGS-B), Regularisierung
      predict/
        matrix.py     Params + Paarung -> Torematrix (0..10) inkl. tau
        outcomes.py   Matrix -> 1X2, exaktes Ergebnis, Over/Under
      simulation/
        table.py      Spiele -> Tabelle (Punkte, Tordifferenz, Tore)
        season.py     Monte-Carlo Restsaison -> Platzverteilung je Team
      evaluation/
        backtest.py   Walk-forward, RPS/LogLoss/Brier gegen Baselines
        tuning.py     Grid-Search der Hyperparameter ueber den Backtest
      pipeline.py     update -> fit -> predict -> simulate -> JSON
      config.py       Decay, Saison-Malus, Sim-Anzahl, Platz-Regeln
      cli.py          update | fit | simulate | backtest

### Querschnittliche Festlegungen
- **Zeitachse ist `date`, nicht `matchday`.** football-data liefert keinen
  Spieltag, deshalb ist `matchday` fuer die historischen Saisons leer. Alle
  Schnitte in Modell und Backtest laufen ueber das Datum; `matchday` ist reine
  Anzeige-Information der laufenden Saison.
- **Platz-Regeln in `config.py`**, nicht hart in der Simulation (Anzahl der
  CL-/EL-/Conference-Plaetze aendert sich mit dem UEFA-Koeffizienten).
- **Reproduzierbarkeit:** fester Seed fuer alle Monte-Carlo-Laeufe.

## 1. Projektsetup — erledigt
Python-Projektstruktur, Abhängigkeiten, historische Saisons von
football-data.co.uk, laufende Saison über OpenLigaDB, Team-Mapping,
einheitliches Format in `data/processed/matches.csv`.

## 2. Modellarchitektur
`log lambda_home = alpha_home - beta_away + gamma`,
`log lambda_away = alpha_away - beta_home`, Identifizierbarkeit ueber
`Summe alpha = 0`. Dazu `rho` fuer die Dixon-Coles-Korrektur der Zellen
0:0/1:0/0:1/1:1.

Zeitgewichtung: `w = exp(-xi * delta_t_Tage) * delta^(Saisonwechsel dazwischen)`.
Der Saisonwechsel-Malus ist bewusst ein eigener Parameter neben dem stetigen
Decay, weil Kaderumbruch sprunghaft wirkt und nicht linear.

Aufsteiger: keine 2.-Liga-Daten, stattdessen Regularisierung Richtung 0 mit
Staerke proportional zu 1/effektive Spielzahl. Teams mit wenig gewichteten
Daten werden zum Ligadurchschnitt gezogen, statt aus drei Spielen als Topteam
geschaetzt zu werden. Ob das reicht, zeigt der Backtest.

## 3. Training & Fitting
Maximum-Likelihood per `scipy.optimize.minimize` (L-BFGS-B) auf der
vektorisierten gewichteten Log-Likelihood. `xi`, `delta` und die
Regularisierungsstaerke sind Hyperparameter und werden nicht mitgeschaetzt,
sondern per Grid-Search ueber den Backtest optimiert.

Backtest (walk-forward): fuer jeden Spieltag ab 2018/19 nur mit Daten davor
fitten, dann RPS, Log-Loss und Brier der 1X2-Vorhersage messen. Baselines:
konstante Ligadurchschnitts-Wahrscheinlichkeiten als Untergrenze und die
margenbereinigten Buchmacherquoten aus `data/raw/historic_data/` als praktische
Obergrenze -- beide werden wie das Modell gegen die tatsaechlichen Ergebnisse
bewertet, der Markt ist also Vergleichsmassstab und keine Zielgroesse, die es
zu treffen gilt.

Tests: Parameter-Recovery (synthetische Saison aus bekannten Parametern
simulieren, zurueckfitten, Parameter vergleichen), Tabellenberechnung gegen
eine echte Abschlusstabelle, Symmetrie der tau-Korrektur.

## 4. Saison-Simulation
Monte-Carlo der offenen Spiele (`finished == False`). Gezogen wird aus der
abgeflachten Torematrix, nicht aus zwei unabhaengigen Poissons — sonst geht
die tau-Korrektur verloren. Vektorisiert ueber ~10.000 Laeufe. Auswertung zu
Wahrscheinlichkeiten je Team fuer Meisterschaft, CL/EL/Conference, Relegation
und Abstieg, plus erwartete Punkte und Platzverteilung.

## 5. Frontend
Statische Seite: die Pipeline schreibt JSON nach `data/output/`, eine einzelne
HTML-Seite mit Vanilla-JS rendert Spielvorhersagen, simulierte Tabelle und die
Wahrscheinlichkeiten. Kein Server, ueber GitHub Pages deploybar. Kein
Modellcode im Frontend.
