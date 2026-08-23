# Projektplan: Bundesliga-Vorhersage

Ziel: ML-Modell (Dixon-Coles/Poisson), das Spielergebnisse, Tabelle und
Meister-/Abstiegs-/Europapokal-Wahrscheinlichkeiten für die laufende Saison
vorhersagt, samt kleinem Frontend zur Anzeige. Läuft nach jedem Spieltag neu.

Zusätzlich Portfolio-/Übungsprojekt: `documentation.md` im Projekt-Root wird
laufend gepflegt (was gebaut wurde, welche Entscheidungen
warum getroffen wurden) als Basis für einen späteren Blogpost.

## Aktueller Stand
- Projektsetup und Datenpipeline abgeschlossen (`data/processed/matches.csv`,
  3366 Spiele = 11 Saisons a 306, 2016/17–2026/27, mit Spieltagsnummer).
- Architektur festgelegt (siehe unten).
- Modellkern gebaut: `config`, `model/params`, `model/weights`,
  `model/likelihood`, `model/fit`, `predict/matrix`, abgesichert durch
  Parameter-Recovery-Test. Fit auf echten Daten laeuft (0,3 s, 30 Teams).
- Vorhersage-Ebene gebaut: `predict/outcomes` (Matrix -> 1X2, exaktes
  Ergebnis, Over/Under).
- Backtest steht: `evaluation/metrics`, `evaluation/baselines`,
  `evaluation/backtest`, `evaluation/report`, plus `cli.py backtest`.
  Walk-forward ueber 2448 Spiele (2018/19-2025/26), ~72 s Laufzeit
- Blockbildung auf echte Spieltagsnummern umgestellt (`matchday_source`),
  `team_mapping` auf alle Vereine 2016/17-2026/27 vervollstaendigt.
  352 Bloecke, keiner mischt Spieltage, 12 abgetrennte Nachholpartien.
- Backtest nach der Umstellung neu erhoben (23.08.2026), `documentation.md`
  aktualisiert. Referenzwerte: Modell RPS 0.2049 / Log-Loss 0.9983 /
  Brier 0.5957, Ligadurchschnitt 0.2320, Markt 0.1978.
- Aufsteiger-Prior gebaut (`model/prior.py`, `PriorConfig`): Shrinkage zieht
  auf einen gemessenen Mittelwert (Angriff -0.25, Abwehr -0.14) statt auf den
  Ligadurchschnitt. Vom Backtest bestaetigt, RPS 0.2049 -> 0.2044; der
  Sonderfall "Team ohne Historie" im Backtest ist damit weg. Details in
  `documentation.md`.

### Naechster Schritt (Stand 23.08.2026)
1. ~~Backtest neu laufen lassen und `documentation.md` aktualisieren.~~ erledigt
2. ~~Aufsteiger-Prior gegen den Backtest testen.~~ erledigt
3. Hyperparameter-Grid-Search (`evaluation/tuning.py`) -- Zuschnitt unten.
   Erst danach sind die Defaults in `config.py` belegt statt geraten.
4. Dann Schritt 4 (Simulation) und Schritt 5 (Frontend).

### Zuschnitt Grid-Search (festgelegt 23.08.2026)

**Tuning/Holdout.** Der Grid laeuft nur auf 2018/19-2023/24 (1836 Spiele).
2024/25-2025/26 (612 Spiele) bleiben unangetastet und werden ganz am Ende
einmal ausgewertet. Sonst waere die berichtete Modellguete geschoent -- bei
~150 Kombinationen gewinnt sonst auch Rauschen mit. Dafuer braucht
`BacktestConfig` ein `end_season`.

**Zwei Stufen.** Stufe A grob (108 Laeufe, ~16 min bei 6 Workern):

    half_life_days   90, 180, 365, 730
    season_penalty   0.6, 0.8, 1.0
    prior_sd         0.15, 0.25, 0.35
    prior_scale      0.0, 1.0, 2.0      (Faktor auf die gemessenen -0.25/-0.14)

`prior_scale` ist eine Achse statt zwei: das Verhaeltnis Angriff/Abwehr kommt
aus der Messung, offen ist nur die Staerke. Stufe B verfeinert um den Sieger
und nimmt `PRIOR_MATCH_WEIGHT` (jetzt fest 17.0) als fuenfte Achse dazu; der
Wert wandert dafuer aus `config.py` in `PriorConfig`.

**Regeln, vorher festgelegt:**
- Zielgroesse ist RPS. Log-Loss und Brier laufen mit, sind aber Kontrolle und
  kein Auswahlkriterium -- sonst sucht man sich hinterher das Mass aus.
- Gewinnt ein Randwert, wird der Grid erweitert, nicht der Rand genommen.
- Nicht das Argmin nehmen, sondern das Plateau: liegen viele Kombinationen
  innerhalb von ~0.0005, ist das Argmin Rauschen. RPS je Saison wird
  mitgeschrieben, um zu sehen, ob ein Sieger nur eine Saison traegt.

**Betrieb:** Ergebnisse zeilenweise in eine CSV anhaengen, bekannte
Kombinationen beim Neustart ueberspringen (fortsetzbar).

**Stand:** `tuning.py` und `cli.py tune` gebaut, Smoke-Tests bestanden --
Defaults ueber alle acht Saisons reproduzieren exakt RPS 0.2044, auf der
Tuning-Teilmenge sind es 0.2043 (1836 Spiele), und die RPS je Saison sind in
beiden Laeufen identisch (`end_season` schneidet also wirklich nur die
Auswertung). 6 Laeufe mit 6 Workern brauchen 31 s, Stufe A also ~10 min.
Stufe A ist noch nicht gestartet.

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
      data/           historic_source, live_source, matchday_source,
                      team_mapping, build_dataset
      model/
        params.py     Dataclass DixonColesParams (attack, defense, home_adv, rho)
        weights.py    Zeitgewichtung (Tages-Decay + Saisonwechsel-Malus)
        prior.py      PriorConfig (sd + Mittelwert), Teams ohne Historie
        likelihood.py vektorisierte neg. gewichtete Log-Likelihood + tau
        fit.py        scipy.optimize.minimize (L-BFGS-B), Regularisierung
      predict/
        matrix.py     Params + Paarung -> Torematrix (0..10) inkl. tau
        outcomes.py   Matrix -> 1X2, exaktes Ergebnis, Over/Under
      simulation/
        table.py      Spiele -> Tabelle (Punkte, Tordifferenz, Tore)
        season.py     Monte-Carlo Restsaison -> Platzverteilung je Team
      evaluation/
        metrics.py    RPS, Log-Loss, Brier fuer 1X2
        baselines.py  Ligadurchschnitt + margenbereinigte Buchmacherquoten
        backtest.py   Walk-forward: Bloecke, Fit je Block, Vorhersagen
        report.py     Vergleichstabellen Modell vs. Baselines
        tuning.py     Grid-Search der Hyperparameter ueber den Backtest (offen)
      pipeline.py     update -> fit -> predict -> simulate -> JSON
      config.py       Decay, Saison-Malus, Sim-Anzahl, Platz-Regeln
      cli.py          update | fit | simulate | backtest

### Querschnittliche Festlegungen
- **Zeitachse ist `date`, nicht `matchday`.** Alle Schnitte in Modell und
  Backtest laufen ueber das Datum. `matchday` gruppiert nur (ein Block im
  Backtest = ein Spieltag), es schneidet nie.
- **`matchday` ist fuer alle Saisons gefuellt.** football-data liefert keine
  Spieltagsnummer, OpenLigaDB dagegen auch fuer vergangene Saisons.
  Abgeschlossene Saisons werden einmal abgerufen und liegen in
  `data/raw/matchdays.csv`; laufend geht nur die aktuelle Saison uebers Netz.
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

Aufsteiger: keine 2.-Liga-Daten, stattdessen Regularisierung mit Staerke
proportional zu 1/effektive Spielzahl, gezogen auf einen gemessenen
Aufsteiger-Mittelwert statt auf den Ligadurchschnitt. Weil das
Shrinkage-Gewicht mit wachsender Datenmenge gegen 0 geht, braucht es keine
Liste, wer Aufsteiger ist.

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
