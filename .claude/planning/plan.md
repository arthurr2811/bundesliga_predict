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
- Grid-Search erledigt (`evaluation/tuning.py`, `cli.py tune`): 504 Laeufe in
  zwei Stufen, getunt auf 2018/19-2023/24, geprueft auf dem zurueckgehaltenen
  Holdout 2024/25-2025/26. Defaults in `config.py` jetzt belegt statt geraten:
  Halbwertszeit 480 (war 180), Saison-Abschlag 0.65 (war 0.8), `prior_sd` 0.25
  (war 0.35), Prior-Mittelwert -0.50/-0.28 (doppelter Messwert).
  `PRIOR_MATCH_WEIGHT` in `PriorConfig` verschoben, Wert unveraendert 17.
- **Aktuelle Referenzwerte:** Modell RPS 0.2031 / Log-Loss 0.9918 / Brier
  0.5914 ueber 2448 Spiele; Holdout 0.2022; Ligadurchschnitt 0.2320,
  Markt 0.1978. Das Modell holt ~85 % des Abstands Baseline-Markt.
  Der Aufsteiger-Rueckstand ist geschlossen (+0.0054 gegen +0.0053 fuer die
  uebrigen Partien); offen bleibt der Rueckrunden-Rueckstand (+0.0100).

### Naechster Schritt (Stand 23.08.2026)
1. ~~Backtest neu erheben, `documentation.md` aktualisieren.~~ erledigt
2. ~~Aufsteiger-Prior gegen den Backtest testen.~~ erledigt
3. ~~Hyperparameter-Grid-Search.~~ erledigt (504 Laeufe, siehe Stand oben)
4. Schritt 4 (Simulation): `simulation/table.py` und `simulation/season.py`,
   Monte-Carlo der offenen Spiele -> Platz-/Titel-/Abstiegs-Wahrscheinlich-
   keiten. Detailplan unter "4. Saison-Simulation". Dann Schritt 5 (Frontend).

Modellseitig ist damit vorerst Schluss: der Grid-Search hat gezeigt, dass an
diesen vier Schrauben nichts mehr zu holen ist (breites Plateau). Wer das
Modell weiter verbessern will, braucht eine neue Idee, keine feinere
Einstellung -- offener Ansatzpunkt ist der Aufsteiger-Rueckstand in der
Rueckrunde (siehe `documentation.md`).

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

Ausgangslage (23.08.2026): 2026/27 liegt komplett als Fixture-Liste vor,
306 offene Spiele, erster Spieltag 28.08. Simuliert wird also zunaechst die
ganze Saison.

### Leitgedanke: Neuberechnung statt Fortschreibung
Es gibt kein Delta-Update nach einem Spieltag. Die Simulation ist eine reine
Funktion von (gespielte Spiele, offene Spiele, Params, Config); nach Spieltag n
laeuft derselbe Pfad wie vor Spieltag 1, nur mit mehr Historie:

    build_dataset (OpenLigaDB) -> fit(reference_date=heute,
    reference_season="2026/27") -> alle offenen Spiele vorhersagen ->
    Monte-Carlo -> data/output/*.json

Das entspricht genau dem, was der Backtest je Block schon tut, inklusive
`with_unknown_teams` fuer Aufsteiger ohne BL-Historie (2026/27: Elversberg).
Kein Cache, keine Zustandsdatei. `--as-of DATE` faellt gratis ab, weil ohnehin
alles ueber das Datum schneidet -- damit laesst sich die Prognose-Historie
(z. B. Meisterwahrscheinlichkeit ueber die Spieltage) nachtraeglich erzeugen,
statt sie mitzuschreiben.

### Festlegungen
- **Feste Parameter je Lauf (v1).** Ein Fit, dieselben Params in allen 10.000
  Laeufen; simuliert wird nur die Ergebnis-Unsicherheit, nicht die Unsicherheit
  ueber die Teamstaerke. Die Verteilungen sind dadurch etwas zu eng, vor dem
  1. Spieltag am staerksten. Ein Bootstrap ueber Refits bleibt als spaetere
  Erweiterung moeglich, blockiert aber nicht die erste Zahl.
- **Ausgabe: Endtabelle plus alle Einzelspiele.** Keine Spieltags-Verlaeufe im
  Sim-Lauf; Zwischenstaende kommen bei Bedarf ueber `--as-of`.
- **Teamstaerken bleiben innerhalb eines Laufs konstant** -- simulierte
  Ergebnisse fliessen nicht ins Modell zurueck.
- **Platz-Regeln in `config.py`** (Annahme: CL 1-4, EL 5, Conference 6,
  Relegation 16, Abstieg 17-18; Pokalsieger-Startplatz bleibt aussen vor).

### Module
`simulation/table.py`
- `table(matches, teams=None) -> DataFrame`: Spiele, S/U/N, Tore, Differenz,
  Punkte, Platz. `teams` explizit, damit Teams ohne Spiel (Saisonstart) mit
  Nullen auftauchen.
- Sortierung Punkte, Tordifferenz, erzielte Tore. Der direkte Vergleich nur
  hier, falls der Test gegen die echte Abschlusstabelle ihn braucht; in der
  Simulation wird bei Gleichstand gelost (betrifft <1 % der Laeufe, spart
  Faktor 10 Laufzeit).

`simulation/season.py`
- `sample_scores(params, fixtures, rng, n_sims)`: je Paarung einmal
  `score_matrix`, flach kumuliert, dann `np.searchsorted(cdf_i, u[:, i])`
  -> `(n_sims, n_fixtures)` Tore je Seite. Ein `searchsorted` pro Spiel,
  306 x 10.000 Ziehungen, unter einer Sekunde.
- Aggregation per `np.add.at` auf Team-Indizes -> Punkte/Tore/Differenz als
  `(n_sims, 18)`; Platzierung per `np.lexsort(..., axis=-1)` mit Zufalls-Key
  als Tiebreak, Platzverteilung per `bincount`.
- Ergebnis `SeasonForecast`: Platzverteilung (18x18), erwartete Punkte plus
  Quantile, Ereignis-Wahrscheinlichkeiten aus den Platz-Regeln.

`config.py`: `N_SIMULATIONS = 10_000`, fester Seed, Punkteregel, `PlaceRules`.

`pipeline.py` + `cli.py simulate [--as-of]`: Datei-I/O nur hier. Ausgabe nach
`data/output/`: `fixtures.json` (jedes offene Spiel mit 1X2, erwarteten Toren,
wahrscheinlichstem Ergebnis), `table.json` (aktuelle + erwartete
Abschlusstabelle), `probabilities.json` (Platzverteilung und Ereignisse je
Team), `meta.json` (Stand, Seed, Anzahl Laeufe).

### Reihenfolge
1. `table.py` + Test gegen die echte Abschlusstabelle 2025/26.
2. `season.py` + Tests: Reproduzierbarkeit bei festem Seed, Platzverteilung
   summiert zeilen- und spaltenweise auf 1, degenerierte Params (ein
   uebermaechtiges Team wird immer Erster), Monte-Carlo-Fehler bei 10.000
   Laeufen ~ +/-0,5 Prozentpunkte.
3. Config, Pipeline, CLI mit `--as-of`.
4. JSON-Ausgabe, dann `documentation.md`.
5. Kalibrierungs-Check (nach dem Frontend): `--as-of` ueber jeden Spieltag
   2024/25 und 2025/26, pruefen ob "30 % auf Top 4" in ~30 % der Faelle
   eingetreten ist. Die einzige echte Pruefung der Simulationsschicht.

## 5. Frontend
Statische Seite: die Pipeline schreibt JSON nach `data/output/`, eine einzelne
HTML-Seite mit Vanilla-JS rendert Spielvorhersagen, simulierte Tabelle und die
Wahrscheinlichkeiten. Kein Server, ueber GitHub Pages deploybar. Kein
Modellcode im Frontend.
