# Projektplan: Bundesliga-Vorhersage

Ziel: Dixon-Coles-Poisson-Modell, das Spielergebnisse, Tabelle und
Meister-/Abstiegs-/Europapokal-Wahrscheinlichkeiten der laufenden Saison
vorhersagt, samt kleinem Frontend. Laeuft nach jedem Spieltag neu.

Zusaetzlich Portfolio-/Uebungsprojekt: `documentation.md` haelt fest, was
gebaut wurde und warum -- Basis fuer einen spaeteren Blogpost. Dort stehen die
Begruendungen; hier nur der Plan.

## Stand (23.08.2026)

Schritte 1-4 erledigt: Datenpipeline, Modellkern, Backtest, Grid-Search,
Simulation und Ausgabe. `cli.py update` faehrt die ganze Kette und schreibt
`data/output/{meta,matches,table,probabilities}.json`. 91 Tests gruen.

Referenzwerte Backtest (2448 Spiele, 2018/19-2025/26): Modell RPS 0.2031,
Ligadurchschnitt 0.2320, Markt 0.1978 -- das Modell holt ~85 % des Abstands.
Erste Prognose 2026/27 steht (Bayern 93 % Meister, Elversberg 94 % Abstieg).

## Naechste Schritte

1. **Frontend.** Eine statische HTML-Seite mit Vanilla-JS, die nur
   `data/output/*.json` liest -- kein Server, kein Modellcode, ueber GitHub
   Pages deploybar. Inhalte: Spielvorhersagen je Spieltag, aktuelle und
   erwartete Tabelle, Platzverteilung je Team.
2. **Kalibrierungs-Check.** Die einzige echte Pruefung der Simulationsschicht:
   per `--as-of` fuer jeden Spieltag vergangener Saisons prognostizieren, am
   Saisonende nachzaehlen. Zwei Auswertungen -- treten Ereignisse mit 90 %
   Prognose auch in 90 % der Faelle ein, und liegt die echte Endpunktzahl in
   90 % der Faelle im 90-%-Intervall? Ergebnis entscheidet ueber Punkt 3.
3. **Offen gelassen: Parameter-Unsicherheit.** Die Simulation haelt die
   Modellparameter in allen Laeufen fest, die Verteilungen sind dadurch zu eng.
   Machbar per Exp(1)-Multiplikator auf die vorhandenen Spielgewichte
   (Bayesian Bootstrap, ~100 Refits, ~30 s je Update) plus Ziehen aus der
   gemessenen Aufsteiger-Streuung fuer Teams ohne Historie. Bewusst
   zurueckgestellt, bis Punkt 2 zeigt, ob es noetig ist.

Modellseitig ist sonst vorerst Schluss: der Grid-Search hat ein breites
Plateau gefunden, weitere Verbesserung braucht eine neue Idee statt feinerer
Einstellung. Offener Ansatzpunkt bleibt der Aufsteiger-Rueckstand in der
Rueckrunde (siehe `documentation.md`).

## Architektur

Vier entkoppelte Schichten, reine Funktionen (DataFrame/Dataclass rein und
raus). Datei-I/O ausschliesslich in `pipeline.py`/`cli.py`, damit Backtest und
`--as-of` dieselben Funktionen mit abgeschnittenen Daten aufrufen koennen.

    data/processed/matches.csv
          |
     [1] model/       Parameter schaetzen
          |
     [2] predict/     Params -> Spiel-Wahrscheinlichkeiten
          |
     [3] simulation/  Monte-Carlo Restsaison -> Platz-/Titel-/Abstiegs-Wkt.
          |
     [4] data/output/*.json  <- Frontend liest nur das

### Modul-Layout

    src/bundesliga_predict/
      historic_source, live_source, matchday_source, team_mapping,
      build_dataset
      model/
        params.py     DixonColesParams (attack, defense, home_adv, rho)
        weights.py    Zeitgewichtung (Tages-Decay + Saisonwechsel-Malus)
        prior.py      PriorConfig, Teams ohne Historie
        likelihood.py vektorisierte neg. gewichtete Log-Likelihood + tau
        fit.py        scipy.optimize.minimize (L-BFGS-B)
      predict/
        matrix.py     Params + Paarung -> Torematrix (0..10) inkl. tau
        outcomes.py   Matrix -> 1X2, exaktes Ergebnis, Over/Under
      simulation/
        table.py      Spiele -> Tabelle; eine Sortierregel fuer beide Ebenen
        season.py     Monte-Carlo Restsaison -> SeasonForecast
      evaluation/
        metrics, baselines, backtest, report, tuning
      pipeline.py     fit -> predict -> simulate -> JSON
      config.py       Decay, Saison-Malus, Sim-Anzahl, Seed, Platz-Regeln
      cli.py          update | simulate | backtest | tune

### Querschnittliche Festlegungen

- **Zeitachse ist `date`, nicht `matchday`.** Alle Schnitte laufen ueber das
  Datum; `matchday` gruppiert nur, es schneidet nie.
- **Kein Fortschreiben.** Der Stand steckt allein in Daten und Stichtag; nach
  jedem Spieltag laeuft dieselbe Kette neu. Deshalb faellt `--as-of` gratis ab:
  alles nach dem Stichtag gilt als ungespielt.
- **Platz-Regeln in `config.py`** (`PLACE_RULES`), nicht hart in der
  Simulation -- die Zahl der Europapokal-Plaetze aendert sich.
- **Reproduzierbarkeit:** fester Seed fuer alle Monte-Carlo-Laeufe.
- **Kein Modellcode im Frontend.** Es liest ausschliesslich `data/output/`.
