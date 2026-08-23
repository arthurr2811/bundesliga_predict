# Bundesliga-Vorhersage – Projektdokumentation

Übungs-/Portfolio-Projekt: Vorhersage der Bundesliga-Saison (Ergebnisse,
Tabelle, Meister-/Europapokal-/Abstiegswahrscheinlichkeiten) mit einem
Dixon-Coles-Poisson-Modell plus Monte-Carlo-Simulation. Grundlage für einen
späteren Blogpost.

## Idee & Ansatz

Jedes Team bekommt Angriffs-/Abwehrstärke plus Heimvorteil (Dixon-Coles,
Erweiterung des Poisson-Tormodells). Daraus folgen Ergebniswahrscheinlichkeiten
je Spiel; die Restsaison wird per Monte-Carlo tausendfach simuliert, um
Tabellenplatz-, Meister-, Europapokal- und Abstiegswahrscheinlichkeiten zu
bekommen. Orientiert an Dixon/Coles (1997) und öffentlich dokumentierten
Umsetzungen (siehe Quellen).

## Datenquellen

- **football-data.co.uk**: historische Saisons 2016/17–2025/26 als CSV,
  lokal unter `data/raw/historic_data/`.
- **OpenLigaDB**: Spielplan/Ergebnisse der laufenden Saison als JSON, inkl.
  Spieltagsnummern (auch für vergangene Saisons – wichtig für den Backtest).

Unterschiedliche Datumsformate/Teamnamen werden über eine
Vereinheitlichungs-Pipeline (`src/bundesliga_predict/`) auf ein kanonisches
Format gemappt.

## Einheitliches Datenformat

`data/processed/matches.csv`: `season`, `date`, `matchday`, `home_team`,
`away_team`, `home_goals`, `away_goals`, `finished`. Bewusst weggelassen:
Halbzeitstände, Schüsse, Karten, Torschützen, Wettquoten (letztere werden nur
für den Backtest live aus den Rohdaten gelesen).

## Modellkern

    log lambda_heim = intercept + attack[heim] - defense[gast] + heimvorteil
    log lambda_gast = intercept + attack[gast] - defense[heim]

Normierung `sum(attack) = sum(defense) = 0`, sodass ein Team mit 0/0 exakt
Ligadurchschnitt ist – der Punkt, auf den Aufsteiger gezogen werden. Dazu die
**Dixon-Coles-Korrektur** `rho` (gemessen ≈ -0.04) für die Unter-/Überschätzung
von 0:0/1:1 bzw. 1:0/0:1 durch zwei unabhängige Poissons.

Drei Design-Entscheidungen jenseits des Papers:

- **Zeitgewichtung**: stetiger Zerfall (Halbwertszeit) *plus* Abschlag pro
  Saisonwechsel (Kaderumbruch wirkt sprunghaft, nicht stetig).
- **Aufsteiger per Shrinkage**: Normal-Prior zieht Angriff/Abwehr Richtung
  Mittelwert, stärker je weniger gewichtete Spielmasse ein Team hat.
- **Hyperparameter** (Halbwertszeit, Saison-Abschlag, Prior-Stärke) stehen in
  `config.py` und werden separat per Grid-Search bestimmt statt mitgefittet.

Fit per Maximum-Likelihood (`scipy`, L-BFGS-B), ~0,3s für 30 Teams/3060 Spiele.
Getestet über **Parameter-Recovery**: synthetische Saisons aus bekannten
Parametern ziehen, zurückfitten, prüfen ob die Wahrheit rauskommt (Fit ist
erwartungstreu, Fehler halbiert sich bei 4x Daten).

## Backtest

**Walk-forward** über Spieltags-Blöcke: Modell wird vor jedem Block nur auf
früheren Spielen gefittet. Blöcke werden über echte OpenLigaDB-Spieltagsnummern
gebildet (nicht per Datums-Heuristik – zwei frühere Heuristikversuche scheiterten
an englischen Wochen bzw. Nachholspielen). Innerhalb eines Spieltags wird nur
nach >3 Tagen Abstand getrennt (verlegte Spiele). Ergebnis: 352 Blöcke über
8 Saisons, keiner mischt zwei Spieltage.

Drei Maße: **RPS** (kennt die Ordnung der Ausgänge), **Log-Loss** (bestraft
selbstsichere Fehler hart, ist das Fit-Ziel), **Brier** (beschränkt, robust).
Zwei Baselines: Ligadurchschnitt (untere Grenze) und margenbereinigter
Buchmachermarkt (Vergleichsmaßstab, kennt Aufstellungen/Verletzungen).

Ergebnis über 2448 Spiele (2018/19–2025/26):

| | RPS | Log-Loss | Brier |
|---|---|---|---|
| Ligadurchschnitt | 0.2320 | 1.0736 | 0.6499 |
| Modell, erster Stand | 0.2049 | 0.9983 | 0.5957 |
| **Modell, aktuell** | **0.2031** | **0.9918** | **0.5914** |
| Markt | 0.1978 | 0.9741 | 0.5788 |

Modell holt ~85% des Abstands Baseline→Markt, schlägt die Baseline in allen
8 Saisons. Aufgeteilt nach Aufsteiger-Beteiligung zeigte sich anfangs ein
doppelt so großer Rückstand zum Markt bei Aufsteiger-Partien (+0.0118 vs.
+0.0057) – gleichmäßig über Hin-/Rückrunde verteilt, kein Saisonverlaufs-Effekt
(der frühere Anschein war ein Artefakt der alten Blockbildung).

### Statistische Signifikanz des Backtests

Naiver Standardfehler des RPS (0.0027) wäre entmutigend, ist aber falsch, weil
Vergleiche gepaart sind (gleiche Spiele). Standardfehler der **gepaarten
Differenz** ist 12-40x kleiner (0.00007–0.00023). Auswahl aus K Kombinationen
kostet nur ~√log(K) an Overfitting-Bonus (klein). Konsequenz: Grid **breit**
statt fein, ein Optimum am Rand ist ein echter Fehler.

## Aufsteiger-Prior

Gemessen über 19 Aufsteiger-Saisons (eigener Fit je Saison, ohne Shrinkage):
Angriff im Schnitt -0.25, Abwehr -0.14 unter Ligaschnitt (95% bzw. 74% der
Fälle unter Durchschnitt) – der alte "neutrale" Prior (Ziel = 0) war also zu
optimistisch. Korrektur: Prior zieht auf den gemessenen Mittelwert statt auf 0,
Gewicht bleibt an der gewichteten Spielmasse (kein Aufsteiger-Flag nötig – für
etablierte Teams verschwindet der Effekt automatisch).

| Prior-Stärke | RPS gesamt | RPS Aufsteiger | Abstand Markt |
|---|---|---|---|
| 0.0 (vorher) | 0.2049 | 0.2032 | +0.0118 |
| 1.0 (gemessen, Default) | 0.2044 | 0.2010 | +0.0096 |
| 1.5 | 0.2042 | 0.2002 | +0.0089 |

Gewinn sitzt komplett in der Hinrunde (+0.0125 → +0.0079 bei Stärke 1.0),
Rückrunde unverändert – sobald ein Aufsteiger eigene Daten hat, verblasst der
Prior. Default bleibt beim gemessenen Wert 1.0 statt dem besser messenden 1.5,
da die vier Hyperparameter gemeinsam getunt werden (kein lokales Optimum vorab).

## Grid-Search (Hyperparameter)

504 Läufe in zwei Stufen, getunt nur auf 2018/19–2023/24 (Holdout: die zwei
jüngsten Saisons). Ergebnis: **flaches Plateau**, 49 von 324 Kombinationen in
Stufe B liegen ununterscheidbar gleichauf. Gewählt wurde der über alle Achsen
stabilste Wert statt des Argmin. Auffällig: optimale Halbwertszeit liegt bei
~1,5 Jahren (viel länger als ursprünglich angenommene 180 Tage); Saison-Abschlag
bringt messbar etwas.

Holdout-Vergleich (gepaart, vorher/nachher Tuning):

| Teilmenge | n | vorher | nachher | Gewinn |
|---|---|---|---|---|
| Tuning-Menge | 1836 | 0.2043 | 0.2034 | +0.0009 |
| **Holdout** | 612 | 0.2046 | **0.2022** | **+0.0023** |

Gewinn im Holdout größer als in der Tuning-Menge → kein Overfitting.
Aufsteiger-Rückstand zum Markt ist nach dem Tuning in der Hinrunde praktisch
verschwunden (+0.0007), sitzt aber jetzt komplett in der Rückrunde (+0.0100,
unverändert über alle Tuning-Stufen) – ungeklärt, Vermutung: Winterpause/
Trainerwechsel treffen Abstiegskandidaten härter, der Markt weiß das, das
Modell nicht.

## Tabelle & Monte-Carlo

Sortierregel arbeitet auf der letzten Achse (`(n_teams,)` bzw.
`(n_sim, n_teams)`), eine Implementierung für Ist-Stand und Simulation.
Direkter Vergleich (DFL-Regel) bewusst weggelassen: entschied in 10 Saisons
kein einziges Mal über eine Platzierung. Getestet gegen die OpenLigaDB-
Abschlusstabellen aller 10 Saisons – exakte Übereinstimmung.

Monte-Carlo zieht aus der flachgelegten Torematrix (behält die Dixon-Coles-
Korrektur, anders als zwei unabhängige Poissons). 306 offene Spiele in 0,2s,
Fit davor 0,3s. **Modellparameter bleiben über alle Läufe fest** – simuliert
wird nur Ergebnis-Unsicherheit, nicht Teamstärken-Unsicherheit (siehe
Kalibrierung/Bootstrap unten).

## Pipeline & Ausgabe

Kein Zustand, kein Fortschreiben – jeder Lauf nimmt nur einen Stichtag
(`--as-of`): alles danach gilt als ungespielt. Das erlaubt Rekonstruktion
jeder historischen Prognose (Grundlage für Kalibrierung). `cli.py update`
= holen + fitten + simulieren + schreiben; `simulate` nur der Rechenteil.

Ausgabe in `data/output/` (~130 KB): `meta.json`, `matches.json` (inkl. Top-3
wahrscheinlichste Ergebnisse), `table.json`, `probabilities.json`.

## Frontend

`frontend/`: statische Vanilla-JS-Seite, liest nur die vier JSON-Dateien
(kein Server/Build, deploybar via GitHub Pages). Erwartete Tabelle oben,
Spiele mit Spieltag-Navigation darunter. Zwei Detailentscheidungen: nie 0%/
100% anzeigen (Simulation hat nur 10.000 Läufe); "wahrscheinlichstes Ergebnis"
≠ "wahrscheinlichster Ausgang" (bei 202/306 Partien ist 1:1 das Modus-Ergebnis,
aber nie X der wahrscheinlichste Ausgang – Siegwahrscheinlichkeit verteilt
sich auf viele Scorelines) – deshalb Top-3 mit Wahrscheinlichkeit beim Hovern.

## Kalibrierung

`cli.py calibrate`: 272 Stichtage über 8 Saisons, je 10.000 Simulationen,
gegen den echten Saisonausgang geprüft.

**Ereignis-Wahrscheinlichkeiten** (Meister/Europapokal/Abstieg): brauchbar
kalibriert, Brier 0.0409, Lücken bis max. 4 Prozentpunkte, konsistentes Muster
"unten zu niedrig, Mitte zu hoch" (Signatur zu enger Verteilungen).

**Endpunktzahl-Intervalle** (90%-Intervall): 87.1% Abdeckung statt 90%,
mit klarem Verlauf – vor Saisonstart nur 75.0%, ab Spieltag 26 94.0%. Passt zur
bekannten Vereinfachung (feste Parameter je Lauf): Teamstärken-Unsicherheit
trägt vor allem früh in der Saison, wenn noch viel offen ist.

## Parameter-Unsicherheit (Bootstrap)

`model/bootstrap.py`: 100 Fits je Lauf mit Bayesian Bootstrap (Exp(1)-Faktor
je Spielgewicht), Simulation verteilt 10.000 Läufe auf diese 100 Parametersätze.
Kosten: ~30s statt 0,5s je Update. Aufsteiger ohne Bundesliga-Historie bekommen
Prior-Mittelwert + gemessene Streuung (RMS über 12 historische Fälle: 0.40
Angriff, 0.32 Abwehr).

Überraschung: die Parameter-Streuung selbst ist über die Saison **konstant**
(~0.05-0.06 SD), nicht abnehmend – was schrumpft, ist ihre *Wirkung*, weil sie
nur noch offene Spiele betrifft.

Gegenprobe (96 Stichtage, mit/ohne Bootstrap):

| Phase | Abdeckung ohne | mit |
|---|---|---|
| vor Saisonstart | 75.0% | **84.0%** |
| gesamt | 84.9% | **87.7%** |
| ab Spieltag 26 | 92.0% | 92.0% |

Größter Zuwachs genau dort, wo die Lücke war; am Saisonende keine Veränderung.
Behoben ist der Befund nur halbiert (früh in der Saison fehlen weiterhin
6-9 Prozentpunkte) – vermutlich weil die Zeitgewichtung kurz nach Saisonstart
noch zu stark an der Vorsaison hängt. Nebenwirkung: Prognose 2026/27 verschiebt
sich sichtbar (Bayern Meister 93%→90%, Elversberg Abstieg 94%→76%).

## Log (Stationen)

1. Datenpipeline vereinheitlicht (historisch + laufende Saison).
2. Modellkern + Recovery-Test. Erster Fit: Heimvorteil 0.19, Ligaschnitt 1.39.
3. Backtest: Walk-forward, 3 Maße, 2 Baselines. Rückstand zum Markt lokalisiert
   bei Aufsteigern.
4. Blöcke auf echte Spieltage umgestellt (RPS 0.2052→0.2049, Aufsteiger-Befund
   schärfer).
5. Aufsteiger-Prior: RPS 0.2049→0.2044.
6. Grid-Search: 504 Läufe, RPS 0.2044→0.2031 (Holdout: 0.2046→0.2022).
7. Simulation (`simulation/table.py`, `season.py`), gegen 10 echte Tabellen
   geprüft.
8. Pipeline + Ausgabe (`pipeline.py`, `cli.py`). Prognose 2026/27 vor Spieltag 1:
   Bayern 93% Meister, Elversberg 94% Abstieg.
9. Frontend (`frontend/`, Vanilla-JS, 4 JSON-Dateien).
10. Kalibrierungs-Check: Ereignisse gut kalibriert, Punkte-Intervalle 87.1%
    statt 90% (75%→94% über die Saison).
11. Parameter-Unsicherheit (Bootstrap): Abdeckung vor Saisonstart 75.0%→84.0%,
    gesamt 84.9%→87.7%.

## Quellen / Inspiration

- Dixon, Coles (1997): Modelling Association Football Scores and Inefficiencies in the Betting Market
- https://pena.lt/y/2021/06/24/predicting-football-results-using-python-and-dixon-and-coles/
- https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling-dixon-coles-and-time-weighting/
- https://github.com/opisthokonta/goalmodel
