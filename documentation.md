# Bundesliga-Vorhersage – Projektdokumentation

Vorhersage der Bundesliga-Saison (Ergebnisse, Tabelle, Meister-/Europapokal-/Abstiegswahrscheinlichkeiten) mit einem
Dixon-Coles-Poisson-Modell plus Monte-Carlo-Simulation.

## Idee & Ansatz

Jedes Team bekommt Angriffs-/Abwehrstärke plus Heimvorteil (Dixon-Coles,
Erweiterung des Poisson-Tormodells). Daraus folgen Ergebniswahrscheinlichkeiten
je Spiel; die Restsaison wird per Monte-Carlo tausendfach simuliert, um
Tabellenplatz-, Meister-, Europapokal- und Abstiegswahrscheinlichkeiten zu
bekommen. Orientiert an Dixon/Coles.

## Datenquellen

- **football-data.co.uk**: historische Saisons 2016/17–2025/26 als CSV,
  lokal unter `data/raw/historic_data/`.
- **OpenLigaDB**: Spielplan/Ergebnisse der laufenden Saison als JSON, inkl.
  Spieltagsnummern (auch für vergangene Saisons – wichtig für den Backtest).

Unterschiedliche Datumsformate/Teamnamen werden über eine
Vereinheitlichungs-Pipeline (`src/bundesliga_predict/`) auf ein Format gemappt.

## Einheitliches Datenformat

`data/processed/matches.csv`: `season`, `date`, `matchday`, `home_team`,
`away_team`, `home_goals`, `away_goals`, `finished`. Bewusst weggelassen:
Halbzeitstände, Schüsse, Karten, Torschützen, Wettquoten (letztere werden nur
für den Backtest aus den Rohdaten gelesen).

## Modellkern

    log lambda_heim = intercept + attack[heim] - defense[gast] + heimvorteil
    log lambda_gast = intercept + attack[gast] - defense[heim]

Normierung `sum(attack) = sum(defense) = 0`, sodass ein Team mit 0/0 exakt
Ligadurchschnitt ist. Dazu die **Dixon-Coles-Korrektur** `rho` (gemessen ≈ -0.04) für die Unter-/Überschätzung
von 0:0/1:1 bzw. 1:0/0:1 durch zwei unabhängige Poissons.

Weitere Design-Entscheidungen:

- **Zeitgewichtung**: stetiger Zerfall (Halbwertszeit) *plus* Abschlag pro
  Saisonwechsel (Kaderumbruch wirkt sprunghaft, nicht stetig).
- **Aufsteiger per Shrinkage**: Normal-Prior zieht Angriff/Abwehr Richtung
  Mittelwert, stärker je weniger gewichtete Spielmasse ein Team hat.
- **Hyperparameter** (Halbwertszeit, Saison-Abschlag, Prior-Stärke) stehen in
  `config.py` und werden separat per Grid-Search bestimmt statt mitgefittet.

## Backtest

Drei Maße: **RPS** (kennt die Ordnung der Ausgänge), **Log-Loss** (bestraft
selbstsichere Fehler hart, ist das Fit-Ziel), **Brier** (beschränkt, robust).
Zwei Baselines: Ligadurchschnitt (untere Grenze) und margenbereinigter
Buchmachermarkt.

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
+0.0057) – gleichmäßig über Hin-/Rückrunde verteilt, kein Saisonverlaufs-Effekt.

## Aufsteiger-Prior

Aktuell bekommt ein Aufsteiger den Wert des Ligaschnitts. Das ist intiutiv natürlich quatsch.
Aufsteiger sind meistens die schwästen Teams: Gemessen über 19 Aufsteiger-Saisons:
Angriff im Schnitt -0.25, Abwehr -0.14 unter Ligaschnitt Korrektur: Prior zieht auf den gemessenen Mittelwert
statt auf 0.

| Prior-Stärke | RPS gesamt | RPS Aufsteiger | Abstand Markt |
|---|---|---|---|
| 0.0 (vorher) | 0.2049 | 0.2032 | +0.0118 |
| 1.0 (gemessen, Default) | 0.2044 | 0.2010 | +0.0096 |
| 1.5 | 0.2042 | 0.2002 | +0.0089 |

Gewinn sitzt komplett in der Hinrunde (+0.0125 → +0.0079 bei Stärke 1.0),
Rückrunde unverändert – sobald ein Aufsteiger eigene Daten hat, verblasst der
Prior.

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
verschwunden (+0.0007).


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
12. Deployment: GitHub Pages + tägliche GitHub Action. Historie ohne Quoten
    als `historic.csv` committet, damit CI ohne Roh-CSVs rechnet; nur das
    Spieltags-Archiv wird versioniert.

## Quellen / Inspiration

- Dixon, Coles (1997): Modelling Association Football Scores and Inefficiencies in the Betting Market
- https://pena.lt/y/2021/06/24/predicting-football-results-using-python-and-dixon-and-coles/
- https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling-dixon-coles-and-time-weighting/
- https://github.com/opisthokonta/goalmodel
