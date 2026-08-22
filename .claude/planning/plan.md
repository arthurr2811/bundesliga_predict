# Projektplan: Bundesliga-Vorhersage

Ziel: ML-Modell (Dixon-Coles/Poisson), das Spielergebnisse, Tabelle und
Meister-/Abstiegs-/Europapokal-Wahrscheinlichkeiten für die laufende Saison
vorhersagt, samt kleinem Frontend zur Anzeige. Läuft nach jedem Spieltag neu.

Zusätzlich Portfolio-/Übungsprojekt: `documentation.md` im Projekt-Root wird
laufend gepflegt (was gebaut wurde, welche Entscheidungen
warum getroffen wurden) als Basis für einen späteren Blogpost.

## Aktueller Stand
- Projektsetup abgeschlossen. Weiter mit Modellarchitektur.

## 1. Projektsetup
Python-Projektstruktur aufsetzen (venv/Poetry, Abhängigkeiten wie pandas,
scipy, numpy). Datenbeschaffung: historische Saisons von football-data.co.uk
laden, tägliche Ergebnisse der laufenden Saison über OpenLigaDB abrufen.
Team-Namen zwischen beiden Quellen mappen. Rohdaten in ein einheitliches
Format bringen und lokal ablegen.

## 2. Modellarchitektur
Dixon-Coles-Poisson-Modell implementieren: Angriffs-/Abwehrstärke pro Team
plus Heimvorteil-Faktor, inkl. Zeit-Gewichtung neuerer Spiele (nicht linear,
Saisonwechsel gesondert berücksichtigen wegen Kaderumbruch/Aufsteigern) und
der Dixon-Coles-Korrektur für knappe Ergebnisse. Daraus Wahrscheinlichkeiten
für Ergebnisse einzelner Spiele ableiten.

## 3. Training & Fitting
Parameter per Maximum-Likelihood auf historischen Daten schätzen. Backtesting
gegen vergangene Saisons zur Prüfung der Vorhersagequalität. Pipeline zum
Neufitten der Parameter nach jedem Spieltag.

## 4. Saison-Simulation
Monte-Carlo-Simulation der verbleibenden Spieltage auf Basis der Modell-
Wahrscheinlichkeiten. Auswertung über viele Durchläufe zu Wahrscheinlichkeiten
je Team für Meisterschaft, internationale Plätze, Abstieg.

## 5. Frontend
Kleine Oberfläche zur Anzeige: Spielvorhersagen je Spieltag, simulierte
Tabelle, Meister-/Europapokal-/Abstiegs-Wahrscheinlichkeiten. Aktualisierung
nach jedem Spieltag.
