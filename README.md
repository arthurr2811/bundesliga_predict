# Bundesliga-Vorhersage

Prognose der laufenden Bundesliga-Saison: Ergebniswahrscheinlichkeiten je
Spiel, erwartete Abschlusstabelle sowie Meister-, Europapokal- und
Abstiegswahrscheinlichkeiten. Dixon-Coles-Poisson-Modell plus
Monte-Carlo-Simulation der Restsaison, mit Bootstrap für die
Parameter-Unsicherheit.

**Live-Demo:** https://arthurr2811.github.io/bundesliga_predict/ (täglich
aktualisiert)

Hintergrund, Modell, Backtest und Kalibrierung: [documentation.md](documentation.md).

## Setup

Python ≥ 3.11.

    python -m venv .venv
    .venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
    pip install -r requirements.txt
    pip install -e .

## Befehle

    # Ergebnisse von OpenLigaDB holen, Prognose rechnen, JSON nach data/output/
    python -m bundesliga_predict.cli update

    # Nur rechnen, auf vorhandenem data/processed/matches.csv
    python -m bundesliga_predict.cli simulate [--as-of 2026-09-01]

    # Auswertung (brauchen die Roh-CSVs, siehe unten)
    python -m bundesliga_predict.cli backtest
    python -m bundesliga_predict.cli calibrate
    python -m bundesliga_predict.cli tune --stage a

    pytest

Frontend lokal ansehen (braucht einen Server, `file://` blockiert `fetch`):

    python -m http.server
    # dann http://localhost:8000/frontend/

## Daten

- **Laufende Saison:** OpenLigaDB, wird bei jedem `update` live geholt.
- **Historie 2016/17–2025/26:** `data/processed/historic.csv` (committet; nur
  Datum, Spieltag, Teams, Tore). Reicht für `update` und `simulate`.
- **Roh-CSVs mit Wettquoten:** nicht im Repo. Nötig für `backtest`,
  `calibrate` und `tune` (Markt-Baseline). Herunterladen von
  [football-data.co.uk](https://www.football-data.co.uk/germanym.php)
  (Bundesliga 1) und als `data/raw/historic_data/D1_<saison>.csv` ablegen,
  z. B. `D1_2025-26.csv`. Sind sie vorhanden, haben sie Vorrang vor
  `historic.csv`.

Nach einem Saisonende: Roh-CSV der abgelaufenen Saison ergänzen und
`historic.csv` neu schreiben:

    python -m bundesliga_predict.build_dataset --historic

Bis dahin holt `update` die abgelaufene Saison automatisch von OpenLigaDB.

## Deployment

GitHub Action [`.github/workflows/update.yml`](.github/workflows/update.yml),
täglich 06:00 UTC und manuell startbar:

1. Tests, dann `cli update`.
2. Hat ein neuer Spieltag begonnen, wird die eingefrorene Vorab-Prognose unter
   `data/output/archive/` vom Bot zurück ins Repo committet. Die aktuellen
   JSONs werden nicht versioniert, nur deployt.
3. `python -m bundesliga_predict.build_site` legt Frontend und Daten in
   `_site/` zusammen, das wird auf GitHub Pages deployt.

Hinweis: GitHub deaktiviert geplante Workflows nach 60 Tagen ohne Commit im
Repo, was in der Sommerpause passieren kann. Wieder einschalten unter
*Actions → Prognose aktualisieren → Enable workflow*.
