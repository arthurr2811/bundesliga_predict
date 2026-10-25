# Bundesliga-Vorhersage

Prognose der laufenden Bundesliga-Saison: Ergebniswahrscheinlichkeiten je
Spiel, erwartete Abschlusstabelle sowie Meister-, Europapokal- und
Abstiegswahrscheinlichkeiten. Dixon-Coles-Poisson-Modell plus
Monte-Carlo-Simulation der Restsaison, mit Bootstrap für die
Parameter-Unsicherheit.

**Live-Demo:** https://arthurr2811.github.io/bundesliga_predict/ (täglich
aktualisiert)

**Blogpost:** https://www.arthurraffel.dev/projekte/bundesliga-prognose

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
- **Historie 2016/17–2025/26:** `data/processed/historic.csv`

Nach einem Saisonende: Raw-CSV der abgelaufenen Saison ergänzen und
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
