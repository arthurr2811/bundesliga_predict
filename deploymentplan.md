# Deployment

Ziel: Frontend öffentlich auf GitHub Pages, Ergebnisse und Prognose werden
täglich automatisch aktualisiert. **Status: umgesetzt** (Oktober 2026).

URL: https://arthurr2811.github.io/bundesliga_predict/ (von arthurraffel.dev
verlinkt, dieses Repo bleibt eigenständig).

## Ansatz

GitHub Actions statt lokalem Cron: läuft kostenlos und unabhängig vom eigenen
Rechner. Ein Lauf (`cli.py update`) dauert lokal ~40 s (Fit + Bootstrap), bei
neuem Spieltag zusätzlich ~30 s je Snapshot.

## Ablauf (`.github/workflows/update.yml`)

1. Auslöser: `schedule` (täglich 06:00 UTC) plus `workflow_dispatch`.
2. Checkout, Python 3.13, `pip install -r requirements.txt` und `pip install -e .`.
3. `pytest -q`: Ein kaputter Stand wird nicht deployt.
4. `python -m bundesliga_predict.cli update`.
5. Geänderte Dateien unter `data/output/archive/` committet und pusht der Bot
   (`pull --rebase` davor). Ohne neuen Spieltag ist der Diff leer und es
   entsteht kein Commit.
6. `python -m bundesliga_predict.build_site _site`: Frontend und Daten in einen
   Ordner, `DATA` in `app.js` wird von `../data/output/` auf `data/` umgeschrieben.
7. `actions/upload-pages-artifact` und `actions/deploy-pages`.

`concurrency: pages` verhindert parallele Läufe. Die Schreibrechte setzt der
Workflow selbst (`permissions:`), in den Repo-Settings ist dafür nichts nötig.

## Was persistiert wird

- **Im Repo:** `data/output/archive/` (eingefrorene Prognosen, nicht
  reproduzierbar, falls sich Modell oder Daten ändern) und
  `data/processed/historic.csv` (abgeschlossene Saisons ohne Quoten).
- **Nicht im Repo:** `meta.json`, `matches.json`, `table.json`,
  `probabilities.json` in `data/output/` (per `.gitignore`). Werden im Workflow
  gebaut und nur deployt.

## Daten in CI

`build_dataset` nimmt die Roh-CSVs aus `data/raw/historic_data/`, wenn sie da
sind, sonst `historic.csv`. Die laufende Saison ergibt sich aus dem Datum (ab
Juli die neue). Alle Saisons zwischen der letzten historischen und der
laufenden kommen von OpenLigaDB, damit entsteht nach einem Saisonwechsel keine
Lücke, auch wenn `historic.csv` noch nicht nachgezogen ist.

## Einmalig in GitHub eingestellt

- Settings → Pages → Source: **GitHub Actions**.

## Offen

- Optional: Lauf abbrechen, wenn sich seit gestern nichts geändert hat
  (spart nur Rechenzeit, Deploy ist idempotent).
- Zum Saisonwechsel prüfen, wie sich Archiv und Spieltagsauswahl verhalten:
  Das Archiv gilt nur für die laufende Saison, `md_XX` der Vorsaison würden mit
  der neuen kollidieren. Spätestens im Juli 2027 lösen (z. B. Archiv je Saison
  in Unterordner).
- Geplante Workflows werden nach 60 Tagen ohne Commit deaktiviert
  (Sommerpause). Dann in Actions wieder einschalten.
