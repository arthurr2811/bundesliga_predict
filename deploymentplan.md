# Deployment-Plan

Ziel: Frontend öffentlich auf GitHub Pages, Ergebnisse und Prognose werden
täglich automatisch aktualisiert. **Status: geplant, noch nichts umgesetzt.**

## Ansatz

GitHub Actions statt lokalem Cron: läuft kostenlos und unabhängig vom eigenen
Rechner. Ein Lauf (`cli.py update`) dauert unter zwei Minuten (Fit + Bootstrap
~30 s, bei neuem Spieltag zusätzlich ein Snapshot ~30 s).

## Ablauf des täglichen Workflows

1. Auslöser: `schedule` (täglich, ca. 06:00 UTC) plus `workflow_dispatch`
   zum manuellen Starten.
2. Repo auschecken, Python einrichten, `pip install -r requirements.txt` und
   `pip install -e .`.
3. `python -m bundesliga_predict.cli update`: Ergebnisse von OpenLigaDB holen,
   Prognose neu rechnen, bei neu begonnenem Spieltag den Snapshot unter
   `data/output/archive/md_XX/` anlegen.
4. Site zusammenbauen: `frontend/` und `data/output/` in ein gemeinsames
   Verzeichnis kopieren (das Frontend liest `../data/output/`, Pages liefert
   nur einen Ordner aus). Alternativ den Pfad `DATA` in `app.js` anpassen.
5. Auf GitHub Pages deployen (`actions/deploy-pages`).
6. Nur wenn ein neuer Snapshot entstanden ist: `data/output/archive/` als Commit
   zurück ins Repo pushen. Das passiert höchstens einmal pro Spieltag.

## Was persistiert wird

- **Im Repo (committet):** `data/output/archive/` – die eingefrorenen
  Prognosen sind nicht reproduzierbar, falls sich Modell oder Daten ändern.
- **Nicht mehr im Repo:** `meta.json`, `matches.json`, `table.json`,
  `probabilities.json` in `data/output/`. Sie werden im Workflow gebaut und nur
  deployt, sonst entsteht täglich ein Diff mit tausenden Zeilen. Dafür in
  `.gitignore` aufnehmen und aus dem Index entfernen.
  (Alternative, falls einfacher gewünscht: der Bot committet täglich alles.)

## Nötige Codeänderungen

- **Rohdaten fehlen in CI.** `build_dataset` liest `data/raw/historic_data/`
  (per `.gitignore` draußen und das soll so bleiben). Lösung: einmal lokal
  `data/processed/historic.csv` erzeugen (nur Datum, Teams, Tore, Spieltag,
  keine Quoten) und committieren; `build_dataset` nutzt sie, wenn die Roh-CSVs
  fehlen. Das gilt auch für den Spieltags-Cache `data/raw/matchdays.csv`, dessen
  Inhalt dann in `historic.csv` steckt.
- **`CURRENT_SEASON = 2026`** in `build_dataset.py` ist fest eincodiert und
  muss zum Saisonwechsel manuell erhöht werden (oder aus dem Datum abgeleitet
  werden).
- **`.github/workflows/update.yml`** neu anlegen (Ablauf oben).
- **`README.md`** neu anlegen: Projektidee, Setup, Befehle, Hinweis, woher die
  Rohdaten für Backtest/Tuning kommen (football-data.co.uk, Dateinamen
  `D1_<saison>.csv`).

## Einmalig in GitHub einzustellen

- Settings → Pages → Source: **GitHub Actions**.
- Settings → Actions → General → Workflow permissions: **Read and write**
  (nötig, damit der Bot `archive/` zurückpushen darf).

## Offene Punkte

- Entscheidung bestätigen: `historic.csv` (nur Spieldaten, keine Quoten) ins
  öffentliche Repo, Roh-CSVs bleiben draußen.
- Entscheidung bestätigen: JSONs nicht mehr tracken, nur das Archiv.
- Optional: Lauf abbrechen, wenn sich seit gestern nichts geändert hat.
- Zum Saisonende / Saisonwechsel prüfen, wie sich Archiv und Spieltagsauswahl
  verhalten (Archiv gilt bisher nur für die laufende Saison).
