"""Das Mapping ist die Nahtstelle zwischen den beiden Datenquellen.

Eine Lücke darin führt nicht zu einem Fehler, sondern zu einem Team, das im
Datensatz zweimal unter verschiedenen Namen auftaucht -- und damit zu einem
Modell, das dessen halbe Historie nicht sieht.
"""

import pytest

from bundesliga_predict.team_mapping import (
    CANONICAL_TEAMS,
    OPENLIGADB_TO_CANONICAL,
    normalize_openligadb_team,
)


def test_alle_ziele_sind_kanonisch():
    unbekannt = set(OPENLIGADB_TO_CANONICAL.values()) - CANONICAL_TEAMS
    assert not unbekannt, f"Ziel nicht in CANONICAL_TEAMS: {sorted(unbekannt)}"


def test_jedes_kanonische_team_ist_erreichbar():
    """Sonst fehlt ein Verein, sobald OpenLigaDB ihn liefert."""
    fehlend = CANONICAL_TEAMS - set(OPENLIGADB_TO_CANONICAL.values())
    assert not fehlend, f"Kein OpenLigaDB-Name gemappt auf: {sorted(fehlend)}"


def test_zuordnung_ist_eindeutig():
    ziele = list(OPENLIGADB_TO_CANONICAL.values())
    assert len(ziele) == len(set(ziele))


def test_unbekannter_name_faellt_auf():
    with pytest.raises(KeyError, match="Unbekannter OpenLigaDB-Teamname"):
        normalize_openligadb_team("SV Musterstadt 1900")


def test_normalisierung_trifft_die_kanonische_form():
    assert normalize_openligadb_team("VfL Wolfsburg") == "Wolfsburg"
    assert normalize_openligadb_team("Borussia Mönchengladbach") == "M'gladbach"
