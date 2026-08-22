"""Kanonische Team-Namen und Mapping der OpenLigaDB-Namen darauf.

Kanonische Namen orientieren sich an football-data.co.uk, da diese Quelle
die 10 historischen Saisons abdeckt und die Schreibweise dort über die Zeit
stabil ist. OpenLigaDB verwendet vollständigere Vereinsnamen und muss daher
auf die kanonische Form gemappt werden.
"""

CANONICAL_TEAMS = {
    "Augsburg",
    "Bayern Munich",
    "Bielefeld",
    "Bochum",
    "Darmstadt",
    "Dortmund",
    "Ein Frankfurt",
    "Elversberg",
    "FC Koln",
    "Fortuna Dusseldorf",
    "Freiburg",
    "Greuther Furth",
    "Hamburg",
    "Hannover",
    "Heidenheim",
    "Hertha",
    "Hoffenheim",
    "Holstein Kiel",
    "Ingolstadt",
    "Leverkusen",
    "M'gladbach",
    "Mainz",
    "Nurnberg",
    "Paderborn",
    "RB Leipzig",
    "Schalke 04",
    "St Pauli",
    "Stuttgart",
    "Union Berlin",
    "Werder Bremen",
    "Wolfsburg",
}

# OpenLigaDB-Teamname -> kanonischer Name. Wird ergänzt, sobald ein neuer
# Aufsteiger zum ersten Mal in einer OpenLigaDB-Antwort auftaucht.
OPENLIGADB_TO_CANONICAL = {
    "1. FC Köln": "FC Koln",
    "1. FC Union Berlin": "Union Berlin",
    "1. FSV Mainz 05": "Mainz",
    "Bayer 04 Leverkusen": "Leverkusen",
    "Borussia Dortmund": "Dortmund",
    "Borussia Mönchengladbach": "M'gladbach",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "FC Augsburg": "Augsburg",
    "FC Bayern München": "Bayern Munich",
    "FC Schalke 04": "Schalke 04",
    "Hamburger SV": "Hamburg",
    "RB Leipzig": "RB Leipzig",
    "SC Freiburg": "Freiburg",
    "SC Paderborn 07": "Paderborn",
    "SV 07 Elversberg": "Elversberg",
    "SV Werder Bremen": "Werder Bremen",
    "TSG Hoffenheim": "Hoffenheim",
    "VfB Stuttgart": "Stuttgart",
}


def normalize_openligadb_team(name: str) -> str:
    """Mappt einen OpenLigaDB-Teamnamen auf die kanonische Schreibweise.

    Wirft KeyError bei unbekannten Namen, statt sie unverändert durchzulassen -
    ein neuer Aufsteiger soll auffallen und explizit ergänzt werden, damit er
    nicht versehentlich als eigenständiges "neues" Team im Datensatz landet.
    """
    try:
        return OPENLIGADB_TO_CANONICAL[name]
    except KeyError:
        raise KeyError(
            f"Unbekannter OpenLigaDB-Teamname {name!r} - in "
            "team_mapping.OPENLIGADB_TO_CANONICAL ergaenzen."
        ) from None
