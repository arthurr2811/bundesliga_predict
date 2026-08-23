"""Kanonische Team-Namen und Mapping der OpenLigaDB-Namen darauf.

Kanonische Namen orientieren sich an football-data.co.uk, da diese Quelle
die 10 historischen Saisons abdeckt und die Schreibweise dort über die Zeit
stabil ist. OpenLigaDB verwendet vollständigere Vereinsnamen und muss daher
auf die kanonische Form gemappt werden.

Das Mapping deckt alle Vereine ab, die zwischen 2016/17 und 2026/27
erstklassig gespielt haben. Ein Aufsteiger, der noch nie dabei war, muss hier
ergänzt werden -- `normalize_openligadb_team` wirft dafür einen KeyError,
statt den Verein stillschweigend als neues Team durchzulassen.
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

OPENLIGADB_TO_CANONICAL = {
    "1. FC Heidenheim 1846": "Heidenheim",
    "1. FC Köln": "FC Koln",
    "1. FC Nürnberg": "Nurnberg",
    "1. FC Union Berlin": "Union Berlin",
    "1. FSV Mainz 05": "Mainz",
    "Bayer 04 Leverkusen": "Leverkusen",
    "Borussia Dortmund": "Dortmund",
    "Borussia Mönchengladbach": "M'gladbach",
    "DSC Arminia Bielefeld": "Bielefeld",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "FC Augsburg": "Augsburg",
    "FC Bayern München": "Bayern Munich",
    "FC Ingolstadt 04": "Ingolstadt",
    "FC Schalke 04": "Schalke 04",
    "FC St. Pauli": "St Pauli",
    "Fortuna Düsseldorf": "Fortuna Dusseldorf",
    "Hamburger SV": "Hamburg",
    "Hannover 96": "Hannover",
    "Hertha BSC": "Hertha",
    "Holstein Kiel": "Holstein Kiel",
    "RB Leipzig": "RB Leipzig",
    "SC Freiburg": "Freiburg",
    "SC Paderborn 07": "Paderborn",
    "SV 07 Elversberg": "Elversberg",
    "SV Darmstadt 98": "Darmstadt",
    "SV Werder Bremen": "Werder Bremen",
    "SpVgg Greuther Fürth": "Greuther Furth",
    "TSG Hoffenheim": "Hoffenheim",
    "VfB Stuttgart": "Stuttgart",
    "VfL Bochum": "Bochum",
    "VfL Wolfsburg": "Wolfsburg",
}


def normalize_openligadb_team(name: str) -> str:
    """Mappt einen OpenLigaDB-Teamnamen auf die kanonische Schreibweise."""
    try:
        return OPENLIGADB_TO_CANONICAL[name]
    except KeyError:
        raise KeyError(
            f"Unbekannter OpenLigaDB-Teamname {name!r} - in "
            "team_mapping.OPENLIGADB_TO_CANONICAL ergaenzen."
        ) from None
