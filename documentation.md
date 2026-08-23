# Bundesliga-Vorhersage – Projektdokumentation

Kleines Übungs- und Portfolio-Projekt: Vorhersage der Bundesliga-Saison
(Spielergebnisse, Tabelle, Meister-/Europapokal-/Abstiegswahrscheinlichkeiten)
mit einem Dixon-Coles-Poisson-Modell. Diese Datei hält fest, was gebaut wurde
und warum – als Grundlage für einen späteren Blogpost, nicht als vollständige
technische Doku.

## Idee & Ansatz

Jedes Team bekommt eine Angriffs- und eine Abwehrstärke sowie einen
Heimvorteil-Faktor (Dixon-Coles-Modell, eine Erweiterung des klassischen
Poisson-Tormodells). Daraus lassen sich Wahrscheinlichkeiten für jedes
Spielergebnis berechnen. Die restliche Saison wird per Monte-Carlo-Simulation
tausendfach durchgespielt, um Wahrscheinlichkeiten für Tabellenplätze,
Meisterschaft, internationale Plätze und Abstieg zu bekommen.

Der Ansatz orientiert sich an etablierten, öffentlich dokumentierten Modellen
(siehe Quellen unten).

## Datenquellen

- **football-data.co.uk** liefert die historischen Saisons 2016/17 bis 2025/26
  als CSV (~150 Spalten, v. a. Wettquoten verschiedener Buchmacher), lokal
  abgelegt unter `data/raw/historic_data/D1_<saison>.csv`.
- **OpenLigaDB** (`api.openligadb.de`) liefert den Spielplan und die
  Ergebnisse der laufenden Saison 2026/27 als JSON (`getmatchdata/bl1/2026`),
  inkl. noch nicht gespielter Spiele (`matchIsFinished: false`) sowie
  Torschützen/Torfolge für bereits ausgetragene Spiele.

Die beiden Quellen unterscheiden sich in Datumsformat, Teamnamen-Schreibweise
(z. B. `Bayern Munich` vs. `FC Bayern München`) und Detailtiefe – dafür sorgt
die Vereinheitlichungs-Pipeline (`src/bundesliga_predict/`).

## Einheitliches Datenformat

Nach der Vereinheitlichung landet jedes Spiel als eine Zeile in
`data/processed/matches.csv` mit den Spalten `season`, `date`, `matchday`,
`home_team`, `away_team`, `home_goals`, `away_goals`, `finished`.

Mehr braucht das Modell nicht: Tore sind die Zielgröße, `date`/`season`
steuern die Zeitgewichtung, `matchday` gruppiert, `finished` trennt Gespieltes
von Offenem. Teamnamen werden auf eine kanonische Schreibweise gemappt (Basis:
football-data.co.uk, über alle Saisons konsistent).

Weggelassen: Halbzeitstände, Schüsse, Karten, Schiedsrichter, Torschützen und
sämtliche Wettquoten. Nichts davon fließt in ein reines Tor-Modell ein. Die
Quoten werden später für den Backtest gebraucht, dann aber aus den Rohdaten
gelesen statt dauerhaft mitgeführt.

## Modellkern

Das Modell beschreibt die Tore beider Teams als Poisson-verteilt:

    log lambda_heim = intercept + attack[heim] - defense[gast] + heimvorteil
    log lambda_gast = intercept + attack[gast] - defense[heim]

`intercept` ist das Torniveau der Liga. Damit Angriffs- und Abwehrwerte
eindeutig sind, gilt `sum(attack) = sum(defense) = 0`. Angenehmer Nebeneffekt:
ein Team mit 0/0 ist exakt Ligadurchschnitt -- genau der Punkt, auf den
Aufsteiger gezogen werden.

Dazu die **Dixon-Coles-Korrektur** `tau`: zwei unabhaengige Poisson-
Verteilungen unterschaetzen systematisch 0:0 und 1:1 und ueberschaetzen
1:0/0:1. Ein einzelner Parameter `rho` korrigiert diese vier Zellen. Auf den
echten Daten faellt er negativ aus (rund -0.04), also genau in die von Dixon
und Coles beschriebene Richtung.

Drei Entscheidungen, die nicht aus dem Paper folgen:

- **Zeitgewichtung mit zwei Effekten.** Ein stetiger Zerfall ueber die Tage
  (Halbwertszeit) *plus* ein zusaetzlicher Abschlag je Saisonwechsel. Zwischen
  zwei Saisons wechseln Spieler, Trainer und drei Vereine -- dieser Bruch ist
  sprunghaft und laesst sich mit einem stetigen Zerfall nicht abbilden.
- **Aufsteiger per Shrinkage.** Ein Normal-Prior zieht Angriff und Abwehr zum
  Ligadurchschnitt, und zwar umso staerker, je weniger gewichtete Spielmasse
  ein Team hat. Ohne das wuerde ein Aufsteiger nach drei guten Spielen als
  Topteam geschaetzt. Die Alternative -- 2.-Liga-Daten mitfitten -- ist
  bewusst zurueckgestellt, bis der Backtest zeigt, ob sich der Aufwand lohnt.
- **Hyperparameter werden nicht mitgeschaetzt.** Halbwertszeit, Saison-Abschlag
  und Prior-Staerke stehen in `config.py` und werden spaeter per Grid-Search
  ueber den Backtest bestimmt. Das trennt sauber, was die Daten sagen, von dem,
  was man einstellt.

Geschaetzt wird per Maximum-Likelihood (`scipy`, L-BFGS-B) auf der
gewichteten Log-Likelihood. Bei 30 Teams und 3060 Spielen dauert ein Fit
etwa 0,3 Sekunden -- schnell genug, um im Backtest tausende Fits zu rechnen.

### Wie man so etwas testet

Bei einem statistischen Modell ist der naheliegende Test -- Ergebnis gegen
fest verdrahtete Zahlen -- wertlos: er zementiert nur, was der Code heute
tut. Stattdessen **Parameter-Recovery**: aus bekannten Parametern werden
synthetische Saisons gezogen, daraus wird zurueckgefittet, und geprueft wird,
ob die Wahrheit wieder herauskommt. Der Fit erweist sich als erwartungstreu,
und der Fehler halbiert sich sauber, wenn man die Datenmenge vervierfacht.
Ein Vorzeichenfehler, ein vertauschter Index oder eine falsch normierte
Torematrix faellt damit sofort auf.

## Backtest: die Messlatte

Ab hier hoert das Raten auf. Jede weitere Aenderung am Modell -- der
Aufsteiger-Prior, die Halbwertszeit, der Saison-Abschlag -- muss sich an einer
Zahl messen lassen, statt an Plausibilitaet.

**Walk-forward.** Die Spiele werden in Spieltags-Bloecke zerlegt. Vor jedem
Block wird das Modell ausschliesslich auf Spielen *davor* neu gefittet und
sagt dann dessen Partien vorher. Kein Spiel wird also mit Wissen ueber sich
selbst oder spaetere Partien bewertet. Das ist genau der Ablauf, in dem die
Pipeline spaeter auch produktiv laeuft.

**Spieltage, die es angeblich nicht gab.** football-data.co.uk liefert keine
Spieltagsnummer, deshalb sollten die Bloecke aus dem Kalender rekonstruiert
werden. Der naheliegende Ansatz -- neuer Block nach genuegend Tagen Pause --
ist unloesbar: zum naechsten Spieltag sind es normalerweise fuenf Tage, in
einer englischen Woche aber nur zwei, also genauso viel wie innerhalb eines
gestreckten Spieltags. In acht Saisons kam das 35 Mal vor.

Die naechste Idee war besser, aber auch falsch: jedes Team spielt pro Spieltag
einmal, also neuer Block, sobald ein Team sich wiederholt. Das loest die
englischen Wochen, scheitert aber an einer Nachholpartie am Vortag eines
Spieltags -- deren Teams treten am Auftakttag nicht an, es gibt keine
Kollision, und der Spieltag wird spaeter an der falschen Stelle zerschnitten
(06.05.2021, Hertha - Freiburg vom 30. Spieltag, gefolgt vom 32.).

Gefunden wurde das erst beim Abgleich mit echten Spieltagsnummern -- und dabei
stellte sich die Ausgangsannahme als falsch heraus: **OpenLigaDB liefert die
Spieltagsnummern auch fuer vergangene Saisons.** Dieselbe API, die das Projekt
ohnehin benutzt, Abgleich bei 3060 von 3060 Partien ohne eine einzige
Abweichung. Damit verschwindet die Heuristik ersatzlos: ein Block *ist* ein
Spieltag. Abgeschlossene Saisons werden einmal abgerufen und in
`data/raw/matchdays.csv` abgelegt.

Uebrig bleibt eine harmlose Datumsregel: innerhalb eines Spieltags wird
abgetrennt, was mehr als drei Tage entfernt liegt, also verlegte Partien. Der
Schwellenwert ist jetzt beliebig, weil er nur zwei weit auseinanderliegende
Faelle trennen muss (gestreckter Spieltag 1-2 Tage, echte Verlegung 10-94
Tage); jeder Wert zwischen 3 und 9 liefert dasselbe. Verlegte Spiele bleiben
bewusst eigene Bloecke -- sie sind ein eigener Vorhersage-Zeitpunkt, an dem
das Modell mehr weiss. Ergebnis: 352 Bloecke, keiner mischt zwei Spieltage,
12 abgetrennte Nachholpartien.

**Die Lehre** steckt weniger in der Regel als im Weg dorthin. Drei Runden lang
wurde eine Heuristik verfeinert, deren Grundannahme ("wir haben die Spieltage
nicht") nie geprueft worden war. Und die Kontrolle, die zwischendurch "passt"
meldete, war eine Tautologie: dass sich die Bloecke je Saison zu 34 Spieltagen
aufaddieren, ist bei 306 Spielen und je 9 Spielen pro Block gar nicht anders
moeglich. Erst eine unabhaengige Quelle hat beides sichtbar gemacht -- den
Fehler und die ueberfluessige Heuristik.

**Ein Detail, das leicht durchrutscht:** vor dem ersten Spiel einer neuen
Saison ist das letzte gespielte Spiel noch aus der alten. Wer den
Saisonwechsel-Malus daran festmacht, wendet ihn genau dann nicht an, wenn er
am wichtigsten ist. Der Backtest gibt dem Fit deshalb die *Zielsaison* explizit
mit.

**Drei Masse statt einem.** RPS ist das Standardmass fuer Fussball, weil es
die Ordnung der Ausgaenge kennt: statt eines Heimsiegs ein Remis zu erwarten
ist ein kleinerer Fehler als ein Auswaertssieg. Log-Loss bestraft
selbstsichere Fehlprognosen unbeschraenkt hart und ist zugleich genau das, was
der Fit maximiert -- der ehrlichste Blick darauf, ob out-of-sample dasselbe
passiert wie in-sample. Brier ist beschraenkt und robust. Alle drei sind
negativ orientiert.

**Zwei Baselines rahmen ein, was ueberhaupt gut ist.** Unten der
Ligadurchschnitt: immer dieselben drei Wahrscheinlichkeiten, geschaetzt aus
den Spielen vor dem Stichtag -- wer den nicht schlaegt, hat aus Teamstaerken
nichts gelernt. Oben der Buchmachermarkt, margenbereinigt aus den Quoten in
den Rohdaten. Der Markt kennt Aufstellungen, Verletzungen und Transfers; er
ist Vergleichsmassstab, keine Zielgroesse. Bewertet wird er wie das Modell
gegen die tatsaechlichen Ergebnisse.

Ergebnis ueber 2448 Spiele (2018/19 bis 2025/26). Der erste Stand ist das
Modell, bevor Aufsteiger-Prior und Hyperparameter-Tuning liefen:

| | RPS | Log-Loss | Brier |
|---|---|---|---|
| Ligadurchschnitt | 0.2320 | 1.0736 | 0.6499 |
| Modell, erster Stand | 0.2049 | 0.9983 | 0.5957 |
| **Modell, aktuell** | **0.2031** | **0.9918** | **0.5914** |
| Markt | 0.1978 | 0.9741 | 0.5788 |

Das Modell liegt dort, wo es liegen soll: deutlich ueber der Untergrenze und
nah, aber messbar unter dem Markt. Es holt rund 85 % des Abstands zwischen
Baseline und Markt (erster Stand: 79 %). In allen acht Saisons schlaegt es die
Baseline; die Streuung zwischen den Saisons (RPS 0.194 bis 0.209) ist groesser
als der Abstand zum Markt -- eine einzelne Saison beweist also nichts.

Wie es von 0.2049 auf 0.2031 kam, steht in den beiden folgenden Abschnitten.

### Was der Backtest sofort verraten hat

Die Vermutung, dass der Aufsteiger-Prior an der falschen Stelle sitzt, laesst
sich jetzt messen statt begruenden. Aufgeteilt nach Partien mit und ohne
Aufsteiger, jeweils Abstand zum Markt:

| Teilmenge | n | Modell | Markt | Abstand |
|---|---|---|---|---|
| ohne Aufsteiger | 1890 | 0.2054 | 0.1997 | +0.0057 |
| mit Aufsteiger | 558 | 0.2032 | 0.1914 | +0.0118 |
| davon Hinrunde | 279 | 0.2104 | 0.1980 | +0.0125 |
| davon Rueckrunde | 279 | 0.1960 | 0.1848 | +0.0112 |

Der Rueckstand auf den Markt ist bei Aufsteigern gut doppelt so gross wie
sonst. Genau das erwartet man, wenn der Prior datenarme Teams zu optimistisch
behandelt. Absolut sind Aufsteiger-Partien uebrigens *leichter* vorherzusagen
(beide Spalten kleiner) -- die Ergebnisse sind einseitiger. Nur die Baseline
dafuer ist eben auch niedriger, und deshalb zaehlt hier der Abstand zum Markt.

**Eine Teil-Erklaerung ist dabei weggebrochen.** Mit den alten,
kalenderbasierten Bloecken sah es so aus, als konzentriere sich der Rueckstand
auf die Hinrunde (+0.0141 gegen +0.0086) -- die schoene Geschichte "das Modell
kennt die Aufsteiger noch nicht". Mit echten Spieltagsnummern teilt sich die
Menge sauber in 279/279 und der Unterschied schrumpft auf fast nichts. Der
scheinbare Saisonverlauf war ueberwiegend ein Artefakt der Blockbildung.
Uebrig bleibt der Befund, der zaehlt: der Rueckstand haengt am Aufsteiger,
nicht am Zeitpunkt. Das passt zu "der Prior zieht auf den falschen Wert" --
und der wirkt die ganze Saison, weil ein Aufsteiger auch im Mai noch die
duennste Datenbasis hat.

## Der Aufsteiger-Prior: die erste Aenderung, die sich beweisen musste

Die Shrinkage zog datenarme Teams bis hierher Richtung 0 -- wegen
`sum(attack) = 0` also exakt auf den Ligadurchschnitt. Das ist ein neutraler
Prior im Sinne von "wir wissen nichts ueber das Team". Wir wissen aber mehr:
wer wenig Bundesliga-Historie hat, ist fast immer gerade aufgestiegen.

**Erst messen, dann einstellen.** Fuer jede Saison ein eigener Fit, nur auf
den Spielen dieser Saison, ohne Zeitgewichtung und ohne Shrinkage -- das ist
die unverzerrte Staerke eines Teams relativ zu *seiner* Liga. Ueber 19
Aufsteiger-Saisons (2017/18-2025/26):

    Angriff:  Mittel -0.25   Median -0.26   sd 0.25   95 % unter Ligaschnitt
    Abwehr:   Mittel -0.14   Median -0.15   sd 0.21   74 % unter Ligaschnitt

Der neutrale Prior war also nicht neutral, sondern zu optimistisch. Die
Korrektur ist ein Einzeiler in der Straffunktion: gezogen wird auf einen
Mittelwert statt auf 0.

**Wer Aufsteiger ist, muss nirgends stehen.** Das Shrinkage-Gewicht haengt an
der gewichteten Spielmasse je Team und geht mit wachsender Datenmenge gegen 0
-- fuer etablierte Teams verschwindet der Mittelwert also mit. Keine Liste,
keine Fallunterscheidung, kein Aufsteiger-Flag. Der Backtest bestaetigt das
schaerfer, als ein Test es koennte: ueber alle Prior-Staerken hinweg bleiben
die 1890 Partien *ohne* Aufsteiger bei RPS 0.2054 -- identisch bis zur vierten
Nachkommastelle.

Am haertesten traf der alte Prior Teams ganz ohne Erstliga-Historie
(Elversberg, Heidenheim vor 2023): die tauchen im Fit gar nicht auf und wurden
als exaktes Durchschnittsteam ergaenzt -- das denkbar optimistischste Urteil
ueber einen Aufsteiger. Sie bekommen jetzt den Prior-Mittelwert, also genau
den Wert, den der Fit ihnen geben wuerde.

**Was es bringt.** Skalierung 1.0 heisst: Prior-Mittelwert genau auf dem
gemessenen Wert.

| Prior-Staerke | RPS gesamt | Log-Loss | Brier | RPS Aufsteiger | Abstand Markt |
|---|---|---|---|---|---|
| 0.0 (Ligaschnitt, vorher) | 0.2049 | 0.9983 | 0.5957 | 0.2032 | +0.0118 |
| 0.5 | 0.2046 | 0.9975 | 0.5951 | 0.2020 | +0.0106 |
| 1.0 (gemessen) | 0.2044 | 0.9968 | 0.5947 | 0.2010 | +0.0096 |
| 1.5 | 0.2042 | 0.9963 | 0.5943 | 0.2002 | +0.0089 |

Alle drei Masse verbessern sich monoton und gleichzeitig. Gesamt ist der
Effekt klein -- rund 2 % des Abstands zur Baseline, was bei 558 betroffenen
von 2448 Partien zu erwarten war. Auf der Teilmenge, um die es geht, ist er
deutlich: der Rueckstand auf den Markt sinkt um ein Viertel.

Und der Gewinn sitzt genau da, wo die Erklaerung ihn vorhersagt:

| Prior-Staerke | Abstand Hinrunde | Abstand Rueckrunde |
|---|---|---|
| 0.0 | +0.0125 | +0.0112 |
| 1.0 | +0.0079 | +0.0113 |
| 1.5 | +0.0063 | +0.0114 |

Die Hinrunde halbiert sich, die Rueckrunde ruehrt sich um keinen Zaehlwert.
Sobald ein Aufsteiger eigene Daten hat, verblasst der Prior -- er kann dann
weder helfen noch schaden. Der verbleibende Rueckrunden-Rueckstand hat also
eine andere Ursache und ist mit diesem Hebel nicht zu holen.

**Warum der Default trotzdem auf 1.0 steht, obwohl 1.5 besser misst.** Der
Prior-Mittelwert ist ein Hyperparameter wie Halbwertszeit und Saison-Abschlag,
und die vier wirken aufeinander: ein staerkerer Prior verhaelt sich anders,
wenn gleichzeitig weniger Vergangenheit zaehlt. Ihn jetzt einzeln auf sein
Optimum zu schieben, waere ein lokales Optimum auf Kosten des gemeinsamen. In
`config.py` steht deshalb der *gemessene* Wert -- das trennt sauber, was aus
den Daten kommt, von dem, was der Grid-Search einstellt.

## Wie genau kann der Backtest ueberhaupt messen?

Bevor der Grid-Search laeuft, lohnt eine Frage, die man leicht ueberspringt:
ab welchem Unterschied ist eine Variante *wirklich* besser und nicht nur
zufaellig vorn? Ohne diese Zahl waehlt man am Ende Rauschen aus.

Der naheliegende Weg -- Standardfehler des RPS ueber 2448 Spiele -- gibt
0.0027 und damit eine entmutigende Antwort: der ganze Aufsteiger-Prior hat
0.0005 gebracht, also ein Fuenftel davon. Waere das der Massstab, koennten wir
das Tuning bleiben lassen.

Der Massstab ist aber falsch. Zwei Varianten werden auf *denselben* Spielen
bewertet, der Vergleich ist gepaart. Bei den meisten Partien sind sich beide
einig; unterscheiden tun sie sich nur dort, wo die Hyperparameter wirklich
etwas aendern. Entscheidend ist also die Streuung der **Differenz** je Spiel:

    Standardfehler des RPS selbst              0.0027
    Standardfehler der gepaarten Differenz     0.00007 bis 0.00023

Ein Faktor 12 bis 40. Der Prior-Effekt von 0.00051 hat damit ein
95 %-Intervall von [0.00021, 0.00082] -- klein, aber sauber von null getrennt.

**Und wie viel schoent die Auswahl selbst?** Laesst man K gleichwertige
Kombinationen gegeneinander antreten, gewinnt die beste allein durch Zufall.
Simuliert mit dem gemessenen Standardfehler waechst dieser Bonus mit
Wurzel-Log-K, also praktisch gar nicht: 0.00010 bei K = 10, 0.00021 bei
K = 500. Ein groesserer Grid ist also nicht das Problem -- die
Modellkomplexitaet aendert sich durch mehr Grid-Punkte ueberhaupt nicht,
ueberangepasst werden kann nur die *Auswahl* der Hyperparameter.

Praktische Folge: der Grid wird **breit** statt fein. Feinere Aufloesung auf
einer flachen Flaeche siebt nur Rauschen; ein Optimum am Rand des Grids ist
dagegen ein echter Fehler, weil man nicht weiss, was dahinter kommt -- genau
das war beim Aufsteiger-Prior passiert.

Ein Vorbehalt bleibt: die 0.00007 stammen aus Varianten, die sich nur im Prior
unterscheiden. Bei sehr verschiedenen Konfigurationen war der Wert schon
0.00023, die Plateau-Schwelle von 0.0002 ist also eher die untere Kante.

## Der Grid-Search: 504 Laeufe, und das Ergebnis ist ein Plateau

Halbwertszeit, Saison-Abschlag und die Prior-Groessen waren bis hierher
geraten. Jetzt werden sie gemessen -- in zwei Stufen, getunt ausschliesslich
auf 2018/19 bis 2023/24, damit die beiden juengsten Saisons als unangetasteter
Holdout uebrig bleiben.

**Stufe A** (180 Kombinationen) sucht breit und findet klare Struktur:

    half_life_days   60 Tage 0.2073 -> 240 Tage 0.2034 -> 480 Tage 0.2033 -> gar kein Zerfall 0.2037
    season_penalty   ohne Abschlag (1.0) durchgaengig ~0.0005 schlechter
    prior_sd         0.40 zu weich
    prior_scale      saettigt bei Faktor 2.0

Zwei Dinge daran sind bemerkenswert. Erstens hat die Halbwertszeit ein
sauberes Optimum im Inneren, und es liegt bei rund anderthalb Jahren -- viel
laenger als die urspruenglich gesetzten 180 Tage. Zweitens verdient der
Saison-Abschlag seinen Platz: ihn abzuschalten kostet messbar. Er war als
Bauchentscheidung eingefuehrt worden ("Kaderumbruch wirkt sprunghaft") und
haelt der Pruefung stand.

**Stufe B** (324 Kombinationen) verfeinert und nimmt `prior_match_weight` als
fuenfte Achse dazu -- und findet praktisch nichts mehr: 0.2032 statt 0.2033,
ein Unterschied unterhalb der Messgenauigkeit. **49 von 324 Kombinationen
liegen gleichauf**, und zwar hochgradig verschiedene: 720/0.50/0.15/2.0/8 und
720/0.50/0.25/4.0/34 liefern dieselbe Zahl auf vier Nachkommastellen.

Das ist das eigentliche Ergebnis: **die Flaeche ist flach.** Es gibt keinen
scharfen Optimalpunkt, den man verfehlen koennte. Gewaehlt wurde deshalb nicht
das Argmin -- das waere bei 49 gleichwertigen Kandidaten Rauschen --, sondern
je Achse der Wert, der ueber die anderen Achsen hinweg am stabilsten vorn lag.
Der kostet 0.0002 gegenueber dem Argmin und ist damit ununterscheidbar, dafuer
aber nicht von einer gluecklichen Ecke abhaengig.

**Ein Regler, der anders wirkt als sein Name.** `prior_match_weight` legt fest,
ab welcher Datenmenge ein Team als datenreich gilt. Erhoeht man ihn, steigt die
Zugkraft des Priors fuer *jedes* Team -- der Aufsteiger von 0.80 auf 0.97, der
Dauergast aber von 0.07 auf 0.41. Und weil `sum(attack) = 0` gilt, faellt
heraus, was alle gleich betrifft. Uebrig bleibt nur der *Unterschied* in der
Zugkraft, und der schrumpft: ein groesserer Wert macht den Prior stumpfer,
nicht schaerfer (Abstand des Aufsteigers zum Ligadurchschnitt -0.139 bei 8,
-0.084 bei 68). Aufgefallen ist das, weil ein Test mit der intuitiven
Erwartung durchfiel.

### Der Holdout

612 Spiele aus 2024/25 und 2025/26, waehrend des gesamten Tunings nicht
angefasst. Verglichen wird gegen den Stand vor dem Tuning, gepaart auf
denselben Partien:

| Teilmenge | n | vorher | nachher | Gewinn |
|---|---|---|---|---|
| Tuning 2018/19-2023/24 | 1836 | 0.2043 | 0.2034 | +0.0009 (SE 0.00054) |
| **Holdout 2024/25-2025/26** | 612 | 0.2046 | **0.2022** | **+0.0023 (SE 0.00094)** |

Der Gewinn im Holdout ist *groesser* als auf der Tuning-Menge -- also das
Gegenteil der Ueberanpassung, gegen die die Aufteilung schuetzen sollte. Bei
2.4 Standardfehlern ist das kein Zufallsbefund, auch wenn 612 Spiele fuer
feinere Aussagen zu wenig waeren.

(Der Standardfehler ist hier deutlich groesser als die frueher gemessenen
0.00007 bis 0.00023. Das passt: die beiden verglichenen Konfigurationen
unterscheiden sich in allen vier Achsen gleichzeitig, ihre Vorhersagen laufen
also viel weiter auseinander als bei zwei benachbarten Grid-Punkten.)

### Was aus dem Aufsteiger-Problem geworden ist

Der Befund, der diese ganze Linie ausgeloest hat, war: bei Aufsteigern ist der
Rueckstand auf den Markt doppelt so gross wie sonst. Ueber die drei Stufen:

| | ohne Aufsteiger | mit Aufsteiger | davon Hinrunde | davon Rueckrunde |
|---|---|---|---|---|
| erster Stand | +0.0057 | +0.0118 | +0.0125 | +0.0112 |
| mit Prior | +0.0057 | +0.0096 | +0.0079 | +0.0113 |
| **nach Tuning** | **+0.0053** | **+0.0054** | **+0.0007** | **+0.0100** |

Aufsteiger-Partien sind jetzt nicht mehr schlechter vorhergesagt als der Rest
(+0.0054 gegen +0.0053), und in der Hinrunde ist der Abstand zum Markt
praktisch verschwunden. Das Problem ist geschlossen.

Dafuer steht ein neues da, das vorher unter dem alten lag: der gesamte
verbleibende Aufsteiger-Rueckstand sitzt jetzt in der **Rueckrunde**
(+0.0100), und dort hat sich ueber alle drei Stufen fast nichts bewegt. Das
ist die Umkehrung der urspruenglichen Erklaerung -- am Saisonanfang, wo am
wenigsten Daten vorliegen, ist das Modell inzwischen auf Markthoehe, und
abgehaengt wird es, wenn es das Team eigentlich kennen sollte. Eine Vermutung
waere die Winterpause: Trainerwechsel und Wintertransfers treffen
Abstiegskandidaten haerter als andere, und der Markt weiss davon, das Modell
nicht. Belegt ist das nicht.

## Die Tabelle: eine Regel, zwei Ebenen

Die Simulation braucht nicht *eine* Tabelle, sondern zehntausend gleichzeitig.
Deshalb steckt die Sortierregel in einer Funktion, die auf der letzten Achse
arbeitet: `(n_teams,)` fuer den aktuellen Stand, `(n_simulationen, n_teams)`
fuer den Monte-Carlo. Zwei Implementierungen, die auseinanderlaufen koennen,
gibt es damit gar nicht erst.

**Der direkte Vergleich fehlt bewusst.** Die DFL-Regeln sehen nach Punkten,
Tordifferenz und erzielten Toren den direkten Vergleich vor. Nachgesehen, wie
oft er in zehn Saisons ueber eine Platzierung entschieden hat: **kein einziges
Mal** -- die erzielten Tore haben immer schon getrennt. Er waere also
ungetesteter Code fuer einen Fall, der praktisch nicht vorkommt, und in der
Simulation teuer, weil er sich nicht vektorisieren laesst. Voellige
Gleichstaende werden im Monte-Carlo per Zufall aufgeloest, was die DFL im
Extremfall ebenfalls tut.

**Getestet gegen eine unabhaengige Quelle.** Ein Test, der die Tabelle mit
denselben Daten nachrechnet, aus denen sie stammt, prueft nichts. Also kommen
die Abschlusstabellen 2016/17-2025/26 aus dem Tabellen-Endpunkt von
OpenLigaDB, waehrend unsere Tabelle aus den Ergebnissen von football-data.co.uk
gerechnet wird. Alle zehn stimmen Zeile fuer Zeile ueberein -- Punktevergabe,
Torzaehlung, Sortierung und nebenbei auch das Team-Mapping zwischen den beiden
Quellen.

## Der Monte-Carlo

Aus Wahrscheinlichkeiten je Spiel werden Wahrscheinlichkeiten je Tabellenplatz,
indem die Restsaison zehntausendmal durchgespielt wird. Gezogen wird aus der
flachgelegten Torematrix -- ein `searchsorted` je Partie zieht alle Laeufe auf
einmal. Zwei unabhaengige Poissons waeren einfacher, wuerden aber genau die
Dixon-Coles-Korrektur wegwerfen, die 0:0 und 1:1 anhebt: also die Ergebnisse,
in denen sich Punkte entscheiden. Der ganze Lauf ueber 306 offene Spiele
dauert 0,2 Sekunden, der Fit davor 0,3.

**Die Parameter bleiben in allen Laeufen dieselben.** Simuliert wird die
Unsicherheit ueber die *Ergebnisse*, nicht die ueber die *Teamstaerken*. Das
ist eine bewusste Vereinfachung, und man sieht ihr Vorzeichen an der ersten
echten Zahl: vor dem ersten Spieltag 2026/27 gibt das Modell Bayern 93 % auf
die Meisterschaft, deutlich mehr als der Wettmarkt. Ein Teil davon ist
Modellmeinung, ein Teil fehlende Parameter-Unsicherheit -- wer die Staerken je
Lauf aus ihrer Schaetzverteilung zoege (Bootstrap ueber Refits), bekaeme
breitere Verteilungen. Nachgereicht werden kann das jederzeit; die Simulation
selbst aendert sich dafuer nicht.

## Die Pipeline: ein Stichtag, sonst nichts

Nach jedem Spieltag laeuft dieselbe Kette noch einmal -- Daten holen, fitten,
Spiele vorhersagen, Restsaison simulieren, JSON schreiben. Es gibt kein
Fortschreiben und keine Zustandsdatei; der Stand steckt allein in den Daten
und im Stichtag. `python -m bundesliga_predict.cli update` macht beides
hintereinander, `simulate` nur den Rechenteil.

Genau daraus faellt `--as-of` gratis ab: **alles nach dem Stichtag gilt als
ungespielt, auch wenn im Datensatz laengst ein Ergebnis steht.** Ein Lauf mit
einem vergangenen Datum rekonstruiert damit exakt die Prognose, die es an dem
Tag gegeben haette -- die Grundlage fuer den spaeteren Kalibrierungs-Check und
fuer Verlaufsdarstellungen ("wie stand Bayern nach dem 12. Spieltag da?").
Der Stichtag waehlt auch die Saison: die fruehste, in der noch gespielt wird.
Zwischen zwei Saisons ist das die kommende, was richtig ist -- die alte ist
entschieden.

Das Frontend bekommt vier Dateien in `data/output/` und keinerlei Modellcode:
`meta.json` (Stand, Seed, Hyperparameter, Zahl der Parameter-Ziehungen), `matches.json` (alle 306 Partien --
gespielte mit Ergebnis, offene mit 1X2, erwarteten Toren und den drei
wahrscheinlichsten Ergebnissen in `likely_scores`), `table.json` (aktuelle und
erwartete Abschlusstabelle) und `probabilities.json` (Platzverteilung und die
Ereignisse aus `PLACE_RULES`). Zusammen rund 130 KB.

## Das Frontend: drei Dateien, kein Build

`frontend/` ist eine statische Seite aus Vanilla-JS, die per `fetch` nur die
vier JSON-Dateien liest -- kein Server, kein Build-Schritt, keine Abhaengigkeit,
ueber GitHub Pages deploybar. Oben die erwartete Abschlusstabelle, beim
Hovern einer Zeile die Wahrscheinlichkeiten und die volle Platzverteilung;
darunter die Spiele mit Blaettern durch die Spieltage.

Zwei Kleinigkeiten, die beim Bauen dazukamen:

**Nie 0 % oder 100 % anzeigen.** Beides waere gelogen -- die Simulation zieht
10.000 Laeufe, mehr als "in keinem davon" oder "in allen" sagt eine Null nicht
aus. Werte, die auf der angezeigten Genauigkeit an einen Rand runden, werden
feiner formatiert; bleibt es dabei, steht dort `<0,01 %` bzw. `>99,99 %`.

**"Wahrscheinlichstes Ergebnis" ist nicht "wahrscheinlichster Ausgang".** In
der Prognose 2026/27 steht bei 202 von 306 Partien ein 1:1 als
wahrscheinlichstes Einzelergebnis -- und in keiner einzigen ist X der
wahrscheinlichste Ausgang. Kein Widerspruch: die Siegwahrscheinlichkeit
verteilt sich auf viele Ergebnisse (2:1, 2:0, 3:1, ...), das Remis
konzentriert sich auf wenige. Der Modus einer flachen Verteilung ist eine
schwache Aussage, und das sieht man erst, wenn Platz zwei und drei danebenstehen:
Bayern -- Stuttgart geht mit 3:1 (7,8 %), 2:1 (7,5 %), 4:1 (6,1 %) hinein.
Deshalb zeigt die Tipp-Spalte beim Hovern die Top 3 samt Wahrscheinlichkeit.

## Kalibrierung: heisst 90 % auch 90 %?

Der Backtest misst die Spielvorhersagen. Ueber die Simulationsschicht sagt er
nichts -- und genau deren Ausgaben liest am Ende jemand ab ("93 % Meister",
"Endpunktzahl 72 bis 91"). Weil `--as-of` die Prognose eines beliebigen
Stichtags rekonstruiert, laesst sich das nachzaehlen: fuer jeden Spieltag der
acht abgeschlossenen Saisons einmal die volle Kette rechnen, am Saisonende
gegen die Wahrheit halten. 272 Stichtage, 4.896 Team-Prognosen, je 10.000
Simulationen; `cli.py calibrate` faehrt das in rund zwei Minuten.

**Erste Frage: treten Ereignisse mit 90-%-Prognose in 90 % der Faelle ein?**
Ueber alle Ereignisse aus `PLACE_RULES`, gebinnt nach prognostizierter
Wahrscheinlichkeit (Brier 0.0409):

| Bin | n | prognostiziert | eingetreten | Luecke |
|---|---|---|---|---|
| 0-2 % | 18940 | 0.2 % | 0.4 % | -0.2 |
| 2-5 % | 2090 | 3.3 % | 4.7 % | -1.4 |
| 5-10 % | 2092 | 7.3 % | 8.9 % | -1.6 |
| 10-25 % | 3290 | 16.1 % | 14.5 % | +1.6 |
| 25-50 % | 1121 | 34.7 % | 33.6 % | +1.0 |
| 50-75 % | 558 | 62.8 % | 59.5 % | +3.3 |
| 75-90 % | 399 | 82.4 % | 78.5 % | +4.0 |
| 90-95 % | 184 | 92.7 % | 92.9 % | -0.2 |
| 95-98 % | 133 | 96.6 % | 96.2 % | +0.4 |
| 98-100 % | 569 | 99.7 % | 100.0 % | -0.3 |

Die Vorzeichen sind bemerkenswert konsistent: unten zu niedrig, in der Mitte
zu hoch. Das ist genau die Signatur zu enger Verteilungen -- Unwahrscheinliches
passiert oefter als versprochen, Wahrscheinliches seltener. Deutlich wird es
in den Raendern: Ereignissen mit hoechstens 1 % Wahrscheinlichkeit wurden
insgesamt 14.3 Eintritte zugestanden, tatsaechlich waren es 27. Der Betrag
bleibt aber klein; die groesste Luecke sind 4 Prozentpunkte.

**Zweite Frage: liegt die echte Endpunktzahl im 90-%-Intervall?** Hier wird es
deutlicher:

| Phase | n | Abdeckung | Intervallbreite | MAE |
|---|---|---|---|---|
| vor Saisonstart | 144 | 75.0 % | 22.9 | 8.31 |
| Spieltag 1-8 | 1152 | 79.3 % | 21.4 | 7.11 |
| Spieltag 9-17 | 1296 | 85.8 % | 17.9 | 5.31 |
| Spieltag 18-25 | 1152 | 91.0 % | 13.7 | 3.46 |
| ab Spieltag 26 | 1152 | 94.0 % | 7.8 | 2.13 |
| **gesamt** | **4896** | **87.1 %** | **15.5** | **4.64** |

Ueber alles gerechnet 87.1 % statt 90 % -- das allein waere ein Achselzucken
wert. Der Verlauf ist der Befund: **die Intervalle sind am Saisonanfang zu
eng und werden gegen Ende zu weit.** Vor dem ersten Spieltag deckt das
90-%-Intervall nur drei von vier Faellen ab.

Das passt exakt zur bekannten Vereinfachung. Die Simulation haelt die
Modellparameter in allen 10.000 Laeufen fest und zieht nur die Ergebnisse; sie
kennt die Unsicherheit ueber die *Spiele*, nicht die ueber die *Teamstaerken*.
Wie stark Letztere ins Gewicht faellt, haengt daran, wie viel von der Saison
noch offen ist: vor dem ersten Spieltag traegt sie fast alles, im April kaum
noch etwas. Genau so sieht die Tabelle aus. Dass es am Saisonende umschlaegt
(94 %), hat einen anderen Grund: Punkte sind ganzzahlig, und Perzentile einer
schmalen diskreten Verteilung fallen konservativ aus.

**Was das nicht zeigt.** Acht Saisons sind acht Meister, nicht 272 -- die
Stichtage innerhalb einer Saison sind stark korreliert, die Fallzahlen in den
Bins also viel weniger aussagekraeftig, als sie aussehen. Die Abdeckung je
Saison schwankt entsprechend zwischen 77.8 % (2018/19) und 92.2 % (2022/23).
Und die auffaelligste Einzelzahl geht in die andere Richtung: vor Saisonstart
war Bayern achtmal Favorit, im Schnitt mit 76 % -- gewonnen haben sie sieben
davon. Auf der Spitzenposition ist das Modell also eher zu vorsichtig als zu
forsch. Bei n = 8 heisst das nichts, es warnt nur davor, aus dem Gesamtbefund
mehr zu machen, als drinsteht.

**Konsequenz.** Der Check war die Entscheidung ueber die zurueckgestellte
Parameter-Unsicherheit, und er faellt sie: sie lohnt sich, aber nur fuer den
frueh-saisonalen Fall. Ein Bayesian Bootstrap ueber die vorhandenen
Spielgewichte wuerde genau dort ansetzen, wo die Luecke sitzt, und dort
nichts kaputtmachen, wo die Intervalle ohnehin schon eher zu weit sind.

## Parameter-Unsicherheit: was der Bootstrap wirklich tut

Der Kalibrierungs-Check hatte den Auftrag erteilt, also wurde die
zurueckgestellte Sache gebaut: statt eines Fits rechnet `model/bootstrap.py`
jetzt 100 Fits, jeder mit einem zusaetzlichen Exp(1)-Faktor auf jedem
Spielgewicht (Bayesian Bootstrap). Die Simulation verteilt ihre 10.000 Laeufe
gleichmaessig auf diese 100 Parametersaetze -- jeder Lauf sieht damit eine
Liga, deren Staerken *eine* plausible Schaetzung sind statt immer derselben.
Kosten: rund 30 s je Update statt einer halben Sekunde.

Aufsteiger ohne jede Bundesliga-Historie brauchen eine Sonderbehandlung. Sie
kommen im Fit nicht vor, der Bootstrap kann aus ihnen also nichts ziehen --
ausgerechnet die Teams, ueber die am wenigsten bekannt ist, haetten sonst gar
keine Unsicherheit. Sie bekommen den Prior-Mittelwert plus eine Streuung, die
an den zwoelf Aufsteigern seit 2017/18 gemessen ist (RMS ihrer am Saisonende
geschaetzten Staerke um den Prior-Mittelwert: 0.40 im Angriff, 0.32 in der
Abwehr). Die Zahl ist grosszuegig, weil sie neben der echten Streuung auch
enthaelt, dass der Prior die Aufsteiger im Schnitt zu schwach ansetzt -- und
zwoelf Faelle sind eine duenne Grundlage.

### Der Mechanismus ist nicht der, den man vermutet

Die naheliegende Erwartung: die Parameter-Streuung ist am Saisonanfang gross
und schrumpft, je mehr gespielt wird. **Das stimmt nicht.** Gemessen an
2025/26 ist sie ueber die ganze Saison praktisch konstant:

| Stichtag | mittlere SD der Angriffsstaerke |
|---|---|
| vor Saisonstart | 0.052 |
| Spieltag 10 | 0.063 |
| Spieltag 30 | 0.055 |

Kein Wunder: der Fit steht auf mehreren Jahren Historie, 25 zusaetzliche
Spiele verschieben die Datenmenge kaum.

Was schrumpft, ist die *Wirkung* dieser Streuung. Eine falsch geschaetzte
Teamstaerke verzerrt nur die Partien, die ueberhaupt noch ausstehen -- vor
Saisonstart 34 Spieltage, am 30. noch vier. Deshalb faellt das Profil
trotzdem richtig aus, ohne dass irgendwo eine Schraube "Unsicherheit je
Spieltag" haengt:

| Stichtag | Intervallbreite ohne | mit | Zuwachs |
|---|---|---|---|
| vor Saisonstart | 23.5 | 25.7 | +2.2 |
| Spieltag 5 | 21.6 | 24.1 | +2.4 |
| Spieltag 17 | 12.7 | 13.0 | +0.3 |
| Spieltag 30 | 8.1 | 8.1 | 0.0 |

### Die Gegenprobe

Derselbe Kalibrierungs-Check, zweimal auf denselben 96 Stichtagen (acht
Saisons, jeder dritte Spieltag) -- einmal mit `--bootstrap 0`, einmal mit 40
Ziehungen:

| Phase | Abdeckung ohne | mit | Breite ohne | mit |
|---|---|---|---|---|
| vor Saisonstart | 75.0 % | **84.0 %** | 22.9 | 26.6 |
| Spieltag 1-8 | 77.3 % | 80.8 % | 21.9 | 24.1 |
| Spieltag 9-17 | 85.2 % | 88.4 % | 18.4 | 20.0 |
| Spieltag 18-25 | 90.7 % | 92.4 % | 13.9 | 14.6 |
| ab Spieltag 26 | 92.0 % | 92.0 % | 9.1 | 9.3 |
| **gesamt** | **84.9 %** | **87.7 %** | 17.0 | 18.4 |

Genau das gewuenschte Profil: der groesste Zuwachs (+9 Prozentpunkte) dort,
wo die Luecke war, und am Saisonende exakt null Veraenderung -- die Stelle,
an der die Intervalle ohnehin schon eher zu weit waren.

Die Ereignis-Wahrscheinlichkeiten bessern sich mit (Brier 0.0441 -> 0.0437),
und zwar dort, wo die Ueberheblichkeit sass:

| Bin | Luecke ohne | mit |
|---|---|---|
| 0-2 % | -0.0022 | -0.0001 |
| 50-75 % | +0.0501 | +0.0224 |
| 75-90 % | +0.0423 | +0.0292 |
| 90-95 % | +0.0319 | -0.0122 |

Die groesste Einzelluecke faellt von 5.0 auf 3.5 Prozentpunkte.

### Was offen bleibt

Behoben ist der Befund nicht, nur halbiert. Vor Saisonstart fehlen weiterhin
6 Prozentpunkte, im Fenster Spieltag 1-8 sogar 9 -- ausgerechnet dort hat der
Bootstrap am wenigsten geholfen. Die Vermutung: kurz nach Saisonstart liegen
ein paar echte Ergebnisse vor, die die Zeitgewichtung noch kaum beruecksichtigt;
das Modell haengt dann laenger an der Vorsaison, als es sollte. Das waere eine
Frage an die Gewichtung, nicht an die Simulation -- und der Grid-Search hat
diese Schrauben auf RPS je Spiel optimiert, nicht auf Saison-Kalibrierung.

Nebenwirkung, die man im Blick behalten sollte: die Prognose fuer 2026/27
verschiebt sich sichtbar. Bayern faellt von 93 % auf 90 % Meisterschaft,
Elversberg von 94 % auf 76 % Abstieg. Der zweite Sprung geht fast ganz auf die
Aufsteiger-Streuung von 0.40/0.32 zurueck -- die am duennsten belegte Zahl im
ganzen Modell. Wer sie halbiert, bekommt Elversberg wieder deutlich naeher an
90 %.

## Log

Nur die Stationen; das Warum steht in den Abschnitten oben.

**Datenpipeline.** Historische und laufende Saison vereinheitlicht.

**Modellkern.** Parameter, Zeitgewichtung, Likelihood, Fit, abgesichert durch
den Recovery-Test. Erster Fit auf echten Daten: Heimvorteil 0.19 (rund 20 %
mehr Tore zu Hause), Ligaschnitt 1.39 Tore je Team.

**Backtest.** Walk-forward ueber acht Saisons, drei Masse, zwei Baselines.
Beim Bauen aufgefallen: der Saisonwechsel-Malus lief vor dem ersten Spieltag
einer Saison ins Leere, weil er sich an der Saison des letzten gespielten
Spiels orientierte -- der Fit nimmt die Zielsaison jetzt explizit entgegen.
Erster belegter Befund: der Rueckstand auf den Markt sitzt bei den Aufsteigern.

**Bloecke auf echte Spieltage umgestellt.** Gesamtbild kaum veraendert
(RPS 0.2052 -> 0.2049), die Aufteilung nach Aufsteigern dagegen deutlich.

**Aufsteiger-Prior.** RPS 0.2049 -> 0.2044. Erste Modellaenderung, die der
Backtest genehmigt hat statt der Plausibilitaet.

**Grid-Search.** 504 Laeufe, RPS 0.2044 -> 0.2031, im Holdout 0.2046 ->
0.2022. Damit sind die Defaults in `config.py` belegt statt geraten, und
weiteres Feintuning an diesen vier Schrauben lohnt nicht.

**Simulation.** `simulation/table.py` und `simulation/season.py`, geprueft
gegen die zehn echten Abschlusstabellen und gegen die Torematrix selbst.

**Pipeline und Ausgabe.** `pipeline.py` plus `cli.py simulate|update`. Die
Prognose fuer 2026/27 vor dem 1. Spieltag steht in `data/output/`: Bayern 93 %
Meister, Elversberg 94 % Abstieg, erwartete Punkte von 81.9 bis 19.7. Ein
kompletter Lauf dauert eine halbe Sekunde.

**Frontend.** Statische Seite unter `frontend/`, Vanilla-JS auf den vier
JSON-Dateien. Dabei nachgezogen: `matches.json` liefert jetzt die drei
wahrscheinlichsten Ergebnisse statt nur des einen, weil der Modus allein eine
irrefuehrend starke Aussage ist.

**Kalibrierungs-Check.** `evaluation/calibration.py` plus `cli.py calibrate`:
272 Stichtage aus acht Saisons gegen den echten Ausgang. Ergebnis: die
Ereignis-Wahrscheinlichkeiten sind brauchbar kalibriert (Luecken bis 4
Prozentpunkte, Vorzeichen konsistent Richtung "zu eng"), die
Punkte-Intervalle decken 87.1 % statt 90 % ab -- mit klarem Verlauf von 75 %
vor Saisonstart auf 94 % ab Spieltag 26. Damit ist die zurueckgestellte
Parameter-Unsicherheit belegt noetig, und zwar fuer die frueh-saisonalen
Prognosen.

**Parameter-Unsicherheit.** `model/bootstrap.py`, 100 Fits je Lauf, in der
Simulation auf die 10.000 Laeufe verteilt. Abdeckung des 90-%-Intervalls vor
Saisonstart 75.0 % -> 84.0 %, gesamt 84.9 % -> 87.7 %, am Saisonende
unveraendert. Beim Bauen widerlegt: die Parameter-Streuung schrumpft im
Saisonverlauf *nicht*: sie ist konstant, nur ihre Wirkung auf die
Endpunktzahl haengt an der Zahl offener Spiele.

## Quellen / Inspiration

- Dixon, Coles (1997): Modelling Association Football Scores and Inefficiencies in the Betting Market
- https://pena.lt/y/2021/06/24/predicting-football-results-using-python-and-dixon-and-coles/
- https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling-dixon-coles-and-time-weighting/
- https://github.com/opisthokonta/goalmodel
