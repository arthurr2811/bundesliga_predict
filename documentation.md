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

Begründung für die Auswahl:
- **Gebraucht:** `home_goals`/`away_goals` sind die einzige Zielgröße, die das
  Dixon-Coles-Poisson-Modell braucht; `date`/`season` werden für die
  Zeit-Gewichtung und die Saisonwechsel-Behandlung benötigt; `matchday` zum
  Gruppieren (Neufitten nach Spieltag, offene Spiele für die Simulation
  bestimmen); `finished` unterscheidet gespielte von noch ausstehenden
  Spielen (letztere braucht die Saison-Simulation als Startpunkt).
  Teamnamen werden auf eine feste, kanonische Schreibweise gemappt (Basis:
  football-data.co.uk-Namen, da sie über alle Saisons konsistent sind),
  damit ein Team über beide Quellen hinweg als dasselbe Team erkannt wird.
- **Weggelassen:** Halbzeitstände, Schüsse, Ecken, Karten, Schiedsrichter,
  Zuschauerzahlen, Torschützen/Torminuten und sämtliche Wettquoten – all das
  fließt in ein reines Tor-basiertes Dixon-Coles-Modell nicht ein und bläht
  nur den Datensatz auf. (Wettquoten könnten später optional für einen
  Modell-Vergleich/Backtest gegen den Wettmarkt interessant sein, werden
  dafür aber aus den Rohdaten neu gelesen statt dauerhaft mitgeführt.)

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
Spieltagsnummer, deshalb stand im Plan, die Bloecke muessten aus dem Kalender
rekonstruiert werden. Das war ein langer Umweg. Der naheliegende Ansatz --
neuer Block nach genuegend Tagen Pause -- ist naemlich unloesbar: innerhalb
eines Spieltags liegen Freitag bis Montag lauter Ein-Tages-Schritte, zum
naechsten Spieltag sind es normalerweise fuenf, in einer englischen Woche aber
nur zwei. Zwei Tage Abstand bedeuten also mal "derselbe Spieltag" und mal
"naechster Spieltag", und das kam in acht Saisons 35 Mal vor.

Die naechste Idee war besser, aber auch nicht richtig: jedes Team spielt pro
Spieltag genau einmal, also neuer Block, sobald ein Team sich wiederholt. Das
loest die englischen Wochen -- scheitert aber an einer Nachholpartie am Vortag
eines Spieltags. Deren zwei Teams treten am Auftakttag nicht an, es gibt also
keine Kollision, das Nachholspiel klebt am Spieltag fest, und der wird spaeter
an der Stelle zerschnitten, an der das nachgeholte Team regulaer antritt. Genau
das ist am 06.05.2021 passiert (Hertha - Freiburg vom 30. Spieltag, gefolgt vom
32.).

Gefunden wurde das erst, als echte Spieltagsnummern zum Abgleich herangezogen
wurden -- und dabei stellte sich die Ausgangsannahme als falsch heraus:
**OpenLigaDB liefert die Spieltagsnummern auch fuer vergangene Saisons**,
nicht nur fuer die laufende. Dieselbe API, die das Projekt ohnehin benutzt.
Der Abgleich passte bei 3060 von 3060 Partien, ohne eine einzige
Datumsabweichung zwischen den Quellen.

Damit verschwindet die Heuristik ersatzlos: ein Block *ist* ein Spieltag.
Abgeschlossene Saisons aendern sich nicht mehr, also werden sie einmal
abgerufen und in `data/raw/matchdays.csv` abgelegt; danach geht nur noch die
laufende Saison ueber das Netz.

Uebrig bleibt eine einzige Datumsregel, und die ist jetzt harmlos: innerhalb
eines Spieltags wird abgetrennt, was mehr als drei Tage entfernt liegt --
also verlegte Partien. Der Schwellenwert war vorher unmoeglich zu waehlen und
ist jetzt beliebig, weil er nur noch zwei weit auseinanderliegende Faelle
trennen muss:

    regulaer gestreckter Spieltag:   1-2 Tage
    echte Verlegung:                10-94 Tage  (alle 12 Faelle)

Dazwischen liegt eine Woche Luft; jeder Wert zwischen 3 und 9 liefert dasselbe
Ergebnis. Ein verlegtes Spiel bleibt bewusst ein eigener Block: es findet
Wochen spaeter statt und ist damit ein eigener Vorhersage-Zeitpunkt, an dem
das Modell mehr weiss. Es zurueck in seinen nominellen Spieltag zu zwingen
hiesse, es mit veralteten Parametern vorherzusagen.

Ergebnis auf den echten Daten: 352 Bloecke, kein einziger mischt zwei
Spieltage, Nummerierung chronologisch, 12 abgetrennte Nachholpartien.

**Die Lehre daraus** steckt weniger in der Regel als im Weg dorthin. Drei
Runden lang wurde eine Heuristik verfeinert, deren Grundannahme ("wir haben
die Spieltage nicht") nie geprueft worden war. Und die Kontrolle, die
zwischendurch "passt" meldete, war eine Tautologie: dass sich die Bloecke je
Saison zu 34 Spieltagen aufaddieren, ist bei 306 Spielen und Bloecken zu je 9
Spielen rechnerisch gar nicht anders moeglich. Erst eine unabhaengige Quelle
hat sowohl den Fehler als auch die ueberfluessige Heuristik sichtbar gemacht.

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
dafuer ist eben auch niedriger, und deshalb zaehlt hier der Abstand zum Markt,
nicht die absolute Zahl.

**Eine Teil-Erklaerung ist dabei weggebrochen.** Solange die Bloecke aus dem
Kalender rekonstruiert wurden, sah es so aus, als konzentriere sich der
Rueckstand auf die Hinrunde (+0.0141 gegen +0.0086) -- die schoene Geschichte
"das Modell kennt die Aufsteiger noch nicht". Mit echten Spieltagsnummern
teilt sich die Menge sauber in 279/279 und der Unterschied schrumpft auf
+0.0125 gegen +0.0112, also fast nichts. Die alte Aufteilung war schief (258
und 286 ergeben nicht einmal die 558); der scheinbare Verlauf ueber die Saison
war ueberwiegend ein Artefakt der Blockbildung. Uebrig bleibt der Befund, der
zaehlt: der Rueckstand haengt am Aufsteiger, nicht am Zeitpunkt. Das passt
weniger zu "zu wenig Daten am Saisonanfang" als zu "der Prior selbst zieht auf
den falschen Wert" -- und ein falscher Prior-Mittelwert wirkt tatsaechlich die
ganze Saison ueber, weil ein Aufsteiger auch im Mai noch die mit Abstand
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
-- fuer etablierte Teams verschwindet der Mittelwert also mit. Es braucht
keine Liste, keine Fallunterscheidung, keinen Aufsteiger-Flag im Datensatz.
Der Backtest bestaetigt das schaerfer, als ein Test es koennte: ueber alle
Prior-Staerken hinweg bleiben die 1890 Partien *ohne* Aufsteiger bei RPS
0.2054, Abstand zum Markt +0.0057 -- identisch bis zur vierten Nachkommastelle.

Am haertesten traf der alte Prior Teams ganz ohne Erstliga-Historie
(Elversberg, Heidenheim vor 2023): die tauchen im Fit ueberhaupt nicht auf,
weil es keine Partie von ihnen gibt, und wurden vom Backtest als exaktes
Durchschnittsteam ergaenzt -- das denkbar optimistischste Urteil ueber einen
Aufsteiger. Sie bekommen jetzt den Prior-Mittelwert, also genau den Wert, den
der Fit ihnen geben wuerde (ihre Shrinkage waere 1, der Prior damit allein
bestimmend).

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
Simuliert man das mit dem gemessenen Standardfehler:

    K = 10    Kombinationen -> bestes scheint 0.00010 besser
    K = 108   Kombinationen -> bestes scheint 0.00017 besser
    K = 500   Kombinationen -> bestes scheint 0.00021 besser

Das waechst mit Wurzel-Log-K, also praktisch gar nicht: von 108 auf 500
Kombinationen kostet 0.00004. Damit ist eine verbreitete Sorge widerlegt --
ein groesserer Grid ist nicht das Problem. Wichtig ist die begriffliche
Trennung: die Modellkomplexitaet aendert sich durch mehr Grid-Punkte
ueberhaupt nicht, es bleiben dieselben Parameter auf denselben Daten.
Ueberangepasst werden kann nur die *Auswahl* der Hyperparameter.

Praktische Folge fuer den Zuschnitt: der Grid wird **breit** statt fein.
Feinere Aufloesung auf einer flachen Flaeche siebt nur Rauschen; ein Optimum,
das am Rand des Grids liegt, ist dagegen ein echter Fehler -- man weiss dann
schlicht nicht, ob dahinter noch etwas kommt. Genau das war beim
Aufsteiger-Prior passiert.

Ein Vorbehalt bleibt: die 0.00007 stammen aus Varianten, die sich nur im
Prior unterscheiden. Bei sehr verschiedenen Konfigurationen war der Wert schon
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
heraus, was alle gleich betrifft: eine Verschiebung der ganzen Liga kann es in
diesem Modell gar nicht geben. Uebrig bleibt nur der *Unterschied* in der
Zugkraft, und der schrumpft. Ein groesserer Wert macht den Prior also
stumpfer, nicht schaerfer. Gemessen am Abstand des Aufsteigers zum
Ligadurchschnitt: -0.139 bei 8, -0.084 bei 68. Aufgefallen ist das, weil ein
Test mit der intuitiven Erwartung durchfiel.

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

## Log

**Datenpipeline.** Historische Saisons und laufende Saison vereinheitlicht.
Dabei aufgefallen: football-data.co.uk liefert keine Spieltagsnummer. Statt
sie kuenstlich zu rekonstruieren, laufen alle Schnitte in Modell und Backtest
ueber das Datum; `matchday` bleibt reine Anzeige-Information der laufenden
Saison.

**Modellkern.** Parameter-Container, Zeitgewichtung, Likelihood und Fit
gebaut, abgesichert durch den Recovery-Test. Erster Fit auf echten Daten:
Heimvorteil 0.19 (rund 20 % mehr Tore zu Hause), Ligaschnitt 1.39 Tore je
Team, Bayern mit Abstand vorn. Noch offen: Backtest und Hyperparameter-Tuning
-- bis dahin sind die Default-Hyperparameter geraten, nicht belegt.

**Backtest.** Walk-forward ueber acht Saisons gebaut, dazu 1X2-Auswertung,
die drei Masse und beide Baselines. Das Modell schlaegt die Baseline klar und
liegt knapp hinter dem Markt. Beim Bauen aufgefallen: der Saisonwechsel-Malus
lief vor dem ersten Spieltag einer Saison ins Leere, weil er sich an der
Saison des letzten gespielten Spiels orientierte -- der Fit nimmt die
Zielsaison jetzt explizit entgegen. Erster belegter Befund: der Rueckstand auf
den Markt konzentriert sich auf Partien mit Aufsteigern.

**Bloecke auf echte Spieltage umgestellt, Backtest neu erhoben.** Die Zahlen
oben stammen aus diesem Lauf (2448 Spiele, 2018/19-2025/26). Gegenueber der
kalenderbasierten Blockbildung aendert sich das Gesamtbild kaum (RPS 0.2052 ->
0.2049), die Aufteilung nach Aufsteigern dagegen deutlich: der
Hinrunden-/Rueckrunden-Unterschied war groesstenteils ein Artefakt der alten
Bloecke.

**Aufsteiger-Prior.** Shrinkage zieht nicht mehr auf den Ligadurchschnitt,
sondern auf einen gemessenen Mittelwert. Erste Modellaenderung, die der
Backtest genehmigt hat statt der Plausibilitaet: RPS 0.2049 -> 0.2044, und der
Rueckstand auf den Markt bei Aufsteigern in der Hinrunde faellt von +0.0125 auf
+0.0079. Nebenbei ist der Sonderfall "Team ohne jede Historie" aus dem
Backtest verschwunden -- er ist jetzt derselbe Prior wie fuer alle anderen.
Naechstes: Grid-Search ueber Halbwertszeit, Saison-Abschlag, Prior-Staerke und
Prior-Mittelwert gemeinsam.

**Grid-Search.** 504 Laeufe in zwei Stufen, getunt auf sechs Saisons, geprueft
auf zwei zurueckgehaltenen. RPS ueber alle acht Saisons 0.2044 -> 0.2031, im
Holdout 0.2046 -> 0.2022. Die Halbwertszeit war mit 180 Tagen deutlich zu kurz
geraten (jetzt 480), der Prior-Mittelwert mit dem reinen Messwert zu schwach
(jetzt das Doppelte). Wichtigster Nebenbefund: das Optimum ist ein breites
Plateau, keine Spitze -- 49 sehr verschiedene Kombinationen liegen gleichauf.
Damit sind die Defaults in `config.py` belegt statt geraten, und weiteres
Feintuning an diesen vier Schrauben lohnt nicht. Naechstes: Simulation.

## Quellen / Inspiration

- Dixon, Coles (1997): Modelling Association Football Scores and Inefficiencies in the Betting Market
- https://pena.lt/y/2021/06/24/predicting-football-results-using-python-and-dixon-and-coles/
- https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling-dixon-coles-and-time-weighting/
- https://github.com/opisthokonta/goalmodel
