"use strict";

const DATA = "../data/output/";

const state = {
  meta: null,
  matches: [],
  table: null,
  probs: new Map(),
  matchdays: [],
  mdIndex: 0,
  // Tatsaechliche Ergebnisse aus dem aktuellen Lauf, gueltig fuer jede Ansicht.
  actual: new Map(),
  snapshots: [],
};

const matchKey = (m) => m.home_team + "|" + m.away_team;

/**
 * Anzeigenamen. Die Daten nutzen die kanonischen Namen von
 * football-data.co.uk (team_mapping.py); uebersetzt wird nur fuer die Anzeige,
 * damit Abgleich und Archiv unberuehrt bleiben. tests/test_team_mapping.py
 * prueft, dass jedes kanonische Team hier steht.
 */
const TEAM_NAMES = {
  "Augsburg": "FC Augsburg",
  "Bayern Munich": "Bayern München",
  "Bielefeld": "Arminia Bielefeld",
  "Bochum": "VfL Bochum",
  "Darmstadt": "Darmstadt 98",
  "Dortmund": "Borussia Dortmund",
  "Ein Frankfurt": "Eintracht Frankfurt",
  "Elversberg": "SV Elversberg",
  "FC Koln": "1. FC Köln",
  "Fortuna Dusseldorf": "Fortuna Düsseldorf",
  "Freiburg": "SC Freiburg",
  "Greuther Furth": "Greuther Fürth",
  "Hamburg": "Hamburger SV",
  "Hannover": "Hannover 96",
  "Heidenheim": "1. FC Heidenheim",
  "Hertha": "Hertha BSC",
  "Hoffenheim": "TSG Hoffenheim",
  "Holstein Kiel": "Holstein Kiel",
  "Ingolstadt": "FC Ingolstadt",
  "Leverkusen": "Bayer Leverkusen",
  "M'gladbach": "Bor. Mönchengladbach",
  "Mainz": "Mainz 05",
  "Nurnberg": "1. FC Nürnberg",
  "Paderborn": "SC Paderborn",
  "RB Leipzig": "RB Leipzig",
  "Schalke 04": "FC Schalke 04",
  "St Pauli": "FC St. Pauli",
  "Stuttgart": "VfB Stuttgart",
  "Union Berlin": "Union Berlin",
  "Werder Bremen": "Werder Bremen",
  "Wolfsburg": "VfL Wolfsburg",
};

const teamName = (team) => TEAM_NAMES[team] || team;

/**
 * Weder 0 % noch 100 % anzeigen
 */
function pct(x, digits = 1) {
  if (x === null || x === undefined) return "-";
  const value = x * 100;
  for (let d = digits; d <= 2; d++) {
    const text = value.toFixed(d);
    const rounded = Number(text);
    if (rounded > 0 && rounded < 100) return text.replace(".", ",") + " %";
  }
  return value >= 50 ? ">99,99 %" : "<0,01 %";
}

function num(x, digits = 1) {
  if (x === null || x === undefined) return "-";
  return x.toFixed(digits).replace(".", ",");
}

function dateLabel(iso) {
  const d = new Date(iso + "T00:00:00");
  if (isNaN(d)) return iso;
  return d.toLocaleDateString("de-DE", {
    weekday: "short",
    day: "2-digit",
    month: "2-digit",
  });
}

/** Zone eines Platzes nach place_rules aus meta.json. */
function zoneFor(position) {
  const rules = (state.meta && state.meta.place_rules) || {};
  const inRange = (key) => {
    const r = rules[key];
    return r && position >= r[0] && position <= r[1];
  };
  if (inRange("relegated")) return "rel";
  if (inRange("relegation_playoff")) return "po";
  if (inRange("conference_league")) return "ecl";
  if (inRange("europa_league")) return "el";
  if (inRange("champions_league")) return "cl";
  return "";
}

function cell(text, cls) {
  const td = document.createElement("td");
  td.textContent = text;
  if (cls) td.className = cls;
  return td;
}

// --- Erwartete Tabelle -----------------------------------------------------

function renderExpected() {
  const rows = state.table.expected
    .slice()
    .sort((a, b) => a.expected_position - b.expected_position);
  const tbody = document.querySelector("#table-expected tbody");
  tbody.innerHTML = "";

  rows.forEach((row, i) => {
    const position = i + 1;
    const p = state.probs.get(row.team) || {};
    const tr = document.createElement("tr");
    const zone = zoneFor(position);
    if (zone) tr.className = "zone-" + zone;

    tr.appendChild(cell(String(position), "num pos"));
    tr.appendChild(cell(teamName(row.team)));
    tr.appendChild(cell(num(row.expected_points), "num"));
    tr.appendChild(cell(String(row.points_p05), "num"));
    tr.appendChild(cell(String(row.points_p95), "num"));
    tr.appendChild(cell(num(row.expected_goal_difference), "num"));
    tr.appendChild(cell(pct(p.champion), "num"));
    tr.appendChild(cell(pct(p.champions_league), "num"));
    tr.appendChild(cell(pct(p.relegated), "num"));

    tr.addEventListener("mouseenter", (e) => showTip(row, p, e));
    tr.addEventListener("mousemove", moveTip);
    tr.addEventListener("mouseleave", hideTip);
    tbody.appendChild(tr);
  });
}

// --- Tooltip ---------------------------------------------------------------

const tip = document.getElementById("tip");

function showTip(row, p, event) {
  const lines = [
    ["Meister", p.champion],
    ["Champions League", p.champions_league],
    ["Europa League", p.europa_league],
    ["Conference League", p.conference_league],
    ["Relegation", p.relegation_playoff],
    ["Abstieg", p.relegated],
  ];

  let html = "<h3>" + teamName(row.team) + "</h3><table>";
  html +=
    "<tr><td>Punkte (&#216;)</td><td>" +
    num(row.expected_points) +
    "</td></tr>";
  html +=
    "<tr><td>90-%-Intervall</td><td>" +
    row.points_p05 +
    " &ndash; " +
    row.points_p95 +
    " Punkte</td></tr>";
  html +=
    "<tr><td>Platz (&#216;)</td><td>" +
    num(row.expected_position, 2) +
    "</td></tr>";
  html +=
    "<tr><td>Tordifferenz (&#216;)</td><td>" +
    num(row.expected_goal_difference) +
    "</td></tr>";
  html += "</table>";

  html += '<table style="margin-top:.4rem">';
  for (const [label, value] of lines) {
    html +=
      "<tr><td>" + label + "</td><td>" + pct(value) + "</td></tr>";
  }
  html += "</table>";

  const positions = p.positions || [];
  if (positions.length) {
    const max = Math.max(...positions);
    html += '<p class="sub">Platzverteilung</p><table>';
    positions.forEach((value, i) => {
      const width = max > 0 ? Math.round((value / max) * 110) : 0;
      html +=
        "<tr><td>" +
        (i + 1) +
        ".</td><td><span class='bar' style='width:" +
        width +
        "px'></span></td><td>" +
        pct(value, 0) +
        "</td></tr>";
    });
    html += "</table>";
  }

  tip.innerHTML = html;
  tip.hidden = false;
  moveTip(event);
}

/**
 * Die drei wahrscheinlichsten Einzelergebnisse einer Partie.
 */
function showScoreTip(match, result, event) {
  let html =
    "<h3>" + teamName(match.home_team) + " &ndash; " + teamName(match.away_team) + "</h3>" +
    "<table><thead><tr><th>Ergebnis</th><th>Wahrscheinlichkeit</th></tr></thead><tbody>";
  (match.likely_scores || []).forEach(([home, away, probability]) => {
    const cls = result ? gradeOf([home, away], result) : "";
    html +=
      "<tr><td class='" + cls + "'>" + home + ":" + away +
      "</td><td class='num'>" + pct(probability) + "</td></tr>";
  });
  html += "</tbody></table>";
  if (result) html += LEGEND_HTML;
  tip.innerHTML = html;
  tip.hidden = false;
  moveTip(event);
}

function moveTip(event) {
  const pad = 14;
  let x = event.pageX + pad;
  let y = event.pageY + pad;
  const rect = tip.getBoundingClientRect();
  if (x + rect.width > window.scrollX + document.documentElement.clientWidth) {
    x = event.pageX - rect.width - pad;
  }
  if (y + rect.height > window.scrollY + document.documentElement.clientHeight) {
    y = Math.max(window.scrollY, event.pageY - rect.height - pad);
  }
  tip.style.left = x + "px";
  tip.style.top = y + "px";
}

function hideTip() {
  tip.hidden = true;
}

// --- Aktuelle Tabelle ------------------------------------------------------

function renderCurrent() {
  document.getElementById("current-section").hidden = !state.meta.matches_played;
  if (!state.meta.matches_played) return;
  const tbody = document.querySelector("#table-current tbody");
  tbody.innerHTML = "";

  for (const row of state.table.current) {
    const tr = document.createElement("tr");
    const zone = zoneFor(row.position);
    if (zone) tr.className = "zone-" + zone;
    tr.appendChild(cell(String(row.position), "num pos"));
    tr.appendChild(cell(teamName(row.team)));
    tr.appendChild(cell(String(row.played), "num"));
    tr.appendChild(cell(String(row.won), "num"));
    tr.appendChild(cell(String(row.drawn), "num"));
    tr.appendChild(cell(String(row.lost), "num"));
    tr.appendChild(cell(row.goals_for + ":" + row.goals_against, "num"));
    tr.appendChild(cell(String(row.goal_difference), "num"));
    tr.appendChild(cell(String(row.points), "num"));
    tbody.appendChild(tr);
  }
}

// --- Spiele ----------------------------------------------------------------

const mdSelect = document.getElementById("md-select");
const mdPrev = document.getElementById("md-prev");
const mdNext = document.getElementById("md-next");

function renderMatchdayNav() {
  mdSelect.innerHTML = "";
  state.matchdays.forEach((md, i) => {
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = md + ". Spieltag";
    mdSelect.appendChild(opt);
  });
  mdSelect.value = String(state.mdIndex);
  mdPrev.disabled = state.mdIndex === 0;
  mdNext.disabled = state.mdIndex === state.matchdays.length - 1;
}

function renderMatches() {
  const md = state.matchdays[state.mdIndex];
  const tbody = document.querySelector("#table-matches tbody");
  tbody.innerHTML = "";

  // Eine Prognose rechnet nur nach vorn: liegt der Spieltag komplett vor dem
  // ghewählten Stand, gibt es hier nichts zu zeigen
  const games = state.matches.filter((m) => m.matchday === md);
  const before = games.length > 0 && games.every((m) => m.finished);
  const notice = document.getElementById("matches-notice");
  notice.hidden = !before;
  document.getElementById("table-matches").hidden = before;
  if (before) {
    notice.textContent =
      "Du hast die Prognose vom Stand " +
      dateLabel(state.meta.as_of) +
      " ausgewählt. Der " +
      md +
      ". Spieltag liegt davor, deshalb gibt es hier keine Prognose für ihn. " +
      "Wähle oben einen früheren Prognose-Stand, um die damalige Prognose " +
      "mit den tatsächlichen Ergebnissen zu vergleichen.";
    return;
  }

  for (const m of games) {
    const hasPrediction = m.p_home !== undefined;
    const result = state.actual.get(matchKey(m));
    const tr = document.createElement("tr");
    tr.appendChild(cell(dateLabel(m.date)));
    tr.appendChild(cell(teamName(m.home_team)));
    tr.appendChild(cell(teamName(m.away_team)));
    tr.appendChild(cell(pct(m.p_home, 0), "num"));
    tr.appendChild(cell(pct(m.p_draw, 0), "num"));
    tr.appendChild(cell(pct(m.p_away, 0), "num"));

    tr.appendChild(
      cell(
        hasPrediction
          ? num(m.expected_home_goals) + " : " + num(m.expected_away_goals)
          : "-",
        "num"
      )
    );

    const tipCell = cell(m.likely_score ? m.likely_score.join(":") : "-", "num");
    if (m.likely_score) grade(tipCell, m.likely_score, result);
    if (m.likely_scores && m.likely_scores.length > 1) {
      hover(tipCell, (e) => showScoreTip(m, result, e));
    }
    tr.appendChild(tipCell);
    tr.appendChild(resultCell(m));
    tbody.appendChild(tr);
  }
}

function hover(td, show) {
  td.classList.add("hoverable");
  td.addEventListener("mouseenter", show);
  td.addEventListener("mousemove", moveTip);
  td.addEventListener("mouseleave", hideTip);
}

/** 1, 0 oder 2: Ausgang (Heimsieg, Remis, Auswaertssieg) eines Ergebnisses. */
const outcomeOf = (home, away) => (home > away ? 1 : home < away ? 2 : 0);

/**
 * Wertung wie im Tippspiel
 */
function gradeOf(predicted, result) {
  if (predicted[0] === result.home_goals && predicted[1] === result.away_goals) {
    return "hit";
  }
  return outcomeOf(predicted[0], predicted[1]) ===
    outcomeOf(result.home_goals, result.away_goals)
    ? "tend"
    : "miss";
}

function grade(td, predicted, result) {
  if (result) td.classList.add(gradeOf(predicted, result));
}

const LEGEND_HTML =
  '<p class="legend-tip">' +
  '<span class="key hit"></span> exakt &nbsp;' +
  '<span class="key tend"></span> Tendenz (1/X/2) &nbsp;' +
  '<span class="key miss"></span> daneben</p>';

function resultCell(m) {
  const result = state.actual.get(matchKey(m));
  if (!result) return cell("-", "num");
  return cell(result.home_goals + ":" + result.away_goals, "num played");
}

function setMatchday(index) {
  state.mdIndex = Math.min(
    Math.max(index, 0),
    state.matchdays.length - 1
  );
  renderMatchdayNav();
  renderMatches();
}

mdPrev.addEventListener("click", () => setMatchday(state.mdIndex - 1));
mdNext.addEventListener("click", () => setMatchday(state.mdIndex + 1));
mdSelect.addEventListener("change", () => setMatchday(Number(mdSelect.value)));

// --- Start -----------------------------------------------------------------

function renderMeta() {
  const m = state.meta;
  document.getElementById("meta").textContent =
    "Saison " +
    m.season +
    " · Stand " +
    dateLabel(m.as_of) +
    " · " +
    m.matches_played +
    " von " +
    (m.matches_played + m.matches_open) +
    " Spielen gespielt · " +
    m.n_simulations.toLocaleString("de-DE") +
    " Simulationen";
}

/** `folder` ist "" für den aktuellen Stand, sonst ein Archivordner. */
async function loadJson(folder, name) {
  const res = await fetch(DATA + folder + name);
  if (!res.ok) throw new Error(folder + name + ": HTTP " + res.status);
  return res.json();
}

async function loadSnapshot(folder) {
  const [meta, matches, table, probs] = await Promise.all([
    loadJson(folder, "meta.json"),
    loadJson(folder, "matches.json"),
    loadJson(folder, "table.json"),
    loadJson(folder, "probabilities.json"),
  ]);
  return { meta, matches, table, probs };
}

function showSnapshot(snapshot, startMatchday) {
  state.meta = snapshot.meta;
  state.matches = snapshot.matches;
  state.table = snapshot.table;
  state.probs = new Map(snapshot.probs.map((p) => [p.team, p]));
  state.matchdays = [...new Set(state.matches.map((m) => m.matchday))].sort(
    (a, b) => a - b
  );

  // Aktuell: erster Spieltag mit offenen Spielen. Archiv: der Spieltag des Stands.
  const target =
    startMatchday ?? (state.matches.find((m) => !m.finished) || {}).matchday;
  state.mdIndex = Math.max(state.matchdays.indexOf(target), 0);

  renderMeta();
  renderExpected();
  renderCurrent();
  setMatchday(state.mdIndex);
}

function snapshotLabel(s) {
  const stand = dateLabel(s.as_of);
  return s.matchday === 1
    ? "Originalprognose (vor Saisonstart, " + stand + ")"
    : "Vor dem " + s.matchday + ". Spieltag (" + stand + ")";
}

function setupSnapshotSelect(current) {
  const select = document.getElementById("snap-select");
  const row = document.getElementById("snap-row");
  if (!state.snapshots.length) return;

  select.innerHTML = "";
  const now = document.createElement("option");
  now.value = "";
  now.textContent = "Aktuell (Stand " + dateLabel(current.meta.as_of) + ")";
  select.appendChild(now);
  // Neueste zuerst.
  for (const s of state.snapshots.slice().reverse()) {
    const opt = document.createElement("option");
    opt.value = s.path;
    opt.textContent = snapshotLabel(s);
    select.appendChild(opt);
  }
  row.hidden = false;

  select.addEventListener("change", async () => {
    try {
      const path = select.value;
      row.classList.toggle("archived", path !== "");
      if (!path) return showSnapshot(current);
      const entry = state.snapshots.find((s) => s.path === path);
      showSnapshot(await loadSnapshot(path), entry.matchday);
    } catch (err) {
      showError(err);
    }
  });
}

function showError(err) {
  document.getElementById("meta").innerHTML =
    '<span class="err">Daten konnten nicht geladen werden (' +
    err.message +
    "). Seite ueber einen HTTP-Server oeffnen, nicht per file://.</span>";
}

async function init() {
  try {
    const current = await loadSnapshot("");
    state.actual = new Map(
      current.matches.filter((m) => m.finished).map((m) => [matchKey(m), m])
    );
    // Ohne Archiv (noch nie gerechnet) bleibt es bei der aktuellen Ansicht.
    try {
      state.snapshots = await loadJson("archive/", "index.json");
    } catch (_) {
      state.snapshots = [];
    }
    showSnapshot(current);
    setupSnapshotSelect(current);
  } catch (err) {
    showError(err);
  }
}

init();
