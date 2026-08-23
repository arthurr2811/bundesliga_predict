"use strict";

const DATA = "../data/output/";

const state = {
  meta: null,
  matches: [],
  table: null,
  probs: new Map(),
  matchdays: [],
  mdIndex: 0,
};

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
    tr.appendChild(cell(row.team));
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

  let html = "<h3>" + row.team + "</h3><table>";
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

/** Die drei wahrscheinlichsten Einzelergebnisse einer Partie. */
function showScoreTip(match, event) {
  const rank = ["1.", "2.", "3."];
  let html =
    "<h3>" + match.home_team + " &ndash; " + match.away_team + "</h3><table>";
  match.likely_scores.forEach(([home, away, probability], i) => {
    html +=
      "<tr><td>" +
      (rank[i] || i + 1 + ".") +
      "</td><td>" +
      home +
      ":" +
      away +
      "</td><td>" +
      pct(probability) +
      "</td></tr>";
  });
  html += "</table>";
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
  if (!state.meta.matches_played) return;
  document.getElementById("current-section").hidden = false;
  const tbody = document.querySelector("#table-current tbody");
  tbody.innerHTML = "";

  for (const row of state.table.current) {
    const tr = document.createElement("tr");
    const zone = zoneFor(row.position);
    if (zone) tr.className = "zone-" + zone;
    tr.appendChild(cell(String(row.position), "num pos"));
    tr.appendChild(cell(row.team));
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

  for (const m of state.matches.filter((m) => m.matchday === md)) {
    const tr = document.createElement("tr");
    tr.appendChild(cell(dateLabel(m.date)));
    tr.appendChild(cell(m.home_team));
    tr.appendChild(cell(m.away_team));
    tr.appendChild(cell(pct(m.p_home, 0), "num"));
    tr.appendChild(cell(pct(m.p_draw, 0), "num"));
    tr.appendChild(cell(pct(m.p_away, 0), "num"));
    tr.appendChild(
      cell(
        num(m.expected_home_goals) + " : " + num(m.expected_away_goals),
        "num"
      )
    );
    const score = m.likely_score ? m.likely_score.join(":") : "-";
    const tipCell = cell(m.finished ? score + " (gespielt)" : score, "num");
    if (m.likely_scores && m.likely_scores.length > 1) {
      tipCell.classList.add("hoverable");
      tipCell.addEventListener("mouseenter", (e) => showScoreTip(m, e));
      tipCell.addEventListener("mousemove", moveTip);
      tipCell.addEventListener("mouseleave", hideTip);
    }
    tr.appendChild(tipCell);
    tbody.appendChild(tr);
  }
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

async function loadJson(name) {
  const res = await fetch(DATA + name);
  if (!res.ok) throw new Error(name + ": HTTP " + res.status);
  return res.json();
}

async function init() {
  try {
    const [meta, matches, table, probs] = await Promise.all([
      loadJson("meta.json"),
      loadJson("matches.json"),
      loadJson("table.json"),
      loadJson("probabilities.json"),
    ]);
    state.meta = meta;
    state.matches = matches;
    state.table = table;
    state.probs = new Map(probs.map((p) => [p.team, p]));
    state.matchdays = [...new Set(matches.map((m) => m.matchday))].sort(
      (a, b) => a - b
    );

    // Auf den ersten Spieltag mit offenen Spielen springen.
    const firstOpen = matches.find((m) => !m.finished);
    if (firstOpen) {
      state.mdIndex = Math.max(
        state.matchdays.indexOf(firstOpen.matchday),
        0
      );
    }

    renderMeta();
    renderExpected();
    renderCurrent();
    setMatchday(state.mdIndex);
  } catch (err) {
    document.getElementById("meta").innerHTML =
      '<span class="err">Daten konnten nicht geladen werden (' +
      err.message +
      "). Seite ueber einen HTTP-Server oeffnen, nicht per file://.</span>";
  }
}

init();
