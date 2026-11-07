/* Evidence: one topic per claim. Each opens with a question, a figure, its denominator and its source;
   the exact table is one click away. Everything comes from local artifacts in data/mirror/ (works offline). */
(function () {
  "use strict";
  const P = window.PokerSim, V = window.Viz;
  const { api, el, toast, fmt, pct } = P;
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));

  const G6 = "Every table size and real players", G2 = "Two-player tables";
  const TOPICS = [
    { id: "wide", group: G6, kicker: "Data", q: "How much more data does every table size give?" },
    { id: "situation", group: G6, kicker: "Forecasts", q: "Which predicts a fold better: the situation or the person?" },
    { id: "bluff", group: G6, kicker: "Real decisions", q: "Does a better forecast win chips from real players?" },
    { id: "realbench", group: G6, kicker: "RL bettors", q: "Do the RL bettors gain against real players?", file: "benchmark_real.json" },
    { id: "strength", group: G6, kicker: "Strength table", q: "Do Pluribus hands improve the strength table?", file: "strength_pluribus.json" },
    { id: "neural", group: G6, kicker: "GPU forecaster", q: "Does a neural network forecast better?", file: "neural_forecaster.json" },
    { id: "typeswide", group: G6, kicker: "Player types", q: "Do player types help on every table size?", file: "styles_wide.json" },
    { id: "filter", group: G2, kicker: "Historical data", q: "Where did the folds go?" },
    { id: "forecast", group: G2, kicker: "Forecasts", q: "Does a player's own history sharpen the forecast?" },
    { id: "benchmark", group: G2, kicker: "Simulated bettors", q: "When does knowing the opponent help?" },
    { id: "types", group: G2, kicker: "Player types", q: "Are there player types worth using?" },
    { id: "range", group: G2, kicker: "Range readout", q: "Can the range readout be trusted?" },
    { id: "training", group: G2, kicker: "Training", q: "Did the bots train enough?" },
    { id: "sessions", group: G2, kicker: "Humans", q: "What do the volunteer sessions show?" },
  ];
  const OPP = { caller: ["Frequent caller", "folds 15%"], folder: ["Frequent folder", "folds 65%"], population: ["Population player", "historical average"], strength: ["Strength-dependent", "folds weak, raises strong"] };
  const E = { data: null, styles: null, built: false, probes: 64, filter: "known" };
  const n0 = (x) => Math.round(x).toLocaleString();
  const milli = (x) => (x > 0 ? "+" : x < 0 ? "−" : "") + Math.abs(1000 * x).toFixed(1);
  const milliTick = (x) => (x > 0.00001 ? "+" : x < -0.00001 ? "−" : "") + Math.abs(1000 * x).toFixed(0);
  const plotW = () => Math.min(1000, ($("#evTopics") || {}).clientWidth || 900);

  function shell() {
    const root = $("#view-evidence");
    root.innerHTML = "";
    const nav = el("nav", { class: "ev-nav", "aria-label": "Evidence topics" });
    const main = el("div", { class: "ev-main" },
      el("header", { class: "ev-head" }, el("div", { class: "scene-eyebrow mono", text: "Evidence · every number names its source and denominator" }),
        el("h1", { class: "q", tabindex: "-1" }, "What the data ", el("em", { text: "can and can’t" }), " say."),
        el("p", { class: "lede", text: "Heads-up river pots from 2009 online cash games: one bet-size band, one street. Synthetic opponents are labelled synthetic; claims about people come only from real recorded answers or held-out volunteer sessions." })),
      el("div", { id: "evTopics" }));
    root.appendChild(el("div", { class: "evidence" }, nav, main));
    E.built = true;
    // highlight the topic in view
    if ("IntersectionObserver" in window) {
      E.io = new IntersectionObserver((ents) => ents.forEach((en) => { if (en.isIntersecting) markNav(en.target.dataset.topic); }), { rootMargin: "-30% 0px -60% 0px" });
    }
  }
  // nav lists only the topics that rendered, grouped, numbered in page order
  function buildNav(ids) {
    const nav = $(".ev-nav");
    nav.innerHTML = "";
    let group = null, ol = null;
    ids.forEach((id, i) => {
      const t = TOPICS.find((x) => x.id === id);
      if (t.group !== group) { group = t.group; nav.appendChild(el("div", { class: "ev-nav-group mono", text: group })); ol = el("ol"); nav.appendChild(ol); }
      ol.appendChild(el("li", {}, el("a", { href: "#evidence=" + id, "data-topic": id, onclick: (e) => { e.preventDefault(); goTopic(id, true); } }, el("span", { class: "num", text: String(i + 1) }), el("span", { text: t.q }))));
    });
  }
  function markNav(id) { $$(".ev-nav a").forEach((a) => { const on = a.dataset.topic === id; a.classList.toggle("active", on); if (on) a.setAttribute("aria-current", "true"); else a.removeAttribute("aria-current"); }); }
  function goTopic(id, user) {
    const sec = $("#ev-" + id);
    if (!sec) return;
    markNav(id);
    try { history.replaceState(null, "", "#evidence=" + id); } catch (e) { /* ignore */ }
    const y = sec.getBoundingClientRect().top + window.scrollY - 72;
    window.scrollTo({ top: y, behavior: user && !V.reduceMotion() ? "smooth" : "auto" });
    const h = $("h2", sec);
    if (h && user) h.focus({ preventScroll: true });
  }
  function topic(id, figure, opts) {
    const t = TOPICS.find((x) => x.id === id);
    const sec = el("section", { class: "ev-topic", id: "ev-" + t.id, "data-topic": t.id, "aria-labelledby": "evq-" + t.id },
      el("div", { class: "ev-kicker mono", text: t.kicker }),
      el("h2", { class: "ev-q", id: "evq-" + t.id, tabindex: "-1", text: t.q }),
      opts.answer ? (opts.answer instanceof Node && opts.answer.tagName === "P" ? opts.answer : el("p", { class: "ev-answer" }, ...[].concat(opts.answer))) : null,
      opts.present === false ? el("p", { class: "hint", text: "Not generated yet. " + (opts.missing || "") }) : figure,
      opts.denom ? el("p", { class: "ev-denom" }, el("span", { class: "ev-label", text: "Denominator" }), " ", ...[].concat(opts.denom)) : null,
      opts.source ? el("p", { class: "ev-src mono", text: "Source: " + opts.source }) : null,
      opts.table ? el("details", { class: "ev-table" }, el("summary", { text: opts.tableLabel || "Exact table" }), opts.table) : null,
      opts.after || null);
    if (E.io) E.io.observe(sec);
    return sec;
  }
  // rows: arrays of cells, or {attrs, cells}; a cell is text or a Node
  function table(head, rows, cls) {
    const td = (c) => el("td", {}, c instanceof Node ? c : String(c === null || c === undefined ? "" : c));
    return el("div", { class: "ev-table-wrap" }, el("table", { class: "mini " + (cls || "") },
      el("thead", {}, el("tr", {}, head.map((h) => el("th", { text: h })))),
      el("tbody", {}, rows.map((r) => Array.isArray(r) ? el("tr", {}, r.map(td)) : el("tr", r.attrs || {}, r.cells.map(td))))));
  }
  const src = (p) => `${p.source}${p.generated ? " · generated " + p.generated : ""}`;

  // ------------------------------------------------------------ Phase 6: every table size, real players
  const ARM = { pooled: "One rate for everyone", population: "Average player, by turn history", situation: "Situation model",
    personal: "This player's history", situation_personal: "Situation + this player", neural: "Neural network" };
  const verdict = (b) => (b.better ? "a real gain" : b.worse ? "a real loss" : "no clear effect");
  const sg2 = (x, d = 2) => (x === null || x === undefined ? "-" : V.sgn(x, d));
  const ci2 = (v, d = 2) => (v ? `${sg2(v[0], d)} to ${sg2(v[1], d)}` : "-");
  const TABLE_LABEL = { "2": "Two players", "3-6": "3–6 players", "7-10": "7–10 players" };
  function protoSeg(protos, current, onPick) {
    const NAMES = { frozen: "History up to the cutoff", online: "Learning as you go" };
    return el("div", { class: "seg", role: "group", "aria-label": "Evaluation protocol" }, el("span", { class: "seg-label", text: "Protocol" }),
      protos.map((k) => el("button", { class: "seg-btn" + (k === current ? " on" : ""), "aria-pressed": String(k === current), "data-k": k, onclick: (e) => {
        $$(".seg-btn", e.currentTarget.parentNode).forEach((b) => { const on = b.dataset.k === k; b.classList.toggle("on", on); b.setAttribute("aria-pressed", String(on)); });
        onPick(k);
      } }, NAMES[k] || k)));
  }
  // single-series horizontal bars, value at the tip; `ref` rows draw as an outline
  function hbars(rows) {
    const max = Math.max(1, ...rows.map((r) => r.v));
    return el("div", { class: "hbars", role: "img", "aria-label": rows.map((r) => `${r.label}: ${n0(r.v)}`).join("; ") },
      rows.map((r) => el("div", { class: "hb-row" + (r.ref ? " ref" : "") }, el("span", { class: "hb-k", text: r.label }),
        el("div", { class: "hb-track" }, el("div", { class: "hb-bar", style: `width:${(100 * r.v / max).toFixed(2)}%` }), el("span", { class: "hb-v num", text: n0(r.v) })))));
  }

  function wideTopic(ws) {
    const d = ws.data, bt = d.by_table_size || {}, m = d.main_cohort || {};
    const before = (bt["2"] || {}).exact_half || 12407;
    const rows = ["2", "3-6", "7-10"].filter((k) => bt[k]).map((k) => ({ label: TABLE_LABEL[k], v: bt[k].main }));
    rows.push({ label: "Before: exact half pot, two-player tables", v: before, ref: true });
    const fig = el("figure", { class: "ev-fig" }, el("div", { class: "ev-sub", text: "Decisions in the main cohort, by table size" }), hbars(rows));
    return topic("wide", fig, {
      answer: [`About `, el("b", { class: "num", text: `${Math.round(m.decisions / before)}×` }), ` more: ${n0(m.decisions)} decisions by ${n0(m.players)} players, from every table size where two players reach the river, against ${n0(before)} before.`],
      denom: [`${n0(d.rows)} rows scanned; ${n0(d.kept)} unique river spots. Main cohort: ${m.definition}; train ${n0(m.train)} / test ${n0(m.test)}.`],
      source: `${src(ws)} · takara-ai/poker_hands @ ${(d.revision || "").slice(0, 10)} · ${(d.shards_scanned || []).length} shards`,
      table: table(["table size", "any river bet", "bet 40–60% of pot", "main cohort", "fold / call / raise", "train / test"],
        ["2", "3-6", "7-10"].filter((k) => bt[k]).map((k) => { const b = bt[k]; return [TABLE_LABEL[k], n0(b.any), n0(b.band), n0(b.main), `${n0(b.main_fold)} / ${n0(b.main_call)} / ${n0(b.main_raise)}`, `${n0(b.main_train)} / ${n0(b.main_test)}`]; })),
    });
  }

  function situationTopic(fw) {
    const d = fw.data, protos = Object.keys(d.protocols || {});
    const LIFTS = [["situation_vs_population", "Situation model vs average player", (d.situation_model && d.situation_model.features) ? `${d.situation_model.features.length} public features` : "public features"],
      ["personal_vs_population", "Player history vs average player", `prior strength ${d.prior_strength}`],
      ["situation_personal_vs_situation", "Player history on top of the situation", "both together vs situation alone"]];
    const holder = el("div");
    let cur = protos.includes("frozen") ? "frozen" : protos[0];
    const all = protos.flatMap((k) => LIFTS.map(([key]) => (d.protocols[k].lift_log_loss || {})[key]).filter(Boolean));
    const lo = Math.min(0, ...all.map((b) => b.ci95[0])), hi = Math.max(0, ...all.map((b) => b.ci95[1]));
    function draw(k) {
      cur = k;
      const L = d.protocols[k].lift_log_loss || {};
      const rows = LIFTS.filter(([key]) => L[key]).map(([key, label, sub]) => ({ label, sub, mean: L[key].mean, lo: L[key].ci95[0], hi: L[key].ci95[1], highlight: key === "situation_vs_population", extra: verdict(L[key]) }));
      holder.innerHTML = "";
      holder.appendChild(V.intervalPlot(rows, { width: plotW(), domain: [lo, hi], fmt: milli, tickFmt: milliTick, step: 0.005, labelW: 280, valueW: 170,
        xTitle: "Log-loss improvement, ×10⁻³ per decision (right = better), 95% intervals over players", unit: "×10⁻³ log loss per decision", aria: "Forecast improvements" }));
    }
    const fig = el("figure", { class: "ev-fig" }, protoSeg(protos, cur, draw), holder);
    draw(cur);
    const L = (d.protocols[cur] || {}).lift_log_loss || {}, a = L.situation_vs_population, b = L.personal_vs_population, c = L.situation_personal_vs_situation;
    const tb = d.cohort && d.cohort.test_by_table_size;
    return topic("situation", fig, {
      answer: a && b ? [`The situation, by a wide margin. The situation model cuts log loss by `, el("b", { class: "num", text: milli(a.mean) }), ` ×10⁻³ per decision; the player's own history adds ${milli(b.mean)} on its own` + (c ? ` and ${milli(c.mean)} on top of the situation (${verdict(c)}).` : ".")] : null,
      denom: [`${n0((d.cohort || {}).test)} test decisions by ${n0((d.cohort || {}).test_players)} players` + (tb ? ` (${Object.entries(tb).map(([k, v]) => `${TABLE_LABEL[k] || k} ${n0(v)}`).join(", ")}).` : ".") + ` Situation model: ${(d.situation_model || {}).kind || "?"}.`],
      source: `${src(fw)} · ${d.label || ""}`,
      table: table(["forecaster", ...protos.map((k) => (k === "frozen" ? "log loss, history up to cutoff" : "log loss, learning as you go"))],
        ["pooled", "population", "situation", "personal", "situation_personal"].map((k) => [ARM[k], ...protos.map((pk) => ((d.protocols[pk].overall || {})[k] === undefined ? "-" : d.protocols[pk].overall[k].toFixed(4)))])),
    });
  }

  function bluffTopic(bt) {
    const d = bt.data, protos = Object.keys(d.protocols || {});
    let cur = protos.includes("online") ? "online" : protos[0];
    const holder = el("div");
    const KEYS = [["situation", "Situation model"], ["personal", "Player history"], ["situation_personal", "Situation + this player"]];
    function draw(k) {
      cur = k;
      const pr = d.protocols[k], c = pr.chips_per_decision, vs = pr.vs_population || {};
      const best = KEYS.map(([kk]) => kk).filter((kk) => c[kk] !== undefined).sort((x, y) => c[y] - c[x])[0];
      const marks = [["never", "never bluff", c.never], ["always", "always bluff = average-player forecast", c.always], [best, `best forecast: ${ARM[best].toLowerCase()}`, c[best]], ["perfect", "perfect foresight (ceiling)", c.perfect]].filter((x) => x[2] !== undefined);
      const max = Math.max(...marks.map((x) => x[2])) * 1.05 || 1;
      holder.innerHTML = "";
      holder.appendChild(el("div", { class: "ev-sub", text: "Chips per decision for a bluffer who bets half the pot (pot = 100)" }));
      holder.appendChild(el("div", { class: "cscale", role: "img", "aria-label": marks.map((x) => `${x[1]}: ${x[2].toFixed(2)}`).join("; ") },
        el("div", { class: "cs-track" }), marks.map((x, i) => el("div", { class: "cs-mark " + (i % 2 ? "below" : "above") + (x[0] === best ? " hl" : ""), style: `left:${(100 * x[2] / max).toFixed(2)}%` },
          el("span", { class: "cs-label" }, el("b", { class: "num", text: x[2].toFixed(1) }), " ", x[1])))));
      const rows = KEYS.filter(([kk]) => vs[kk]).map(([kk, label]) => ({ label, sub: `${pct(vs[kk].decisions_changed, 1)} of decisions changed`, mean: vs[kk].mean, lo: vs[kk].ci95[0], hi: vs[kk].ci95[1], highlight: kk === "situation_personal", extra: verdict(vs[kk]) }));
      holder.appendChild(el("div", { class: "ev-sub", text: "Gain over the average-player forecast" }));
      holder.appendChild(V.intervalPlot(rows, { width: plotW(), labelW: 230, valueW: 170, step: 0.05, xTitle: "Δ chips per decision vs the average-player forecast, 95% intervals over players", unit: "chips per decision", aria: "Bluff test gains" }));
    }
    const fig = el("figure", { class: "ev-fig" }, protoSeg(protos, cur, draw), holder);
    draw(cur);
    const pr = d.protocols[cur], sp = (pr.vs_population || {}).situation_personal, c = pr.chips_per_decision;
    return topic("bluff", fig, {
      answer: sp ? [`Only a little. Scored against ${n0(d.decisions)} real recorded answers, a bluffer using the situation and the player wins `, el("b", { class: "num", text: sg2(sp.mean) }), ` chips per decision more than one using the average player (${ci2(sp.ci95)}, ${verdict(sp)}) — small next to the ${(c.perfect - c.always).toFixed(0)}-chip gap to perfect foresight. The average player already folds often enough that bluffing nearly always pays.`] : null,
      denom: [`${n0(d.decisions)} held-out decisions by ${n0(d.players)} players. Rule: ${d.rule}. ${d.units || ""}`],
      source: `${src(bt)} · ${d.label || ""}`,
      table: table(["the bluffer uses", "chips per decision", "bluffs"], [["never", "never bluff"], ["always", "always bluff"], ["population", "average-player forecast"], ["situation", "situation model"], ["personal", "player history"], ["situation_personal", "situation + player"], ["perfect", "perfect foresight (ceiling)"]]
        .filter(([k]) => pr.chips_per_decision[k] !== undefined).map(([k, label]) => [label, pr.chips_per_decision[k].toFixed(2), k === "always" ? "100%" : k === "never" ? "0%" : (pr.bluff_rate || {})[k] !== undefined ? pct(pr.bluff_rate[k], 1) : "-"])),
    });
  }

  function realBenchTopic(rb) {
    const d = rb.data, rows = d.rows || [];
    const plotRows = rows.map((r) => ({ label: `${r.probes} answers seen`, sub: `${pct(r.share_helped, 0)} helped · ${pct(r.share_hurt, 0)} hurt`, mean: r.mean_gain_chips, lo: r.ci95[0], hi: r.ci95[1], extra: `${pct(r.decisions_changed, 0)} of decisions changed` }));
    const last = rows[rows.length - 1];
    const q = d.qlearning_check, g = d.by_true_fold_rate || {};
    return topic("realbench", el("figure", { class: "ev-fig" }, V.intervalPlot(plotRows, { width: plotW(), labelW: 200, valueW: 190, xTitle: "Gain in chips per decision vs the population bettor, 95% intervals", unit: "chips per decision" })), {
      answer: last ? [`After ${last.probes} answers, the bettor gains `, el("b", { class: "num", text: sg2(last.mean_gain_chips) }), ` chips per decision (${ci2(last.ci95)}).`] : null,
      denom: `${n0(d.players)} real players. ${d.label || ""}`,
      source: src(rb),
      after: el("div", {}, Object.keys(g).length ? el("p", { class: "hint", text: "By how often the player really folds: " + Object.entries(g).map(([k, v]) => `${k} ${sg2(v.mean_gain_chips)} (${v.players} players)`).join(" · ") }) : null,
        el("p", { class: "hint", text: `With 0 answers seen the two bettors are ${d.zero_probe_identical ? "identical, as they must be" : "NOT identical: check the setup"}.` + (q ? ` Q-learning re-check on ${q.players.length} players: ${q.cells_agree} of ${q.cells} first-decision cells match the exact best response; value gap ${sg2(q.mean_value_gap_chips)} chips.` : "") })),
      table: table(["answers seen", "gain, chips per decision", "95% interval", "players helped", "players hurt", "decisions changed"], rows.map((r) => [r.probes, sg2(r.mean_gain_chips), ci2(r.ci95), pct(r.share_helped, 0), pct(r.share_hurt, 0), pct(r.decisions_changed, 0)])),
    });
  }

  function strengthTopic(sp) {
    const d = sp.data, c = d.check || {};
    return topic("strength", el("figure", { class: "ev-fig" }, el("div", { class: "stat-row" },
      stat(sg2(c.improvement, 3), `log-score change with the calibrated table (95% ${ci2(c.ci95, 3)})`),
      stat(`${pct(c.top_bucket_right_default, 0)} → ${pct(c.top_bucket_right_calibrated, 0)}`, "top strength bucket right, default → calibrated"),
      stat(d.use ? "used" : "not used", d.use ? "by the twins" : "the twins keep the labelled default"))), {
      answer: [d.use ? "Yes: the twins now use the Pluribus-calibrated table." : "Not clearly, so the twins keep the labelled default table."],
      denom: `Fitted on ${n0(d.fit_spots)} Pluribus spots, checked on the other ${n0(d.check_spots)}. Log score ${c.log_score_default} default, ${c.log_score_calibrated} calibrated.`,
      source: `${src(sp)} · ${d.rule || ""}`,
    });
  }

  function neuralTopic(nf) {
    const d = nf.data, lifts = d.neural_lift_vs || {};
    const rows = Object.entries(lifts).map(([k, b]) => ({ label: `vs ${(ARM[k] || k.replace(/_/g, " ")).toLowerCase()}`, mean: b.mean, lo: b.ci95[0], hi: b.ci95[1], extra: verdict(b) }));
    return topic("neural", el("figure", { class: "ev-fig" }, rows.length ? V.intervalPlot(rows, { width: plotW(), fmt: milli, tickFmt: milliTick, labelW: 260, valueW: 170, xTitle: "Neural network's log-loss improvement, ×10⁻³ per decision (right = better)", unit: "×10⁻³ log loss per decision" }) : null), {
      answer: d.bluff_vs_population ? [`In the bluff test the network wins `, el("b", { class: "num", text: sg2(d.bluff_vs_population.mean) }), ` chips per decision against the average-player forecast (${ci2(d.bluff_vs_population.ci95)}).`] : null,
      denom: `${d.device || ""}${d.quick ? ", quick check only" : ""}. Same frozen test half as the other forecasters.`,
      source: `${src(nf)} · ${d.label || ""}`,
      table: table(["forecaster", "test log loss"], Object.entries(d.test_log_loss || {}).map(([k, v]) => [ARM[k] || k.replace(/_/g, " "), v.toFixed(4)])),
    });
  }

  function typesWideTopic(tw) {
    const g = (tw.data || {}).gate || {}, a = g.type_vs_population_zero_history, b = g.type_vs_personal;
    const rows = [];
    if (a) rows.push({ label: "Type prior vs average player", sub: `no river history · ${n0(a.players)} players`, mean: a.lift, lo: a.ci95[0], hi: a.ci95[1] });
    if (b) rows.push({ label: "Types + history vs history", sub: b.players ? `${n0(b.players)} players` : "", mean: b.lift, lo: b.ci95[0], hi: b.ci95[1] });
    return topic("typeswide", el("figure", { class: "ev-fig" }, rows.length ? V.intervalPlot(rows, { width: plotW(), fmt: milli, tickFmt: milliTick, labelW: 230, valueW: 170, xTitle: "Log-loss improvement, ×10⁻³ per decision (right = types help)", unit: "×10⁻³ log loss per decision" }) : null,
      el("p", { class: "gate " + (g.pass ? "pass" : "fail") }, el("span", { class: "tag " + (g.pass ? "good" : "warn"), text: g.pass ? "✓ gate passed" : "✗ gate failed" }), " ", g.pass ? "Both intervals sit above zero." : "Types ship only if both intervals sit above zero.")), {
      source: `${src(tw)} · rule: ${g.rule || ""}`,
    });
  }

  function pendingNote(ids) {
    const items = ids.map((id) => TOPICS.find((t) => t.id === id)).filter(Boolean);
    return el("section", { class: "ev-pending", "aria-label": "Not generated yet" }, el("div", { class: "ev-kicker mono", text: "Not generated yet" }),
      el("ul", {}, items.map((t) => el("li", {}, el("span", { text: t.q }), " ", el("span", { class: "mono muted", text: `data/mirror/${t.file || ""}` })))),
      el("p", { class: "hint", text: "Each appears here as a full topic once its artifact exists; see PROGRESS.md (Phase 6) for the commands." }));
  }

  // ------------------------------------------------------------ 1 filter
  function filterTopic(p) {
    if (!p.present) return topic("filter", null, { present: false, missing: "python -m pokersim mirror-prepare" });
    const two = p.data.two_shard.cohorts, full = p.data.full_corpus.cohorts;
    const strip = V.filterStrip(two);
    const big = el("div", { class: "fx-big small" }, el("span", { class: "fx-num" }), el("span", { class: "fx-sub" }));
    const keys = [["all", "All eligible offers", "all_eligible"], ["known", "Known defender cards", "defender_cards_known"], ["clean", "Warning-free hands", "warning_free"]];
    const seg = el("div", { class: "seg", role: "group", "aria-label": "Filter" }, keys.map(([k, label]) => el("button", { class: "seg-btn", "data-k": k, "aria-pressed": "false", onclick: () => set(k) }, label)));
    function set(k) {
      E.filter = k;
      $$(".seg-btn", seg).forEach((b) => { const on = b.dataset.k === k; b.classList.toggle("on", on); b.setAttribute("aria-pressed", String(on)); });
      strip.set(k);
      const c = two[keys.find((x) => x[0] === k)[2]];
      $(".fx-num", big).textContent = `${n0(c.folds)} of ${n0(c.opportunities)}`;
      $(".fx-sub", big).textContent = `folded · ${pct(c.folds / c.opportunities, 0)}${k === "all" ? "" : ` · ${n0(two.all_eligible.folds)} of ${n0(two.all_eligible.opportunities)} before filtering`}`;
    }
    const labels = { all_eligible: "All eligible offers", half_checked_through: "Exact half pot, turn checked through", half_bettor_aggr: "Exact half pot, bettor bet the turn", half_defender_aggr: "Exact half pot, defender bet the turn", defender_cards_known: "Defender's cards became known", warning_free: "Warning-free hands" };
    const rows = Object.keys(full).map((k) => { const a = two[k], b = full[k]; return [labels[k] || k, `${n0(a.folds)} / ${n0(a.opportunities)}`, pct(a.folds / (a.opportunities || 1), 1), `${n0(b.folds)} / ${n0(b.opportunities)}`, pct(b.folds / (b.opportunities || 1), 1)]; });
    const fig = el("figure", { class: "ev-fig" }, seg, big, strip.root);
    const sec = topic("filter", fig, {
      answer: [`Filtering to hands where the defender's cards became known keeps ${n0(two.defender_cards_known.opportunities)} offers and `, el("b", { text: "none of the folds" }), `. Folds never reach showdown, so a model trained on revealed hands never sees one.`],
      denom: [`${n0(two.all_eligible.opportunities)} eligible offers in shards 35 and 36; `, `${n0(p.scan.eligible_unique)} unique eligible offers from ${n0(p.scan.rows)} hands across all 37 shards. Modelling cohort: ${n0(p.scan.eligible_modelling)} exact half-pot offers (train ${n0(p.scan.manifest.train)} / test ${n0(p.scan.manifest.test)}).`],
      source: src(p),
      table: table(["cohort", "shards 35+36: folds / offers", "fold rate", "all 37 shards: folds / offers", "fold rate"], rows),
    });
    set(E.filter);
    return sec;
  }

  // ------------------------------------------------------------ 2 forecast ladder
  function forecastTopic(fl, popP) {
    if (!fl.present) return topic("forecast", null, { present: false, missing: "python -m pokersim mirror-eval" });
    const d = fl.data, f = d.frozen || d, lift = f.paired_lift_log_loss || {};
    const NAMES = { personal: ["Population + this player's history", "the live model"], type: ["Player type + history", "not shipped (gate failed)"], pooled: ["One pooled model", "ignores the turn history"] };
    const rows = ["personal", "type", "pooled"].filter((k) => lift[k]).map((k) => ({ label: NAMES[k][0], sub: NAMES[k][1], mean: lift[k].lift_vs_population, lo: lift[k].ci95[0], hi: lift[k].ci95[1], highlight: k === "personal" }));
    const plot = V.intervalPlot(rows, { width: plotW(), fmt: milli, tickFmt: milliTick, step: 0.004, xTitle: "Log-loss improvement over the population model, ×10⁻³ per decision (right = better)", unit: "×10⁻³ log loss per decision", aria: "Forecast improvement over the population model", labelW: 260, valueW: 170 });
    const ladder = d.ladder || f.ladder || [];
    const maxGain = Math.max(0.001, ...ladder.map((r) => r.log_loss.population - r.log_loss.personal));
    const ladderTbl = el("div", { class: "ev-table-wrap" }, el("table", { class: "mini ladder" },
      el("thead", {}, el("tr", {}, ["earlier river decisions seen", "test decisions", "population", "+ history", "gain"].map((h) => el("th", { text: h })))),
      el("tbody", {}, ladder.map((r) => { const g = r.log_loss.population - r.log_loss.personal; return el("tr", {}, el("td", { class: "num", text: r.bin }), el("td", { class: "num", text: n0(r.n) }), el("td", { class: "num", text: r.log_loss.population.toFixed(4) }), el("td", { class: "num", text: r.log_loss.personal.toFixed(4) }),
        el("td", { class: "gain" }, el("span", { class: "gain-bar", style: `width:${Math.max(0, 100 * g / maxGain)}%` }), el("span", { class: "num", text: milli(g) }))); }))));
    const pop = popP && popP.present ? popP.data : null;
    const ct = pop && pop.by_context.checked_through;
    const fig = el("figure", { class: "ev-fig" }, plot,
      el("h3", { class: "ev-sub", text: "The gain grows with what the model has seen of the player" }), ladderTbl,
      el("p", { class: "hint", text: "Three-action log loss per decision on the frozen 20% test split (lower is better); gain = population − history, ×10⁻³. Brier scores are in the artifact." }));
    const p = lift.personal;
    return topic("forecast", fig, {
      answer: p ? [`Yes, slightly: `, el("b", { class: "num", text: milli(p.lift_vs_population) }), ` ×10⁻³ log loss per decision (95% interval ${milli(p.ci95[0])} to ${milli(p.ci95[1])}), most of it for players seen 16 or more times.`] : null,
      denom: [`${n0(f.overall ? f.overall.n : d.test)} held-out river decisions from ${n0(f.overall ? f.overall.players : 0)} players (train ${n0(d.train)} / test ${n0(d.test)}).`,
        ct ? ` The population starting point after a checked turn is ${n0(ct.fold)} folds of ${n0(ct.fold + ct.call + ct.raise)} (${pct(ct.fold / (ct.fold + ct.call + ct.raise), 0)}).` : ""],
      source: `${src(fl)} · ${d.label || ""}`,
      table: table(["earlier obs.", "n", "pooled", "population", "type + history", "history"], ladder.map((r) => [r.bin, n0(r.n), ...["pooled", "population", "type", "personal"].map((k) => r.log_loss[k] === undefined || r.log_loss[k] === null ? "-" : r.log_loss[k].toFixed(4))])),
    });
  }

  // ------------------------------------------------------------ 3 benchmark
  function benchmarkTopic(b) {
    if (!b.present) return topic("benchmark", null, { present: false, missing: "python -m pokersim mirror-rl benchmark" });
    const d = b.data;
    const all = d.rows || [];
    const lo = Math.min(0, ...all.map((r) => r.ci95_chips[0])), hi = Math.max(0, ...all.map((r) => r.ci95_chips[1]));
    const holder = el("div", { class: "bench-plot" });
    const answer = el("p", { class: "ev-answer" });
    const probes = Array.from(new Set(all.map((r) => r.probes))).sort((a, b2) => a - b2);
    const seg = el("div", { class: "seg", role: "group", "aria-label": "Probes the bettor saw of its opponent" },
      el("span", { class: "seg-label", text: "Probes of the opponent" }),
      probes.map((n) => el("button", { class: "seg-btn", "data-k": n, "aria-pressed": "false", onclick: () => set(n) }, String(n))));
    function set(n) {
      E.probes = n;
      $$(".seg-btn", seg).forEach((x) => { const on = Number(x.dataset.k) === n; x.classList.toggle("on", on); x.setAttribute("aria-pressed", String(on)); });
      const rows = all.filter((r) => r.probes === n).map((r) => ({ label: (OPP[r.opponent] || [r.opponent])[0], sub: (OPP[r.opponent] || [, ""])[1], mean: r.mean_diff_chips, lo: r.ci95_chips[0], hi: r.ci95_chips[1],
        failed: r.oracle_pass === false, hollow: r.oracle_pass === false, note: "oracle failed: training miss", extra: `${pct(r.fraction_changed, 0)} of decisions changed` }));
      holder.innerHTML = "";
      holder.appendChild(V.intervalPlot(rows, { width: plotW(), domain: [lo, hi], xTitle: "Δ chips per spot, bettor trained on the opponent − population bettor (opening pot = 100)", unit: "chips per spot", aria: `Benchmark at ${n} probes`, labelW: 200, valueW: 190 }));
      const ok = rows.filter((r) => !r.failed);
      const best = ok.slice().sort((a, c) => c.mean - a.mean)[0], worst = ok.slice().sort((a, c) => a.mean - c.mean)[0];
      answer.innerHTML = "";
      if (n === 0) answer.append("With no probes the two bettors are the same policy, so every difference is exactly zero — the matched setup works.");
      else answer.append(`At ${n} probes, against the ${best.label.toLowerCase()} the bettor gains `, el("b", { class: "num", text: V.sgn(best.mean) }), " chips per spot", ` (${V.sgn(best.lo)} to ${V.sgn(best.hi)}); against the ${worst.label.toLowerCase()} it ${worst.mean < 0 ? "loses" : "gains"} `, el("b", { class: "num", text: `${V.sgn(worst.mean)}` }), ` (${V.sgn(worst.lo)} to ${V.sgn(worst.hi)}). Knowing the opponent helps when it is unusual, and can hurt when it is the average.`);
    }
    const c = d.control;
    const fig = el("figure", { class: "ev-fig" }, seg, holder,
      el("p", { class: "hint" }, c ? `Control: the same population bettor retrained with another seed changes ${pct(c.fraction_changed, 0)} of decisions, ${V.sgn(c.mean_diff_chips)} chips per spot (${V.sgn(c.ci95_chips[0])} to ${V.sgn(c.ci95_chips[1])}). ` : "",
        d.zero_probe_identical !== undefined && d.zero_probe_identical !== null ? (d.zero_probe_identical ? "At 0 probes the two bettors are identical, as they must be." : "At 0 probes the bettors are NOT identical: check the setup.") : ""));
    const sec = topic("benchmark", fig, {
      answer: answer,
      denom: [`${n0(d.pairs)} matched deals per row; ${n0(d.episodes)} training episodes per bettor; seed ${d.seed}. Opponents are `, el("b", { text: "simulated" }), ", not people."],
      source: `${src(b)} · ${d.label}`,
      table: table(["opponent", "probes", "Δ chips / spot", "95% interval", "decisions changed", "oracle"], all.map((r) => ({ attrs: r.oracle_pass === false ? { class: "muted" } : {}, cells: [(OPP[r.opponent] || [r.opponent])[0], r.probes, V.sgn(r.mean_diff_chips), `${V.sgn(r.ci95_chips[0])} to ${V.sgn(r.ci95_chips[1])}`, pct(r.fraction_changed, 1), r.oracle_pass ? "pass" : "FAIL"] }))),
    });
    // answer paragraph is managed by set(); move it to the answer slot
    set(E.probes);
    return sec;
  }

  // ------------------------------------------------------------ 4 types
  function typesTopic(tg) {
    if (!tg.present) return topic("types", null, { present: false });
    const g = tg.data.gate || {}, a = g.type_vs_population_zero_history, b = g.type_vs_personal;
    const rows = [];
    if (a) rows.push({ label: "Type prior vs population", sub: `players with no river history · ${n0(a.players)} players`, mean: a.lift, lo: a.ci95[0], hi: a.ci95[1] });
    if (b) rows.push({ label: "Types + history vs history", sub: `all test decisions · ${n0(b.players)} players`, mean: b.lift, lo: b.ci95[0], hi: b.ci95[1] });
    const det = el("details", { class: "ev-explore" }, el("summary", {}, "Explore the style map ", el("span", { class: "tag", text: "exploratory · 3,000 players" })));
    let built = false;
    det.addEventListener("toggle", async () => {
      if (!det.open || built) return;
      built = true;
      const t0 = performance.now();
      if (!E.styles) { try { E.styles = await api("/api/mirror/styles"); } catch (e) { E.styles = { present: false }; } }
      if (!E.styles.present) { det.appendChild(el("p", { class: "hint", text: "Style map not generated." })); return; }
      det.appendChild(styleMap(E.styles));
      requestAnimationFrame(() => { E.styleMs = Math.round(performance.now() - t0); det.dataset.renderMs = E.styleMs; });
    });
    const fig = el("figure", { class: "ev-fig" },
      rows.length ? V.intervalPlot(rows, { width: plotW(), fmt: milli, tickFmt: milliTick, step: 0.005, xTitle: "Log-loss improvement, ×10⁻³ per decision (right = types help)", unit: "×10⁻³ log loss per decision", labelW: 230, valueW: 170 }) : null,
      el("p", { class: "gate " + (g.pass ? "pass" : "fail") }, el("span", { class: "tag " + (g.pass ? "good" : "warn"), text: g.pass ? "✓ gate passed" : "✗ gate failed" }), " ",
        g.pass ? "Both intervals sit above zero, so the live model uses types." : "Types ship only if both intervals sit above zero. Neither does, so the live model ignores them and the map below only describes how people play."),
      det);
    return topic("types", fig, {
      answer: [`Not for forecasting. Clustering finds ${tg.data.k} styles, but adding them `, el("b", { text: "does not beat" }), " the simpler models on held-out players."],
      denom: a ? `${n0(a.n)} decisions by ${n0(a.players)} players with no earlier river decisions; ${b ? n0(b.n) : "?"} decisions by ${b ? n0(b.players) : "?"} players overall. Prior strength ${g.prior_strength}.` : null,
      source: `${src(tg)} · rule: ${g.rule || ""}`,
    });
  }
  // Small multiples, one per type: that type highlighted, everyone else grey (no colour has to carry identity).
  function styleMap(st) {
    const pts = st.map_points || [];
    const wrap = el("div", { class: "stylemap" });
    if (!pts.length) return el("p", { class: "hint", text: "No map points." });
    const xs = pts.map((p) => p.x).sort((a, b) => a - b), ys = pts.map((p) => p.y).sort((a, b) => a - b);
    const qt = (arr, f) => arr[Math.max(0, Math.min(arr.length - 1, Math.floor(f * (arr.length - 1))))];
    const x0 = qt(xs, 0.01), x1 = qt(xs, 0.99), y0 = qt(ys, 0.01), y1 = qt(ys, 0.99);
    const W = 240, H = 170, m = 10;
    const clamp = (v, a, b) => Math.min(Math.max(v, a), b);
    const sx = (x) => m + (W - 2 * m) * (clamp(x, x0, x1) - x0) / ((x1 - x0) || 1);
    const sy = (y) => H - m - (H - 2 * m) * (clamp(y, y0, y1) - y0) / ((y1 - y0) || 1);
    // the grey background cloud is identical in every panel: draw it once as a path, reuse via <use>
    const bgId = "smbg" + Math.random().toString(36).slice(2, 7);
    const defs = V.svg("svg", { width: 0, height: 0, style: "position:absolute", "aria-hidden": "true" });
    const d = pts.map((p) => `M${sx(p.x).toFixed(1)} ${sy(p.y).toFixed(1)}h0`).join("");
    const sym = V.svg("symbol", { id: bgId, viewBox: `0 0 ${W} ${H}` });
    sym.appendChild(V.svg("path", { d, stroke: "var(--muted)", "stroke-opacity": 0.28, "stroke-width": 2.6, "stroke-linecap": "round" }));
    defs.appendChild(sym);
    wrap.appendChild(defs);
    st.names.forEach((name, t) => {
      const svg = V.svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `${name}: ${pct(st.shares[t], 0)} of players` });
      svg.appendChild(V.svg("use", { href: "#" + bgId, x: 0, y: 0, width: W, height: H }));
      const mine = pts.filter((p) => p.type === t);
      svg.appendChild(V.svg("path", { d: mine.map((p) => `M${sx(p.x).toFixed(1)} ${sy(p.y).toFixed(1)}h0`).join(""), stroke: "var(--ink)", "stroke-opacity": 0.8, "stroke-width": 3.2, "stroke-linecap": "round" }));
      const cxy = st.centroid_xy[t];
      svg.appendChild(V.svg("circle", { cx: sx(cxy[0]).toFixed(1), cy: sy(cxy[1]).toFixed(1), r: 6, fill: "var(--amber)", stroke: "var(--surface)", "stroke-width": 2 }));
      const c = st.centroids[t];
      const tip = el("div", { class: "chart-tip", hidden: true });
      const cell = el("div", { class: "stylemap-cell" },
        el("div", { class: "row between" }, el("b", { text: name }), el("span", { class: "muted", text: `${pct(st.shares[t], 0)} of players` })), svg,
        el("div", { class: "muted", text: `VPIP ${pct(c.vpip, 0)} · PFR ${pct(c.pfr, 0)} · aggression ${pct(c.af, 0)} · folds to bets ${pct(c.fold_to_bet, 0)}` }), tip);
      svg.addEventListener("mousemove", (e) => {
        const rect = svg.getBoundingClientRect();
        const mx = (e.clientX - rect.left) * W / rect.width, my = (e.clientY - rect.top) * H / rect.height;
        let best = null, bd = 64;
        mine.forEach((p) => { const dd = (sx(p.x) - mx) ** 2 + (sy(p.y) - my) ** 2; if (dd < bd) { bd = dd; best = p; } });
        if (!best) { tip.hidden = true; return; }
        tip.hidden = false;
        tip.textContent = `${name} · ${best.venue} · ${best.hands} hands`;
        tip.style.left = Math.min(e.clientX - cell.getBoundingClientRect().left + 10, cell.clientWidth - 150) + "px";
        tip.style.top = (e.clientY - cell.getBoundingClientRect().top + 10) + "px";
      });
      svg.addEventListener("mouseleave", () => { tip.hidden = true; });
      wrap.appendChild(cell);
    });
    return el("div", {}, wrap, el("p", { class: "hint", text: `Each dot is a player (${n0(pts.length)} sampled, at least ${st.n_min} hands each). Position summarises five style statistics; players close together play alike. Bright: the type named above the panel; grey: everyone else; amber: the type's centre.` }));
  }

  // ------------------------------------------------------------ 5 range readout
  function rangeTopic(rc) {
    if (!rc.present) return topic("range", null, { present: false });
    const d = rc.data, pl = d.pluribus, h = d.held_out_sessions || {}, t = d.twin_unit_check;
    let fig = null;
    if (pl) {
      const r = pl.by_response || {};
      const rows = ["fold", "call", "raise"].filter((k) => r[k]).map((k) => ({ label: `Defender ${k === "fold" ? "folded" : k === "call" ? "called" : "raised"}`, sub: `${n0(r[k].n)} spots`, mean: r[k].mean_equity, lo: r[k].ci95[0], hi: r[k].ci95[1] }));
      fig = el("figure", { class: "ev-fig" }, V.intervalPlot(rows, { width: plotW(), fmt: (v) => pct(v, 0), tickFmt: (v) => pct(v, 0), step: 0.25, domain: [0, 1], xTitle: "Defender's mean river equity, 95% interval (Pluribus, every card known)", unit: "", labelW: 200, valueW: 150 }),
        el("p", { class: "hint", text: pl.label || "" }));
    }
    return topic("range", fig, {
      answer: pl ? [pl.monotone ? "Directionally, yes. " : "Not yet. ", `In ${n0(pl.spots)} Pluribus river spots where every card is known, players who fold hold weaker hands than callers, and callers weaker than raisers — the ordering the readout assumes.`] : null,
      denom: [pl ? `${n0(pl.spots)} heads-up river spots (${n0(pl.half_pot_band)} in the half-pot band). ` : "", h.decisions ? `Held-out humans: ${h.sessions} session(s), ${h.decisions} hands.` : el("span", {}, el("b", { text: "No held-out human sessions yet" }), ", so nothing is claimed about humans.")],
      source: src(rc),
      after: t ? el("p", { class: "hint", text: `Unit test only: on the twin's own responses the readout scores ${t.mean_log_score_readout} against ${t.mean_log_score_blind} without it (top bucket right ${pct(t.top_bucket_accuracy, 0)} of the time). That checks the code, not the model.` }) : null,
    });
  }

  // ------------------------------------------------------------ 6 training
  function trainingTopic(ts, pb) {
    const parts = [];
    if (ts.present) {
      const d = ts.data;
      parts.push(el("div", { class: "stat-row" },
        stat(pct(d.stage1_agreement, 0), "of first-decision cells agree across three seeds"),
        stat(d.disagreements_are_near_ties ? "near-ties" : "not ties", "where the seeds disagree"),
        stat(d.all_oracle_pass ? "all pass" : "some fail", "Q* oracle check, every seed")));
    }
    if (pb.present) {
      const m = pb.data.meta;
      parts.push(el("p", { class: "hint", text: `Population bettor: ${n0(m.episodes)} episodes, seed ${m.seed}, ${m.train_seconds}s, oracle ${m.oracle_check ? (m.oracle_check.pass ? "pass" : "fail") : "n/a"}.` }));
    }
    const fig = parts.length ? el("figure", { class: "ev-fig" }, ...parts) : null;
    const cells = ts.present ? ts.data.cells || [] : [];
    return topic("training", fig, {
      present: ts.present || pb.present,
      answer: ts.present ? [`Enough for the comparison: three seeds agree on ${pct(ts.data.stage1_agreement, 0)} of first decisions, and every disagreement is a near-tie that costs almost nothing.`] : null,
      denom: ts.present ? `${ts.data.cells ? ts.data.cells.length : 0} state cells × ${ts.data.seeds ? ts.data.seeds.length : 3} seeds × ${n0(ts.data.episodes)} episodes.` : null,
      source: [ts.present ? src(ts) : "", pb.present ? pb.source : ""].filter(Boolean).join(" · "),
      table: cells.length ? table(["state", "greedy action by seed", "agree", "Q* check / bet", "gap", "min visits"], cells.map((c) => [c.state, c.greedy_by_seed.join(", "), c.agree ? "yes" : "no", c.q_star ? c.q_star.join(" / ") : "-", c.q_star_gap, c.min_visits])) : null,
      after: pb.present && pb.data.checkpoints ? el("details", { class: "ev-table" }, el("summary", { text: "Population bettor checkpoints" }),
        table(["episode", "validation mean reward (pots)"], pb.data.checkpoints.map((c) => [n0(c.episode), c.validation_mean_reward]))) : null,
    });
  }
  function stat(v, l) { return el("div", { class: "stat" }, el("div", { class: "stat-v", text: v }), el("div", { class: "stat-l", text: l })); }

  // ------------------------------------------------------------ 7 sessions
  const TAGS = [["practice", "practice"], ["calibration", "calibration (development)"], ["heldout", "held-out evaluation"]];
  function sessionsTopic(ss) {
    const d = ss.data || {}, lc = d.live_counts || {}, ev = d.evaluation, cal = d.calibration;
    const fig = el("figure", { class: "ev-fig" },
      el("div", { class: "stat-row" }, TAGS.map(([t, label]) => stat(n0((lc[t] || {}).sessions || 0), `${label} session${((lc[t] || {}).sessions || 0) === 1 ? "" : "s"} · ${n0((lc[t] || {}).decisions || 0)} decisions`))));
    let answer = "Nothing yet. Held-out scores need at least one held-out session; an interval needs three.";
    if (ev && ev.heldout_forecast && ev.heldout_forecast.decisions) {
      const f = ev.heldout_forecast;
      answer = `Held-out: ${f.sessions} session(s), ${f.decisions} decisions. Log loss per decision: population ${f.log_loss.population}, personal ${f.log_loss.personal}.` + (f.lift_ci95_over_sessions ? ` Lift 95% interval over sessions: ${f.lift_ci95_over_sessions.join(" to ")}.` : " An interval needs at least three held-out sessions.");
    }
    return topic("sessions", fig, {
      answer: [answer],
      denom: cal ? `Strength calibration: ${cal.sessions} session(s), ${cal.decisions} decisions.` : "Strength table: the labelled synthetic default until calibration sessions exist.",
      source: ss.source,
      after: el("div", { class: "row" }, el("button", { class: "btn small", onclick: recompute }, "Recompute from session logs"), el("span", { class: "hint", text: "Operator: tag judges in Mirror → Operator (calibration first, then held-out)." })),
    });
  }
  async function recompute() {
    try { await api("/api/mirror/sessions/recompute", {}); toast("Recomputed calibration and held-out scores from the session logs."); E.data = null; render(); } catch (e) { toast(e.message, true); }
  }

  // ------------------------------------------------------------ entry
  async function render() {
    if (!E.built) shell();
    const host = $("#evTopics");
    if (!E.data) {
      host.innerHTML = "";
      host.appendChild(el("p", { class: "hint", text: "Loading artifacts…" }));
      try { E.data = await api("/api/mirror/evidence"); } catch (e) { host.innerHTML = ""; host.appendChild(el("p", { class: "scene-error", role: "alert", text: e.message })); return; }
      const p = E.data.panels;
      host.innerHTML = "";
      if (E.io) E.io.disconnect();
      const secs = [];
      const add = (fn, ...args) => { try { const x = fn(...args); if (x) secs.push(x); } catch (e) { console.error(e); } };
      // Phase 6 first (the newer, wider cohort), then the original two-player-table evidence
      const p6 = { wide: p.wide_scan, situation: p.forecast_wide, bluff: p.bluff_test, realbench: p.benchmark_real, strength: p.strength_pluribus, neural: p.neural_forecaster, typeswide: p.type_gate_wide };
      const R6 = { wide: wideTopic, situation: situationTopic, bluff: bluffTopic, realbench: realBenchTopic, strength: strengthTopic, neural: neuralTopic, typeswide: typesWideTopic };
      const missing = [];
      Object.keys(R6).forEach((k) => { if (p6[k] && p6[k].present) add(R6[k], p6[k]); else missing.push(k); });
      add(filterTopic, p.filter_exhibit); add(forecastTopic, p.forecast_ladder, p.population); add(benchmarkTopic, p.decision_benchmark); add(typesTopic, p.type_gate);
      add(rangeTopic, p.range_checks); add(trainingTopic, p.training_sufficiency, p.population_bot); add(sessionsTopic, p.sessions);
      let group = null;
      secs.forEach((sec, i) => {
        const t = TOPICS.find((x) => x.id === sec.dataset.topic);
        if (t.group !== group) {
          if (group === G6 && missing.length) host.appendChild(pendingNote(missing));
          group = t.group;
          host.appendChild(el("p", { class: "ev-group mono", text: group }));
        }
        $(".ev-kicker", sec).textContent = `${i + 1} · ${t.kicker}`;
        host.appendChild(sec);
      });
      buildNav(secs.map((x) => x.dataset.topic));
      E.order = secs.map((x) => x.dataset.topic);
    }
  }
  async function onShow() {
    await render();
    const params = new URLSearchParams(location.hash.replace(/^#/, ""));
    const t = params.get("evidence");
    if (t && TOPICS.some((x) => x.id === t)) requestAnimationFrame(() => goTopic(t, false));
    else { markNav((E.order || ["filter"])[0]); const h = $("#view-evidence h1"); if (h) h.focus({ preventScroll: true }); }
  }
  window.Evidence = { onShow, state: E, styleMap };
})();
