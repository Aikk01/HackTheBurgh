/* Poker Mirror panes: replay reveal, filter exhibit, defender drills, bettor play vs. twin, bot comparison, evidence. */
(function () {
  "use strict";
  const P = window.PokerSim;
  const { api, el, cardEl, toast, fmt, pct } = P;
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));

  const M = {
    token: null, state: null, step: "reveal", demo: null, exhibit: null, revealed: false, guess: null,
    filter: "all", evidence: null, lastDrill: null, pollTimer: null,
  };
  const CTX_LABEL = { checked_through: "turn checked through", bettor_aggr: "bettor bet/raised the turn", defender_aggr: "defender bet/raised the turn" };
  const RESP = ["fold", "call", "raise"];

  function token() {
    if (M.token) return M.token;
    try { M.token = localStorage.getItem("mirrorToken") || ""; } catch (e) { M.token = ""; }
    return M.token;
  }
  function remember(tok) { M.token = tok; try { localStorage.setItem("mirrorToken", tok); } catch (e) { /* ignore */ } }
  const q = () => "?session=" + encodeURIComponent(token());
  const body = (extra) => Object.assign({ session: token() }, extra || {});

  async function refresh() {
    const st = await api("/api/mirror/state" + q());
    remember(st.token);
    M.state = st;
    $("#mirrorSession").textContent = "judge " + st.token.slice(0, 6) + " · " + st.profile.observations + " observations";
    return st;
  }

  // ------------------------------------------------------------ table view for a river spot
  function spotHand(o) {
    const mk = (i, name, cards, bet, folded, shown, hand) => ({
      index: i, name, seat: i + 1, position: i === 0 ? "BB" : "BTN", stack: o.stacks ? o.stacks[i] : 150, starting_stack: 200,
      bet: bet || 0, contributed: 50 + (bet || 0), folded: !!folded, all_in: false, acted: true, mucked: false, shown: !!shown,
      has_cards: true, cards, cards_known: cards.every((c) => c !== "??" && c !== "XX"), hand: hand || null,
      payout: o.payouts ? o.payouts[i] : 0, net: null, is_button: i === 1, agent: null,
    });
    const players = [
      mk(0, o.defenderName, o.defenderCards, o.defenderBet, o.defenderFolded, o.defenderShown),
      mk(1, o.bettorName, o.bettorCards, o.bettorBet, o.bettorFolded, o.bettorShown),
    ];
    if (o.bettorHuman) players[1].agent = { kind: "human", label: "you" };
    if (o.defenderHuman) players[0].agent = { kind: "human", label: "you" };
    const bets = (o.defenderBet || 0) + (o.bettorBet || 0);
    return {
      hand_id: o.id || "mirror", variant: "NT", status: o.status || "betting", street: "river", street_index: 3, board: o.board,
      pot: o.pot === undefined ? 100 : o.pot, total_pot: (o.pot === undefined ? 100 : o.pot) + bets, max_bet: Math.max(o.defenderBet || 0, o.bettorBet || 0),
      actor: o.actor === undefined ? null : o.actor, button: 1, blinds: [1, 2], antes: [0, 0], min_bet: 2, players, legal: null,
      pots: null, payouts: o.payouts || null, unresolved: false, warnings: [],
    };
  }
  function showSpot(o, viewer) {
    const hand = spotHand(o);
    const slots = [{ slot: viewer, seat: viewer + 1, name: hand.players[viewer].name, player: viewer },
                   { slot: 1 - viewer, seat: 2 - viewer, name: hand.players[1 - viewer].name, player: 1 - viewer }];
    P.state.prevHand = null;
    P.renderTable({ hand, slots, viewer, reveal: !!o.reveal, bubble: o.bubble || null, lens: o.lens || null, payouts: false });
    $("#centerMsg").innerHTML = "";
    if (o.message) $("#centerMsg").appendChild(el("span", { html: o.message }));
  }

  // ------------------------------------------------------------ step 1: replay reveal
  async function renderReveal() {
    const root = $("#mirror-reveal");
    root.innerHTML = "";
    if (!M.demo) {
      try { M.demo = await api("/api/mirror/demo_hands"); } catch (e) { root.appendChild(el("div", { class: "card-box hint", text: "Demo hands not available: " + e.message })); return; }
    }
    if (!M.exhibit) { try { M.exhibit = await api("/api/mirror/exhibit"); } catch (e) { M.exhibit = { present: false }; } }
    const hands = M.demo.hands || [];
    if (!hands.length) { root.appendChild(el("div", { class: "card-box hint", text: "No demo hands bundled. Run: python -m pokersim mirror-prepare" })); return; }
    const box = el("div", { class: "card-box" }, el("h4", { text: "Two river decisions by the same Absolute Poker 100NL player" }),
      el("p", { class: "hint", text: "Same defender, same spot: checks the river, faces a half-pot bet. One time they folded, one time they called. Which one folded?" }));
    hands.forEach((h, i) => {
      const row = el("div", { class: "demo-hand" },
        el("div", { class: "row between" }, el("b", { text: `Hand ${i + 1} · #${h.hand}` }), el("span", { class: "muted", text: `${fmt(h.bet)} into ${fmt(h.pot_before_bet)}` })),
        el("div", { class: "row" }, P.cardsInline(h.board), el("button", { class: "btn small", onclick: () => showDemoHand(i) }, "Show on table"),
          el("button", { class: "btn small primary", onclick: () => { M.guess = i; M.revealed = true; renderReveal(); showDemoHand(i); }, disabled: M.revealed }, "This one folded")),
        M.revealed ? el("div", { class: "reveal" },
          el("div", {}, el("span", { class: "tag", text: "turn: " + CTX_LABEL[h.context] }), " ", el("span", { class: "tag " + (h.response === "fold" ? "good" : "warn"), text: "recorded action: " + h.response })),
          M.guess === i ? el("div", { class: "muted", text: h.response === "fold" ? "Your guess was right." : "Not this one." }) : null) : null);
      box.appendChild(row);
    });
    root.appendChild(box);
    if (M.revealed && M.exhibit && M.exhibit.present) {
      const two = M.exhibit.exhibit.two_shard.cohorts;
      const ct = two.half_checked_through, ba = two.half_bettor_aggr, da = two.half_defender_aggr;
      const ebox = el("div", { class: "card-box" }, el("h4", { text: "The turn history changes the answer" }),
        el("p", { class: "hint", text: "Exact half-pot offers in the two demo shards (Takara 35 + 36), folds over opportunities:" }),
        bars([["turn checked through", ct.folds / ct.opportunities, fo(ct)],
              ["bettor bet/raised the turn", ba.folds / ba.opportunities, fo(ba)],
              ["defender bet/raised the turn", da.folds / da.opportunities, fo(da)]]),
        el("p", { class: "hint", text: `Across all 37 shards: ${fmt(M.exhibit.scan.eligible_unique)} unique eligible offers from ${fmt(M.exhibit.scan.rows)} hands; ${fmt(M.exhibit.scan.eligible_modelling)} exact half-pot offers with all three responses legal form the modelling cohort.` }));
      root.appendChild(ebox);
    }
  }
  function showDemoHand(i) {
    const h = M.demo.hands[i];
    const step = h.replay.steps[Math.min(h.decision_step, h.replay.steps.length - 1)];
    const hand = step.state;
    const viewer = 0;
    const order = hand.players.slice().sort((a, b) => ((a.index - viewer + 100) % 100) - ((b.index - viewer + 100) % 100));
    const slots = order.map((p) => ({ slot: p.index, seat: p.index + 1, name: p.name, stack: p.stack, player: p.index }));
    P.state.prevHand = null;
    P.renderTable({ hand, slots, viewer: null, reveal: false, bubble: { player: 1, text: `Bet ${fmt(h.bet)}`, kind: "aggr" }, payouts: false });
    $("#centerMsg").innerHTML = "";
    $("#centerMsg").appendChild(el("span", { html: `${h.defender.slice(0, 8)}… checks, faces ${fmt(h.bet)} into ${fmt(h.pot_before_bet)}<span class="sub">${M.revealed ? "recorded action: " + h.response : "what did they do?"}</span>` }));
  }

  function bars(rows) {
    const root = el("div", { class: "bars" });
    rows.forEach(([label, frac, val]) => root.appendChild(el("div", { class: "bar-row" }, el("span", { text: label }), el("div", { class: "track" }, el("div", { class: "fill", style: `width:${(100 * frac).toFixed(1)}%` })), el("span", { class: "val", text: val === undefined ? pct(frac) : val }))));
    return root;
  }
  // "66.5% · 565/849" for a cohort of folds over opportunities
  const fo = (c) => `${pct(c.folds / c.opportunities)} · ${c.folds.toLocaleString()}/${c.opportunities.toLocaleString()}`;

  // ------------------------------------------------------------ step 2: filters
  async function renderFilters() {
    const root = $("#mirror-filters");
    root.innerHTML = "";
    if (!M.exhibit) { try { M.exhibit = await api("/api/mirror/exhibit"); } catch (e) { M.exhibit = { present: false }; } }
    if (!M.exhibit.present) { root.appendChild(el("div", { class: "card-box hint", text: "Exhibit not generated yet (python -m pokersim mirror-prepare)." })); return; }
    const two = M.exhibit.exhibit.two_shard.cohorts, full = M.exhibit.exhibit.full_corpus.cohorts;
    const views = { all: ["all observed actions", "all_eligible"], known: ["only hands where the defender's cards became known", "defender_cards_known"], clean: ["only warning-free hands", "warning_free"] };
    const box = el("div", { class: "card-box" }, el("h4", { text: "What a filter hides" }),
      el("div", { class: "subtabs" }, ...Object.entries(views).map(([k, [label]]) => el("button", { class: M.filter === k ? "active" : "", onclick: () => { M.filter = k; renderFilters(); } }, label))));
    const key = views[M.filter][1];
    const t = two[key], f = full[key];
    box.appendChild(el("div", { class: "tiles" },
      P.tile("folds / offers (shards 35+36)", `${t.folds}/${t.opportunities}`, t.opportunities ? pct(t.folds / t.opportunities) : "no data"),
      P.tile("folds / offers (all 37 shards)", `${f.folds}/${f.opportunities}`, f.opportunities ? pct(f.folds / f.opportunities) : "no data")));
    const msg = M.filter === "known" ? "A fold never reaches a showdown, so a dataset filtered to hands with known cards contains zero folds. Only 18.9% of eligible offers in the two shards ever reveal the defender's cards; models that train on showdowns learn from players who did not fold."
      : M.filter === "clean" ? "Dropping every hand with a replay warning also drops valid earlier decisions: here the warnings concern cards unknown at a later showdown, which says nothing about the river decision."
      : "All recorded responses to a 40-60% pot bet on the river after a check, heads-up no-limit hold'em.";
    box.appendChild(el("p", { class: "hint", text: msg }));
    box.appendChild(el("p", { class: "muted", text: `Source: takara-ai/poker_hands @ ${(M.exhibit.scan.revision || "").slice(0, 10)}, generated ${M.exhibit.exhibit.generated}` }));
    root.appendChild(box);
    // style map (type layer): who these players are; descriptive only while the type gate fails
    if (M.styles === undefined) { try { M.styles = await api("/api/mirror/styles"); } catch (e) { M.styles = null; } }
    if (M.styles && M.styles.present && M.step === "filters" && root.isConnected) {
      const g = M.styles.gate || {};
      root.appendChild(el("div", { class: "card-box" }, el("h4", { text: "Style map: who these players are" }),
        el("p", { class: "hint" }, el("span", { class: "tag warn", text: "description only" }),
          ` Model estimate, not an observation. The type prior ${g.pass ? "passed" : "failed"} its gate, so live forecasts ignore player types.`),
        styleMap(M.styles)));
    }
  }

  // ------------------------------------------------------------ step 3: you defend
  function forecastBox(pred, title) {
    const p = pred.probs;
    return el("div", { class: "forecast" }, el("h4", { text: title }),
      bars([["fold", p.fold, `${pct(p.fold)} (90% band ${pct(pred.fold_p05)}–${pct(pred.fold_p95)})`], ["call", p.call, pct(p.call)], ["raise", p.raise, pct(p.raise)]]),
      el("div", { class: "muted", text: `${pred.arm} arm · ${pred.observations} of your responses seen · ${pred.model_version}` }),
      pred.type_posterior ? el("div", { class: "muted", text: "type: " + Object.entries(pred.type_posterior).map(([k, v]) => `${k} ${pct(v, 0)}`).join(", ") }) : null);
  }

  async function renderDrill() {
    const root = $("#mirror-drill");
    root.innerHTML = "";
    const st = M.state || await refresh();
    const d = st.drill.pending;
    const box = el("div", { class: "card-box" }, el("h4", { text: "You defend: you checked the river and face a half-pot bet" }));
    box.appendChild(tagRow(st));
    if (!d) {
      box.appendChild(el("p", { class: "hint", text: "Each spot commits a forecast of your response before you click. Deals alternate the turn history and your hand strength." }));
      box.appendChild(el("div", { class: "row" }, el("button", { class: "btn primary", onclick: newDrill }, "Deal a spot"), el("span", { class: "muted", text: `${st.drill.drills} spots played` })));
      if (M.lastDrill) box.appendChild(lastDrillBox(M.lastDrill));
    } else {
      box.appendChild(el("p", {}, el("b", { text: "Turn history: " }), d.extra.context_label, el("br"), el("span", { class: "muted", text: d.extra.story.join(" · ") })));
      box.appendChild(forecastBox(d.prediction, "Committed forecast of your response"));
      box.appendChild(el("div", { class: "row" },
        el("button", { class: "btn danger", onclick: () => respond(d.decision_id, "fold") }, "Fold"),
        el("button", { class: "btn", onclick: () => respond(d.decision_id, "call") }, `Call ${d.bet}`),
        el("button", { class: "btn primary", onclick: () => respond(d.decision_id, "raise") }, `Raise to ${st.drill.raise_to}`)));
      showSpot({ board: d.board, defenderCards: d.own_cards, bettorCards: ["XX", "XX"], defenderName: "You", bettorName: "Bettor", bettorBet: d.bet, pot: d.pot_before_bet, actor: 0, defenderHuman: true, bubble: { player: 1, text: `Bet ${d.bet}`, kind: "aggr" }, message: `Pot ${d.pot_before_bet}, bet ${d.bet} · ${d.extra.context_label}` }, 0);
    }
    root.appendChild(box);
    root.appendChild(profileBox(st.profile));
  }
  const TAGS = [["practice", "practice"], ["calibration", "calibration (development)"], ["heldout", "held-out evaluation"]];
  function tagRow(st) {
    return el("div", { class: "tag-row" },
      el("div", { class: "row" }, el("span", { class: "muted", text: "This judge's answers are for:" }),
        ...TAGS.map(([t, label]) => el("button", { class: "chip-btn" + (st.tag === t ? " active" : ""), onclick: () => setTag(t) }, label))),
      el("div", { class: "hint", text: st.tag === "heldout" ? "Held-out: someone who has not seen the model or its forecasts. Their answers are scored, never trained on."
        : st.tag === "calibration" ? "Calibration: development sessions that teach the twin how hand strength changes responses."
        : "Practice: nothing is used for evaluation. Raw answers never leave this laptop." }));
  }
  async function setTag(t) {
    try { await api("/api/mirror/tag", body({ tag: t })); await refresh(); renderDrill(); } catch (e) { toast(e.message, true); }
  }

  function lastDrillBox(r) {
    const dec = r.decision, pred = dec.prediction, out = r.outcome;
    return el("div", { class: "result" }, el("h4", { text: "Last spot" }),
      el("div", {}, `Forecast fold ${pct(pred.fold_mean)} (${pred.arm}, ${pred.observations} seen) → you ${dec.actual}. `, out ? `${out.result}. Bettor had ${out.bettor_cards.join(" ")}; your hand was ${out.your_bucket} (${pct(out.your_equity)} equity).` : ""),
      el("div", { class: "muted", text: `Updated forecast for this history: fold ${pct(r.after.fold_mean)} after ${r.after.observations} responses.` }));
  }
  async function newDrill() {
    try { const r = await api("/api/mirror/drill", body()); remember(r.token); await refresh(); renderDrill(); } catch (e) { toast(e.message, true); }
  }
  async function respond(id, action) {
    try {
      const r = await api("/api/mirror/respond", body({ decision_id: id, action }));
      M.lastDrill = r;
      await refresh();
      const d = r.decision, out = r.outcome;
      showSpot({ board: d.board, defenderCards: d.own_cards, bettorCards: out ? out.bettor_cards : ["XX", "XX"], defenderName: "You", bettorName: "Bettor",
        bettorBet: action === "fold" ? 50 : 0, defenderFolded: action === "fold", defenderHuman: true, reveal: true, status: "finished",
        bubble: { player: 0, text: action === "raise" ? "Raise to 100" : action === "call" ? "Call 50" : "Fold", kind: action === "raise" ? "aggr" : action === "fold" ? "fold" : "" },
        message: `${out ? out.result : ""}<span class="sub">forecast was fold ${pct(d.prediction.fold_mean)}, you ${action}</span>` }, 0);
      renderDrill();
    } catch (e) { toast(e.message, true); }
  }
  function profileBox(prof) {
    const box = el("div", { class: "card-box" }, el("h4", { text: `Your profile · ${prof.observations} responses · ${prof.arm} arm` }));
    Object.entries(prof.contexts).forEach(([c, v]) => {
      const pop = prof.population[c];
      box.appendChild(el("div", { class: "ctx-row" }, el("div", { class: "muted", text: `${v.label} (${v.observations} seen)` }),
        bars([["you: fold", v.probs.fold, `${pct(v.probs.fold)} [${pct(v.fold_p05, 0)}–${pct(v.fold_p95, 0)}]`], ["population: fold", pop.fold, pct(pop.fold)]])));
    });
    if (prof.type_posterior) box.appendChild(el("div", {}, el("b", { text: "Type: " }), prof.type_posterior.names.map((n, i) => `${n} ${pct(prof.type_posterior.weights[i], 0)}`).join(" · ")));
    box.appendChild(el("div", { class: "muted", text: `Strength model: ${prof.strength_sources.prior} (${prof.strength_sources.calibration_sessions} calibration sessions)` + (prof.frozen ? ` · frozen twin ${prof.frozen}` : "") }));
    box.appendChild(el("div", { class: "row" }, el("button", { class: "btn", onclick: freeze }, prof.frozen ? "Re-freeze twin" : "Freeze my twin")));
    return box;
  }
  async function freeze() {
    try { const r = await api("/api/mirror/freeze", body()); toast(`Twin frozen (${r.frozen.observations} observations, hash ${r.frozen.hash})`); await refresh(); renderCurrent(); } catch (e) { toast(e.message, true); }
  }

  // ------------------------------------------------------------ step 4: you bet vs your twin
  async function renderPlay() {
    const root = $("#mirror-play");
    root.innerHTML = "";
    const st = M.state || await refresh();
    const v = st.play;
    const real = st.opponent && st.opponent.kind === "real";
    const box = el("div", { class: "card-box" }, el("h4", { text: real ? "You bet against a real player's twin" : "You bet against your frozen twin" }));
    box.appendChild(await opponentPicker(st));
    if (!st.frozen) box.appendChild(el("p", { class: "hint", text: "Freeze your profile first (step 3), or pick a real player from the dataset above. With no responses, your twin is the population player." }));
    box.appendChild(el("div", { class: "row" }, el("button", { class: "btn primary", onclick: newSpot }, "New spot"), el("span", { class: "muted", text: v.frozen ? "twin " + v.frozen : "" })));
    if (v.spot) {
      const s = v.spot;
      box.appendChild(el("p", {}, el("b", { text: "Turn history: " }), s.context_label, el("br"), el("span", { class: "muted", text: s.story.join(" · ") })));
      if (v.advice && s.stage === 1) box.appendChild(adviceBox(v.advice));
      if (s.stage === 1) box.appendChild(el("div", { class: "row" }, el("button", { class: "btn", onclick: () => act("check") }, "Check back"), el("button", { class: "btn primary", onclick: () => act("bet") }, `Bet ${s.bet}`)));
      if (s.stage === 2) box.appendChild(el("div", { class: "row" }, el("span", { text: `${real ? "The player's twin" : "Your twin"} raised to ${s.raise_to}. ` }), el("button", { class: "btn danger", onclick: () => act("fold") }, "Fold"), el("button", { class: "btn primary", onclick: () => act("call") }, "Call 50")));
      if (v.readout) box.appendChild(readoutBox(v.readout, v.response));
      if (v.result) box.appendChild(el("div", { class: "result" }, el("b", { text: `Result: ${v.result.reward_chips >= 0 ? "+" : ""}${fmt(v.result.reward_chips)} chips` }), ` · twin had ${v.result.defender_cards.join(" ")} (${v.result.defender_bucket}).`));
      const bettorBet = s.stage === 1 ? 0 : 50, defBet = s.stage === 2 ? 100 : (v.response === "call" ? 50 : 0);
      const done = !!v.result;
      showSpot({ board: s.board, bettorCards: s.own_cards, defenderCards: done ? v.result.defender_cards : ["XX", "XX"], defenderName: "Your twin", bettorName: "You",
        bettorBet: done ? 0 : bettorBet, defenderBet: done ? 0 : defBet, actor: done ? null : 1, bettorHuman: true, reveal: done, status: done ? "finished" : "betting",
        defenderFolded: v.response === "fold", bubble: v.response ? { player: 0, text: v.response === "raise" ? "Raise to 100" : v.response === "call" ? "Call 50" : "Fold", kind: v.response === "raise" ? "aggr" : v.response === "fold" ? "fold" : "" } : null,
        message: done ? `${v.result.reward_chips >= 0 ? "+" : ""}${fmt(v.result.reward_chips)} chips` : `Pot ${s.pot} · ${s.context_label}` }, 1);
    }
    root.appendChild(box);
    if (v.log && v.log.length) {
      const lb = el("div", { class: "card-box" }, el("h4", { text: "Spots against the twin" }));
      v.log.slice().reverse().forEach((x) => lb.appendChild(el("div", { class: "muted", text: `${x.board.join(" ")} · you ${x.own.join(" ")} · ${x.actions.join(", ")}${x.response ? " → " + x.response : ""} · ${x.reward_chips >= 0 ? "+" : ""}${fmt(x.reward_chips)}` })));
      root.appendChild(lb);
    }
  }
  function adviceBox(a) {
    const f = a.forecast;
    return el("div", { class: "advice" }, el("h4", { text: "EV advisor (opening pot = 100 chips)" }),
      el("div", { class: "tiles" }, P.tile("check back", fmt(Math.round(a.ev_check * 100)), "chips, equity vs any two"),
        P.tile("bet 50", fmt(Math.round(a.ev_bet * 100)) + (a.ev_bet_band ? ` (${fmt(Math.round(a.ev_bet_band[0] * 100))}…${fmt(Math.round(a.ev_bet_band[1] * 100))})` : ""), "chips, band from the fold interval")),
      el("div", {}, el("b", { text: `Recommendation: ${a.recommend}. ` }), a.reason),
      el("div", { class: "muted", text: `Forecast fold ${pct(f.fold)} [${pct(f.interval[0], 0)}–${pct(f.interval[1], 0)}] from the ${f.arm} arm (${f.observations} responses) · equity vs calling range ${pct(a.equity.vs_call_range)}, vs raising range ${pct(a.equity.vs_raise_range)} · pure bluff breaks even at ${pct(a.break_even_fold, 0)}` }));
  }
  function readoutBox(r, response) {
    const box = el("div", { class: "readout" }, el("h4", { text: `Range X-ray after the twin's ${response}` }),
      bars([["weak", r.buckets.weak, `${pct(r.buckets.weak)} (prior ${pct(r.prior_buckets.weak)})`], ["medium", r.buckets.medium, `${pct(r.buckets.medium)} (prior ${pct(r.prior_buckets.medium)})`], ["strong", r.buckets.strong, `${pct(r.buckets.strong)} (prior ${pct(r.prior_buckets.strong)})`]]),
      el("div", { class: "muted", text: `Estimate at ${r.resolution}, ${r.holdings} holdings, from the twin's strength table. Not an observation.` }));
    const ranks = "AKQJT98765432".split("");
    const grid = el("div", { class: "grid13" });
    ranks.forEach((r1, i) => ranks.forEach((r2, j) => {
      const cls = i === j ? r1 + r2 : i < j ? r1 + r2 + "s" : r2 + r1 + "o";
      const w = r.grid[cls] || 0;
      const frac = r.grid_max ? w / r.grid_max : 0;
      const own = r.own_cells.includes(cls);
      grid.appendChild(el("div", { class: "cell" + (own ? " own" : ""), style: `background:${w ? P.seqColor(frac) : "transparent"};color:${frac > 0.5 ? "#fff" : "var(--ink-2)"}`, title: `${cls}: ${(100 * w).toFixed(2)}%`, text: cls }));
    }));
    box.appendChild(grid);
    return box;
  }
  async function realPlayers() {
    if (!M.realPlayers) { try { M.realPlayers = await api("/api/mirror/real_players"); } catch (e) { M.realPlayers = { present: false, players: [] }; } }
    return M.realPlayers;
  }
  async function opponentPicker(st) {
    const o = st.opponent || { kind: "judge" };
    const row = el("div", { class: "row" }, el("span", { class: "muted", text: o.kind === "real"
      ? `Opponent: real player ${o.id} · ${o.venue} · ${o.decisions} recorded decisions · folds ${pct(o.fold_rate, 0)}` : "Opponent: your frozen twin" }));
    const rp = await realPlayers();
    if (rp.present && rp.players.length) {
      const sel = el("select", { class: "small", title: "Players from the dataset with enough recorded river answers" });
      rp.players.forEach((pl) => sel.appendChild(el("option", { value: pl.id, text: `${pl.id} · ${pl.venue} · ${pl.decisions} hands · folds ${pct(pl.fold_rate, 0)}` })));
      if (o.kind === "real") sel.value = o.id;
      row.appendChild(sel);
      row.appendChild(el("button", { class: "btn small", onclick: () => loadReal(sel.value) }, "Play this real player"));
    }
    if (o.kind === "real") row.appendChild(el("button", { class: "btn small", onclick: async () => { await freeze(); await newSpot(); } }, "Back to my twin"));
    return row;
  }
  async function loadReal(id) {
    try { const r = await api("/api/mirror/real/load", body({ player_id: id })); remember(r.token); toast(`Twin of ${id} loaded`); await refresh(); await newSpot(); } catch (e) { toast(e.message, true); }
  }
  async function newSpot() { try { const r = await api("/api/mirror/play/new", body()); remember(r.token); await refresh(); renderPlay(); } catch (e) { toast(e.message, true); } }
  async function act(a) { try { await api("/api/mirror/play/act", body({ action: a })); await refresh(); renderPlay(); } catch (e) { toast(e.message, true); } }

  // ------------------------------------------------------------ step 5: bots
  async function renderCompare() {
    const root = $("#mirror-compare");
    root.innerHTML = "";
    let c;
    try { c = await api("/api/mirror/compare" + q()); } catch (e) { root.appendChild(el("div", { class: "card-box hint", text: e.message })); return; }
    const box = el("div", { class: "card-box" }, el("h4", { text: "Two counter-bots, same deal" }),
      el("p", { class: "hint", text: "Tabular Q-learning, 20,000 simulated river episodes each, same seeds and deal distribution. One learned against the population twin, one against your frozen twin. Values are simulated estimates in chips." }));
    const t = c.training || {};
    box.appendChild(el("div", { class: "row" }, el("button", { class: "btn primary", onclick: startTraining, disabled: t.status === "running" }, t.status === "done" ? "Retrain personal bot" : "Train bot vs my twin"),
      el("span", { class: "muted", text: t.status === "running" ? `training… ${t.progress}/${t.episodes}` : t.status === "done" ? `trained on twin ${t.profile_hash} · validation ${t.validation} pots · oracle ${t.oracle && t.oracle.pass ? "pass" : "fail"}` : t.status === "error" ? "error: " + t.error : "" })));
    if (c.deal) {
      const d = c.deal;
      const table = el("table", { class: "mini" }, el("thead", {}, el("tr", {}, el("th", { text: "bettor" }), el("th", { text: "action" }), el("th", { text: "Q check / bet (chips)" }), el("th", { text: "visits" }), el("th", { text: "oracle" }))));
      const tb = el("tbody");
      ["population", "personal"].forEach((k) => {
        const b = c.on_deal && c.on_deal[k];
        const nice = (a) => ({ bet_half: "bet 50", check_back: "check", fold: "fold", call: "call" }[a] || a);
        tb.appendChild(el("tr", {}, el("td", { text: k }), el("td", { text: b ? nice(b.action) : "(not trained)" }), el("td", { text: b ? b.q_chips.join(" / ") : "" }), el("td", { text: b ? b.visits.join(" / ") : "" }), el("td", { text: b && b.oracle ? (b.oracle.pass ? "pass" : "fail") : "" })));
      });
      if (c.advice) tb.appendChild(el("tr", {}, el("td", { text: "EV advisor" }), el("td", { text: c.advice.recommend }), el("td", { text: `${fmt(Math.round(c.advice.ev_check * 100))} / ${fmt(Math.round(c.advice.ev_bet * 100))}` }), el("td", { text: "closed form" }), el("td", { text: "" })));
      table.appendChild(tb);
      box.appendChild(el("div", {}, el("b", { text: `Deal: ${d.board.join(" ")} · you ${d.bettor_cards.join(" ")} (${d.bettor_bucket}) · ${CTX_LABEL[d.context]}` })));
      box.appendChild(table);
      box.appendChild(el("p", { class: "hint", text: "Identical actions are a valid result: the twin may not differ from the population in this spot." }));
    } else box.appendChild(el("p", { class: "hint", text: "Play a spot in step 4 to compare the bots on it." }));
    root.appendChild(box);
    [["population", c.population_chart, c.population_meta], ["personal", c.personal_chart, c.personal_meta]].forEach(([name, chart, meta]) => {
      if (!chart) return;
      const cb = el("div", { class: "card-box" }, el("h4", { text: `${name} bot policy chart (stage 1: check back vs bet)` }));
      const tbl = el("table", { class: "matrix" }, el("tr", {}, el("th", { text: "turn ↓ strength →" }), ...["weak", "medium", "strong"].map((b) => el("th", { text: b }))));
      Object.entries(chart["1"]).forEach(([ctx, row]) => {
        const tr = el("tr", {}, el("th", { text: CTX_LABEL[ctx] }));
        ["weak", "medium", "strong"].forEach((b) => { const cell = row[b]; tr.appendChild(el("td", { class: cell.action === "bet_half" ? "pc-bet" : "", html: `<b>${cell.action === "bet_half" ? "bet" : "check"}</b><br><span class="muted">${cell.q.map((x) => Math.round(x * 100)).join(" / ")}</span><br><span class="muted">${cell.visits.join("+")}</span>` })); });
        tbl.appendChild(tr);
      });
      cb.appendChild(tbl);
      if (meta) cb.appendChild(el("div", { class: "muted", text: `${meta.episodes} episodes, seed ${meta.seed}, ${meta.train_seconds}s · oracle check ${meta.oracle_check ? (meta.oracle_check.pass ? "pass" : meta.oracle_check.failures + " cell(s) off") : "n/a"}` }));
      root.appendChild(cb);
    });
    if (t.status === "running") { clearTimeout(M.pollTimer); M.pollTimer = setTimeout(renderCompare, 1500); }
  }
  async function startTraining() { try { await api("/api/mirror/train", body({ episodes: 20000 })); renderCompare(); } catch (e) { toast(e.message, true); } }

  // ------------------------------------------------------------ evidence: Phase 6 (every table size, real players)
  const sg = (x, d = 2) => (x === null || x === undefined ? "-" : (x > 0 ? "+" : x < 0 ? "\u2212" : "") + Math.abs(x).toFixed(d));
  const ci = (v, d = 2) => (v ? `${sg(v[0], d)} to ${sg(v[1], d)}` : "-");
  const fmtN = (n) => (n === null || n === undefined ? "-" : Number(n).toLocaleString("en-GB"));
  const ARM_LABEL = { pooled: "one rate for everyone", population: "average player, by context", situation: "situation model",
    personal: "this player's history", situation_personal: "situation + this player", personal_online: "this player's history",
    situation_personal_online: "situation + this player", neural: "neural network" };
  const verdict = (b) => (b.better ? "real gain" : b.worse ? "real loss" : "no clear effect");
  function miniTable(head, rows) {
    const t = el("table", { class: "mini" }, el("thead", {}, el("tr", {}, ...head.map((h) => el("th", { text: h })))));
    const tb = el("tbody");
    rows.forEach((r) => tb.appendChild(el("tr", {}, ...r.map((c) => el("td", { text: c === null || c === undefined ? "-" : String(c) })))));
    t.appendChild(tb);
    return t;
  }
  function wideScanBox(d) {
    const rows = Object.entries(d.by_table_size || {}).map(([k, v]) => [k === "2" ? "two players" : `${k} players`, fmtN(v.any), fmtN(v.band), fmtN(v.main)]);
    const m = d.main_cohort || {};
    return el("div", {}, miniTable(["table size", "any bet", "bet 40-60% of pot", "main cohort"], rows),
      el("p", { text: `Main cohort: ${fmtN(m.decisions)} decisions by ${fmtN(m.players)} players, ${fmtN(m.train)} to train and ${fmtN(m.test)} to test. The original cohort had 12,407 decisions from two-player tables.` }),
      el("div", { class: "muted", text: `${fmtN(d.rows)} rows scanned, ${fmtN(d.kept)} unique river spots, ${d.seconds} s.` }));
  }
  function forecastWideBox(d) {
    const out = el("div", {});
    [["frozen", "History up to the cutoff (as before)"], ["online", "Learning as you go"]].forEach(([proto, title]) => {
      const pr = (d.protocols || {})[proto];
      if (!pr) return;
      out.appendChild(el("h5", { text: title }));
      out.appendChild(miniTable(["forecaster", "log loss"], ["pooled", "population", "situation", "personal", "situation_personal"].map((a) => [ARM_LABEL[a], pr.overall[a].toFixed(4)])));
      const L = pr.lift_log_loss || {};
      [["situation_vs_population", "Situation over the average player"], ["personal_vs_population", "Player history over the average player"],
        ["situation_personal_vs_situation", "Player history on top of the situation"]].forEach(([k, label]) => {
        const b = L[k];
        if (b) out.appendChild(el("div", { text: `${label}: ${sg(b.mean, 4)} (95% ${ci(b.ci95, 4)}), ${verdict(b)}.` }));
      });
    });
    out.appendChild(el("div", { class: "muted", text: `${fmtN((d.cohort || {}).test)} test decisions by ${fmtN((d.cohort || {}).test_players)} players. ${d.label}.` }));
    return out;
  }
  function bluffBox(d) {
    const pr = (d.protocols || {}).online || (d.protocols || {}).frozen;
    const c = pr.chips_per_decision, br = pr.bluff_rate || {};
    const rows = [["never", "never bluff"], ["always", "always bluff"], ["population", "average-player forecast"], ["situation", "situation model"],
      ["personal", "player history"], ["situation_personal", "situation + player"], ["perfect", "perfect foresight (ceiling)"]]
      .filter(([k]) => c[k] !== undefined).map(([k, label]) => [label, c[k].toFixed(2), k === "always" ? "100%" : k === "never" ? "0%" : br[k] !== undefined ? pct(br[k], 0) : "-"]);
    const out = el("div", {}, miniTable(["the bluffer uses", "chips per decision", "bluffs"], rows));
    [["situation", "Situation model"], ["personal", "Player history"], ["situation_personal", "Situation + player"]].forEach(([k, label]) => {
      const b = (pr.vs_population || {})[k];
      if (b) out.appendChild(el("div", { text: `${label} vs the average-player forecast: ${sg(b.mean)} chips per decision (95% ${ci(b.ci95)}), ${pct(b.decisions_changed, 0)} of decisions changed, ${verdict(b)}.` }));
    });
    out.appendChild(el("div", { class: "muted", text: `${fmtN(d.decisions)} held-out decisions by ${fmtN(d.players)} players, learning as you go. Rule: ${d.rule}. Pot = 100 chips.` }));
    return out;
  }
  function realBenchBox(d) {
    const rows = (d.rows || []).map((r) => [r.probes, sg(r.mean_gain_chips), ci(r.ci95), pct(r.share_helped, 0), pct(r.share_hurt, 0), pct(r.decisions_changed, 0)]);
    const out = el("div", {}, miniTable(["answers seen", "gain, chips per decision", "95% interval", "players helped", "players hurt", "decisions changed"], rows));
    const g = d.by_true_fold_rate || {};
    out.appendChild(el("div", { text: "By how often the player really folds: " + Object.entries(g).map(([k, v]) => `${k} ${sg(v.mean_gain_chips)} (${v.players} players)`).join(" · ") }));
    out.appendChild(el("div", { class: "muted", text: `Sanity: with 0 answers seen the two bettors are ${d.zero_probe_identical ? "identical, as they must be" : "NOT identical, check the setup"}.` }));
    const q = d.qlearning_check;
    if (q) out.appendChild(el("div", { class: "muted", text: `Q-learning re-check on ${q.players.length} players: ${q.cells_agree} of ${q.cells} first-decision cells match the exact best response; value gap ${sg(q.mean_value_gap_chips)} chips.` }));
    out.appendChild(el("div", { class: "muted", text: `${d.players} real players. ${d.label}.` }));
    return out;
  }
  function strengthBox(d) {
    const c = d.check || {};
    return el("div", {}, el("p", { text: `Fitted on ${d.fit_spots} Pluribus spots and checked on the other ${d.check_spots}. Readout log score ${c.log_score_default} with the default table and ${c.log_score_calibrated} calibrated: ${sg(c.improvement, 3)} (95% ${ci(c.ci95, 3)}). The top bucket is right ${pct(c.top_bucket_right_default, 0)} of the time with the default and ${pct(c.top_bucket_right_calibrated, 0)} calibrated.` }),
      el("div", { text: d.use ? "Used by the twins." : "Not used: the improvement is not clear, so the twins keep the labelled default." }), el("div", { class: "muted", text: d.rule }));
  }
  function neuralBox(d) {
    const out = el("div", {}, miniTable(["forecaster", "test log loss"], Object.entries(d.test_log_loss || {}).map(([k, v]) => [ARM_LABEL[k] || k.replace(/_/g, " "), v.toFixed(4)])));
    Object.entries(d.neural_lift_vs || {}).forEach(([k, b]) => out.appendChild(el("div", { text: `Network vs ${ARM_LABEL[k] || k.replace(/_/g, " ")}: ${sg(b.mean, 4)} (95% ${ci(b.ci95, 4)}), ${verdict(b)}.` })));
    if (d.bluff_vs_population) out.appendChild(el("div", { text: `Bluff test: ${sg(d.bluff_vs_population.mean)} chips per decision against the average-player forecast (95% ${ci(d.bluff_vs_population.ci95)}).` }));
    out.appendChild(el("div", { class: "muted", text: `${d.device}${d.quick ? ", quick check only" : ""}. ${d.label}.` }));
    return out;
  }
  function typeGateWideBox(d) {
    const g = d.gate || {}, a = g.type_vs_population_zero_history || {}, b = g.type_vs_personal || {};
    return el("div", {}, el("p", {}, el("b", { text: g.pass ? "Gate passed. " : "Gate failed. " }),
      `Players with no river history, type prior vs average player: ${sg(a.lift, 4)} (95% ${ci(a.ci95, 4)}, ${fmtN(a.players)} players). Types plus history vs history alone: ${sg(b.lift, 4)} (95% ${ci(b.ci95, 4)}).`),
      el("div", { class: "muted", text: g.rule || "" }));
  }
  function phase6Panels(root, p, panel) {
    root.appendChild(el("div", { class: "card-box" }, el("h4", { text: "Every table size and real players" }),
      el("p", { class: "hint", text: "Two-player river pots from every table size, scored against real players' recorded answers. The original two-player-table panels follow." })));
    const W = (key, title, fn) => { const x = p[key]; if (x) root.appendChild(panel(title, x.present, x.source, x.generated, x.present ? fn(x.data) : null)); };
    W("wide_scan", "A · Data from every table size", wideScanBox);
    W("forecast_wide", "B · Forecasts: situation, person, both", forecastWideBox);
    W("bluff_test", "C · Bluff test on real decisions", bluffBox);
    W("benchmark_real", "D · RL bettors against real players", realBenchBox);
    W("strength_pluribus", "E · Strength table from Pluribus hands", strengthBox);
    W("neural_forecaster", "F · Neural forecaster (GPU)", neuralBox);
    W("type_gate_wide", "G · Player types on every table size", typeGateWideBox);
  }

  // ------------------------------------------------------------ evidence
  async function renderEvidence() {
    const root = $("#evidencePanels");
    root.innerHTML = "";
    let ev;
    try { ev = await api("/api/mirror/evidence"); } catch (e) { root.appendChild(el("div", { class: "card-box hint", text: e.message })); return; }
    const p = ev.panels;
    const panel = (title, present, source, generated, content) => el("div", { class: "card-box" }, el("div", { class: "row between" }, el("h4", { text: title }), el("span", { class: "tag " + (present ? "good" : "warn"), text: present ? "generated" : "missing" })),
      present ? content : el("p", { class: "hint", text: "Not generated yet." }), el("div", { class: "muted", text: `source: ${source}${generated ? " · " + generated : ""}` }));
    phase6Panels(root, p, panel);
    // 1 filter exhibit
    const fe = p.filter_exhibit;
    root.appendChild(panel("1 · Filter exhibit", fe.present, fe.source, fe.generated, fe.present ? (() => {
      const rows = el("table", { class: "mini" }, el("thead", {}, el("tr", {}, el("th", { text: "cohort" }), el("th", { text: "shards 35+36" }), el("th", { text: "all 37 shards" }))));
      const tb = el("tbody");
      const labels = { all_eligible: "all eligible offers", half_checked_through: "half pot, turn checked through", half_bettor_aggr: "half pot, bettor bet the turn",
        half_defender_aggr: "half pot, defender bet the turn", defender_cards_known: "defender's cards became known", warning_free: "warning-free hands" };
      const cnt = (c) => `${c.folds.toLocaleString()}/${c.opportunities.toLocaleString()}`;
      Object.keys(fe.data.full_corpus.cohorts).forEach((k) => { const a = fe.data.two_shard.cohorts[k], b = fe.data.full_corpus.cohorts[k]; tb.appendChild(el("tr", {}, el("td", { text: labels[k] || k }), el("td", { text: cnt(a) }), el("td", { text: cnt(b) }))); });
      rows.appendChild(tb);
      return el("div", {}, rows, el("div", { class: "muted", text: `${fmt(fe.scan.rows)} rows scanned, ${fmt(fe.scan.eligible_unique)} unique eligible offers, ${fmt(fe.scan.eligible_modelling)} in the modelling cohort; train ${fmt(fe.scan.manifest.train)} / test ${fmt(fe.scan.manifest.test)}` }));
    })() : null));
    // 2 forecast ladder
    const fl = p.forecast_ladder;
    root.appendChild(panel("2 · Forecast ladder (held-out test)", fl.present, fl.source, fl.generated, fl.present ? ladderTable(fl.data) : null));
    // 3 type gate + style map
    const tg = p.type_gate;
    let st = null;
    try { st = await api("/api/mirror/styles"); } catch (e) { st = null; }
    root.appendChild(panel("3 · Player types and the type-prior gate", tg.present, tg.source, tg.generated, tg.present ? (() => {
      const g = tg.data.gate || {};
      return el("div", {},
        el("p", {}, el("b", { text: `${tg.data.k} types from ${(st && st.players_used || 0).toLocaleString()} players with at least ${st && st.n_min} hands. Gate ${g.pass ? "passed" : "failed"}.` })),
        el("p", { class: "hint", text: gateText(g) }),
        st && st.present ? styleMap(st) : null);
    })() : null));
    // 4 benchmark
    const db = p.decision_benchmark;
    root.appendChild(panel("4 · Decision benchmark (matched deals)", db.present, db.source, db.generated, db.present ? benchTable(db.data) : null));
    // 5 range checks
    const rc = p.range_checks;
    root.appendChild(panel("5 · Range readout checks", rc.present, rc.source, rc.generated, rc.present ? rangeChecksBox(rc.data) : null));
    // population bot
    const pb = p.population_bot;
    root.appendChild(panel("Population bot training", pb.present, pb.source, pb.generated, pb.present ? el("div", {}, el("div", { text: `${pb.data.meta.episodes} episodes, seed ${pb.data.meta.seed}, ${pb.data.meta.train_seconds}s, oracle check ${pb.data.meta.oracle_check ? (pb.data.meta.oracle_check.pass ? "pass" : "fail") : "n/a"}` }),
      el("div", { class: "muted", text: "checkpoints: " + pb.data.checkpoints.map((c) => `${c.episode}: ${c.validation_mean_reward}`).join(" · ") })) : null));
    // training sufficiency
    const ts = p.training_sufficiency;
    root.appendChild(panel("Training sufficiency (three seeds)", ts.present, ts.source, ts.generated, ts.present ? el("div", {},
      el("div", { text: `Seeds agree on ${pct(ts.data.stage1_agreement, 0)} of first-decision cells. Disagreements only on near-ties: ${ts.data.disagreements_are_near_ties ? "yes" : "no"}. Every seed passes the oracle: ${ts.data.all_oracle_pass ? "yes" : "no"}.` })) : null));
    // human sessions
    const ss = p.sessions;
    root.appendChild(panel("6 · Human sessions", true, ss.source, ss.generated, sessionsBox(ss.data)));
  }

  function rangeChecksBox(d) {
    const out = el("div", {});
    const h = d.held_out_sessions || {};
    out.appendChild(el("p", {}, el("b", { text: "Held-out humans: " }), h.decisions
      ? `${h.sessions} session(s), ${h.decisions} hands. Log score of the true strength bucket ${h.mean_log_score_readout} with the readout, ${h.mean_log_score_blind} without it.`
      : "none yet, so nothing is claimed about humans."));
    const pl = d.pluribus;
    if (pl) {
      const r = pl.by_response || {};
      const line = ["fold", "call", "raise"].filter((k) => r[k]).map((k) => `${k} ${pct(r[k].mean_equity, 0)} (${r[k].n} spots)`).join(", ");
      out.appendChild(el("p", {}, el("b", { text: "Pluribus, every card known: " }),
        `${pl.spots} heads-up river spots. Mean defender equity by response: ${line}. ` +
        (pl.monotone ? "Weaker hands fold more, which is what the readout assumes." : "The ordering does not hold, so treat the readout with care.")));
    }
    const t = d.twin_unit_check;
    if (t) out.appendChild(el("p", { class: "muted", text: `Unit check on the twin's own responses: log score ${t.mean_log_score_readout} with the readout, ${t.mean_log_score_blind} without; top bucket right ${pct(t.top_bucket_accuracy, 0)} of the time. This only tests the code.` }));
    return out;
  }

  function gateText(g) {
    const a = g.type_vs_population_zero_history, b = g.type_vs_personal;
    if (!a || !b) return `Log-loss lift over the population model ${g.lift_vs_population_log_loss}, 95% interval ${(g.ci95 || []).join(" to ")}.`;
    const show = (c) => (c && c.ci95 ? `${c.lift} per decision, 95% interval ${c.ci95.join(" to ")}, ${c.players} players` : "no data");
    return `Players with no river history, type prior against the population model: ${show(a)}. All test decisions, types plus history against history alone: ${show(b)}. ` +
      (g.pass ? "Both intervals sit above zero, so the live model uses types."
        : "Types ship only if both intervals sit above zero, so the live model ignores them and the map below only describes how people play.");
  }

  function sessionsBox(d) {
    const lc = d.live_counts || {};
    const t = el("table", { class: "mini" }, el("thead", {}, el("tr", {}, el("th", { text: "tag" }), el("th", { text: "sessions" }), el("th", { text: "decisions" }))));
    const tb = el("tbody");
    TAGS.forEach(([tag, label]) => tb.appendChild(el("tr", {}, el("td", { text: label }), el("td", { text: (lc[tag] || {}).sessions || 0 }), el("td", { text: (lc[tag] || {}).decisions || 0 }))));
    t.appendChild(tb);
    const ev = d.evaluation;
    let res = el("p", { class: "hint", text: "No held-out scores yet. Tag judges in step 3 (calibration first, then held-out) and recompute. Nothing is claimed about humans until then." });
    if (ev && ev.heldout_forecast && ev.heldout_forecast.decisions) {
      const f = ev.heldout_forecast, r = ev.heldout_readout;
      res = el("div", {},
        el("p", { text: `Held-out: ${f.sessions} session(s), ${f.decisions} decisions. Log loss per decision: population ${f.log_loss.population}, personal ${f.log_loss.personal}.` + (f.lift_ci95_over_sessions ? ` Lift 95% interval over sessions: ${f.lift_ci95_over_sessions.join(" to ")}.` : " An interval needs at least three held-out sessions.") }),
        r && r.decisions ? el("p", { text: `Range readout on held-out hands: log score ${r.mean_log_score_readout} against ${r.mean_log_score_blind} for the response-blind prior (strength table: ${r.strength_table}).` }) : null);
    }
    const cal = d.calibration;
    return el("div", {}, t, res,
      el("div", { class: "muted", text: cal ? `Strength calibration: ${cal.sessions} session(s), ${cal.decisions} decisions.` : "Strength table: synthetic default until calibration sessions exist." }),
      el("div", { class: "row" }, el("button", { class: "btn small", onclick: recompute }, "Recompute from sessions")));
  }
  async function recompute() {
    try { await api("/api/mirror/sessions/recompute", {}); toast("Recomputed calibration and held-out scores from the session logs."); renderEvidence(); } catch (e) { toast(e.message, true); }
  }

  // Small multiples: one panel per type, that type highlighted and everyone else grey (no colour has to carry identity).
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
    const ns = "http://www.w3.org/2000/svg";
    const mk = (tag, attrs) => { const e = document.createElementNS(ns, tag); Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v)); return e; };
    st.names.forEach((name, t) => {
      const svg = mk("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `${name}: ${pct(st.shares[t], 0)} of players` });
      const mine = [];
      pts.forEach((p) => { if (p.type !== t) svg.appendChild(mk("circle", { cx: sx(p.x).toFixed(1), cy: sy(p.y).toFixed(1), r: 1.3, fill: "var(--muted)", "fill-opacity": 0.3 })); else mine.push(p); });
      mine.forEach((p) => svg.appendChild(mk("circle", { cx: sx(p.x).toFixed(1), cy: sy(p.y).toFixed(1), r: 1.6, fill: "var(--series-1)", "fill-opacity": 0.75 })));
      const cxy = st.centroid_xy[t];
      svg.appendChild(mk("circle", { cx: sx(cxy[0]).toFixed(1), cy: sy(cxy[1]).toFixed(1), r: 6, fill: "var(--series-1)", stroke: "var(--surface)", "stroke-width": 2 }));
      const c = st.centroids[t];
      const tip = el("div", { class: "chart-tip", hidden: true });
      const cell = el("div", { class: "stylemap-cell" },
        el("div", { class: "row between" }, el("b", { text: name }), el("span", { class: "muted", text: `${pct(st.shares[t], 0)} of players` })),
        svg,
        el("div", { class: "muted", text: `VPIP ${pct(c.vpip, 0)} · PFR ${pct(c.pfr, 0)} · aggression ${pct(c.af, 0)} · folds to bets ${pct(c.fold_to_bet, 0)}` }),
        tip);
      svg.addEventListener("mousemove", (e) => {
        const rect = svg.getBoundingClientRect();
        const mx = (e.clientX - rect.left) * W / rect.width, my = (e.clientY - rect.top) * H / rect.height;
        let best = null, bd = 64;
        mine.forEach((p) => { const d = (sx(p.x) - mx) ** 2 + (sy(p.y) - my) ** 2; if (d < bd) { bd = d; best = p; } });
        if (!best) { tip.hidden = true; return; }
        tip.hidden = false;
        tip.textContent = `${name} · ${best.venue} · ${best.hands} hands`;
        tip.style.left = Math.min(e.clientX - cell.getBoundingClientRect().left + 10, cell.clientWidth - 150) + "px";
        tip.style.top = (e.clientY - cell.getBoundingClientRect().top + 10) + "px";
      });
      svg.addEventListener("mouseleave", () => { tip.hidden = true; });
      wrap.appendChild(cell);
    });
    return el("div", {}, wrap, el("div", { class: "muted", text: "Each dot is a player (3,000 sampled). Position summarises five style statistics: players close together play alike. Highlighted: the type named above the panel; grey: everyone else." }));
  }
  function ladderTable(d) {
    const t = el("table", { class: "mini" }, el("thead", {}, el("tr", {}, el("th", { text: "earlier obs." }), el("th", { text: "n" }), el("th", { text: "pooled" }), el("th", { text: "population" }), el("th", { text: "type + history" }), el("th", { text: "history" }))));
    const tb = el("tbody");
    (d.ladder || []).forEach((r) => tb.appendChild(el("tr", {}, el("td", { text: r.bin }), el("td", { text: r.n }), ...["pooled", "population", "type", "personal"].map((k) => el("td", { text: r.log_loss && r.log_loss[k] !== undefined && r.log_loss[k] !== null ? r.log_loss[k].toFixed(4) : "-" })))));
    t.appendChild(tb);
    return el("div", {}, el("div", { class: "muted", text: "three-action log loss per decision (lower is better); Brier scores in the artifact" }), t);
  }
  function benchTable(d) {
    const t = el("table", { class: "mini" }, el("thead", {}, el("tr", {}, el("th", { text: "opponent" }), el("th", { text: "probes" }), el("th", { text: "Δ chips / spot" }), el("th", { text: "95% CI" }), el("th", { text: "changed" }))));
    const tb = el("tbody");
    const sg = (x) => (x > 0 ? "+" : x < 0 ? "−" : "") + Math.abs(x).toFixed(2);
    const ci = (v) => `${sg(v[0])} .. ${sg(v[1])}`;
    const missed = (d.rows || []).filter((r) => r.oracle_pass === false);
    (d.rows || []).forEach((r) => {
      const miss = r.oracle_pass === false;
      tb.appendChild(el("tr", miss ? { class: "muted" } : {}, el("td", { text: r.opponent }), el("td", { text: r.probes }),
        el("td", { text: sg(r.mean_diff_chips) + (miss ? " *" : "") }), el("td", { text: ci(r.ci95_chips) }), el("td", { text: pct(r.fraction_changed) })));
    });
    t.appendChild(tb);
    const c = d.control;
    return el("div", {}, t,
      missed.length ? el("div", { class: "muted", text: `* This bot failed its oracle check: it did not learn its own best response, so the row shows a training miss, not the value of knowing the opponent (${missed.length} of ${d.rows.length} rows).` }) : null,
      c ? el("div", { text: `Control: the same population bot trained with another seed changes ${pct(c.fraction_changed)} of decisions, ${sg(c.mean_diff_chips)} chips per spot (interval ${ci(c.ci95_chips)}). An effect must be larger than this to count.` }) : null,
      d.zero_probe_identical !== undefined && d.zero_probe_identical !== null ? el("div", { class: "muted", text: `Sanity: with 0 probes the two bettors are ${d.zero_probe_identical ? "identical, as they must be" : "NOT identical, check the setup"}.` }) : null,
      el("div", { class: "muted", text: d.label || "" }));
  }

  // ------------------------------------------------------------ wiring
  function renderCurrent() {
    const fn = { reveal: renderReveal, filters: renderFilters, drill: renderDrill, play: renderPlay, compare: renderCompare }[M.step];
    return fn ? fn() : null;
  }
  function selectStep(step) {
    M.step = step;
    $$("#mirrorSteps button").forEach((x) => x.classList.toggle("active", x.dataset.step === step));
    $$(".mstep").forEach((p) => p.classList.toggle("active", p.id === "mirror-" + step));
  }

  async function onShow(tab) {
    if (tab === "evidence") { renderEvidence(); return; }
    try { await refresh(); } catch (e) { toast("Mirror backend unavailable: " + e.message, true); return; }
    P.showOverlay(false);
    if (!$("#seats").children.length && M.step !== "drill" && M.step !== "play") P.showOverlay(true, "Poker Mirror: use the steps on the right.");
    // deep links for demos and screenshots: #mirror=<step>&deal=1&spot=1&reveal=1
    const params = new URLSearchParams(location.hash.replace(/^#/, ""));
    if (params.has("mirror") && !M.deepLinked) {
      M.deepLinked = true;
      selectStep(params.get("mirror") || "reveal");
      if (params.has("reveal")) { M.revealed = true; M.guess = 0; }
      if (params.has("deal")) await newDrill();
      else if (params.has("respond")) { await newDrill(); const d = M.state.drill.pending; if (d) await respond(d.decision_id, params.get("respond")); }
      else if (params.has("spot")) { await freeze(); await newSpot(); if (params.has("act")) await act(params.get("act")); }
      else if (params.has("real")) { const rp = await realPlayers(); const id = params.get("real") || (rp.players[0] || {}).id; if (id) await loadReal(id); if (params.has("act")) await act(params.get("act")); }
      if (params.has("deal") || params.has("respond")) return;  // the drill renderers already drew the table
      if ((params.has("spot") || params.has("real")) && M.step === "play") return;
    }
    await renderCurrent();
    if (M.step === "reveal" && M.revealed && M.demo) showDemoHand(0);
  }
  $$("#mirrorSteps button").forEach((b) => b.addEventListener("click", () => { selectStep(b.dataset.step); renderCurrent(); }));
  $("#mirrorReset").addEventListener("click", async () => {
    try { const st = await api("/api/mirror/reset", body()); remember(st.token); M.state = st; M.lastDrill = null; M.revealed = false; M.guess = null; toast("Judge reset: new profile, no observations."); renderCurrent(); } catch (e) { toast(e.message, true); }
  });
  window.Mirror = { onShow, state: M };
})();
