/* Poker Mirror: a judge-operated, full-width sequence of five scenes.
   1 the historical guess  2 the known-cards filter  3 you defend (the model predicts you)
   4 you bet against your frozen twin  5 two bettors on the same deal.
   Visual identity is keyed by scene + decision/spot so a forecast update never looks like a fresh deal. */
(function () {
  "use strict";
  const P = window.PokerSim, V = window.Viz;
  const { api, el, toast, fmt, pct } = P;
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));

  const STEPS = [
    { id: "reveal", label: "Which bet got the fold?" },
    { id: "filters", label: "Where did the folds go?" },
    { id: "drill", label: "You defend" },
    { id: "play", label: "You bet" },
    { id: "compare", label: "Same deal, two bettors" },
  ];
  const M = {
    token: null, state: null, step: "reveal", demo: null, exhibit: null, evidence: null, guess: null, filter: "all",
    lastDrill: null, pollTimer: null, busy: false, built: false, resumeChecked: false, deepLinked: false, hu: null, sceneId: null,
    rangeOpen: false, flyFrom: null,
  };
  const TURN = { checked_through: "both players checked the turn", bettor_aggr: "the bettor also bet the turn", defender_aggr: "the defender bet the turn" };
  // The defender in steps 4-5 is either the judge's frozen twin or a real player's twin (Phase 6).
  function opp() {
    const o = (M.state && M.state.opponent) || { kind: "judge" };
    const real = o.kind === "real";
    return { o, real, name: real ? `${o.id}'s twin` : "Your twin", the: real ? "the player's twin" : "your twin", The: real ? "The player's twin" : "Your twin" };
  }
  const ctxBettor = (c) => ({ checked_through: "both players checked the turn", bettor_aggr: "you bet the turn", defender_aggr: `${opp().the} bet the turn` }[c]);
  async function realPlayers() {
    if (!M.realPlayers) { try { M.realPlayers = await api("/api/mirror/real_players"); } catch (e) { M.realPlayers = { present: false, players: [] }; } }
    return M.realPlayers;
  }
  function setHead(root, eyebrow, h1Parts, lede) {
    const h = $(".scene-head", root);
    if (!h) return;
    $(".scene-eyebrow", h).textContent = eyebrow;
    const h1 = $("h1", h);
    h1.innerHTML = "";
    h1Parts.forEach((x) => h1.appendChild(typeof x === "string" ? document.createTextNode(x) : x));
    const l = $(".lede", h);
    if (l) l.textContent = lede;
  }
  const CTX_SHORT = { checked_through: "Turn checked through", bettor_aggr: "Bettor bet the turn", defender_aggr: "Defender bet the turn" };
  const OPP_LABEL = { caller: ["Frequent caller", "folds 15% of the time"], folder: ["Frequent folder", "folds 65% of the time"], population: ["Population player", "the historical average"], strength: ["Strength-dependent", "folds weak, raises strong"] };
  const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
  const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;

  // ------------------------------------------------------------ session
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode: fine */ } },
    del(k) { try { localStorage.removeItem(k); } catch (e) { /* ignore */ } },
  };
  function token() { if (M.token === null) M.token = store.get("mirrorToken") || ""; return M.token; }
  function remember(tok) { M.token = tok; store.set("mirrorToken", tok); }
  const q = () => "?session=" + encodeURIComponent(token());
  const body = (extra) => Object.assign({ session: token() }, extra || {});
  async function refresh() {
    const st = await api("/api/mirror/state" + q());
    remember(st.token);
    M.state = st;
    paintJudge();
    return st;
  }
  const heldout = () => M.state && M.state.tag === "heldout";
  function frozenInfo() {
    const st = M.state;
    if (!st || !st.frozen) return null;
    try { const f = JSON.parse(store.get("mirrorFrozen") || "null"); if (f && f.hash === st.frozen) return f; } catch (e) { /* ignore */ }
    return { hash: st.frozen, observations: null };
  }

  // ------------------------------------------------------------ shell
  function buildShell() {
    const root = $("#view-mirror");
    root.innerHTML = "";
    const steps = el("nav", { class: "m-steps", "aria-label": "Poker Mirror steps" }, el("ol", {},
      STEPS.map((s, i) => el("li", {}, el("button", { class: "m-step", "data-step": s.id, onclick: () => go(s.id, { focus: true }) },
        el("span", { class: "m-n num", text: String(i + 1) }), el("span", { class: "m-l", text: s.label }))))));
    const judge = el("div", { class: "m-judge" }, el("span", { class: "m-judge-id", id: "mJudge" }),
      el("button", { class: "btn small ghost", id: "mOpBtn", "aria-expanded": "false", "aria-controls": "mOperator", onclick: toggleOperator }, "Operator"));
    const head = el("header", { class: "m-head" }, steps, judge, el("div", { class: "m-operator", id: "mOperator", hidden: true, role: "region", "aria-label": "Operator controls" }));
    root.appendChild(el("div", { class: "mirror" }, head, el("div", { class: "m-scene", id: "mScene" })));
    M.built = true;
  }
  function paintJudge() {
    const st = M.state;
    const j = $("#mJudge");
    if (!j || !st) return;
    j.innerHTML = "";
    j.appendChild(el("span", { class: "mono", text: "judge " + st.token.slice(0, 6) }));
    j.appendChild(el("span", { "aria-hidden": "true", text: "·" }));
    j.appendChild(el("span", { text: plural(st.profile.observations, "response") }));
    if (st.tag !== "practice") j.appendChild(el("span", { class: "tag warn", text: st.tag === "heldout" ? "held-out" : st.tag }));
    if (!$("#mOperator").hidden) paintOperator();
  }
  function toggleOperator() {
    const op = $("#mOperator"), b = $("#mOpBtn");
    op.hidden = !op.hidden;
    b.setAttribute("aria-expanded", String(!op.hidden));
    if (!op.hidden) { paintOperator(); const f = op.querySelector("button"); if (f) f.focus(); }
  }
  const TAGS = [["practice", "Practice", "Nothing is used for evaluation. Raw answers never leave this laptop."],
    ["calibration", "Calibration", "Development sessions that teach the twin how hand strength changes responses."],
    ["heldout", "Held-out", "Someone who has not seen the model or its forecasts. Forecasts stay hidden; answers are scored, never trained on."]];
  function paintOperator() {
    const op = $("#mOperator"), st = M.state;
    if (!st) return;
    op.innerHTML = "";
    const cur = TAGS.find((t) => t[0] === st.tag) || TAGS[0];
    op.appendChild(el("div", { class: "op-row" }, el("span", { class: "op-k", text: "This judge's answers are for" }),
      el("div", { class: "seg", role: "group", "aria-label": "Session tag" }, TAGS.map(([t, label]) => el("button", { class: "seg-btn" + (st.tag === t ? " on" : ""), "aria-pressed": String(st.tag === t), onclick: () => setTag(t) }, label)))));
    op.appendChild(el("p", { class: "hint", text: cur[2] }));
    op.appendChild(el("div", { class: "op-row" },
      el("span", { class: "mono muted", text: `session ${st.token} · ${plural(st.drill.drills, "spot")} · twin ${st.frozen ? "#" + st.frozen.slice(0, 7) : "not frozen"}` }),
      el("button", { class: "btn small", onclick: restartDemo }, "Back to the opening"),
      el("button", { class: "btn small danger", onclick: resetJudge }, "New judge"),
      el("a", { class: "btn small ghost", href: "#evidence=sessions", onclick: (e) => { e.preventDefault(); openEvidence("sessions"); } }, "Session evidence")));
  }
  async function setTag(t) {
    try { await api("/api/mirror/tag", body({ tag: t })); await refresh(); paintOperator(); if (M.step === "drill") renderScene(); } catch (e) { toast(e.message, true); }
  }
  async function resetJudge() {
    try {
      const st = await api("/api/mirror/reset", body());
      remember(st.token);
      M.lastDrill = null; M.guess = null; M.filter = "all"; M.rangeOpen = false;
      store.del("mirrorFrozen"); store.del("mirrorGuess"); store.del("mirrorStep");
      await refresh();
      toast("New judge: fresh profile, no observations.");
      go("reveal", { focus: true });
    } catch (e) { toast(e.message, true); }
  }
  function restartDemo() { M.guess = null; M.filter = "all"; store.del("mirrorGuess"); go("reveal", { focus: true }); }
  function openEvidence(topic) {
    try { history.replaceState(null, "", "#evidence=" + topic); } catch (e) { /* ignore */ }
    P.setTab("evidence");
  }

  // ------------------------------------------------------------ navigation
  function go(step, opts) {
    opts = opts || {};
    if (!STEPS.some((s) => s.id === step)) step = "reveal";
    M.step = step;
    store.set("mirrorStep", step);
    try { if (P.state.tab === "mirror") history.replaceState(null, "", "#mirror=" + step); } catch (e) { /* ignore */ }
    $$(".m-step").forEach((b) => { const on = b.dataset.step === step; b.classList.toggle("active", on); if (on) b.setAttribute("aria-current", "step"); else b.removeAttribute("aria-current"); });
    return renderScene(opts);
  }
  async function renderScene(opts) {
    opts = opts || {};
    clearTimeout(M.pollTimer);
    const fn = { reveal: sceneReveal, filters: sceneFilters, drill: sceneDrill, play: scenePlay, compare: sceneCompare }[M.step];
    await fn(opts);
    if (opts.focus) { const h = $("#mScene h1"); if (h) h.focus({ preventScroll: true }); window.scrollTo(0, 0); }
  }
  // Mount a scene root; returns {root, fresh}. Re-rendering the same scene keeps its root (and its table) in place.
  function mount(id, build) {
    const host = $("#mScene");
    if (M.sceneId === id && host.firstChild) return { root: host.firstChild, fresh: false };
    host.innerHTML = "";
    M.sceneId = id;
    M.hu = null;
    const root = build();
    host.appendChild(root);
    V.anim(root, [{ opacity: 0, transform: "translateY(6px)" }, { opacity: 1, transform: "none" }], { duration: 160 });
    return { root, fresh: true };
  }
  function sceneHead(eyebrow, h1Parts, lede, extra) {
    return el("div", { class: "scene-head" },
      el("div", { class: "scene-eyebrow mono", text: eyebrow }),
      el("h1", { class: "q", tabindex: "-1" }, ...h1Parts),
      lede !== null && lede !== undefined ? el("p", { class: "lede", text: lede }) : null, extra || null);
  }
  function nextLink(label, step) {
    return el("button", { class: "btn next-btn", onclick: () => go(step, { focus: true }) }, label, el("span", { "aria-hidden": "true", text: " →" }));
  }
  const evLink = (label, topic) => el("a", { class: "ev-link", href: "#evidence=" + topic, onclick: (e) => { e.preventDefault(); openEvidence(topic); } }, label);
  function errorBox(root, msg) { root.appendChild(el("div", { class: "scene-error", role: "alert" }, msg)); }

  // ------------------------------------------------------------ 1 · which river bet got the fold?
  async function loadDemo() {
    if (!M.demo) M.demo = await api("/api/mirror/demo_hands");
    if (!M.exhibit) { try { M.exhibit = await api("/api/mirror/exhibit"); } catch (e) { M.exhibit = { present: false }; } }
  }
  async function sceneReveal() {
    const { root, fresh } = mount("reveal", () => el("section", { class: "scene opening" }));
    if (!fresh) return;
    try { await loadDemo(); } catch (e) { errorBox(root, "Demo hands are not available: " + e.message); return; }
    const hands = M.demo.hands || [];
    if (hands.length < 2) { errorBox(root, "The two demo hands are not bundled. Run: python -m pokersim mirror-prepare"); return; }
    const meta = (hands[0].replay.record || {}).meta || {};
    const when = meta.month ? `${MONTHS[meta.month - 1]} ${meta.year}` : "";
    root.appendChild(sceneHead(`${hands[0].venue || "Historical hands"} · 100NL${when ? " · " + when : ""}`,
      ["Which river bet ", el("em", { text: "got the fold?" })],
      "One player faced a half-pot river bet twice. Pick the hand they folded."));
    const cohorts = M.exhibit && M.exhibit.present ? M.exhibit.exhibit.two_shard.cohorts : null;
    const grid = el("div", { class: "op-hands" });
    const result = el("div", { class: "op-result", hidden: true });
    hands.slice(0, 2).forEach((h, i) => {
      const letter = "AB"[i];
      const c = cohorts && cohorts["half_" + h.context];
      const verdict = el("div", { class: "op-verdict" }, el("span", { class: "v " + (h.response === "fold" ? "fold" : "call"), text: h.response === "fold" ? "Folded" : h.response === "call" ? "Called" : "Raised" }),
        el("span", { class: "op-turn", text: " · " + (TURN[h.context] || h.context_label) }));
      // supporting detail: inline on wide screens, behind a disclosure on a 1024px projector
      const detail = [c ? el("p", {}, el("b", { class: "num", text: `${c.folds.toLocaleString()}/${c.opportunities.toLocaleString()}` }), ` comparable half-pot offers folded after this turn (${pct(c.folds / c.opportunities, 0)}). `,
        el("button", { class: "link-btn", onclick: () => openDemoReplay(i) }, "Replay this hand")) : el("button", { class: "link-btn", onclick: () => openDemoReplay(i) }, "Replay this hand")];
      const more = window.innerWidth >= 1100 ? el("div", { class: "op-more" }, ...detail) : el("details", { class: "op-more" }, el("summary", { text: "Comparable offers in the data" }), ...detail);
      const reveal = el("div", { class: "op-reveal", hidden: true }, verdict, more);
      const choice = el("button", { class: "op-choice", "data-i": i, onclick: () => makeGuess(i) }, `Hand ${letter} folded`);
      grid.appendChild(el("article", { class: "op-hand", "data-i": i, "aria-label": `Hand ${letter}` },
        el("div", { class: "op-hand-head" }, el("h2", { text: `Hand ${letter}` }), el("span", { class: "mono op-bet", text: `${fmt(h.bet)} into ${fmt(h.pot_before_bet)}` })),
        V.board(h.board, "xl"), choice, reveal));
    });
    root.appendChild(grid);
    root.appendChild(result);
    root.appendChild(el("footer", { class: "scene-foot" },
      el("span", { class: "mono muted", text: `Hand records: ${M.demo.source} · ${M.demo.license}` })));
    const saved = store.get("mirrorGuess");
    if (M.guess === null && saved !== null && saved !== "" && token()) M.guess = Number(saved);
    if (M.guess !== null) showGuess(false);
  }
  function makeGuess(i) {
    if (M.guess !== null) return;
    M.guess = i;
    store.set("mirrorGuess", String(i));
    showGuess(true);
  }
  function showGuess(animate) {
    const root = $("#mScene .opening");
    if (!root) return;
    const hands = M.demo.hands, i = M.guess;
    const foldIdx = hands.findIndex((h) => h.response === "fold");
    const right = i === foldIdx;
    $$(".op-hand", root).forEach((card, k) => {
      card.classList.toggle("picked", k === i);
      card.classList.toggle("other", k !== i);
      const btn = $(".op-choice", card);
      btn.disabled = true;
      btn.classList.toggle("selected", k === i);
      btn.textContent = k === i ? `Your guess: Hand ${"AB"[k]} folded` : `Hand ${"AB"[k]} folded`;
      const rev = $(".op-reveal", card);
      rev.hidden = false;
      if (animate) V.anim(rev, [{ opacity: 0, transform: "translateY(-4px)" }, { opacity: 1, transform: "none" }], { duration: 200, delay: k === i ? 0 : 40 });
    });
    const res = $(".op-result", root);
    res.hidden = false;
    res.innerHTML = "";
    const L = (k) => "AB"[k];
    const what = `Hand ${L(foldIdx)} folded; Hand ${L(1 - foldIdx)} was ${hands[1 - foldIdx].response === "call" ? "called" : "raised"}.`;
    res.appendChild(el("p", { class: "op-verdict-line" }, el("b", { text: right ? "Right. " : "Not this time. " }), what,
      " Same player, same bet size: what differed was the turn before it."));
    const btn = nextLink("Next: where did the folds go?", "filters");
    res.appendChild(btn);
    if (animate) {
      V.anim(res, [{ opacity: 0 }, { opacity: 1 }], { duration: 200, delay: 60 });
      V.announce(`${right ? "Right." : "Not this time."} ${what}`);
      btn.focus({ preventScroll: true });
    }
  }
  async function openDemoReplay(i) {
    const h = M.demo.hands[i];
    if (!P.openReplay) { toast("Replay is not available in this build.", true); return; }
    try { await P.openReplay({ source: "path", path: M.demo.source, i, step: Math.max(0, h.decision_step - 1) }); } catch (e) { toast(e.message, true); }
  }

  // ------------------------------------------------------------ 2 · where did the folds go?
  const FILTERS = [
    ["all", "All eligible offers", "all_eligible"],
    ["known", "Known defender cards", "defender_cards_known"],
    ["clean", "Warning-free hands", "warning_free"],
  ];
  const FILTER_TEXT = {
    all: "Every recorded response to a 40–60% pot bet on the river after a check, heads-up no-limit hold'em.",
    known: "A fold never reaches a showdown, so a dataset filtered to hands where the defender's cards became known contains no folds at all. A model trained only on revealed hands learns from players who did not fold.",
    clean: "Dropping every hand with a replay warning removes continuing offers only: the warnings concern cards unknown at a later showdown, which says nothing about the river decision.",
  };
  async function sceneFilters() {
    const { root, fresh } = mount("filters", () => el("section", { class: "scene filters" }));
    if (!fresh) return;
    try { await loadDemo(); } catch (e) { /* exhibit loaded below */ }
    if (!M.exhibit || !M.exhibit.present) { errorBox(root, "The filter exhibit is not generated yet (python -m pokersim mirror-prepare)."); return; }
    const ex = M.exhibit.exhibit, two = ex.two_shard.cohorts, full = ex.full_corpus.cohorts;
    root.appendChild(sceneHead("Historical offers · filter exhibit", ["Where did the ", el("em", { text: "folds go?" })],
      "River bets of 40–60% of the pot across two historical data shards. Pick a filter a modeller might apply."));
    const strip = V.filterStrip(two);
    const toggles = el("div", { class: "seg big", role: "group", "aria-label": "Filter" },
      FILTERS.map(([k, label]) => el("button", { class: "seg-btn", "data-k": k, "aria-pressed": "false", onclick: () => setFilter(k) }, label)));
    const big = el("div", { class: "fx-big" }, el("span", { class: "fx-num" }), el("span", { class: "fx-sub" }));
    const before = el("div", { class: "fx-before" });
    const expl = el("p", { class: "fx-expl" });
    const fullLine = el("p", { class: "fx-full muted" });
    root.appendChild(toggles);
    root.appendChild(el("figure", { class: "fx-figure" }, big, before, strip.root, el("figcaption", { class: "sr-only", text: "Each offer keeps its position on the full 2,277-offer scale; offers a filter drops stay visible as hatching." })));
    root.appendChild(expl);
    const styleHost = el("div", { class: "fx-styles", id: "fxStyles", hidden: true });
    const styleBtn = el("button", { class: "link-btn", "aria-expanded": "false", "aria-controls": "fxStyles", onclick: () => toggleStyles(styleBtn, styleHost) }, "Who these players are: the style map");
    root.appendChild(el("div", { class: "fx-meta" }, fullLine, styleBtn));
    root.appendChild(styleHost);
    root.appendChild(el("footer", { class: "scene-foot" },
      el("span", { class: "mono muted", text: `Takara · shards 35 and 36 · revision ${(M.exhibit.scan.revision || "").slice(0, 10)} · generated ${ex.generated}` }),
      el("span", { class: "foot-links" }, el("button", { class: "link-btn", onclick: () => go("reveal", { focus: true }) }, "Back to the hands"), evLink("Evidence: the full table", "filter"), nextLink("Next: you defend", "drill"))));
    function setFilter(k) {
      M.filter = k;
      $$(".seg-btn", toggles).forEach((b) => { const on = b.dataset.k === k; b.classList.toggle("on", on); b.setAttribute("aria-pressed", String(on)); });
      strip.set(k);
      const key = FILTERS.find((f) => f[0] === k)[2];
      const t = two[key], f = full[key], a = two.all_eligible;
      const num = $(".fx-num", big), sub = $(".fx-sub", big);
      num.textContent = `${t.folds.toLocaleString()} of ${t.opportunities.toLocaleString()}`;
      sub.textContent = `folded · ${t.opportunities ? pct(t.folds / t.opportunities, 0) : "no data"} of ${k === "all" ? "eligible" : "retained"} offers`;
      before.textContent = k === "all" ? "Shards 35 and 36, no filter." : `${a.folds.toLocaleString()} of ${a.opportunities.toLocaleString()} folded before filtering.`;
      V.anim(big, [{ opacity: 0.35 }, { opacity: 1 }], { duration: 160 });
      expl.textContent = FILTER_TEXT[k];
      fullLine.textContent = `All 37 shards, same filter: ${f.folds.toLocaleString()} of ${f.opportunities.toLocaleString()} folded (${f.opportunities ? pct(f.folds / f.opportunities, 1) : "no data"}).`;
      V.announce(`${$(".fx-num", big).textContent} ${sub.textContent}.`);
    }
    setFilter(M.filter);
  }

  async function toggleStyles(btn, host) {
    const open = host.hidden;
    host.hidden = !open;
    btn.setAttribute("aria-expanded", String(open));
    if (!open || host.dataset.built) return;
    host.dataset.built = "1";
    if (M.styles === undefined) { try { M.styles = await api("/api/mirror/styles"); } catch (e) { M.styles = null; } }
    if (!M.styles || !M.styles.present || !window.Evidence) { host.appendChild(el("p", { class: "hint", text: "Style map not generated yet." })); return; }
    const g = M.styles.gate || {};
    host.appendChild(el("p", { class: "hint" }, el("span", { class: "tag warn", text: "description only" }),
      ` A model estimate, not an observation. The type prior ${g.pass ? "passed" : "failed"} its gate, so live forecasts ${g.pass ? "use" : "ignore"} player types.`));
    host.appendChild(window.Evidence.styleMap(M.styles));
    host.appendChild(evLink("Evidence: are player types worth using?", "types"));
  }

  // ------------------------------------------------------------ heads-up table with stable card slots
  function makeHU() {
    const mkSeat = (pos) => {
      const name = el("div", { class: "hu-name" }), meta = el("div", { class: "hu-meta" }), badge = el("div", { class: "hu-badge-slot" });
      const slots = [el("div", { class: "cslot" }), el("div", { class: "cslot" })];
      const label = el("div", { class: "hu-cards-label", hidden: true });
      const say = el("div", { class: "hu-say", hidden: true, "aria-live": "off" });
      const plate = el("div", { class: "hu-plate" }, name, meta, badge);
      const seat = el("div", { class: "hu-seat " + pos }, plate, el("div", { class: "hu-hand" }, el("div", { class: "hu-cards" }, slots), label), say);
      const wager = el("div", { class: "hu-wager " + pos });
      return { seat, plate, name, meta, badge, slots, label, say, wager, amt: 0 };
    };
    const top = mkSeat("top"), bottom = mkSeat("bottom");
    const boardSlots = Array.from({ length: 5 }, () => el("div", { class: "cslot board-slot" }));
    const pot = el("div", { class: "hu-pot num" });
    const note = el("div", { class: "hu-note" });
    const actions = el("div", { class: "hu-actions", role: "group", "aria-label": "Your actions" });
    const center = el("div", { class: "hu-center", hidden: true });
    const felt = el("div", { class: "hu-felt" }, top.seat, top.wager, el("div", { class: "hu-mid" }, el("div", { class: "hu-board", role: "group" }, boardSlots), pot, note), bottom.wager, bottom.seat, center);
    const root = el("div", { class: "hu" }, felt, actions);
    let key = null;

    function setSlot(slot, code, fresh, delay) {
      code = code || "";
      const cur = slot.dataset.code || "";
      if (!fresh && cur === code) return;
      slot.innerHTML = "";
      slot.dataset.code = code;
      if (!code) return;
      const c = V.card(code, "md");
      slot.appendChild(c);
      if (fresh) V.anim(c, [{ opacity: 0, transform: "translateY(-10px) scale(.96)" }, { opacity: 1, transform: "none" }], { duration: 180, delay });
      else if (cur === "XX" && code !== "XX") V.flipIn(c);
    }
    function setWager(s, amt, fresh, delay) {
      const prev = s.amt;
      s.amt = amt;
      if (amt === prev && !fresh) return;
      s.wager.innerHTML = "";
      if (!amt) return;
      const n = Math.min(5, 1 + Math.floor(Math.log2(amt / 10 + 1)));
      const stack = el("div", { class: "chips" }, Array.from({ length: n }, () => el("div", { class: "chip c25" })));
      const w = el("div", { class: "hu-chip" }, stack, el("span", { class: "num", text: fmt(amt) }));
      s.wager.appendChild(w);
      const from = s.plate.getBoundingClientRect();
      if (from.width) V.flyFrom(w, { left: from.left + from.width / 2 - 20, top: from.top }, { duration: 200, delay: delay || 0 });
    }
    function update(v) {
      const fresh = v.key !== key;
      key = v.key;
      const base = v.dealDelay || 0;
      [["top", top], ["bottom", bottom]].forEach(([pos, s]) => {
        const sv = v[pos] || {};
        s.name.textContent = sv.name || "";
        s.meta.textContent = sv.meta || "";
        if (sv.badge !== undefined) { s.badge.innerHTML = ""; if (sv.badge) s.badge.appendChild(sv.badge); }
        s.seat.classList.toggle("actor", v.actor === pos);
        s.seat.classList.toggle("folded", !!sv.folded);
        s.seat.classList.toggle("you", !!sv.you);
        const cards = sv.cards || [];
        s.slots.forEach((slot, i) => setSlot(slot, cards[i], fresh, base + 260));
        s.slots.forEach((slot) => slot.setAttribute("aria-label", ""));
        s.label.hidden = !sv.cardsLabel;
        s.label.textContent = sv.cardsLabel || "";
        const sayText = sv.say ? sv.say.text : "";
        if (s.say.textContent !== sayText) {
          s.say.textContent = sayText;
          if (sayText && !fresh) V.anim(s.say, [{ opacity: 0, transform: "scale(.9)" }, { opacity: 1, transform: "none" }], { duration: 160 });
        }
        s.say.hidden = !sayText;
        s.say.className = "hu-say " + (sv.say && sv.say.kind ? sv.say.kind : "");
        setWager(s, (v.wager || {})[pos] || 0, fresh, fresh ? base + 340 : 0);
      });
      (v.board || []).forEach((c, i) => setSlot(boardSlots[i], c, fresh, base + 45 * i));
      $(".hu-board", felt).setAttribute("aria-label", "Board: " + V.cardsName(v.board || []));
      pot.textContent = v.pot !== undefined && v.pot !== null ? `Pot ${fmt(v.pot)}` : "";
      note.textContent = v.note || "";
      note.className = "hu-note " + (v.noteKind || "");
      center.hidden = !v.center;
      center.innerHTML = "";
      if (v.center) center.appendChild(v.center);
      actions.innerHTML = "";
      (v.actions || []).forEach((a) => { if (a) actions.appendChild(a); });
      return fresh;
    }
    return { root, update, top, bottom, actions };
  }
  function actBtn(label, k, cls, onclick, hint) {
    return el("button", { class: "btn large " + (cls || ""), "data-key": k || "", onclick }, label, k ? el("kbd", { text: k.toUpperCase() }) : null, hint ? el("span", { class: "sr-only", text: hint }) : null);
  }
  function lockActions(on) { $$("#mScene .hu-actions button").forEach((b) => { b.disabled = on; }); }
  function historyCard(story, riverLine, ctx) {
    return el("section", { class: "side-card history", "aria-label": "Hand history" }, el("h3", { text: "How the hand got here" }),
      el("ol", { class: "story" }, ...story.map((s) => {
        const [street, ...rest] = s.split(":");
        return el("li", { class: street.trim() === "turn" ? "turn" : "" }, el("span", { class: "st mono", text: street.trim() }), el("span", { text: rest.join(":").trim() }));
      }), el("li", { class: "river" }, el("span", { class: "st mono", text: "river" }), el("span", { text: riverLine }))),
      el("p", { class: "hint", text: "Turn history: " + ctx + "." }));
  }
  function ensureStage(root) {
    if (!M.hu) {
      M.hu = makeHU();
      $(".ps-stage", root).appendChild(M.hu.root);
    }
    return M.hu;
  }
  function playLayout(cls, head) {
    return el("section", { class: "scene play-scene " + cls }, head, el("div", { class: "ps-grid" }, el("div", { class: "ps-stage" }), el("aside", { class: "ps-side" })));
  }

  // ------------------------------------------------------------ 3 · you defend
  async function sceneDrill() {
    const { root } = mount("drill", () => playLayout("drill", sceneHead("Reverse the roles", ["Now the model predicts ", el("em", { text: "you." })],
      "You checked the river and face a half-pot bet. The forecast of your response is committed before you click.")));
    const st = M.state || await refresh();
    const hu = ensureStage(root);
    const side = $(".ps-side", root);
    side.innerHTML = "";
    const d = st.drill.pending, last = M.lastDrill;
    const ho = heldout();
    if (d) {
      hu.update({
        key: "drill:" + d.decision_id, actor: "bottom",
        top: { name: "Bettor", meta: "acts first on the river", cards: ["XX", "XX"], say: { text: `Bets ${fmt(d.bet)}`, kind: "aggr" }, badge: null },
        bottom: { name: "You", meta: "defender · checked", cards: d.own_cards, you: true, badge: null },
        board: d.board, pot: d.pot_before_bet, wager: { top: d.bet, bottom: 0 },
        note: `${fmt(d.bet)} to call into ${fmt(d.pot_before_bet)}`,
        actions: [
          d.legal.fold ? actBtn("Fold", "f", "danger", () => respond(d.decision_id, "fold")) : null,
          d.legal.call ? actBtn(`Call ${fmt(st.drill.to_call || d.bet)}`, "c", "", () => respond(d.decision_id, "call")) : null,
          d.legal.raise ? actBtn(`Raise to ${fmt(st.drill.raise_to)}`, "r", "", () => respond(d.decision_id, "raise")) : null,
        ],
      });
      side.appendChild(forecastCard(d.prediction, null, null, ho));
      side.appendChild(historyCard(d.extra.story, `you check, the bettor bets ${fmt(d.bet)} into ${fmt(d.pot_before_bet)}`, d.extra.context_label));
      side.appendChild(profileCard(st, false));
    } else if (last) {
      const dec = last.decision, out = last.outcome || {}, a = dec.actual;
      const sayMe = a === "fold" ? { text: "Folds", kind: "fold" } : a === "call" ? { text: `Calls ${fmt(dec.bet)}`, kind: "" } : { text: `Raises to ${fmt(st.drill.raise_to || 100)}`, kind: "aggr" };
      hu.update({
        key: "drill:" + dec.decision_id, actor: null,
        top: { name: "Bettor", meta: "acted first on the river", cards: out.bettor_cards || ["XX", "XX"], cardsLabel: out.bettor_cards ? "shown after the hand" : "", say: { text: `Bet ${fmt(dec.bet)}`, kind: "aggr" } },
        bottom: { name: "You", meta: `your hand: ${out.your_bucket || "?"}`, cards: dec.own_cards, you: true, folded: a === "fold", say: sayMe },
        board: dec.board, pot: dec.pot_before_bet, wager: { top: dec.bet, bottom: a === "call" ? dec.bet : a === "raise" ? (st.drill.raise_to || 100) : 0 },
        note: out.result ? cap(out.result) : "", noteKind: "result",
        actions: [actBtn("Next spot", "n", "primary", newDrill), freezeButton(st, "Freeze my twin")],
      });
      side.appendChild(forecastCard(dec.prediction, last.after, a, ho, !last.shown));
      side.appendChild(outcomeCard(dec, out, ho));
      side.appendChild(profileCard(st, true));
      last.shown = true;
    } else {
      hu.update({
        key: "drill:empty", actor: null, top: { name: "Bettor", meta: "", cards: [] }, bottom: { name: "You", meta: "defender", cards: [], you: true },
        board: [], pot: null, wager: {}, note: "",
        center: el("div", { class: "hu-intro" }, el("p", { text: `Each spot deals you a hand on a finished board. You check, the bettor bets half the pot. Fold, call or raise — the model has already written down what it thinks you'll do.` })),
        actions: [actBtn(st.drill.drills ? "Deal another spot" : "Deal a spot", "n", "primary", newDrill)],
      });
      side.appendChild(el("section", { class: "side-card" }, el("h3", { text: "What the model knows about you" }),
        el("p", { class: "hint", text: st.profile.observations ? `${plural(st.profile.observations, "response")} so far. Each new response updates the forecast for the next similar spot.` : "Nothing yet. Its first forecast is the population starting point: how 2009 players responded in the same spot." })));
      side.appendChild(profileCard(st, false));
    }
  }
  const cap = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);
  function forecastCard(pred, after, actual, ho, animate) {
    const card = el("section", { class: "side-card forecast", "aria-label": "Model forecast" });
    if (ho) {
      card.appendChild(el("h3", { text: actual ? "Response recorded" : "Forecast recorded" }));
      card.appendChild(el("p", { class: "recorded" }, el("span", { class: "lock", "aria-hidden": "true", text: "●" }),
        actual ? " Your answer and the model's sealed forecast are both stored." : " The model has committed its forecast. It stays sealed during held-out sessions."));
      card.appendChild(el("p", { class: "hint", text: "Held-out forecasts are scored later in Evidence, never shown to the judge." }));
      return card;
    }
    const zero = !pred.observations;
    card.appendChild(el("div", { class: "fc-head" }, el("h3", { text: "Committed forecast" }),
      el("span", { class: "tag " + (zero ? "" : "good"), text: zero ? "Population starting point" : `After ${plural(pred.observations, "of your responses")}` })));
    const scale = V.forecastScale(pred, { title: "Chance you fold", beforeLabel: "Before your action" });
    card.appendChild(scale.root);
    card.appendChild(V.responseSplit(pred.probs, actual));
    if (after) {
      scale.showNext(after, animate);
      card.appendChild(el("p", { class: "fc-next" }, "Next estimate for this turn history: ", el("b", { class: "num", text: pct(after.fold_mean, 0) }),
        el("span", { class: "muted", text: ` (90% band ${pct(after.fold_p05, 0)}–${pct(after.fold_p95, 0)}, after ${plural(after.observations, "response")})` })));
    }
    card.appendChild(el("p", { class: "src mono", text: `${pred.arm} model · ${pred.model_version}` }));
    return card;
  }
  function outcomeCard(dec, out, ho) {
    const a = dec.actual;
    return el("section", { class: "side-card outcome" }, el("h3", { text: "What happened" }),
      el("p", {}, el("b", { text: `You ${a === "fold" ? "folded" : a === "call" ? "called" : "raised"}.` }), " ",
        out.result ? cap(out.result) + ". " : "",
        out.bettor_cards ? el("span", {}, "The bettor had ", P.cardsInline(out.bettor_cards), ". ") : null),
      out.your_bucket ? el("p", { class: "hint", text: `Your hand counted as ${out.your_bucket} (${pct(out.your_equity, 0)} equity against a random hand).` }) : null,
      ho ? null : evLink("Evidence: how well these forecasts score", "forecast"));
  }
  function freezeButton(st, label) {
    const n = st.profile.observations;
    return el("button", { class: "btn large amber freeze-btn", "data-key": "z", onclick: (e) => freezeAndPlay(e.currentTarget) }, label, el("span", { class: "freeze-n", text: ` · ${plural(n, "response")}` }));
  }
  function profileCard(st, responded) {
    const prof = st.profile, n = prof.observations;
    const card = el("section", { class: "side-card profile", "aria-label": "Your profile" });
    const chip = el("div", { class: "twin-chip" }, el("span", { class: "tc-mark", "aria-hidden": "true", text: "♠" }), el("span", { text: "Your profile" }), el("b", { class: "num", text: plural(n, "response") }));
    card.appendChild(el("div", { class: "row between" }, chip, responded ? null : el("button", { class: "btn small", onclick: (e) => freezeAndPlay(chip) }, st.frozen ? "Re-freeze twin" : "Freeze my twin")));
    if (!heldout()) {
      const rows = el("div", { class: "pf-rows" });
      Object.entries(prof.contexts).forEach(([c, v]) => {
        const pop = prof.population[c];
        rows.appendChild(el("div", { class: "pf-row" }, el("span", { class: "pf-k", text: CTX_SHORT[c] }),
          el("span", { class: "pf-v" }, "fold ", el("b", { class: "num", text: pct(v.probs.fold, 0) }), el("span", { class: "muted", text: ` · population ${pct(pop.fold, 0)} · ${v.observations} seen` }))));
      });
      card.appendChild(rows);
    }
    card.appendChild(el("p", { class: "hint", text: st.frozen ? `Twin frozen as #${st.frozen.slice(0, 7)}. Freezing again replaces it.` : "Freezing copies this profile into an opponent you can bet against." }));
    return card;
  }
  async function newDrill() {
    if (M.busy) return;
    M.busy = true; lockActions(true);
    try { const r = await api("/api/mirror/drill", body()); remember(r.token); M.lastDrill = null; await refresh(); await sceneDrill(); focusAction(); }
    catch (e) { toast(e.message, true); lockActions(false); }
    finally { M.busy = false; }
  }
  async function respond(id, action) {
    if (M.busy) return;
    M.busy = true; lockActions(true);
    try {
      const r = await api("/api/mirror/respond", body({ decision_id: id, action }));
      M.lastDrill = r;
      await refresh();
      await sceneDrill();
      const out = r.outcome || {};
      V.announce(`You ${action === "fold" ? "folded" : action === "call" ? "called" : "raised"}. ${out.result ? cap(out.result) + "." : ""}` +
        (heldout() ? "" : ` The forecast before your action was fold ${pct(r.decision.prediction.fold_mean, 0)}; next estimate ${pct(r.after.fold_mean, 0)}.`));
      focusAction();
    } catch (e) {
      // A repeated click or an interrupted request: trust the server's record of this decision before allowing another action.
      try { await refresh(); } catch (e2) { /* keep the error below */ }
      const p = M.state && M.state.drill.pending;
      if (!p || p.decision_id !== id) toast("That spot was already answered. Showing the latest state.");
      else toast(e.message, true);
      await sceneDrill();
    } finally { M.busy = false; }
  }
  function focusAction() { const b = $("#mScene .hu-actions button:not([disabled])"); if (b) b.focus({ preventScroll: true }); }

  // ------------------------------------------------------------ freeze: the profile marker moves into the opponent's nameplate
  async function freezeAndPlay(srcEl) {
    if (M.busy) return;
    M.busy = true;
    lockActions(true);
    const marker = $("#mScene .twin-chip") || srcEl;
    const from = marker ? marker.getBoundingClientRect() : null;
    const ghost = marker && from && from.width && !V.reduceMotion() ? marker.cloneNode(true) : null;
    try {
      const r = await api("/api/mirror/freeze", body());
      store.set("mirrorFrozen", JSON.stringify({ hash: r.frozen.hash, observations: r.frozen.observations }));
      const s = await api("/api/mirror/play/new", body());
      remember(s.token);
      await refresh();
      M.flyFrom = ghost ? { ghost, from } : null;
      M.busy = false;
      await go("play", { focus: true });
      V.announce(`Twin frozen from ${plural(r.frozen.observations, "response")}. You are now the bettor.`);
      setTimeout(focusAction, M.flyFrom === null ? 420 : 0);
    } catch (e) { toast(e.message, true); M.busy = false; lockActions(false); }
  }
  function twinBadge(info) {
    return el("div", { class: "twin-badge" }, el("span", { class: "tc-mark", "aria-hidden": "true", text: "♠" }),
      el("span", { class: "tb-text" }, el("b", { text: "Frozen twin" }), el("span", { class: "mono", text: `${info.observations === null ? "" : plural(info.observations, "response") + " · "}#${info.hash.slice(0, 7)}` })));
  }
  function runFly() {
    const f = M.flyFrom;
    M.flyFrom = null;
    const target = $("#mScene .hu-seat.top .twin-badge");
    if (!f || !target) return;
    const to = target.getBoundingClientRect();
    const g = f.ghost;
    g.classList.add("fly-ghost");
    Object.assign(g.style, { left: f.from.left + "px", top: f.from.top + "px", width: f.from.width + "px", height: f.from.height + "px" });
    document.body.appendChild(g);
    target.style.visibility = "hidden";
    const a = V.anim(g, [{ transform: "none", opacity: 1 }, { transform: `translate(${to.left - f.from.left}px, ${to.top - f.from.top}px)`, opacity: 1 }], { duration: 380, fill: "forwards" });
    const done = () => { g.remove(); target.style.visibility = ""; V.anim(target, [{ boxShadow: "0 0 0 3px rgba(255,213,140,.7)" }, { boxShadow: "0 0 0 0 rgba(255,213,140,0)" }], { duration: 420 }); };
    if (a) a.finished.then(done, done); else done();
  }

  // ------------------------------------------------------------ 4 · you bet against the frozen twin
  async function scenePlay(opts) {
    const st = M.state || await refresh();
    const info = frozenInfo();
    const O = opp();
    await realPlayers();
    const { root } = mount("play", () => playLayout("play", sceneHead("", [""], "")));
    if (O.real) setHead(root, "You are the bettor · real player", ["Bet into ", el("em", { text: "a real player's twin." })],
      `The defender answers the way player ${O.o.id} answered ${O.o.decisions} recorded river bets (${O.o.venue}). You hold the bettor's cards.`);
    else setHead(root, "You are now the bettor", ["Bet into ", el("em", { text: "your twin." })],
      "The opponent is a frozen copy of the profile you just built. You hold the bettor's cards; the twin defends.");
    const hu = ensureStage(root);
    const side = $(".ps-side", root);
    side.innerHTML = "";
    const v = st.play;
    if (!info) {
      hu.update({ key: "play:none", actor: null, top: { name: "Your twin", meta: "not frozen yet", cards: [], badge: null }, bottom: { name: "You", meta: "bettor", cards: [], you: true }, board: [], pot: null, wager: {},
        center: el("div", { class: "hu-intro" }, el("p", { text: `Freeze your profile first${M.realPlayers.present ? ", or pick a real player from the dataset" : ""}. With ${plural(st.profile.observations, "response")} your twin ${st.profile.observations ? "leans on what you did" : "is the population player"}.` })),
        actions: [freezeButton(st, "Freeze my twin")] });
      side.appendChild(profileCard(st, true));
      const pick = pickerCard(st);
      if (pick) side.appendChild(pick);
      return;
    }
    const badge = O.real ? realBadge(O.o) : twinBadge(info.observations === null && v.advice ? { hash: info.hash, observations: v.advice.forecast.observations } : info);
    if (!v.spot) {
      hu.update({ key: "play:idle:" + info.hash, actor: null, top: { name: O.name, meta: "defender", cards: [], badge }, bottom: { name: "You", meta: "bettor", cards: [], you: true }, board: [], pot: null, wager: {},
        center: el("div", { class: "hu-intro" }, el("p", { text: `Each spot deals a fresh river. You can check back or bet half the pot; ${O.the} answers the way ${O.real ? "that player's record" : "your profile"} says it would.` })),
        actions: [actBtn("Deal a betting spot", "n", "primary", () => newSpot(false))] });
      side.appendChild(twinCard(st, info));
      if (M.flyFrom) runFly();
      return;
    }
    const s = v.spot, done = !!v.result, res = v.result;
    const resp = v.response;
    const sayTwin = resp === "fold" ? { text: "Folds", kind: "fold" } : resp === "call" ? { text: "Calls 50", kind: "" } : resp === "raise" ? { text: `Raises to ${fmt(s.raise_to)}`, kind: "aggr" } : (s.stage === 1 ? { text: "Checks", kind: "" } : null);
    const myActs = v.actions || [];
    const sayMe = myActs.length ? (myActs[myActs.length - 1] === "bet" ? { text: `Bet ${fmt(s.bet)}`, kind: "aggr" } : myActs[myActs.length - 1] === "check" ? { text: "Checks back", kind: "" } : myActs[myActs.length - 1] === "fold" ? { text: "Folds", kind: "fold" } : { text: "Calls", kind: "" }) : null;
    const betMe = myActs.includes("bet") ? s.bet + (myActs.includes("call") ? s.raise_to - s.bet : 0) : 0;
    const betTwin = resp === "call" ? s.bet : resp === "raise" ? s.raise_to : 0;
    let actions;
    if (done) actions = [actBtn("New spot", "n", "primary", () => newSpot(false)), el("button", { class: "btn large", onclick: () => go("compare", { focus: true }) }, "Compare two bettors on this deal →")];
    else if (s.stage === 1) actions = [actBtn("Check back", "k", "", () => act("check")), actBtn(`Bet ${fmt(s.bet)}`, "b", "", () => act("bet"))];
    else if (s.stage === 2) actions = [actBtn("Fold", "f", "danger", () => act("fold")), actBtn(`Call ${fmt(s.raise_to - s.bet)}`, "c", "", () => act("call"))];
    const fresh = hu.update({
      key: "play:" + s.board.join("") + s.own_cards.join(""), actor: done ? null : "bottom", dealDelay: M.flyFrom ? 380 : 0,
      top: { name: O.name, meta: "defender · checked", cards: done ? res.defender_cards : ["XX", "XX"], cardsLabel: done ? "shown after the hand" : "", badge, say: sayTwin, folded: resp === "fold" },
      bottom: { name: "You", meta: `bettor · ${s.bettor_bucket} hand`, cards: s.own_cards, you: true, say: sayMe, folded: myActs.includes("fold") },
      board: s.board, pot: s.pot, wager: { top: done ? betTwin : betTwin, bottom: betMe },
      note: done ? `${res.reward_chips >= 0 ? "+" : "−"}${fmt(Math.abs(res.reward_chips))} chips` : s.stage === 2 ? `${O.The} raised to ${fmt(s.raise_to)}` : `Pot ${fmt(s.pot)} · ${ctxBettor(s.context) || s.context_label}`,
      noteKind: done ? (res.reward_chips >= 0 ? "result win" : "result loss") : "",
      actions,
    });
    if (fresh) M.rangeOpen = false;
    if (v.advice && s.stage === 1) side.appendChild(adviceCard(v.advice));
    if (v.readout) side.appendChild(readoutCard(v.readout, resp, res));
    else if (done) side.appendChild(el("section", { class: "side-card" }, el("h3", { text: "No response to read" }), el("p", { class: "hint", text: `You checked back, so ${O.the} never had to answer a bet. Its range stays at the prior.` })));
    side.appendChild(historyCard(s.story, done ? "you act after the twin checks" : "the twin checks to you", ctxBettor(s.context) || s.context_label));
    side.appendChild(twinCard(st, info, v.log));
    if (M.flyFrom) runFly();
    else if (opts && opts.announce) V.announce(opts.announce);
  }
  function realBadge(o) {
    return el("div", { class: "twin-badge real" }, el("span", { class: "tc-mark", "aria-hidden": "true", text: "♠" }),
      el("span", { class: "tb-text" }, el("b", { text: "Real player's twin" }), el("span", { class: "mono", text: `${o.decisions} answers · folds ${pct(o.fold_rate, 0)}` })));
  }
  function twinCard(st, info, log) {
    const O = opp();
    const card = O.real
      ? el("section", { class: "side-card twin" }, el("h3", { text: "Opponent: a real player's twin" }),
        el("p", {}, el("span", { class: "mono", text: O.o.id }), ` · ${O.o.venue} · ${O.o.decisions} recorded river answers · folds ${pct(O.o.fold_rate, 0)}.`),
        el("p", { class: "hint", text: "Built from that player's recorded answers on every table size, so no drills are needed. Its cards still come from a strength model: a revealed hand and an estimated range carry different labels." }))
      : el("section", { class: "side-card twin" }, el("h3", { text: "Opponent: your frozen twin" }),
        el("p", {}, `Snapshot `, el("span", { class: "mono", text: "#" + info.hash.slice(0, 12) }), info.observations === null ? "" : ` from ${plural(info.observations, "response")}.`),
        el("p", { class: "hint", text: "It answers bets the way your profile predicted. Its cards come from a strength model, so a revealed hand and an estimated range carry different labels." }));
    const pick = pickerControls(st);
    if (pick) card.appendChild(pick);
    if (log && log.length) {
      const total = log.reduce((a, x) => a + x.reward_chips, 0);
      card.appendChild(el("details", { class: "twin-log" }, el("summary", {}, `${plural(log.length, "spot")} played · `, el("b", { class: "num", text: `${total >= 0 ? "+" : "−"}${fmt(Math.abs(total))} chips` })),
        el("ol", {}, log.slice().reverse().map((x) => el("li", {}, P.cardsInline(x.own), el("span", { class: "muted", text: ` ${x.actions.join(", ")}${x.response ? " → " + x.response : ""} ` }), el("b", { class: "num", text: `${x.reward_chips >= 0 ? "+" : "−"}${fmt(Math.abs(x.reward_chips))}` }))))));
    }
    return card;
  }
  // Real players from the dataset with enough river answers to build a twin (summaries only; answers stay on the server).
  function pickerControls(st) {
    const rp = M.realPlayers;
    const O = opp();
    if (!rp || !rp.present || !rp.players.length) return null;
    const id = "realPick";
    const sel = el("select", { id, class: "rp-select" }, rp.players.map((pl) => el("option", { value: pl.id, text: `${pl.id} · ${pl.venue} · ${pl.decisions} answers · folds ${pct(pl.fold_rate, 0)}` })));
    if (O.real) sel.value = O.o.id;
    return el("div", { class: "rp-pick" },
      el("label", { for: id, class: "rp-label", text: O.real ? "Switch to another real player" : "Or bet against a real player's twin" }),
      el("div", { class: "rp-row" }, sel, el("button", { class: "btn small", onclick: () => loadReal(sel.value) }, "Play this player")),
      O.real ? el("button", { class: "btn small ghost rp-back", onclick: () => freezeAndPlay(null) }, `Back to my twin (${plural(st.profile.observations, "response")})`) : null,
      el("p", { class: "hint", text: `${rp.eligible_players} players with at least ${rp.min_decisions} answers. ${rp.label}.` }));
  }
  function pickerCard(st) {
    const c = pickerControls(st);
    return c ? el("section", { class: "side-card" }, el("h3", { text: "No drills? Use a real player" }), c) : null;
  }
  async function loadReal(id) {
    if (M.busy || !id) return;
    M.busy = true; lockActions(true);
    try {
      const r = await api("/api/mirror/real/load", body({ player_id: id }));
      remember(r.token);
      store.set("mirrorFrozen", JSON.stringify({ hash: r.frozen.hash, observations: r.frozen.observations }));
      await api("/api/mirror/play/new", body());
      await refresh();
      M.busy = false;
      if (M.step !== "play") await go("play", { focus: true }); else await scenePlay();
      const b = $("#mScene .hu-seat.top .twin-badge");
      if (b) V.anim(b, [{ boxShadow: "0 0 0 3px rgba(255,213,140,.7)" }, { boxShadow: "0 0 0 0 rgba(255,213,140,0)" }], { duration: 420 });
      V.announce(`You are betting against the twin of real player ${id}.`);
      focusAction();
    } catch (e) { toast(e.message, true); lockActions(false); M.busy = false; }
  }
  function adviceCard(a) {
    const f = a.forecast;
    return el("section", { class: "side-card advice", "aria-label": "EV advisor" },
      el("div", { class: "fc-head" }, el("h3", { text: "EV advisor" }), el("span", { class: "tag", text: "opening pot = 100 chips" })),
      V.evCompare(a),
      el("p", { class: "adv-rec" }, el("b", { text: `Recommendation: ${a.recommend === "bet" ? "bet 50" : "check back"}. ` }), a.reason),
      el("p", { class: "hint" }, `${opp().The} folds `, el("b", { class: "num", text: pct(f.fold, 0) }), ` (90% band ${pct(f.interval[0], 0)}–${pct(f.interval[1], 0)}, ${f.arm} arm, ${plural(f.observations, "response")}). Equity against its calling range ${pct(a.equity.vs_call_range, 0)}, raising range ${pct(a.equity.vs_raise_range, 0)}.`),
      el("p", { class: "hint", text: a.pure_bluff ? "A pure bluff breaks even when the twin folds one time in three: 50 / (100 + 50)." : `A pure bluff would break even at ${pct(a.break_even_fold, 0)} folds; this hand also wins some showdowns.` }));
  }
  function readoutCard(r, response, res) {
    const card = el("section", { class: "side-card readout", "aria-label": "Range X-ray" },
      el("div", { class: "fc-head" }, el("h3", { text: `Range X-ray after the twin ${response === "fold" ? "folded" : response === "call" ? "called" : "raised"}` }), el("span", { class: "tag warn", text: "estimate" })),
      V.bucketBars(r.prior_buckets, r.buckets),
      el("p", { class: "hint", text: `A model estimate at ${r.resolution} over ${r.holdings} holdings, from the twin's strength table. Not an observation.` }));
    if (res && res.defender_cards) card.appendChild(el("p", { class: "revealed" }, el("span", { class: "tag", text: "revealed" }), " The twin actually held ", P.cardsInline(res.defender_cards), ` (${res.defender_bucket}).`));
    const det = el("details", { class: "range-more" }, el("summary", { text: "Inspect the 13 × 13 estimated range" }));
    det.open = M.rangeOpen;
    let built = false;
    const build = () => { if (built) return; built = true; det.appendChild(V.rangeGrid(r)); };
    if (det.open) build();
    det.addEventListener("toggle", () => { M.rangeOpen = det.open; if (det.open) build(); });
    card.appendChild(det);
    card.appendChild(evLink("Evidence: checking the range readout", "range"));
    return card;
  }
  async function newSpot(afterFreeze) {
    if (M.busy) return;
    M.busy = true; lockActions(true);
    try { const r = await api("/api/mirror/play/new", body()); remember(r.token); await refresh(); await scenePlay(); if (!afterFreeze) focusAction(); else setTimeout(focusAction, 420); }
    catch (e) { toast(e.message, true); lockActions(false); }
    finally { M.busy = false; }
  }
  async function act(a) {
    if (M.busy) return;
    M.busy = true; lockActions(true);
    try {
      await api("/api/mirror/play/act", body({ action: a }));
      await refresh();
      const v = M.state.play;
      const msg = v.result ? `${v.response ? opp().The + " " + (v.response === "fold" ? "folded" : v.response === "call" ? "called" : "raised") + ". " : ""}${v.result.reward_chips >= 0 ? "You won" : "You lost"} ${fmt(Math.abs(v.result.reward_chips))} chips.` : v.spot && v.spot.stage === 2 ? `${opp().The} raised to 100.` : "";
      await scenePlay({ announce: msg });
      focusAction();
    } catch (e) {
      try { await refresh(); } catch (e2) { /* ignore */ }
      toast(e.message, true);
      await scenePlay();
    } finally { M.busy = false; }
  }

  // ------------------------------------------------------------ 5 · two bettors, same deal
  const NICE = { bet_half: "Bet 50", check_back: "Check back", fold: "Fold", call: "Call" };
  async function sceneCompare() {
    const { root, fresh } = mount("compare", () => el("section", { class: "scene compare" },
      sceneHead("", [""], ""), el("div", { class: "cmp-body" })));
    const O = opp();
    setHead(root, "Same deal, two bettors", ["Does knowing ", el("em", { text: O.real ? "this player" : "your twin" }), " change the bet?"],
      `Two Q-learning bettors, 20,000 simulated rivers each, same seeds and deal distribution. One learned against the population, one against ${O.real ? `the twin of real player ${O.o.id}` : "your frozen twin"}. Values are simulated chips; the opening pot is 100.`);
    const bodyEl = $(".cmp-body", root);
    let c;
    try { c = await api("/api/mirror/compare" + q()); } catch (e) { bodyEl.innerHTML = ""; errorBox(bodyEl, e.message); return; }
    if (!M.evidence) { try { M.evidence = await api("/api/mirror/evidence"); } catch (e) { M.evidence = null; } }
    bodyEl.innerHTML = "";
    const t = c.training || {};
    if (!c.deal) {
      bodyEl.appendChild(el("div", { class: "cmp-empty" }, el("p", { text: `Play a betting spot against ${O.the} first; both bettors are then asked about that exact deal.` }), nextLink("Go to: you bet", "play")));
    } else {
      const d = c.deal;
      bodyEl.appendChild(el("div", { class: "cmp-deal" }, V.board(d.board, "md"),
        el("div", { class: "cmp-hands" },
          el("div", {}, el("span", { class: "muted", text: "Bettor (you) " }), P.cardsInline(d.bettor_cards), el("span", { class: "muted", text: ` ${d.bettor_bucket}` })),
          el("div", {}, el("span", { class: "muted", text: O.real ? "Player's twin " : "Twin " }), P.cardsInline(d.defender_cards), el("span", { class: "muted", text: ` ${d.defender_bucket} · revealed` })),
          el("div", { class: "muted", text: cap(ctxBettor(d.context) || TURN[d.context] || d.context) + "." }))));
      const pop = c.on_deal && c.on_deal.population, per = c.on_deal && c.on_deal.personal;
      const cols = el("div", { class: "cmp-cols" }, botCol("Population bettor", "trained against the 2009 population", pop, null),
        botCol(O.real ? `Bettor trained on ${O.o.id}'s twin` : "Bettor trained on your twin", `twin #${(M.state && M.state.frozen || "").slice(0, 7)}`, per, t), advisorCol(c.advice));
      if (pop && per) {
        const same = pop.action === per.action;
        bodyEl.appendChild(el("div", { class: "cmp-verdict", role: "status" }, el("span", { class: "cv-k", text: same ? "Same action" : "Different actions" }),
          el("span", { text: same ? ` Both bettors choose ${NICE[pop.action].toLowerCase()}. On this deal, knowing your twin does not change the decision — a valid result, not a failure.`
            : ` The population bettor would ${NICE[pop.action].toLowerCase()}; the bettor trained on your twin would ${NICE[per.action].toLowerCase()}.` }),
          per.oracle && !per.oracle.pass ? el("span", { class: "cv-warn" }, el("span", { class: "tag warn", text: "✗ oracle check failed" }),
            ` The twin-trained bettor missed its own best response in ${plural(per.oracle.failures, "cell")}, so its choice is a training miss, not evidence. Use the EV advisor (${c.advice ? (c.advice.recommend === "bet" ? "bet 50" : "check back") : "right"}) for ${O.the}.`) : null));
      } else if (pop) {
        bodyEl.appendChild(el("div", { class: "cmp-verdict pending", role: "status" }, el("span", { class: "cv-k", text: "Personal bettor pending" }), el("span", { text: ` Until it is trained, the EV advisor on the right is the comparison available for ${O.the}.` })));
      }
      bodyEl.appendChild(cols);
    }
    // broader result: the prepared benchmark
    const bm = M.evidence && M.evidence.panels.decision_benchmark;
    if (bm && bm.present) {
      const rows = bm.data.rows.filter((r) => r.probes === 64).map((r) => ({ label: (OPP_LABEL[r.opponent] || [r.opponent])[0], sub: (OPP_LABEL[r.opponent] || [, ""])[1], mean: r.mean_diff_chips, lo: r.ci95_chips[0], hi: r.ci95_chips[1], failed: r.oracle_pass === false, note: "oracle failed: training miss" }));
      bodyEl.appendChild(el("section", { class: "cmp-broad" },
        el("h2", { class: "q-sm", text: "Beyond this deal: when does knowing the opponent help?" }),
        el("p", { class: "hint", text: `Change in chips per spot when the bettor trains on 64 probes of a simulated opponent instead of the population, ${bm.data.pairs.toLocaleString()} matched deals per row, 95% intervals.` }),
        V.intervalPlot(rows, { xTitle: "Δ chips per spot vs the population bettor (opening pot = 100)", unit: "chips per spot", aria: "Benchmark at 64 probes", width: Math.min(900, bodyEl.clientWidth || 820), labelW: 200, valueW: 190 }),
        el("p", { class: "src mono", text: `${bm.source} · ${bm.data.label} · generated ${bm.generated}` }),
        evLink("Evidence: the full benchmark, all probe counts", "benchmark")));
    }
    const charts = [["Population bettor", c.population_chart, c.population_meta], ["Bettor trained on your twin", c.personal_chart, c.personal_meta]].filter((x) => x[1]);
    if (charts.length) {
      const det = el("details", { class: "cmp-charts" }, el("summary", { text: "Policy charts: what each bettor does with every hand strength and turn history" }));
      charts.forEach(([name, chart, meta]) => det.appendChild(policyChart(name, chart, meta)));
      bodyEl.appendChild(det);
    }
    if (t.status === "running" && M.step === "compare") M.pollTimer = setTimeout(() => { if (M.step === "compare" && P.state.tab === "mirror") sceneCompare(); }, 1500);
    if (fresh) { /* first paint only */ }
  }
  function botCol(title, sub, b, training) {
    const col = el("article", { class: "cmp-col" + (b ? "" : " pending") }, el("h3", { text: title }), el("p", { class: "muted cmp-sub", text: sub }));
    if (b) {
      col.appendChild(el("div", { class: "cmp-action", text: NICE[b.action] || b.action }));
      col.appendChild(qScale(b.q_chips));
      col.appendChild(el("dl", { class: "cmp-meta" },
        el("dt", { text: "training visits" }), el("dd", { class: "num", text: `${b.visits[0].toLocaleString()} check / ${b.visits[1].toLocaleString()} bet` }),
        el("dt", { text: "oracle check" }), el("dd", {}, b.oracle ? el("span", { class: "tag " + (b.oracle.pass ? "good" : "warn"), text: b.oracle.pass ? "✓ passed" : `✗ ${b.oracle.failures} cell(s) off` }) : "n/a"),
        el("dt", { text: "episodes" }), el("dd", { class: "num", text: (b.episodes || 0).toLocaleString() })));
    } else {
      const t = training || {};
      const running = t.status === "running";
      col.appendChild(el("div", { class: "cmp-action muted", text: running ? "Training…" : t.status === "error" ? "Training failed" : "Not trained yet" }));
      if (running) col.appendChild(el("div", { class: "progress", role: "progressbar", "aria-valuemin": 0, "aria-valuemax": t.episodes, "aria-valuenow": t.progress }, el("div", { style: `width:${(100 * (t.progress || 0) / (t.episodes || 1)).toFixed(1)}%` })));
      col.appendChild(el("p", { class: "hint", text: running ? `${(t.progress || 0).toLocaleString()} of ${(t.episodes || 0).toLocaleString()} episodes against your twin.` : t.status === "error" ? t.error : "About 90 seconds on this laptop. Uses the same seed and deals as the population bettor." }));
      if (!running) col.appendChild(el("button", { class: "btn primary", onclick: startTraining }, "Train against my twin"));
    }
    return col;
  }
  function qScale(q) {
    return V.evCompare({ ev_check: q[0] / 100, ev_bet: q[1] / 100, ev_bet_band: null, recommend: q[1] > q[0] ? "bet" : "check" });
  }
  function advisorCol(a) {
    const col = el("article", { class: "cmp-col advisor" }, el("h3", { text: "EV advisor" }), el("p", { class: "muted cmp-sub", text: "closed form, from the twin's forecast" }));
    if (!a) { col.appendChild(el("p", { class: "hint", text: "No advice for this deal." })); return col; }
    col.appendChild(el("div", { class: "cmp-action", text: a.recommend === "bet" ? "Bet 50" : "Check back" }));
    col.appendChild(V.evCompare(a));
    col.appendChild(el("p", { class: "hint", text: a.reason }));
    return col;
  }
  function policyChart(name, chart, meta) {
    const tbl = el("table", { class: "matrix policy" }, el("caption", { text: `${name}: first decision (check back or bet), Q values in chips, training visits` }),
      el("tr", {}, el("th", { text: "turn ↓ strength →" }), ...["weak", "medium", "strong"].map((b) => el("th", { text: b }))));
    Object.entries(chart["1"]).forEach(([ctx, row]) => {
      const tr = el("tr", {}, el("th", { text: CTX_SHORT[ctx] || ctx }));
      ["weak", "medium", "strong"].forEach((b) => { const cell = row[b]; tr.appendChild(el("td", { class: cell.action === "bet_half" ? "pc-bet" : "" }, el("b", { text: cell.action === "bet_half" ? "bet" : "check" }), el("br"), el("span", { class: "num muted", text: cell.q.map((x) => Math.round(x * 100)).join(" / ") }), el("br"), el("span", { class: "num muted", text: cell.visits.join(" + ") }))); });
      tbl.appendChild(tr);
    });
    return el("div", { class: "policy-wrap" }, tbl, meta ? el("p", { class: "src mono", text: `${meta.episodes.toLocaleString()} episodes · seed ${meta.seed} · ${meta.train_seconds}s · oracle ${meta.oracle_check ? (meta.oracle_check.pass ? "pass" : meta.oracle_check.failures + " cell(s) off") : "n/a"}` }) : null);
  }
  async function startTraining() { try { await api("/api/mirror/train", body({ episodes: 20000 })); sceneCompare(); } catch (e) { toast(e.message, true); } }

  // ------------------------------------------------------------ returning judge
  function resumeChoice(st) {
    const host = $("#mScene");
    host.innerHTML = "";
    M.sceneId = null;
    const saved = store.get("mirrorStep") || "drill";
    const stepLabel = (STEPS.find((s) => s.id === saved) || STEPS[2]).label;
    const facts = [plural(st.profile.observations, "response"), plural(st.drill.drills, "spot")];
    if (st.frozen) facts.push("twin frozen");
    const card = el("section", { class: "scene resume", role: "dialog", "aria-modal": "false", "aria-labelledby": "resumeH" },
      el("div", { class: "scene-eyebrow mono", text: "judge " + st.token.slice(0, 6) }),
      el("h1", { class: "q", id: "resumeH", tabindex: "-1" }, "Welcome ", el("em", { text: "back." })),
      el("p", { class: "lede", text: `This laptop still holds a judge with ${facts.join(", ")}. Carry on with them, or start clean for someone new.` }),
      el("div", { class: "row resume-actions" },
        el("button", { class: "btn large primary", onclick: () => { M.resumeChecked = true; go(saved, { focus: true }); } }, `Resume at “${stepLabel}”`),
        el("button", { class: "btn large", onclick: () => { M.resumeChecked = true; resetJudge(); } }, "Start fresh as a new judge")));
    host.appendChild(card);
    $$(".m-step").forEach((b) => b.classList.remove("active"));
    $("#resumeH").focus({ preventScroll: true });
  }

  // ------------------------------------------------------------ keyboard: F/C/R, K/B, N, Z act on the visible scene's buttons
  document.addEventListener("keydown", (e) => {
    if (P.state.tab !== "mirror") return;
    if (e.key === "Escape" && !$("#mOperator").hidden) { toggleOperator(); $("#mOpBtn").focus(); return; }
    if (e.metaKey || e.ctrlKey || e.altKey || e.target.matches("input, select, textarea, [contenteditable]")) return;
    const k = e.key.toLowerCase();
    if (k.length !== 1) return;
    const b = $(`#mScene .hu-actions button[data-key="${k}"]:not([disabled])`);
    if (b) { e.preventDefault(); b.click(); }
  });

  // ------------------------------------------------------------ entry
  async function onShow() {
    if (!M.built) buildShell();
    try { await refresh(); } catch (e) { $("#mScene").innerHTML = ""; errorBox($("#mScene"), "Mirror backend unavailable: " + e.message); return; }
    const params = new URLSearchParams(location.hash.replace(/^#/, ""));
    if (params.has("mirror") && !M.deepLinked) {
      // deep links for demos and screenshots: #mirror=<step>&deal=1 | &respond=<fold|call|raise> | &spot=1&act=<check|bet> | &real=<id>&act=<a> | &reveal=1
      M.deepLinked = true;
      M.resumeChecked = true;
      const step = params.get("mirror") || "reveal";
      if (params.has("reveal")) { M.guess = 0; }
      if (params.has("deal") || params.has("respond")) {
        await go("drill");
        await newDrill();
        if (params.has("respond") && M.state.drill.pending) await respond(M.state.drill.pending.decision_id, params.get("respond"));
        return;
      }
      if (params.has("real")) {
        const rp = await realPlayers();
        const id = params.get("real") || (rp.players[0] || {}).id;
        await go("play");
        if (id) await loadReal(id);
        if (params.has("act")) await act(params.get("act"));
        return;
      }
      if (params.has("spot")) {
        const r = await api("/api/mirror/freeze", body());
        store.set("mirrorFrozen", JSON.stringify({ hash: r.frozen.hash, observations: r.frozen.observations }));
        await refresh();
        await go("play");
        await newSpot(false);
        if (params.has("act")) await act(params.get("act"));
        return;
      }
      await go(step, { focus: true });
      return;
    }
    const st = M.state;
    const returning = st.profile.observations > 0 || st.drill.drills > 0 || !!st.frozen;
    if (returning && !M.resumeChecked) { resumeChoice(st); return; }
    M.resumeChecked = true;
    await go(M.step);
  }
  function onHide() { clearTimeout(M.pollTimer); }
  window.Mirror = { onShow, onHide, state: M };
})();
