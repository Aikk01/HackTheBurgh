/* PokerSim front end: table renderer + play / replay / evaluate / simulate panes. */
(function () {
  "use strict";

  // ------------------------------------------------------------ utilities
  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    if (attrs) {
      for (const [k, v] of Object.entries(attrs)) {
        if (v === null || v === undefined || v === false) continue;
        if (k === "class") node.className = v;
        else if (k === "style") node.style.cssText = v;
        else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
        else if (k === "html") node.innerHTML = v;
        else if (k === "text") node.textContent = v;
        else node.setAttribute(k, v === true ? "" : v);
      }
    }
    for (const c of children.flat()) {
      if (c === null || c === undefined || c === false) continue;
      node.appendChild(typeof c === "string" || typeof c === "number" ? document.createTextNode(String(c)) : c);
    }
    return node;
  }

  function fmt(x, opts) {
    if (x === null || x === undefined) return "∞";
    const v = Number(x);
    if (!Number.isFinite(v)) return "∞";
    if (Number.isInteger(v)) return v.toLocaleString();
    return v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2, ...(opts || {}) });
  }
  const pct = (x, d = 1) => (x === null || x === undefined) ? "-" : (100 * x).toFixed(d) + "%";

  let toastTimer = null;
  function toast(msg, isError) {
    const t = $("#toast");
    t.textContent = msg;
    t.classList.toggle("err", !!isError);
    t.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { t.hidden = true; }, isError ? 5000 : 2600);
  }

  async function api(path, body, method) {
    const opts = { method: method || (body !== undefined ? "POST" : "GET"), headers: {} };
    if (body !== undefined) { opts.headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(body); }
    const res = await fetch(path, opts);
    const ctype = res.headers.get("Content-Type") || "";
    const data = ctype.includes("json") ? await res.json() : await res.text();
    if (!res.ok) throw new Error((data && data.error) || `${res.status} ${res.statusText}`);
    return data;
  }

  const SUITS = { c: "♣", d: "♦", h: "♥", s: "♠" };
  function cardEl(code, cls) {
    const classes = ["card"].concat(cls ? cls.split(" ") : []);
    if (!code || code === "XX") return el("div", { class: classes.concat("back").join(" ") });
    if (code === "??") return el("div", { class: classes.concat("unknown").join(" ") });
    const r = code[0] === "T" ? "10" : code[0];
    const s = code[1];
    classes.push(s);
    if (s === "h" || s === "d") classes.push("red");
    return el("div", { class: classes.join(" "), title: code },
      el("span", { class: "r", text: r }), el("span", { class: "s", text: SUITS[s] || "?" }));
  }
  function cardsInline(codes) {
    return el("span", { class: "cards-inline" }, (codes || []).map((c) => cardEl(c, "inline")));
  }

  // ------------------------------------------------------------ state
  const S = {
    tab: "play",
    agents: [],
    agentInfo: {},
    game: null,            // live game state from the server
    prevHand: null,        // for diffing animations
    replay: { list: null, data: null, i: null, step: 0, playing: null, lens: null, lensSpec: null, lensBusy: false },
    hfPath: "",
    evalResult: null,
    simResult: null,
    simReq: null,          // the body of the last simulate request (for labels)
    fourColor: false,
    anims: [],             // in-flight Web Animations on the table, cancelled on every render
    evalSel: 0,            // index into the (filtered) disagreement list
    evalFilter: null,      // {actual, predicted} confusion cell filter
    simSel: null,          // seat number highlighted in the simulate chart
    simCursor: null,       // hand index under the simulate crosshair
  };
  const reducedMotion = () => { try { return window.matchMedia("(prefers-reduced-motion: reduce)").matches; } catch (e) { return false; } };

  // ------------------------------------------------------------ tabs
  // Three destinations: Mirror (judging), Simulator (play / replay / evaluate / simulate), Evidence.
  const SIM_TABS = ["play", "replay", "evaluate", "simulate"];
  const destOf = (name) => (SIM_TABS.includes(name) ? "simulator" : name);
  function setTab(name) {
    const prevDest = destOf(S.tab);
    S.tab = name;
    const dest = destOf(name);
    if (dest === "simulator") S.simTab = name;
    $$("#dests .dest").forEach((b) => { const on = b.dataset.dest === dest; b.classList.toggle("active", on); if (on) b.setAttribute("aria-current", "page"); else b.removeAttribute("aria-current"); });
    $$(".view").forEach((v) => { v.hidden = v.id !== "view-" + dest; });
    if (prevDest !== dest) window.scrollTo(0, 0);
    if (window.Mirror && window.Mirror.onHide && dest !== "mirror") window.Mirror.onHide();
    $$("#tabs .tab").forEach((b) => { const on = b.dataset.tab === name; b.classList.toggle("active", on); if (on) b.setAttribute("aria-current", "page"); else b.removeAttribute("aria-current"); });
    $$(".pane").forEach((p) => p.classList.toggle("active", p.id === "pane-" + name));
    $("#transport").hidden = name !== "replay" || !S.replay.data;
    // play / replay use the felt; evaluate / simulate put their results on the stage instead
    const onTable = name === "play" || name === "replay";
    if (dest === "simulator") { $("#tableWrap").hidden = !onTable; $("#simStage").hidden = onTable; }
    if (name === "play") {
      if (S.game && S.game.active) { renderGame(S.game, true); updateActionBar(S.game); }
      else showOverlay(true);
    } else if (name === "replay") {
      $("#actionBar").hidden = true;
      if (S.replay.data) renderReplayStep(false); else showOverlay(true, "Load a hand history on the right to start replaying.");
    } else if (name === "mirror") {
      $("#actionBar").hidden = true;
      if (window.Mirror) window.Mirror.onShow(name);
    } else if (name === "evidence") {
      $("#actionBar").hidden = true;
      if (window.Evidence) window.Evidence.onShow();
    } else {
      $("#actionBar").hidden = true;
      if (name === "evaluate") {
        if (!$("#evalStop").disabled) { /* running: keep the live status */ } else if (!S.evalResult) $("#evalStatus").textContent = S.replay.list ? `${S.replay.list.count} hands loaded (${S.replay.list.label})` : "Load a hand history in Replay first.";
        renderEvalStage();
      } else if (name === "simulate") renderSimStage();
    }
  }
  // User navigation rewrites the hash to a side-effect-free form so a reload returns to the same place.
  const navHash = (name) => { try { history.replaceState(null, "", destOf(name) === "simulator" ? "#tab=" + name : name === "evidence" ? "#evidence" : "#mirror=" + ((window.Mirror && window.Mirror.state.step) || "reveal")); } catch (e) { /* ignore */ } };
  $$("#tabs .tab").forEach((b) => b.addEventListener("click", () => { setTab(b.dataset.tab); navHash(b.dataset.tab); }));
  $$("#dests .dest").forEach((b) => b.addEventListener("click", () => { const t = b.dataset.dest === "simulator" ? (S.simTab || "play") : b.dataset.dest; navHash(t); setTab(t); }));

  // The felt overlay: with no message it is the Play welcome (with "Deal with defaults"); with a message it explains an empty tool.
  const OVERLAY_DEFAULT = { title: $("#overlayTitle").textContent, html: $("#overlayMsg").innerHTML };
  function showOverlay(show, msg) {
    const ov = $("#tableOverlay");
    ov.hidden = !show;
    if (show) {
      $("#overlayTitle").textContent = msg ? (S.tab === "replay" ? "Replay a real hand" : "PokerSim") : OVERLAY_DEFAULT.title;
      if (msg) $("#overlayMsg").textContent = msg; else $("#overlayMsg").innerHTML = OVERLAY_DEFAULT.html;
      $("#overlayDeal").hidden = !!msg;
      cancelAnims();
      $("#seats").innerHTML = ""; $("#board").innerHTML = ""; $("#pot").innerHTML = ""; $("#centerMsg").innerHTML = "";
    }
  }
  function cancelAnims() {
    S.anims.forEach((a) => { try { a.cancel(); } catch (e) { /* already gone */ } });
    S.anims = [];
    const fx = $("#tableWrap .chip-fx");
    if (fx) fx.innerHTML = "";
  }
  // Chips swept into the pot fly on their own layer, so a repeat render of the same state does not cut them off.
  function chipLayer() {
    const wrap = $("#tableWrap");
    let fx = $("#tableWrap .chip-fx");
    if (!fx) { fx = el("div", { class: "chip-fx", "aria-hidden": "true" }); wrap.appendChild(fx); }
    return fx;
  }

  $("#fourColor").addEventListener("change", (e) => document.body.classList.toggle("four-color", e.target.checked));

  // ------------------------------------------------------------ table renderer
  /**
   * view = { hand, slots: [{slot, seat, name, stack, kind, player (index|null), sittingOut}], viewer (player index|null),
   *          reveal, bubble: {player, text, kind}, lens: {player, text, agree}, payouts: bool }
   */
  function renderTable(view) {
    $("#tableOverlay").hidden = true;
    const hand = view.hand;
    const wrap = $("#tableWrap");
    const seatsRoot = $("#seats");
    seatsRoot.innerHTML = "";
    // view.animate === false: scrubbing, jumps, resizes - draw the requested state directly, no effects.
    const still = view.animate === false;
    const motion = !still && !reducedMotion();
    wrap.classList.toggle("no-anim", still);
    const n = view.slots.length;
    const W = wrap.clientWidth || 900, H = wrap.clientHeight || 520;
    const cx = W / 2, cy = H / 2;
    const rx = W * 0.405, ry = H * 0.40;
    const brx = W * 0.255, bry = H * 0.225;
    const players = hand.players;
    const byIndex = {};
    players.forEach((p) => { byIndex[p.index] = p; });
    const prev = S.prevHand && S.prevHand.hand_id === hand.hand_id ? S.prevHand : null;
    // a new hand or a direct jump cancels effects still in flight; a repeat render leaves them alone
    if (still || !prev) cancelAnims();
    else S.anims = S.anims.filter((a) => a.playState !== "finished");
    const prevBy = {};
    if (prev) prev.players.forEach((p) => { prevBy[p.index] = p; });
    const sameStreet = !!prev && prev.street_index === hand.street_index;
    const geo = {};       // player index -> seat and wager coordinates, for chip movement
    const moves = [];     // wagers that grew this render: chips travel seat -> wager

    view.slots.forEach((slot, k) => {
      const a = (Math.PI / 2) + (2 * Math.PI * k) / n;
      const x = cx + rx * Math.cos(a), y = cy + ry * Math.sin(a);
      const p = slot.player !== null && slot.player !== undefined ? byIndex[slot.player] : null;
      const seat = el("div", { class: "seat", style: `left:${x}px;top:${y}px` });
      if (p) {
        const bx = cx + brx * Math.cos(a), by = cy + bry * Math.sin(a);
        geo[p.index] = { x, y, bx, by };
        const isViewer = view.viewer === p.index;
        const visible = view.reveal || isViewer || p.shown;
        const codes = (p.cards || []).map((c) => (visible || c === "??") ? c : "XX");
        const pp = prevBy[p.index];
        const cards = el("div", { class: "cards" });
        codes.forEach((c, ci) => {
          const changed = !pp || (pp.cards || [])[ci] !== c || !pp.has_cards;
          const ce = cardEl(c, "small");
          if (!changed) ce.style.animation = "none";
          cards.appendChild(ce);
        });
        if (p.folded) cards.style.visibility = p.has_cards ? "visible" : "hidden";
        const isActor = hand.status === "betting" && hand.actor === p.index;
        const tags = el("div", { class: "tags" }, el("span", { class: "tag pos", text: p.position }));
        if (isActor) tags.prepend(el("span", { class: "tag act", text: "to act" }));
        if (p.agent && p.agent.kind && p.agent.kind !== "bot") tags.appendChild(el("span", { class: `tag kind-${p.agent.kind}`, text: p.agent.kind === "human" ? "you" : p.agent.label }));
        if (p.all_in && !p.folded) tags.appendChild(el("span", { class: "tag", text: "ALL-IN" }));
        if (p.folded) tags.appendChild(el("span", { class: "tag", text: "folded" }));
        if (visible && p.hand && !p.folded && (hand.street_index > 0 || hand.status === "finished")) tags.appendChild(el("span", { class: "tag hand", text: p.hand }));
        const plate = el("div", { class: "plate" },
          el("div", { class: "name", text: p.name, title: p.name }),
          el("div", { class: "stack" + (p.all_in && !p.folded ? " allin" : ""), text: p.stack === null ? "∞" : fmt(p.stack) }),
          tags, el("div", { class: "thinking" }));
        seat.appendChild(cards);
        seat.appendChild(plate);
        if (p.folded) seat.classList.add("folded");
        if (isActor) { seat.classList.add("actor"); seat.setAttribute("aria-label", `${p.name} to act`); }
        if (view.thinking === p.index) seat.classList.add("thinking");
        if (hand.status === "finished" && p.payout > 0) seat.classList.add("winner");
        // wager: one small stack in front of the seat. When it grew since the last render, it travels from the seat.
        if (p.bet > 0) {
          const bet = el("div", { class: "bet", style: `left:${bx}px;top:${by}px` }, chipsEl(p.bet, hand.min_bet), el("span", { text: fmt(p.bet) }));
          seatsRoot.appendChild(bet);
          const before = sameStreet && pp ? pp.bet : 0;
          if (prev && p.bet > before) moves.push({ node: bet, dx: x - bx, dy: y - by });
        }
        // dealer button
        if (p.is_button) {
          const dx = cx + (rx * 0.72) * Math.cos(a + 0.35), dy = cy + (ry * 0.70) * Math.sin(a + 0.35);
          seatsRoot.appendChild(el("div", { class: "dealer-btn", style: `left:${dx}px;top:${dy}px`, text: "D" }));
        }
        // payout float
        if (view.payouts && hand.status === "finished" && p.payout > 0 && (!prev || prev.status !== "finished")) {
          seatsRoot.appendChild(el("div", { class: "payout", style: `left:${x}px;top:${y - 40}px`, text: "+" + fmt(p.payout) }));
        }
        if (view.bubble && view.bubble.player === p.index) {
          seatsRoot.appendChild(el("div", { class: "bubble " + (view.bubble.kind || ""), style: `left:${x}px;top:${y - 48}px`, text: view.bubble.text }));
        }
        if (view.lens && view.lens.player === p.index) {
          seatsRoot.appendChild(el("div", { class: "bubble lens " + (view.lens.agree === true ? "agree" : view.lens.agree === false ? "disagree" : ""), style: `left:${x}px;top:${y + 36}px`, text: view.lens.text }));
        }
      } else {
        const plate = el("div", { class: "plate", style: "opacity:0.55" },
          el("div", { class: "name", text: slot.name || `Seat ${slot.seat}` }),
          el("div", { class: "stack", text: slot.stack === null || slot.stack === undefined ? "" : fmt(slot.stack) }),
          el("div", { class: "tags" }, el("span", { class: "tag", text: slot.sittingOut ? "sitting out" : "empty" })));
        seat.appendChild(el("div", { class: "cards" }));
        seat.appendChild(plate);
      }
      seatsRoot.appendChild(seat);
    });

    // board
    const board = $("#board");
    board.innerHTML = "";
    const prevBoard = prev ? prev.board : [];
    for (let i = 0; i < 5; i++) {
      const c = hand.board[i];
      if (c) {
        const ce = cardEl(c, "big");
        if (prevBoard[i] === c) ce.style.animation = "none";
        board.appendChild(ce);
      } else board.appendChild(el("div", { class: "slot" }));
    }
    // pot: one value, updated at once (the swept chips land on it)
    const pot = $("#pot");
    pot.innerHTML = "";
    const streetLabel = hand.status === "finished" ? "Showdown" : hand.status === "incomplete" ? "Incomplete" : hand.street.charAt(0).toUpperCase() + hand.street.slice(1);
    pot.appendChild(el("div", { class: "pot-main" }, el("span", { class: "muted", text: streetLabel + " · Pot" }), el("b", { text: fmt(hand.total_pot) })));
    if (hand.pots && hand.pots.length > 1 && hand.status === "finished") {
      hand.pots.forEach((sp, i) => pot.appendChild(el("div", { class: "pot-side", text: `${i === 0 ? "Main" : "Side " + i}: ${fmt(sp.amount)} → ${sp.winners.map((w) => byIndex[w] ? byIndex[w].name : "p" + (w + 1)).join(", ")}` })));
    }
    // chip movement (transform + opacity only, cancelled by the next render)
    if (motion && prev) {
      const ease = "cubic-bezier(0.2, 0.7, 0.3, 1)";
      moves.forEach((m) => {
        S.anims.push(m.node.animate([
          { transform: `translate(calc(-50% + ${m.dx.toFixed(1)}px), calc(-50% + ${m.dy.toFixed(1)}px)) scale(0.8)`, opacity: 0.5 },
          { transform: "translate(-50%, -50%) scale(1)", opacity: 1 }], { duration: 200, easing: ease }));
      });
      // wagers from the previous render that are gone (street over, hand over): sweep them into the pot
      const wr = wrap.getBoundingClientRect();
      const pr = ($(".pot-main", pot) || pot).getBoundingClientRect();
      const tx = pr.left + pr.width / 2 - wr.left, ty = pr.top + pr.height / 2 - wr.top;
      prev.players.forEach((pp) => {
        const p = byIndex[pp.index], g = geo[pp.index];
        if (!g || !(pp.bet > 0) || (p && p.bet > 0 && sameStreet)) return;
        const ghost = el("div", { class: "bet ghost", style: `left:${g.bx}px;top:${g.by}px`, "aria-hidden": "true" }, chipsEl(pp.bet, prev.min_bet), el("span", { text: fmt(pp.bet) }));
        chipLayer().appendChild(ghost);
        const anim = ghost.animate([
          { transform: "translate(-50%, -50%) scale(1)", opacity: 1 },
          { transform: `translate(calc(-50% + ${(tx - g.bx).toFixed(1)}px), calc(-50% + ${(ty - g.by).toFixed(1)}px)) scale(0.7)`, opacity: 0 }],
        { duration: 220, easing: "cubic-bezier(0.4, 0, 0.2, 1)", fill: "forwards" });
        anim.onfinish = () => ghost.remove();
        anim.oncancel = () => ghost.remove();
        S.anims.push(anim);
      });
    }
    // centre message
    const msg = $("#centerMsg");
    msg.innerHTML = "";
    if (hand.status === "finished") {
      const winners = players.filter((p) => p.payout > 0);
      if (winners.length) {
        msg.appendChild(document.createTextNode(winners.map((w) => `${w.name} wins ${fmt(w.payout)}`).join(" · ")));
        const h = winners.find((w) => w.hand && w.cards_known);
        if (h && players.filter((p) => !p.folded).length > 1) msg.appendChild(el("span", { class: "sub", text: h.hand }));
        if (hand.unresolved) msg.appendChild(el("span", { class: "sub", text: "cards unknown - pot awarded from the recorded result" }));
      }
    } else if (hand.status === "incomplete") {
      msg.appendChild(document.createTextNode("Record ends before the hand was resolved"));
    }
    S.prevHand = hand;
  }

  function chipsEl(amount, bb) {
    const unit = bb || 1;
    const ratio = amount / unit;
    const count = Math.min(6, 1 + Math.floor(Math.log2(ratio + 1)));
    const cls = ratio < 1 ? "c1" : ratio < 3 ? "c5" : ratio < 10 ? "c25" : ratio < 30 ? "c100" : ratio < 100 ? "c500" : ratio < 300 ? "c1000" : "c5000";
    const stack = el("div", { class: "chips" });
    for (let i = 0; i < count; i++) stack.appendChild(el("div", { class: "chip " + cls }));
    return stack;
  }

  // ------------------------------------------------------------ PLAY
  function agentOptions(includeHuman) {
    const opts = [];
    if (includeHuman) opts.push(el("option", { value: "human", text: "Human" }));
    S.agents.forEach((a) => opts.push(el("option", { value: a.id, text: a.label })));
    return opts;
  }

  function seatRow(container, i, defaults) {
    const d = defaults || {};
    const row = el("div", { class: "seat-row" },
      el("span", { class: "n", text: String(i + 1) }),
      el("label", {}, "Name", el("input", { type: "text", class: "s-name", value: d.name || "", placeholder: `Seat ${i + 1}` })),
      el("label", {}, "Player", el("select", { class: "s-agent" }, agentOptions(container.id === "seatConfig"))),
      el("label", {}, "Params / URL", el("input", { type: "text", class: "s-params", value: d.params || "", placeholder: "optional" })),
      el("label", {}, "Stack", el("input", { type: "number", class: "s-stack", step: "any", value: d.stack || "" })));
    $(".s-agent", row).value = d.agent || (i === 0 && container.id === "seatConfig" ? "human" : "equity");
    container.appendChild(row);
  }

  function buildSpec(kind, params) {
    params = (params || "").trim();
    if (kind === "human") return "human";
    if (kind === "http") return "http:" + (params || "http://127.0.0.1:5001/act");
    if (kind === "claude") return params ? "claude:" + params : "claude";
    return params ? `${kind}:${params}` : kind;
  }

  function readSeats(container) {
    return $$(".seat-row", container).map((row) => ({
      name: $(".s-name", row).value.trim(),
      agent: buildSpec($(".s-agent", row).value, $(".s-params", row).value),
      stack: $(".s-stack", row).value ? Number($(".s-stack", row).value) : undefined,
    }));
  }

  // The defaults behind "Deal with defaults" (and the initial Table settings form).
  const DEFAULT_SEATS = [{ name: "You", agent: "human" }, { agent: "tag" }, { agent: "equity" }, { agent: "lag" }, { agent: "equity" }, { agent: "random" }];
  const DEFAULT_TABLE = { small_blind: 1, big_blind: 2, ante: 0, ante_mode: "uniform", stack: 200 };
  function defaultBody() {
    return Object.assign({ seats: DEFAULT_SEATS.map((d) => ({ name: d.name || "", agent: d.agent })), bot_delay_ms: Number($("#botDelay").value) || 700, spectate: false }, DEFAULT_TABLE);
  }
  function describeDefaults() {
    const label = (id) => { const a = S.agents.find((x) => x.id === id); return a ? a.label : id; };
    const bots = DEFAULT_SEATS.slice(1).map((d) => label(d.agent));
    return `You in seat 1 against ${bots.slice(0, -1).join(", ")} and ${bots[bots.length - 1]}. Blinds ${DEFAULT_TABLE.small_blind}/${DEFAULT_TABLE.big_blind}, ${DEFAULT_TABLE.stack} chips each.`;
  }

  function initSeatConfig() {
    const c = $("#seatConfig");
    c.innerHTML = "";
    DEFAULT_SEATS.forEach((d, i) => seatRow(c, i, d));
    $("#addSeat").addEventListener("click", () => { if ($$(".seat-row", c).length < 10) seatRow(c, $$(".seat-row", c).length, {}); });
    $("#removeSeat").addEventListener("click", () => { const rows = $$(".seat-row", c); if (rows.length > 2) rows[rows.length - 1].remove(); });
  }

  async function startGame(preset) {
    const body = preset || {
      seats: readSeats($("#seatConfig")), small_blind: Number($("#sb").value), big_blind: Number($("#bb").value), ante: Number($("#ante").value) || 0,
      ante_mode: $("#anteMode").value, stack: Number($("#stack").value) || 200, bot_delay_ms: Number($("#botDelay").value) || 0,
      spectate: $("#spectate").checked,
    };
    try {
      const st = await api("/api/game/new", body);
      $("#setupBox").open = false;
      $("#gameBox").hidden = false;
      $("#revealToggle").checked = st.reveal;
      onGameState(st);
      setTab("play");
    } catch (e) { toast(e.message, true); }
  }

  function onGameState(st) {
    S.game = st;
    if (!st.active) return;
    if (S.tab === "play") renderGame(st, false);
    updateActionBar(st);
    renderLog(st);
    renderSession(st);
    $("#nextHand").disabled = !(st.hand.status === "finished" || st.hand.status === "incomplete");
    $("#exportLive").hidden = !st.hands_recorded;
    $("#gameTitle").textContent = st.mode === "scenario" ? `Scenario: ${st.scenario.label}` : `Hand #${st.hand_no}`;
    const t = st.table;
    $("#gameBlinds").textContent = t && t.big_blind ? `blinds ${fmt(t.small_blind)}/${fmt(t.big_blind)}${t.ante ? " ante " + fmt(t.ante) : ""}` : (st.hand.blinds ? "blinds " + st.hand.blinds.filter((b) => b).map(fmt).join("/") : "");
    $("#gameBox").hidden = false;
  }

  function renderGame(st, force) {
    const hand = st.hand;
    let slots;
    if (st.mode === "table" && st.table && st.table.seats.length) {
      const seats = st.table.seats;
      const viewerSeat = (hand.players.find((p) => p.is_human) || {}).seat || seats[0].seat;
      const order = seats.slice().sort((a, b) => ((a.seat - viewerSeat + 100) % 100) - ((b.seat - viewerSeat + 100) % 100));
      slots = order.map((s) => {
        const p = hand.players.find((pl) => pl.seat === s.seat);
        return { slot: s.seat, seat: s.seat, name: s.name, stack: s.stack, kind: s.kind, player: p ? p.index : null, sittingOut: !p };
      });
    } else {
      const viewerIdx = (hand.players.find((p) => p.is_human) || hand.players.find((p) => p.cards_known) || hand.players[0]).index;
      const order = hand.players.slice().sort((a, b) => ((a.index - viewerIdx + 100) % 100) - ((b.index - viewerIdx + 100) % 100));
      slots = order.map((p) => ({ slot: p.index, seat: p.index + 1, name: p.name, stack: p.stack, player: p.index }));
    }
    let bubble = null;
    const last = st.log && st.log.length ? st.log[st.log.length - 1] : null;
    if (last && last.player !== null && last.player !== undefined && hand.status !== "finished") {
      const text = last.text.replace(/^.*?\s(folds|checks|calls|bets|raises)/, "$1");
      bubble = { player: last.player, text: text.charAt(0).toUpperCase() + text.slice(1), kind: /raise|bet/.test(text) ? "aggr" : /fold/.test(text) ? "fold" : "" };
    }
    renderTable({ hand, slots, viewer: st.viewer, reveal: st.reveal, bubble, thinking: st.thinking, payouts: true });
  }

  function updateActionBar(st) {
    const hand = st.hand;
    const bar = $("#actionBar");
    const legal = hand.legal;
    const myTurn = S.tab === "play" && hand.status === "betting" && hand.actor !== null && hand.players[hand.actor].is_human && legal;
    bar.hidden = !myTurn;
    if (!myTurn) return;
    const me = hand.players[hand.actor];
    $("#actionInfo").innerHTML = "";
    $("#actionInfo").appendChild(el("span", {}, `${me.name}, your turn. `, el("b", { text: me.hand || "" }), legal.call ? ` · ${fmt(legal.call)} to call` : " · no bet to call", ` · pot ${fmt(hand.total_pot)}`));
    $("#btnFold").hidden = !legal.fold;
    $("#btnCheckCall").innerHTML = legal.check ? "Check <kbd>C</kbd>" : `Call ${fmt(legal.call)}${legal.call_is_all_in ? " (all-in)" : ""} <kbd>C</kbd>`;
    const rg = $(".raise-group");
    rg.hidden = !legal.raise;
    if (legal.raise) {
      const slider = $("#raiseSlider"), input = $("#raiseInput");
      const step = Number.isInteger(hand.min_bet) && hand.min_bet >= 1 ? 1 : 0.01;
      slider.min = legal.min_raise_to; slider.max = legal.max_raise_to; slider.step = step;
      input.min = legal.min_raise_to; input.max = legal.max_raise_to; input.step = step;
      const current = Math.min(Math.max(Number(input.dataset.handId) === hand.hand_id && Number(input.value) || legal.min_raise_to, legal.min_raise_to), legal.max_raise_to);
      setRaise(current);
      input.dataset.handId = hand.hand_id;
      $("#btnRaise").firstChild.textContent = (legal.raise_kind === "bet" ? "Bet " : "Raise to ");
    }
  }

  function setRaise(v) {
    const legal = S.game.hand.legal;
    const step = Number($("#raiseSlider").step) || 1;
    v = Math.round(v / step) * step;
    v = Math.min(Math.max(v, legal.min_raise_to), legal.max_raise_to);
    v = Number(v.toFixed(2));
    $("#raiseSlider").value = v;
    $("#raiseInput").value = v;
    $("#raiseAmountLabel").textContent = fmt(v) + (v >= legal.max_raise_to ? " (all-in)" : "");
  }

  async function humanAct(action, amount) {
    if (!S.game || $("#actionBar").hidden) return;
    try {
      const st = await api("/api/game/act", { action, amount });
      onGameState(st);
    } catch (e) { toast(e.message, true); }
  }

  function renderLog(st) {
    const log = $("#gameLog");
    log.innerHTML = "";
    (st.log || []).forEach((e) => {
      const entry = el("div", { class: "entry" }, e.street !== undefined && e.street !== null ? el("span", { class: "street", text: typeof e.street === "string" ? e.street : "" }) : null, document.createTextNode(e.text));
      if (e.note) entry.appendChild(el("span", { class: "note", text: e.note }));
      if (e.reasoning) entry.appendChild(el("span", { class: "reasoning", text: e.reasoning }));
      if (e.error) entry.appendChild(el("span", { class: "error", text: e.error }));
      if (e.latency_ms !== undefined && e.latency_ms !== null && e.latency_ms > 50) entry.appendChild(el("span", { class: "muted", text: ` ${Math.round(e.latency_ms)} ms` }));
      log.appendChild(entry);
    });
    log.scrollTop = log.scrollHeight;
  }

  function renderSession(st) {
    const tb = $("#sessionTable tbody");
    tb.innerHTML = "";
    if (st.table && st.table.seats.length) {
      st.table.seats.forEach((s) => {
        const net = s.stack === null ? null : s.stack - s.buy_in;
        tb.appendChild(el("tr", {}, el("td", { text: s.seat }), el("td", { text: s.name }), el("td", { text: fmt(s.stack) }),
          el("td", { class: net > 0 ? "pos" : net < 0 ? "neg" : "", text: net === null ? "" : (net > 0 ? "+" : "") + fmt(net) })));
      });
    } else {
      st.hand.players.forEach((p) => {
        tb.appendChild(el("tr", {}, el("td", { text: p.seat }), el("td", { text: p.name }), el("td", { text: fmt(p.stack) }),
          el("td", { class: p.net > 0 ? "pos" : p.net < 0 ? "neg" : "", text: p.net === null ? "" : (p.net > 0 ? "+" : "") + fmt(p.net) })));
      });
    }
  }

  function initPlay() {
    initSeatConfig();
    $("#defaultsHint").textContent = describeDefaults();
    $("#startGame").addEventListener("click", () => startGame());
    const dealDefaults = () => { S.prevHand = null; startGame(defaultBody()); };
    $("#dealDefaults").addEventListener("click", dealDefaults);
    $("#overlayDeal").addEventListener("click", dealDefaults);
    $("#nextHand").addEventListener("click", async () => { try { onGameState(await api("/api/game/next", {})); } catch (e) { toast(e.message, true); } });
    $("#revealToggle").addEventListener("change", async (e) => { try { onGameState(await api("/api/game/reveal", { reveal: e.target.checked })); } catch (err) { toast(err.message, true); } });
    $("#btnFold").addEventListener("click", () => humanAct("fold"));
    $("#btnCheckCall").addEventListener("click", () => humanAct("call"));
    $("#btnRaise").addEventListener("click", () => humanAct("raise", Number($("#raiseInput").value)));
    $("#raiseSlider").addEventListener("input", (e) => setRaise(Number(e.target.value)));
    $("#raiseInput").addEventListener("change", (e) => setRaise(Number(e.target.value)));
    $$(".quick .chip-btn").forEach((b) => b.addEventListener("click", () => {
      const legal = S.game.hand.legal, hand = S.game.hand, me = hand.players[hand.actor];
      const size = b.dataset.size;
      if (size === "min") return setRaise(legal.min_raise_to);
      if (size === "max") return setRaise(legal.max_raise_to);
      const potAfterCall = hand.total_pot + legal.call;
      setRaise(me.bet + legal.call + Number(size) * potAfterCall);
    }));
    document.addEventListener("keydown", (e) => {
      if (e.target.matches("input, select, textarea") || !$("#forkModal").hidden) return;
      if (S.tab === "play" && !$("#actionBar").hidden) {
        if (e.key === "f" || e.key === "F") humanAct("fold");
        else if (e.key === "c" || e.key === "C") humanAct("call");
        else if (e.key === "r" || e.key === "R" || e.key === "Enter") { if (!$(".raise-group").hidden) humanAct("raise", Number($("#raiseInput").value)); }
      } else if (S.tab === "play" && S.game && S.game.active && (S.game.hand.status === "finished" || S.game.hand.status === "incomplete") && (e.key === "Enter" || e.key === " ")) {
        e.preventDefault();
        $("#nextHand").click();
      } else if (S.tab === "replay" && S.replay.data) {
        // a held key is a scrub: render each step directly, no effects
        if (e.key === "ArrowRight") { e.preventDefault(); stepReplay(1, { animate: !e.repeat }); }
        else if (e.key === "ArrowLeft") { e.preventDefault(); stepReplay(-1, { animate: !e.repeat }); }
        else if (e.key === " ") { e.preventDefault(); togglePlay(); }
        else if (e.key === "Home") { e.preventDefault(); setStep(0); }
        else if (e.key === "End") { e.preventDefault(); setStep(S.replay.data.steps.length - 1); }
      }
    });
  }

  // ------------------------------------------------------------ REPLAY: loading
  // opts.open === false: the caller opens a specific hand itself (deep links, openReplay).
  async function loadRecords(body, opts) {
    opts = opts || {};
    try {
      $("#handsTitle").textContent = "Loading…";
      $("#handsBox").hidden = false;
      const list = await api("/api/replay/load", body);
      S.replay.list = list;
      S.replay.src = { source: body.source, path: body.path || body.name || null };
      S.replay.data = null;
      S.replay.lens = null;
      S.evalResult = null;
      renderHandList();
      $("#loadBox").open = false;
      if (!opts.quiet) toast(`Loaded ${list.count} hands from ${list.label}`);
      if (opts.open !== false && list.hands.length) await openHand(list.hands[0].i);
      if (S.tab === "evaluate") setTab("evaluate");
      return list;
    } catch (e) { toast(e.message, true); $("#handsTitle").textContent = "Hands"; return null; }
  }

  // Open one recorded hand in Replay, e.g. from Mirror: P.openReplay({ source: "path", path: "data/mirror/demo_hands.phhs", i: 0, step: 15 }).
  async function openReplay(opts) {
    opts = opts || {};
    const source = opts.source || "sample", path = opts.path;
    setTab("replay");
    navHash("replay");
    const loaded = S.replay.list && S.replay.src && S.replay.src.source === source && S.replay.src.path === path;
    if (!loaded) {
      const list = await loadRecords({ source, path, limit: Number($("#loadLimit").value) || null }, { open: false, quiet: true });
      if (!list) return false;
    }
    await openHand(Number(opts.i) || 0, { step: Number(opts.step) || 0 });
    if (opts.label) toast(opts.label);
    return true;
  }

  function renderHandList() {
    const list = S.replay.list;
    const root = $("#handList");
    root.innerHTML = "";
    if (!list) return;
    const onlyShowdown = $("#filterShowdown").checked, onlyKnown = $("#filterKnown").checked;
    const shown = list.hands.filter((h) => (!onlyShowdown || h.showdown) && (!onlyKnown || h.known_hole_cards > 0));
    $("#handsTitle").textContent = `${list.label} — ${shown.length}/${list.count} hands`;
    shown.slice(0, 2000).forEach((h) => {
      const item = el("div", { class: "hand-item" + (h.i === S.replay.i ? " active" : ""), onclick: () => openHand(h.i) },
        el("span", { class: "idx", text: h.index !== null && h.index !== undefined ? h.index : h.i + 1 }),
        el("span", { class: "desc", text: `${h.n}p · ${h.players.map((p) => p.length > 12 ? p.slice(0, 10) + "…" : p).join(", ")}` }),
        el("span", { class: "badges" },
          el("span", { class: "tag", text: h.variant }),
          h.showdown ? el("span", { class: "tag", text: "SD" }) : null,
          h.known_hole_cards ? el("span", { class: "tag", text: `${h.known_hole_cards} known` }) : null,
          !h.supported ? el("span", { class: "tag", text: "unsupported" }) : null));
      root.appendChild(item);
    });
  }

  async function renderSamples() {
    try {
      const local = await api("/api/dataset/local");
      const root = $("#sampleList");
      root.innerHTML = "";
      local.samples.forEach((f) => root.appendChild(el("div", { class: "file", onclick: () => loadRecords({ source: "sample", path: f.name, limit: Number($("#loadLimit").value) || null }) },
        el("span", { text: f.name }), el("span", { class: "meta", text: `${(f.size / 1024).toFixed(0)} kB` }))));
      const cached = $("#cachedList");
      cached.innerHTML = "";
      if (!local.cached.length) cached.appendChild(el("div", { class: "hint", text: "Nothing downloaded yet." }));
      local.cached.forEach((f) => cached.appendChild(el("div", { class: "file", onclick: () => loadRecords({ source: "path", path: f.path, limit: Number($("#loadLimit").value) || null }) },
        el("span", { text: f.name }), el("span", { class: "meta", text: `${(f.size / 1024).toFixed(0)} kB` }))));
    } catch (e) { toast(e.message, true); }
  }

  async function renderFeatured() {
    try {
      const d = await api("/api/dataset/featured");
      const root = $("#featuredList");
      root.innerHTML = "";
      d.featured.forEach((f) => root.appendChild(el("div", { class: "file", title: f.path, onclick: () => loadRecords({ source: "hf", path: f.path, limit: Number($("#loadLimit").value) || null }) },
        el("span", { text: f.title }), el("span", { class: "meta", text: f.path.split("/").pop().slice(0, 22) }))));
    } catch (e) { /* offline */ }
  }

  async function browseHF(path) {
    S.hfPath = path;
    const crumbs = $("#hfCrumbs");
    crumbs.innerHTML = "";
    crumbs.appendChild(el("a", { text: "PokerData", onclick: () => browseHF("") }));
    let acc = "";
    path.split("/").filter(Boolean).forEach((part) => {
      acc += (acc ? "/" : "") + part;
      const p = acc;
      crumbs.appendChild(document.createTextNode(" / "));
      crumbs.appendChild(el("a", { text: part, onclick: () => browseHF(p) }));
    });
    const root = $("#hfList");
    root.innerHTML = "";
    root.appendChild(el("div", { class: "hint", text: "Loading…" }));
    try {
      const d = await api("/api/dataset/list?path=" + encodeURIComponent(path));
      root.innerHTML = "";
      d.items.forEach((it) => {
        const isDir = it.type === "directory";
        root.appendChild(el("div", { class: "file" + (isDir ? " dir" : ""), onclick: () => isDir ? browseHF(it.path) : loadRecords({ source: "hf", path: it.path, limit: Number($("#loadLimit").value) || null }) },
          el("span", { text: it.name }), el("span", { class: "meta", text: isDir ? "" : `${((it.size || 0) / 1024).toFixed(0)} kB` })));
      });
      if (!d.items.length) root.appendChild(el("div", { class: "hint", text: "Empty folder." }));
    } catch (e) { root.innerHTML = ""; root.appendChild(el("div", { class: "hint", text: e.message })); }
  }

  function initLoadPanel() {
    $$("#loadTabs button").forEach((b) => b.addEventListener("click", () => {
      $$("#loadTabs button").forEach((x) => x.classList.toggle("active", x === b));
      $$(".srcpane").forEach((p) => p.classList.toggle("active", p.id === "src-" + b.dataset.src));
      if (b.dataset.src === "hf" && !$("#hfList").children.length) { renderFeatured(); browseHF(""); }
    }));
    $("#uploadFile").addEventListener("change", (e) => {
      const f = e.target.files[0];
      if (!f) return;
      const reader = new FileReader();
      reader.onload = () => loadRecords({ source: "upload", name: f.name, text: reader.result, limit: Number($("#loadLimit").value) || null });
      reader.readAsText(f);
    });
    $("#filterShowdown").addEventListener("change", renderHandList);
    $("#filterKnown").addEventListener("change", renderHandList);
    renderSamples();
  }

  // ------------------------------------------------------------ REPLAY: playback
  async function openHand(i, opts) {
    opts = opts || {};
    stopPlay();
    try {
      let url = `/api/replay/hand?i=${i}`;
      if (S.replay.lensSpec && isFastSpec(S.replay.lensSpec)) url += `&agent=${encodeURIComponent(S.replay.lensSpec)}`;
      const data = await api(url);
      S.replay.data = data;
      S.replay.i = i;
      S.replay.lens = data.predictions || null;
      S.replay.step = 0;
      renderHandList();
      $("#replayBox").hidden = false;
      $("#replayTitle").textContent = data.record.label;
      $("#phhSource").textContent = describeMeta(data.record) + "\n\n" + data.phh;
      $("#replayWarnings").innerHTML = "";
      if (data.warnings.length) $("#replayWarnings").appendChild(el("div", {}, el("b", { text: "Engine notes: " }), data.warnings.join(" · ")));
      if (data.stacks_match === true) $("#replayWarnings").appendChild(el("div", { class: "muted", text: "Replayed finishing stacks match the record exactly." }));
      renderStepList();
      updateLensSummary();
      $("#transport").hidden = S.tab !== "replay";
      S.prevHand = null;
      $("#tpSlider").max = data.steps.length - 1;
      setStep(opts.step !== undefined ? opts.step : 0);
      if (S.replay.lensSpec && !isFastSpec(S.replay.lensSpec) && opts.autoLens) runLens();
    } catch (e) { toast(e.message, true); }
  }

  function describeMeta(rec) {
    const m = rec.meta || {};
    const parts = [];
    if (m.event) parts.push(m.event);
    if (m.venue) parts.push(m.venue);
    if (m.table) parts.push("table " + m.table);
    if (m.year) parts.push([m.day, m.month, m.year].filter(Boolean).join("/"));
    if (m.time) parts.push(m.time);
    parts.push(`${rec.variant}, ${rec.n} players, blinds ${rec.blinds.filter((b) => b).join("/")}` + (rec.antes.some((a) => a) ? `, antes ${rec.antes.filter((a) => a).join("/")}` : ""));
    return parts.join(" · ");
  }

  function renderStepList() {
    const data = S.replay.data;
    const root = $("#stepList");
    root.innerHTML = "";
    data.steps.forEach((s, k) => {
      const item = el("div", { class: "step" + (k === S.replay.step ? " active" : ""), onclick: () => setStep(k) },
        el("span", { class: "st", text: s.state.street || "" }),
        el("span", { text: s.label }));
      const lens = lensFor(s);
      if (lens) item.appendChild(lens);
      root.appendChild(item);
    });
    renderActionStrip();
  }

  // ------------------------------------------------------------ REPLAY: street-labelled action strip
  // One chip per step, grouped under its street. A decision chip selects the decision point (the step before the
  // action): the actor is lit and the model lens shows its suggestion; a second click shows the action itself.
  const STREET_NAMES = { preflop: "Preflop", flop: "Flop", turn: "Turn", river: "River", showdown: "Showdown" };
  const shortName = (n) => (n && n.length > 10 ? n.slice(0, 9) + "…" : n || "");
  function stripText(step) {
    const evs = step.events || [];
    const act = evs.find((e) => e.type === "action");
    if (act) {
      const p = step.state.players[act.player];
      const verb = act.action === "fold" ? "folds" : act.action === "check" ? "checks" : act.action === "call" ? `calls ${fmt(act.amount)}` : act.action === "bet" ? `bets ${fmt(act.to)}` : `raises ${fmt(act.to)}`;
      return { text: `${shortName(p ? p.name : "p" + (act.player + 1))} ${verb}${act.all_in ? " all-in" : ""}`, kind: act.action };
    }
    if (evs.some((e) => e.type === "deal_hole")) return { text: "", kind: "dot" };
    if (evs.some((e) => e.type === "deal_board")) return { text: step.label.replace(/^[^:]*:\s*/, ""), kind: "board" };
    if (evs.some((e) => e.type === "blind")) return { text: "Blinds", kind: "event" };
    const show = evs.find((e) => e.type === "show");
    if (show) { const p = step.state.players[show.player]; return { text: `${shortName(p ? p.name : "")} shows`, kind: "event" }; }
    const t = step.label.replace(/^Pot awarded:\s*/, "");
    return { text: t.length > 26 ? t.replace(/^(\S{10})\S+/, "$1…") : t, kind: "event" };
  }
  const isDecisionStep = (k) => { const st = S.replay.data && S.replay.data.steps[k]; return !!st && st.decision_id !== null && st.decision_id !== undefined; };
  function renderActionStrip() {
    const data = S.replay.data;
    const root = $("#actionStrip");
    root.innerHTML = "";
    if (!data) return;
    let group = null, street = null;
    data.steps.forEach((st, k) => {
      const sname = st.state.street || "";
      if (sname !== street || !group) {
        street = sname;
        group = el("div", { class: "as-street" }, el("div", { class: "as-label", text: STREET_NAMES[sname] || sname }), el("div", { class: "as-chips" }));
        root.appendChild(group);
      }
      const t = stripText(st);
      const d = decisionAt(st);
      let lensCls = "", lensTxt = "";
      if (d && S.replay.lens && S.replay.lens.decisions) {
        const pr = S.replay.lens.decisions[String(d.id)];
        if (pr && !pr.skipped) { lensCls = pr.agree ? " agree" : " disagree"; lensTxt = ` Model: ${actionText(pr.prediction)} (${pr.agree ? "agrees" : "disagrees"}).`; }
      }
      const chip = el("button", {
        class: `as-chip k-${t.kind}${d ? " decision" : ""}${lensCls}`, type: "button", tabindex: "-1", "data-k": k,
        "aria-label": `Step ${k}: ${st.label}.${d ? " Decision." : ""}${lensTxt}`, title: st.label + (lensTxt ? " ·" + lensTxt : ""),
        onclick: () => {
          // decision chip: first click = the moment of decision (step before), second click = the action
          const target = d && S.replay.step !== k - 1 ? k - 1 : k;
          setStep(target);
        },
      }, t.text ? el("span", { text: t.text }) : null, lensCls ? el("span", { class: "as-lens", "aria-hidden": "true", text: lensCls === " agree" ? "✓" : "✗" }) : null);
      $(".as-chips", group).appendChild(chip);
    });
    updateActionStrip();
  }
  function updateActionStrip() {
    const root = $("#actionStrip");
    const s = S.replay.step;
    const hadFocus = root.contains(document.activeElement);
    let cur = null;
    $$(".as-chip", root).forEach((c) => {
      const k = Number(c.dataset.k);
      c.classList.toggle("cur", k === s);
      c.classList.toggle("past", k < s);
      c.classList.toggle("next", k === s + 1 && isDecisionStep(k));
      c.tabIndex = k === s ? 0 : -1;
      if (k === s) { c.setAttribute("aria-current", "step"); cur = c; } else c.removeAttribute("aria-current");
    });
    if (cur) {
      // keep the current chip in view without moving the page
      const r = root.getBoundingClientRect(), cr = cur.getBoundingClientRect();
      if (cr.left < r.left + 24 || cr.right > r.right - 24) root.scrollLeft += (cr.left + cr.width / 2) - (r.left + r.width / 2);
      if (hadFocus) cur.focus({ preventScroll: true });
    }
  }

  function decisionAt(step) {
    if (!step || step.decision_id === null || step.decision_id === undefined) return null;
    return S.replay.data.decisions.find((d) => d.id === step.decision_id) || null;
  }

  function lensFor(step) {
    const d = decisionAt(step);
    if (!d) return null;
    const lens = S.replay.lens;
    if (!lens || !lens.decisions) return S.replay.lensSpec ? el("span", { class: "lens pending", text: "…" }) : null;
    const pr = lens.decisions[String(d.id)];
    if (!pr) return el("span", { class: "lens pending", text: "…" });
    if (pr.skipped) return el("span", { class: "lens pending", text: "no cards" });
    return el("span", { class: "lens " + (pr.agree ? "agree" : "disagree"), text: "model: " + actionText(pr.prediction) });
  }

  function actionText(a) {
    if (!a) return "?";
    if (a.action === "bet" || a.action === "raise") return `${a.action} ${fmt(a.to)}`;
    return a.action;
  }

  // opts.animate === false for scrubs; jumps of more than one step never animate.
  function setStep(k, opts) {
    const data = S.replay.data;
    if (!data) return;
    k = Math.max(0, Math.min(k, data.steps.length - 1));
    const from = S.replay.step;
    S.replay.step = k;
    $("#tpSlider").value = k;
    $("#tpLabel").textContent = `${k} / ${data.steps.length - 1}`;
    $$("#stepList .step").forEach((e, i) => e.classList.toggle("active", i === k));
    const active = $("#stepList .step.active");
    if (active) { const list = $("#stepList"), ar = active.getBoundingClientRect(), lr = list.getBoundingClientRect(); if (ar.top < lr.top || ar.bottom > lr.bottom) list.scrollTop += ar.top - lr.top - lr.height / 2; }
    updateActionStrip();
    renderReplayStep(!(opts && opts.animate === false) && Math.abs(k - from) === 1);
  }

  function renderReplayStep(animate) {
    const data = S.replay.data;
    if (!data) return;
    const step = data.steps[S.replay.step];
    const hand = step.state;
    const reveal = $("#tpReveal").checked;
    const viewer = (hand.players.find((p) => p.cards_known) || hand.players[0]).index;
    const order = hand.players.slice().sort((a, b) => ((a.index - viewer + 100) % 100) - ((b.index - viewer + 100) % 100));
    const slots = order.map((p) => ({ slot: p.index, seat: p.index + 1, name: p.name, stack: p.stack, player: p.index }));
    let bubble = null;
    const ev = (step.events || []).find((e) => e.type === "action");
    if (ev) {
      const t = ev.action === "fold" ? "Fold" : ev.action === "check" ? "Check" : ev.action === "call" ? `Call ${fmt(ev.amount)}` : `${ev.action === "bet" ? "Bet" : "Raise to"} ${fmt(ev.to)}`;
      bubble = { player: ev.player, text: t + (ev.all_in ? " (all-in)" : ""), kind: ev.action === "fold" ? "fold" : (ev.action === "bet" || ev.action === "raise") ? "aggr" : "" };
    }
    const show = (step.events || []).find((e) => e.type === "show");
    if (!bubble && show) bubble = { player: show.player, text: "Shows", kind: "" };
    // lens bubble: at the step *before* a decision (the decision point) show the model's intention
    let lens = null;
    const next = data.steps[S.replay.step + 1];
    const d = decisionAt(next);
    if (d && S.replay.lens && S.replay.lens.decisions) {
      const pr = S.replay.lens.decisions[String(d.id)];
      if (pr && !pr.skipped) lens = { player: d.player, text: `Model: ${actionText(pr.prediction)} · actual: ${actionText(d.actual)}`, agree: pr.agree };
    }
    renderTable({ hand, slots, viewer: reveal ? null : viewer, reveal, bubble, lens, payouts: S.replay.step === data.steps.length - 1, animate: !!animate });
  }

  function stepReplay(delta, opts) { setStep(S.replay.step + delta, opts); }
  function togglePlay() { if (S.replay.playing) stopPlay(); else startPlay(); }
  function startPlay() {
    if (!S.replay.data) return;
    stopPlay();
    $("#tpPlay").innerHTML = "&#10074;&#10074;";
    const tick = () => {
      if (S.replay.step >= S.replay.data.steps.length - 1) { stopPlay(); return; }
      setStep(S.replay.step + 1);
      S.replay.playing = setTimeout(tick, Number($("#tpSpeed").value));
    };
    S.replay.playing = setTimeout(tick, Number($("#tpSpeed").value));
  }
  function stopPlay() {
    if (S.replay.playing) clearTimeout(S.replay.playing);
    S.replay.playing = null;
    $("#tpPlay").innerHTML = "&#9654;";
  }

  const FAST = new Set(["random", "station", "maniac", "equity", "tag", "lag"]);
  const isFastSpec = (spec) => FAST.has(String(spec).split(":")[0]);

  function currentLensSpec() {
    return buildSpec($("#lensAgent").value, $("#lensParams").value);
  }

  async function runLens() {
    if (!S.replay.data) return;
    const spec = currentLensSpec();
    S.replay.lensSpec = spec;
    if (isFastSpec(spec)) { await openHand(S.replay.i, { step: S.replay.step }); return; }
    // slow model: sequential per-decision predictions
    if (S.replay.lensBusy) return;
    S.replay.lensBusy = true;
    const data = S.replay.data;
    S.replay.lens = { agent: { kind: spec }, decisions: {} };
    renderStepList();
    try {
      for (const d of data.decisions) {
        if (S.replay.data !== data) break;
        $("#lensSummary").textContent = `predicting ${Object.keys(S.replay.lens.decisions).length + 1}/${data.decisions.length}…`;
        const pr = await api("/api/replay/predict", { i: S.replay.i, decision_id: d.id, agent: spec });
        S.replay.lens.decisions[String(d.id)] = pr;
        renderStepList();
        renderReplayStep(false);
      }
    } catch (e) { toast(e.message, true); }
    S.replay.lensBusy = false;
    updateLensSummary();
  }

  function updateLensSummary() {
    const lens = S.replay.lens;
    const out = $("#lensSummary");
    if (!lens || !lens.decisions) { out.textContent = ""; return; }
    const vals = Object.values(lens.decisions).filter((p) => !p.skipped);
    const agree = vals.filter((p) => p.agree).length;
    out.textContent = vals.length ? `${lens.agent.label || lens.agent.kind}: agrees on ${agree}/${vals.length} decisions` : "no evaluable decisions (cards unknown)";
  }

  function initReplay() {
    initLoadPanel();
    $("#tpFirst").addEventListener("click", () => setStep(0));
    $("#tpPrev").addEventListener("click", () => stepReplay(-1));
    $("#tpNext").addEventListener("click", () => stepReplay(1));
    $("#tpLast").addEventListener("click", () => setStep(S.replay.data ? S.replay.data.steps.length - 1 : 0));
    $("#tpPlay").addEventListener("click", togglePlay);
    $("#tpSlider").addEventListener("input", (e) => setStep(Number(e.target.value), { animate: false }));
    $("#tpReveal").addEventListener("change", () => renderReplayStep(false));
    const sel = $("#lensAgent");
    sel.innerHTML = "";
    sel.appendChild(el("option", { value: "", text: "(none)" }));
    agentOptions(false).forEach((o) => sel.appendChild(o));
    sel.value = "";
    $("#lensRun").addEventListener("click", () => { if ($("#lensAgent").value) runLens(); else { S.replay.lensSpec = null; S.replay.lens = null; renderStepList(); updateLensSummary(); renderReplayStep(false); } });
    $("#forkBtn").addEventListener("click", openForkModal);
    $("#forkCancel").addEventListener("click", () => { $("#forkModal").hidden = true; });
    $("#forkStart").addEventListener("click", startFork);
    $("#forkBoard").addEventListener("change", renderForkAssumptions);
    $("#forkModal").addEventListener("keydown", (e) => { if (e.key === "Escape") { $("#forkModal").hidden = true; $("#forkBtn").focus(); } });
  }

  function openForkModal() {
    const data = S.replay.data;
    if (!data) return;
    if (S.replay.step >= data.steps.length - 1) { toast("The hand is over at this step - pick an earlier step to fork from.", true); return; }
    const hand = data.steps[S.replay.step].state;
    const root = $("#forkSeats");
    root.innerHTML = "";
    hand.players.forEach((p, i) => {
      const row = el("div", { class: "seat-row" },
        el("span", { class: "n", text: `p${p.index + 1}` }),
        el("label", {}, "Player", el("span", { style: "padding:6px 0;font-size:13px", text: p.name + (p.folded ? " (folded)" : "") })),
        el("label", {}, "Controller", el("select", { class: "s-agent", "data-index": p.index }, agentOptions(true))),
        el("label", {}, "Params / URL", el("input", { type: "text", class: "s-params", placeholder: "optional" })),
        el("label", {}, "Stack", el("span", { style: "padding:6px 0;font-size:13px", text: fmt(p.stack) })));
      $(".s-agent", row).value = (i === 0 && !p.folded) ? "human" : "equity";
      if (p.folded) $(".s-agent", row).disabled = true;
      root.appendChild(row);
    });
    renderForkAssumptions();
    $("#forkModal").hidden = false;
    const first = $("#forkSeats select:not([disabled])");
    if (first) first.focus();
  }

  // What the forked hand will know: hole cards known from the record vs dealt at random, and the board run-out.
  function renderForkAssumptions() {
    const data = S.replay.data;
    if (!data) return;
    let box = $("#forkModal .fork-assume");
    if (!box) { box = el("div", { id: "forkAssume", class: "fork-assume" }); $("#forkSeats").before(box); }
    box.innerHTML = "";
    const hand = data.steps[S.replay.step].state;
    const recorded = data.steps[data.steps.length - 1].state.board || [];
    const known = (p) => (p.cards || []).length > 0 && p.cards.every((c) => c && c !== "??" && c !== "XX");
    const live = hand.players.filter((p) => !p.folded);
    const rows = el("div", { class: "fa-rows" });
    live.forEach((p) => rows.appendChild(el("div", { class: "fa-row" },
      el("span", { class: "fa-name", text: p.name }),
      known(p) ? cardsInline(p.cards) : el("span", { class: "fa-random" }, el("span", { class: "card inline unknown", "aria-hidden": "true" }), el("span", { class: "card inline unknown", "aria-hidden": "true" })),
      el("span", { class: "fa-note", text: known(p) ? "known from the record" : (p.cards || []).length ? "unknown: dealt at random" : "not dealt yet: dealt at random" }))));
    const keep = $("#forkBoard").checked;
    const now = hand.board || [];
    const rest = recorded.slice(now.length);
    let boardNote;
    if (now.length >= 5) boardNote = "The board is complete.";
    else if (!keep) boardNote = `The remaining ${5 - now.length} board card${5 - now.length === 1 ? "" : "s"} will be dealt at random.`;
    else if (!rest.length) boardNote = "The record has no later board cards, so they will be dealt at random.";
    else if (recorded.length < 5) boardNote = `Recorded run-out kept (${rest.length} card${rest.length === 1 ? "" : "s"}); the record stops there, so later cards are dealt at random.`;
    else boardNote = "Recorded run-out kept.";
    box.appendChild(el("h4", { text: "Card assumptions" }));
    box.appendChild(rows);
    box.appendChild(el("div", { class: "fa-board" },
      el("span", { class: "fa-name", text: "Board" }),
      el("span", { class: "fa-cards" }, now.length ? cardsInline(now) : el("span", { class: "muted", text: "none yet" }),
        keep && rest.length && now.length < 5 ? el("span", { class: "fa-runout" }, el("span", { class: "muted", text: " then " }), cardsInline(rest)) : null),
      el("span", { class: "fa-note", text: boardNote })));
    if (hand.players.some((p) => p.folded)) box.appendChild(el("div", { class: "hint", text: "Folded players stay folded." }));
  }

  async function startFork() {
    const seats = {};
    $$("#forkSeats .seat-row").forEach((row) => {
      const sel = $(".s-agent", row);
      if (!sel.disabled) seats[sel.dataset.index] = buildSpec(sel.value, $(".s-params", row).value);
    });
    try {
      const st = await api("/api/scenario/start", {
        i: S.replay.i, step_index: S.replay.step, seats, use_recorded_board: $("#forkBoard").checked,
        spectate: $("#forkSpectate").checked, bot_delay_ms: Number($("#botDelay").value) || 700,
      });
      $("#forkModal").hidden = true;
      $("#setupBox").open = false;
      $("#revealToggle").checked = st.reveal;
      S.prevHand = null;
      onGameState(st);
      setTab("play");
      toast("Playing on from the recorded spot. Use 'Next hand' to re-deal the same scenario.");
    } catch (e) { toast(e.message, true); }
  }

  // ------------------------------------------------------------ EVALUATE
  function initEvaluate() {
    const sel = $("#evalAgent");
    agentOptions(false).forEach((o) => sel.appendChild(o));
    sel.value = "tag";
    $("#evalStart").addEventListener("click", async () => {
      if (!S.replay.list) { toast("Load a hand history in the Replay tab first.", true); return; }
      const body = {
        agent: buildSpec(sel.value, $("#evalParams").value), limit_hands: Number($("#evalHands").value) || null,
        max_decisions: Number($("#evalDecisions").value) || null, use_final_cards: $("#evalFinal").checked,
        only_player: $("#evalPlayer").value.trim() || null,
      };
      try {
        await api("/api/evaluate/start", body);
        $("#evalStop").disabled = false;
        $("#evalProgress").hidden = false;
        $("#evalStatus").textContent = "running…";
        $("#evalResults").innerHTML = "";
        if (!S.sseOk) pollJob("evaluate");
      } catch (e) { toast(e.message, true); }
    });
    $("#evalStop").addEventListener("click", () => api("/api/evaluate/stop", {}).catch(() => {}));
  }

  function onEvalProgress(p) {
    const bar = $("#evalProgress > div");
    if (p.total) bar.style.width = (100 * p.done / p.total).toFixed(1) + "%";
    $("#evalStatus").textContent = `${p.done}/${p.total} hands · ${p.decisions} decisions · agreement ${pct(p.accuracy)}`;
  }

  async function onEvalDone() {
    $("#evalStop").disabled = true;
    try {
      const st = await api("/api/evaluate/state");
      if (st.status === "error") { $("#evalStatus").textContent = "error: " + st.error; return; }
      S.evalResult = st.result;
      S.evalSel = 0;
      S.evalFilter = null;
      $("#evalStatus").textContent = `${st.status} in ${st.result.seconds}s`;
      $("#evalExport").hidden = false;
      renderEvalResults(st.result);
      if (S.tab === "evaluate") renderEvalStage();
    } catch (e) { toast(e.message, true); }
  }

  function tile(label, value, sub) {
    return el("div", { class: "tile" }, el("div", { class: "v", text: value }), el("div", { class: "l", text: label + (sub ? " · " + sub : "") }));
  }

  const SEQ_STEPS = ["--seq-100", "--seq-200", "--seq-300", "--seq-400", "--seq-500", "--seq-600", "--seq-700"];
  const seqIndex = (frac) => Math.max(0, Math.min(SEQ_STEPS.length - 1, Math.floor(frac * SEQ_STEPS.length)));
  function seqColor(frac) {
    // single-hue sequential ramp (ivory on green): --seq-700 is the brightest step
    return `var(${SEQ_STEPS[seqIndex(frac)]})`;
  }
  // text on a seqColor cell: the bright half of the ramp takes dark ink
  function seqInk(frac) { return seqIndex(frac) >= 3 ? "var(--seq-ink-hi)" : "var(--ink)"; }

  const EVAL_ACTIONS = ["fold", "check", "call", "bet", "raise"];
  const evalLabel = (r) => (r.agent && (r.agent.label || r.agent.kind)) || "model";
  function disagreements() {
    const r = S.evalResult;
    if (!r) return [];
    const f = S.evalFilter;
    return (r.rows || []).filter((x) => !x.agree && (!f || (x.actual === f.actual && x.predicted === f.predicted)));
  }

  // Panel: breakdowns and the keyboard-navigable disagreement list. The stage holds the matrix and the selected row.
  function renderEvalResults(r) {
    const root = $("#evalResults");
    root.innerHTML = "";
    const box = el("div", { class: "card-box" });
    if (r.skipped_unknown_cards) box.appendChild(el("p", { class: "hint", text: `${r.skipped_unknown_cards} decisions skipped because the player's cards are unknown in the record. Load hands with known hole cards (Pluribus, WSOP, the iPoker hero-card sample) for full coverage.` }));
    const barsFor = (obj, keys) => {
      const bars = el("div", { class: "bars" });
      keys.filter((k) => obj[k]).forEach((k) => {
        const v = obj[k];
        bars.appendChild(el("div", { class: "bar-row" }, el("span", { text: k }), el("div", { class: "track" }, el("div", { class: "fill", style: `width:${(100 * v.accuracy).toFixed(1)}%` })), el("span", { class: "val", text: `${pct(v.accuracy)} (n=${v.n})` })));
      });
      return bars;
    };
    box.appendChild(el("h4", { text: "Agreement by street" }));
    box.appendChild(barsFor(r.by_street, ["preflop", "flop", "turn", "river"]));
    box.appendChild(el("h4", { text: "Agreement by the human's actual action" }));
    box.appendChild(barsFor(r.by_action, EVAL_ACTIONS));
    if (r.size_error_mean_abs_log_ratio !== null && r.size_error_mean_abs_log_ratio !== undefined) {
      box.appendChild(el("p", { class: "hint", text: `Bet sizing: when both bet/raised, the model's size differed from the human's by a factor of ${Math.exp(r.size_error_mean_abs_log_ratio).toFixed(2)} on average (${r.sized_comparisons} comparisons).` }));
    }
    root.appendChild(box);
    const mbox = el("div", { class: "card-box" });
    mbox.appendChild(el("div", { class: "row between" }, el("h4", { id: "mmTitle", class: "mm-title", style: "margin:0" }), el("button", { class: "btn small mm-clear", onclick: () => setEvalFilter(null) }, "Show all")));
    mbox.appendChild(el("p", { class: "hint", text: "↑ ↓ to move, Enter to open the decision in Replay." }));
    const ml = el("div", { class: "mismatch-list", id: "mmList", role: "listbox", tabindex: "0", "aria-labelledby": "mmTitle" });
    ml.addEventListener("keydown", (e) => {
      const rows = disagreements();
      if (!rows.length) return;
      let k = S.evalSel;
      if (e.key === "ArrowDown") k++; else if (e.key === "ArrowUp") k--;
      else if (e.key === "Home") k = 0; else if (e.key === "End") k = rows.length - 1;
      else if (e.key === "PageDown") k += 10; else if (e.key === "PageUp") k -= 10;
      else if (e.key === "Enter" || e.key === " ") { e.preventDefault(); jumpToDecision(rows[S.evalSel]); return; }
      else return;
      e.preventDefault();
      selectDisagreement(Math.max(0, Math.min(rows.length - 1, k)));
    });
    mbox.appendChild(ml);
    root.appendChild(mbox);
    renderDisagreementList();
  }
  function renderDisagreementList() {
    const ml = $("#evalResults .mismatch-list");
    if (!ml || !S.evalResult) return;
    const r = S.evalResult, rows = disagreements(), f = S.evalFilter;
    const shown = rows.slice(0, 300);
    $("#evalResults .mm-title").textContent = `Disagreements: ${rows.length}${f ? ` · recorded ${f.actual}, model ${f.predicted}` : ""}${r.rows && r.rows.length < r.decisions ? ` (first ${r.rows.length} decisions)` : ""}`;
    $("#evalResults .mm-clear").hidden = !f;
    ml.innerHTML = "";
    shown.forEach((x, i) => {
      ml.appendChild(el("div", { class: "mismatch" + (i === S.evalSel ? " sel" : ""), id: "mm-" + i, "data-i": i, role: "option", "aria-selected": i === S.evalSel ? "true" : "false", onclick: () => selectDisagreement(i), ondblclick: () => jumpToDecision(x) },
        el("span", {}, el("b", { text: x.name }), ` ${x.street} ${x.position} `, cardsInline(x.cards)),
        el("span", { class: "tag", text: "recorded " + actionText({ action: x.actual, to: x.actual_to }) }),
        el("span", { class: "tag", text: "model " + actionText({ action: x.predicted, to: x.predicted_to }) })));
    });
    if (!shown.length) ml.appendChild(el("div", { class: "hint", style: "padding:6px", text: "No disagreements in this selection." }));
    if (shown.length) ml.setAttribute("aria-activedescendant", "mm-" + Math.min(S.evalSel, shown.length - 1)); else ml.removeAttribute("aria-activedescendant");
  }
  function selectDisagreement(i) {
    S.evalSel = i;
    $$("#evalResults .mismatch").forEach((n, k) => { n.classList.toggle("sel", k === i); n.setAttribute("aria-selected", k === i ? "true" : "false"); });
    const cur = $(`#evalResults [data-i="${i}"]`);
    if (cur) { $("#evalResults .mismatch-list").setAttribute("aria-activedescendant", cur.id); const l = $("#evalResults .mismatch-list"), cr = cur.getBoundingClientRect(), lr = l.getBoundingClientRect(); if (cr.top < lr.top || cr.bottom > lr.bottom) l.scrollTop += cr.top - lr.top - lr.height / 2; }
    renderEvalPick();
  }
  function setEvalFilter(f) {
    const same = f && S.evalFilter && S.evalFilter.actual === f.actual && S.evalFilter.predicted === f.predicted;
    S.evalFilter = same ? null : f;
    S.evalSel = 0;
    renderDisagreementList();
    renderEvalStage();
  }

  function emptyStage(title, text, action) {
    return el("div", { class: "stage-empty" }, el("h2", { class: "stage-title", text: title }), el("p", { text }),
      action ? el("button", { class: "btn primary", onclick: action.onclick }, action.label) : null);
  }

  function renderEvalStage() {
    const root = $("#simStage");
    if (!root || S.tab !== "evaluate") return;
    root.innerHTML = "";
    const r = S.evalResult;
    if (!r) {
      root.appendChild(S.replay.list
        ? emptyStage("Shadow evaluation", `${S.replay.list.count} hands loaded from ${S.replay.list.label}. Choose a model on the right and run it: every recorded decision is shown to the model exactly as the player saw it. The confusion matrix and the disagreements appear here.`)
        : emptyStage("Shadow evaluation", "Load a hand history in Replay, then run a model against every recorded decision. The confusion matrix and the disagreements appear here.", { label: "Open Replay", onclick: () => { setTab("replay"); navHash("replay"); } }));
      return;
    }
    root.appendChild(el("header", { class: "stage-head" },
      el("h2", { class: "stage-title", text: `${evalLabel(r)} against recorded decisions` }),
      el("p", { class: "stage-sub" }, el("span", { class: "num", text: fmt(r.decisions) }), ` decisions from ${r.hands_evaluated} of ${r.hands} hands · same action `, el("b", { class: "num", text: pct(r.accuracy) }), " · same intent (fold / passive / aggressive) ", el("b", { class: "num", text: pct(r.accuracy_coarse) }),
        r.latency_ms_mean === null || r.latency_ms_mean === undefined ? null : ` · ${r.latency_ms_mean.toFixed(0)} ms per decision`)));
    const grid = el("div", { class: "eval-grid" });
    // confusion matrix: rows recorded, columns model. Off-diagonal cells filter the disagreement list.
    const maxv = Math.max(1, ...EVAL_ACTIONS.flatMap((a) => EVAL_ACTIONS.map((b) => r.confusion[a][b])));
    const f = S.evalFilter;
    const table = el("table", { class: "cm" }, el("caption", { class: "sr-only", text: "Confusion matrix: recorded action by row, model suggestion by column" }));
    table.appendChild(el("tr", {}, el("th", { class: "cm-corner", scope: "col" }, el("span", { text: "recorded ↓" }), el("span", { text: "model →" })), ...EVAL_ACTIONS.map((a) => el("th", { scope: "col", text: a }))));
    EVAL_ACTIONS.forEach((a) => {
      const row = el("tr", {}, el("th", { scope: "row", text: a }));
      EVAL_ACTIONS.forEach((b) => {
        const v = r.confusion[a][b], frac = v / maxv;
        const on = f && f.actual === a && f.predicted === b;
        const style = `background:${v ? seqColor(frac) : "transparent"};color:${v ? seqInk(frac) : "var(--muted)"}`;
        const label = `recorded ${a}, model ${b}: ${v} decision${v === 1 ? "" : "s"}`;
        const td = el("td", { class: (a === b ? "diag" : "") + (on ? " on" : ""), style });
        if (a !== b && v) td.appendChild(el("button", { class: "cm-cell", "aria-pressed": on ? "true" : "false", "aria-label": label + (on ? " (filtering the list)" : ". Show these disagreements"), title: label, onclick: () => setEvalFilter({ actual: a, predicted: b }) }, String(v)));
        else td.appendChild(el("span", { title: label, text: v ? String(v) : "·" }));
        row.appendChild(td);
      });
      table.appendChild(row);
    });
    grid.appendChild(el("figure", { class: "eval-fig" }, table,
      el("figcaption", { class: "hint", text: "Diagonal: the model chose the recorded action. Select an off-diagonal cell to list only those disagreements; select it again to show all." })));
    grid.appendChild(el("div", { class: "eval-pick", id: "evalPick" }));
    root.appendChild(grid);
    renderEvalPick();
  }

  function renderEvalPick() {
    const box = $("#simStage .eval-pick");
    if (!box) return;
    box.innerHTML = "";
    const rows = disagreements();
    if (!rows.length) { box.appendChild(el("p", { class: "hint", text: S.evalResult && S.evalResult.rows && S.evalResult.rows.length ? "The model agreed with every recorded decision in this selection." : "No per-decision rows were returned for this run." })); return; }
    const i = Math.min(S.evalSel, rows.length - 1), x = rows[i];
    const step = (d) => () => { selectDisagreement(Math.max(0, Math.min(Math.min(rows.length, 300) - 1, i + d))); };
    box.appendChild(el("div", { class: "row between" },
      el("span", { class: "eyebrow", text: `Disagreement ${i + 1} of ${rows.length}` }),
      el("span", { class: "row", style: "margin:0" }, el("button", { class: "icon-btn", "aria-label": "Previous disagreement", disabled: i === 0, onclick: step(-1) }, "‹"), el("button", { class: "icon-btn", "aria-label": "Next disagreement", disabled: i >= Math.min(rows.length, 300) - 1, onclick: step(1) }, "›"))));
    box.appendChild(el("h3", { class: "pick-title", text: x.name }));
    box.appendChild(el("div", { class: "muted", text: `${x.street} · ${x.position} · ${x.hand}` }));
    box.appendChild(el("div", { class: "pick-cards" }, (x.cards || []).map((c) => cardEl(c, "big"))));
    box.appendChild(el("dl", { class: "pick-facts" },
      el("dt", { text: "Pot" }), el("dd", { class: "num", text: fmt(x.pot) }),
      el("dt", { text: "To call" }), el("dd", { class: "num", text: x.to_call ? fmt(x.to_call) : "nothing" })));
    const sized = (a, to) => (a === "bet" || a === "raise") && to !== null && to !== undefined ? `${a} to ${fmt(to)}` : a;
    box.appendChild(el("div", { class: "pick-vs" },
      el("div", { class: "pv" }, el("span", { class: "pv-l", text: "Recorded" }), el("span", { class: "pv-v", text: sized(x.actual, x.actual_to) })),
      el("div", { class: "pv model" }, el("span", { class: "pv-l", text: "Model suggestion" }), el("span", { class: "pv-v", text: sized(x.predicted, x.predicted_to) }))));
    if (x.note) box.appendChild(el("p", { class: "hint", text: "Note: " + x.note }));
    if (x.reasoning) box.appendChild(el("p", { class: "pick-reason", text: x.reasoning }));
    box.appendChild(el("button", { class: "btn primary", onclick: () => jumpToDecision(x) }, "Open this decision →"));
  }

  async function jumpToDecision(row) {
    S.replay.lensSpec = S.evalResult && S.evalResult.agent ? specFromDescribe(S.evalResult.agent) : S.replay.lensSpec;
    setTab("replay");
    navHash("replay");
    await openHand(row.list_index);
    const data = S.replay.data;
    const idx = data.steps.findIndex((s) => s.decision_id === row.decision_id);
    if (idx > 0) setStep(idx - 1);
  }

  function specFromDescribe(desc) {
    if (!desc || !desc.kind) return null;
    const params = desc.params || {};
    const parts = Object.entries(params).filter(([k]) => k !== "seed" && k !== "name").map(([k, v]) => `${k}=${v}`);
    if (desc.kind === "http") return "http:" + (params.url || "");
    if (desc.kind === "claude") return "claude:" + (params.model || "claude-opus-5-5") + (parts.filter((p) => !p.startsWith("model=")).length ? "," + parts.filter((p) => !p.startsWith("model=")).join(",") : "");
    return parts.length ? `${desc.kind}:${parts.join(",")}` : desc.kind;
  }

  // ------------------------------------------------------------ SIMULATE
  function initSimulate() {
    const c = $("#simSeats");
    [{ agent: "equity" }, { agent: "tag" }, { agent: "lag" }, { agent: "random" }].forEach((d, i) => seatRow(c, i, d));
    $("#simAdd").addEventListener("click", () => { if ($$(".seat-row", c).length < 8) seatRow(c, $$(".seat-row", c).length, {}); });
    $("#simRemove").addEventListener("click", () => { const rows = $$(".seat-row", c); if (rows.length > 2) rows[rows.length - 1].remove(); });
    $("#simStart").addEventListener("click", async () => {
      const seats = readSeats(c);
      const body = {
        agents: seats.map((s) => s.agent), names: seats.map((s, i) => s.name || null), hands: Number($("#simHands").value) || 100,
        stack: Number($("#simStack").value) || 200, small_blind: Number($("#simSb").value) || 1, big_blind: Number($("#simBb").value) || 2,
        seed: $("#simSeed").value ? Number($("#simSeed").value) : null, reset_stacks: $("#simReset").checked,
      };
      if (body.names.some((n) => !n)) body.names = null;
      try {
        await api("/api/simulate/start", body);
        S.simReq = body;
        $("#simProgress").hidden = false;
        $("#simStatus").textContent = "running…";
        $("#simResults").innerHTML = "";
        if (!S.sseOk) pollJob("simulate");
      } catch (e) { toast(e.message, true); }
    });
  }

  function onSimProgress(p) {
    const bar = $("#simProgress > div");
    if (p.total) bar.style.width = (100 * p.done / p.total).toFixed(1) + "%";
    $("#simStatus").textContent = `${p.done}/${p.total} hands`;
  }

  async function onSimDone() {
    try {
      const st = await api("/api/simulate/state");
      if (st.status === "error") { $("#simStatus").textContent = "error: " + st.error; return; }
      S.simResult = st.result;
      S.simSel = null;
      S.simCursor = null;
      $("#simStatus").textContent = `${st.result.hands} hands in ${st.result.seconds}s`;
      $("#simExport").hidden = false;
      renderSimResults(st.result);
      if (S.tab === "simulate") renderSimStage();
    } catch (e) { toast(e.message, true); }
  }

  // Panel keeps the controls; the chart and the bb/100 table are on the stage.
  function renderSimResults(r) {
    const root = $("#simResults");
    root.innerHTML = "";
    root.appendChild(el("p", { class: "hint", text: `${r.hands} hands played. The cumulative chart and bb/100 with standard errors are on the left.` }));
  }

  const signed = (v, d) => (v > 0 ? "+" : v < 0 ? "−" : "") + (d === undefined ? fmt(Math.abs(v)) : Math.abs(v).toFixed(d));
  function niceStep(span, target) {
    const raw = span / Math.max(1, target), mag = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / mag;
    return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * mag;
  }
  function simSeries(r) {
    return r.seats.map((st, i) => ({ seat: st.seat, name: st.name, values: r.cumulative[String(st.seat)] || [], color: `var(--series-${(i % 8) + 1})` }));
  }

  function renderSimStage() {
    const root = $("#simStage");
    if (!root || S.tab !== "simulate") return;
    root.innerHTML = "";
    const r = S.simResult;
    if (!r) {
      root.appendChild(emptyStage("Self-play", "Choose the seats on the right and run. Models and bots play each other for many hands; the cumulative chip chart and bb/100 with standard errors appear here."));
      return;
    }
    const q = S.simReq || {};
    root.appendChild(el("header", { class: "stage-head" },
      el("h2", { class: "stage-title", text: `Self-play · ${fmt(r.hands)} hands` }),
      el("p", { class: "stage-sub", text: [`${r.seats.length} players`, q.big_blind ? `blinds ${fmt(q.small_blind)}/${fmt(q.big_blind)}` : null, q.stack ? `${fmt(q.stack)} chip stacks` : null, q.reset_stacks === false ? "stacks carried over" : q.reset_stacks ? "stacks reset every hand" : null, q.seed !== null && q.seed !== undefined ? `seed ${q.seed}` : null, `${r.seconds}s`].filter(Boolean).join(" · ") })));
    const series = simSeries(r);
    root.appendChild(simChart(series, root.clientWidth || 800));
    // results table: bb/100 and its standard error read together
    const table = el("table", { class: "mini sim-table" }, el("thead", {}, el("tr", {}, el("th", { text: "Seat" }), el("th", { text: "Player" }), el("th", { text: "Type" }), el("th", { class: "r", text: "Net, chips" }), el("th", { class: "r", text: "bb/100 ± s.e." }))));
    const tb = el("tbody");
    r.seats.forEach((st, i) => tb.appendChild(el("tr", { class: S.simSel === st.seat ? "sel" : "" },
      el("td", { text: st.seat }),
      el("td", {}, el("span", { class: "swatch", style: `--c:var(--series-${(i % 8) + 1})`, "aria-hidden": "true" }), st.name),
      el("td", { text: st.agent.kind }),
      el("td", { class: "r num", text: signed(st.net) }),
      el("td", { class: "r num" }, el("b", { text: signed(st.bb_per_100, 1) }), el("span", { class: "muted", text: ` ± ${st.bb_per_100_stderr.toFixed(1)}` })))));
    table.appendChild(tb);
    root.appendChild(table);
    root.appendChild(el("p", { class: "hint", text: "bb/100 = big blinds won per 100 hands, ± one standard error. Differences smaller than about two standard errors are noise: run more hands." }));
  }

  // Cumulative net by hand: one y-axis (chips), direct end labels for ≤4 series plus a legend,
  // legend selection highlights one agent, and the crosshair moves with the mouse or Left/Right/Home/End.
  function simChart(series, width) {
    const n = Math.max(1, ...series.map((sr) => sr.values.length));
    const W = Math.max(480, Math.round(width)), H = Math.round(Math.max(260, Math.min(420, W * 0.4, window.innerHeight - 330)));
    const direct = series.length <= 4;
    const m = { l: 64, r: direct ? 150 : 24, t: 26, b: 42 };
    let ymin = 0, ymax = 0;
    series.forEach((sr) => sr.values.forEach((v) => { ymin = Math.min(ymin, v); ymax = Math.max(ymax, v); }));
    if (ymin === ymax) { ymin -= 1; ymax += 1; }
    const step = niceStep(ymax - ymin, 5);
    ymin = Math.floor(ymin / step) * step; ymax = Math.ceil(ymax / step) * step;
    const x = (i) => m.l + (W - m.l - m.r) * (n > 1 ? i / (n - 1) : 0.5);
    const y = (v) => m.t + (H - m.t - m.b) * (1 - (v - ymin) / (ymax - ymin));
    const NS = "http://www.w3.org/2000/svg";
    const mk = (tag, attrs, text) => { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); if (text !== undefined) e.textContent = text; return e; };
    const svg = mk("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: "sim-svg", tabindex: "0", role: "group", "aria-label": `Cumulative net chips by hand for ${series.length} players. Use left and right arrows to move through hands; the readout below the chart updates.` });
    // grid + y ticks (recessive, solid hairlines)
    for (let v = ymin; v <= ymax + step / 2; v += step) {
      svg.appendChild(mk("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: Math.abs(v) < step / 2 ? "var(--ink-2)" : "var(--line)", "stroke-width": Math.abs(v) < step / 2 ? 1.25 : 1 }));
      svg.appendChild(mk("text", { x: m.l - 8, y: y(v) + 4, "text-anchor": "end", class: "tick" }, Math.abs(v) < step / 2 ? "0" : signed(Math.round(v))));
    }
    // x ticks
    const xs = Math.max(1, Math.round(niceStep(n, 6)));
    for (let h = 0; h <= n; h += xs) { const i = Math.max(0, Math.min(n - 1, h === 0 ? 0 : h - 1)); svg.appendChild(mk("text", { x: x(i), y: H - m.b + 18, "text-anchor": "middle", class: "tick" }, String(h === 0 ? 1 : h))); }
    svg.appendChild(mk("text", { x: m.l, y: 14, class: "axis-title" }, "Cumulative net, chips"));
    svg.appendChild(mk("text", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle", class: "axis-title" }, "Hand"));
    const sel = S.simSel;
    const paths = [];
    series.forEach((sr) => {
      if (!sr.values.length) return;
      const d = sr.values.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
      const on = sel === null || sel === sr.seat;
      const path = mk("path", { d, fill: "none", stroke: sr.color, "stroke-width": sel === sr.seat ? 3 : 2, "stroke-linejoin": "round", "stroke-linecap": "round", opacity: on ? 1 : 0.22 });
      paths.push(path);
      svg.appendChild(path);
    });
    // direct end labels with leader lines when they would collide
    if (direct) {
      const ends = series.filter((sr) => sr.values.length).map((sr) => ({ sr, y0: y(sr.values[sr.values.length - 1]), v: sr.values[sr.values.length - 1] })).sort((a, b) => a.y0 - b.y0);
      const gap = 16;
      ends.forEach((e, i) => { e.y = i ? Math.max(e.y0, ends[i - 1].y + gap) : e.y0; });
      const over = ends.length ? ends[ends.length - 1].y - (H - m.b) : 0;
      if (over > 0) ends.forEach((e) => { e.y -= over; });
      ends.forEach((e) => {
        const xe = x(e.sr.values.length - 1), dim = sel !== null && sel !== e.sr.seat;
        if (Math.abs(e.y - e.y0) > 2) svg.appendChild(mk("path", { d: `M${xe + 4},${e.y0} L${xe + 12},${e.y}`, stroke: "var(--muted)", "stroke-width": 1, fill: "none", opacity: dim ? 0.3 : 1 }));
        svg.appendChild(mk("text", { x: xe + 14, y: e.y + 4, class: "end-label" + (dim ? " dim" : "") }, `${e.sr.name.length > 14 ? e.sr.name.slice(0, 13) + "…" : e.sr.name} ${signed(Math.round(e.v))}`));
      });
    }
    // crosshair + markers
    const cross = mk("line", { x1: 0, x2: 0, y1: m.t, y2: H - m.b, stroke: "var(--ink-2)", "stroke-width": 1, visibility: "hidden" });
    svg.appendChild(cross);
    const dots = series.map((sr) => { const c = mk("circle", { r: 4.5, fill: sr.color, stroke: "var(--surface)", "stroke-width": 2, visibility: "hidden" }); svg.appendChild(c); return c; });
    const wrap = el("figure", { class: "sim-chart" });
    const legend = el("div", { class: "sim-legend", role: "group", "aria-label": "Highlight a player" });
    series.forEach((sr) => legend.appendChild(el("button", { class: "sim-lg" + (sel === sr.seat ? " on" : ""), type: "button", "aria-pressed": sel === sr.seat ? "true" : "false", onclick: () => { S.simSel = S.simSel === sr.seat ? null : sr.seat; renderSimStage(); } },
      el("span", { class: "sim-key", style: `--c:${sr.color}`, "aria-hidden": "true" }), sr.name)));
    const readout = el("p", { class: "sim-readout", "aria-live": "polite" });
    const tip = el("div", { class: "chart-tip", hidden: true });
    const setCursor = (i, show) => {
      i = Math.max(0, Math.min(n - 1, i));
      S.simCursor = i;
      const vis = show ? "visible" : "hidden";
      cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("visibility", vis);
      series.forEach((sr, k) => { const v = sr.values[i]; if (v === undefined) { dots[k].setAttribute("visibility", "hidden"); return; } dots[k].setAttribute("cx", x(i)); dots[k].setAttribute("cy", y(v)); dots[k].setAttribute("visibility", vis); });
      const ordered = sel === null ? series : [series.find((sr) => sr.seat === sel)].concat(series.filter((sr) => sr.seat !== sel));
      readout.innerHTML = "";
      readout.appendChild(el("b", { text: `Hand ${i + 1}` }));
      ordered.forEach((sr) => { readout.appendChild(document.createTextNode(" · ")); readout.appendChild(el("span", { class: sel === sr.seat ? "ro-sel" : "" }, el("span", { class: "sim-key", style: `--c:${sr.color}`, "aria-hidden": "true" }), `${sr.name} `, el("span", { class: "num", text: signed(Math.round(sr.values[i] || 0)) }))); });
      readout.appendChild(document.createTextNode(" chips"));
      return i;
    };
    setCursor(S.simCursor === null ? n - 1 : S.simCursor, S.simCursor !== null);
    svg.addEventListener("mousemove", (e) => {
      const rect = svg.getBoundingClientRect();
      const px = (e.clientX - rect.left) * W / rect.width;
      const i = setCursor(Math.round((px - m.l) / (W - m.l - m.r) * (n - 1)), true);
      tip.hidden = false;
      tip.innerHTML = "";
      tip.appendChild(el("b", { text: `Hand ${i + 1}` }));
      series.forEach((sr) => tip.appendChild(el("div", {}, el("span", { class: "sim-key", style: `--c:${sr.color}` }), `${sr.name}: `, el("span", { class: "num", text: signed(Math.round(sr.values[i] || 0)) }))));
      const wr = wrap.getBoundingClientRect();
      tip.style.left = Math.min(e.clientX - wr.left + 14, wr.width - 190) + "px";
      tip.style.top = (e.clientY - wr.top + 12) + "px";
    });
    svg.addEventListener("mouseleave", () => { tip.hidden = true; });
    svg.addEventListener("keydown", (e) => {
      const cur = S.simCursor === null ? n - 1 : S.simCursor, big = Math.max(1, Math.round(n / 10));
      const k = { ArrowRight: cur + 1, ArrowLeft: cur - 1, Home: 0, End: n - 1, PageUp: cur + big, PageDown: cur - big }[e.key];
      if (k === undefined) return;
      e.preventDefault();
      tip.hidden = true;
      setCursor(k, true);
    });
    svg.addEventListener("focus", () => setCursor(S.simCursor === null ? n - 1 : S.simCursor, true));
    wrap.appendChild(legend);
    wrap.appendChild(svg);
    wrap.appendChild(readout);
    wrap.appendChild(tip);
    return wrap;
  }

  // ------------------------------------------------------------ SSE
  function connect() {
    const conn = $("#conn");
    const es = new EventSource("/api/events");
    es.onopen = () => { S.sseOk = true; conn.className = "conn ok"; conn.lastChild.textContent = " live"; };
    es.onerror = () => { S.sseOk = false; conn.className = "conn bad"; conn.lastChild.textContent = " reconnecting"; };
    es.addEventListener("game", (e) => onGameState(JSON.parse(e.data).state));
    es.addEventListener("eval_progress", (e) => onEvalProgress(JSON.parse(e.data)));
    es.addEventListener("eval_done", onEvalDone);
    es.addEventListener("sim_progress", (e) => onSimProgress(JSON.parse(e.data)));
    es.addEventListener("sim_done", onSimDone);
  }

  // ------------------------------------------------------------ boot
  async function boot() {
    try {
      const a = await api("/api/agents");
      S.agents = a.agents;
      S.agentInfo = a;
      const hints = $("#modelHints");
      hints.innerHTML = "";
      if (!a.anthropic_installed) hints.appendChild(el("span", { class: "hint", text: "Claude seats need: pip install anthropic (and an API key)." }));
      else if (!a.anthropic_key) hints.appendChild(el("span", { class: "hint", text: "Claude seats: set ANTHROPIC_API_KEY (or run 'ant auth login') before starting the server." }));
    } catch (e) { toast("Could not reach the server: " + e.message, true); S.agents = [{ id: "equity", label: "Equity bot" }]; }
    initPlay();
    initReplay();
    initEvaluate();
    initSimulate();
    if (!location.hash.includes("nosse")) connect();  // #...&nosse=1 disables live updates (headless screenshots)
    try {
      const st = await api("/api/game/state");
      if (st.active) { $("#setupBox").open = false; $("#revealToggle").checked = st.reveal; onGameState(st); }
    } catch (e) { /* no game yet */ }
    window.addEventListener("resize", () => {
      if (S.tab === "play" && S.game && S.game.active) renderGame(S.game, true);
      else if (S.tab === "replay" && S.replay.data) renderReplayStep(false);
      else if (S.tab === "simulate") renderSimStage();
      else if (S.tab === "evaluate") renderEvalStage();
    });
    // Judges land on the Mirror opening; existing deep links (#replay=, #sim=, #demo, #tab=) still open their tools.
    if (!location.hash) setTab("mirror");
    await handleHash();
  }

  // Deep links: #demo (bot-only table), #replay=<sample file>&i=<hand>&step=<step>&lens=<agent spec>
  async function handleHash() {
    const h = location.hash.replace(/^#/, "");
    if (!h) return;
    const params = new URLSearchParams(h);
    const handled = ["demo", "replay", "sim", "mirror", "evidence", "tab"].some((k) => params.has(k)) || h === "demo";
    if (!handled) setTab("mirror");
    if (params.has("tab") && SIM_TABS.includes(params.get("tab"))) setTab(params.get("tab"));   // #tab=play|replay|evaluate|simulate
    if (h === "demo" || params.has("demo")) {
      setTab("play");
      $$("#seatConfig .s-agent").forEach((sel, i) => { sel.value = ["tag", "equity", "lag", "equity", "random", "maniac"][i % 6]; });
      $("#spectate").checked = true;
      $("#botDelay").value = params.get("delay") || 1200;
      await startGame();
      return;
    }
    if (params.has("replay")) {
      setTab("replay");
      const file = params.get("replay");
      if (params.get("lens")) {
        S.replay.lensSpec = params.get("lens");
        const [kind, ...rest] = params.get("lens").split(":");
        if ($(`#lensAgent option[value="${kind}"]`)) { $("#lensAgent").value = kind; $("#lensParams").value = rest.join(":"); }
      }
      await loadRecords({ source: "sample", path: file, limit: Number($("#loadLimit").value) || null }, { open: false });
      const i = Number(params.get("i") || 0);
      await openHand(i, { step: Number(params.get("step") || 0) });
      if (params.get("lens") && !isFastSpec(params.get("lens"))) runLens();
      if (params.has("eval")) {
        setTab("evaluate");
        $("#evalStatus").textContent = "running…";
        await api("/api/evaluate/start", { agent: params.get("eval") || "tag", limit_hands: Number(params.get("hands") || 100) });
        await pollJob("evaluate");
      }
    }
    if (params.has("sim")) {
      setTab("simulate");
      const agents = (params.get("sim") || "equity,tag,lag,random").split(",").filter(Boolean);
      S.simReq = { agents, hands: Number(params.get("hands") || 100), seed: 1 };
      await api("/api/simulate/start", S.simReq);
      await pollJob("simulate");
    }
    if (params.has("mirror")) setTab("mirror");   // #mirror=<step>&deal=1 | &spot=1 | &reveal=1 (handled in mirror.js)
    if (params.has("evidence")) setTab("evidence");
  }

  // Poll a background job (fallback when live updates are unavailable).
  async function pollJob(kind) {
    for (;;) {
      let st;
      try { st = await api(`/api/${kind}/state`); } catch (e) { return; }
      if (st.status === "running") {
        if (st.progress && st.progress.total) (kind === "evaluate" ? onEvalProgress : onSimProgress)(st.progress);
        await new Promise((r) => setTimeout(r, 400));
        continue;
      }
      if (kind === "evaluate") await onEvalDone(); else await onSimDone();
      return;
    }
  }

  // Shared helpers for the Mirror module (web/mirror.js).
  window.PokerSim = { api, el, cardEl, cardsInline, renderTable, toast, fmt, pct, setTab, showOverlay, tile, seqColor, seqInk, openReplay, state: S };

  boot();
})();
