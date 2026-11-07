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
    fourColor: false,
  };

  // ------------------------------------------------------------ tabs
  function setTab(name) {
    S.tab = name;
    $$("#tabs .tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
    $$(".pane").forEach((p) => p.classList.toggle("active", p.id === "pane-" + name));
    $("#transport").hidden = name !== "replay" || !S.replay.data;
    if (name === "play") {
      if (S.game && S.game.active) renderGame(S.game, true);
      else showOverlay(true);
    } else if (name === "replay") {
      $("#actionBar").hidden = true;
      if (S.replay.data) renderReplayStep(); else showOverlay(true, "Load a hand history to start replaying.");
    } else if (name === "mirror" || name === "evidence") {
      $("#actionBar").hidden = true;
      if (window.Mirror) window.Mirror.onShow(name);
    } else {
      $("#actionBar").hidden = true;
      if (name === "evaluate") $("#evalStatus").textContent = S.replay.list ? `${S.replay.list.count} hands loaded (${S.replay.list.label})` : "Load a hand history in Replay first.";
      if (S.game && S.game.active) renderGame(S.game, true);
    }
  }
  $$("#tabs .tab").forEach((b) => b.addEventListener("click", () => setTab(b.dataset.tab)));

  function showOverlay(show, msg) {
    const ov = $("#tableOverlay");
    ov.hidden = !show;
    if (show && msg) ov.querySelector("p").textContent = msg;
    if (show) { $("#seats").innerHTML = ""; $("#board").innerHTML = ""; $("#pot").innerHTML = ""; $("#centerMsg").innerHTML = ""; }
  }

  $("#fourColor").addEventListener("change", (e) => document.body.classList.toggle("four-color", e.target.checked));

  // ------------------------------------------------------------ table renderer
  /**
   * view = { hand, slots: [{slot, seat, name, stack, kind, player (index|null), sittingOut}], viewer (player index|null),
   *          reveal, bubble: {player, text, kind}, lens: {player, text, agree}, payouts: bool }
   */
  function renderTable(view) {
    showOverlay(false);
    const hand = view.hand;
    const wrap = $("#tableWrap");
    const seatsRoot = $("#seats");
    seatsRoot.innerHTML = "";
    const n = view.slots.length;
    const W = wrap.clientWidth || 900, H = wrap.clientHeight || 520;
    const cx = W / 2, cy = H / 2;
    const rx = W * 0.405, ry = H * 0.40;
    const brx = W * 0.255, bry = H * 0.225;
    const players = hand.players;
    const byIndex = {};
    players.forEach((p) => { byIndex[p.index] = p; });
    const prev = S.prevHand && S.prevHand.hand_id === hand.hand_id ? S.prevHand : null;
    const prevBy = {};
    if (prev) prev.players.forEach((p) => { prevBy[p.index] = p; });

    view.slots.forEach((slot, k) => {
      const a = (Math.PI / 2) + (2 * Math.PI * k) / n;
      const x = cx + rx * Math.cos(a), y = cy + ry * Math.sin(a);
      const p = slot.player !== null && slot.player !== undefined ? byIndex[slot.player] : null;
      const seat = el("div", { class: "seat", style: `left:${x}px;top:${y}px` });
      if (p) {
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
        const tags = el("div", { class: "tags" }, el("span", { class: "tag pos", text: p.position }));
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
        if (hand.status === "betting" && hand.actor === p.index) seat.classList.add("actor");
        if (view.thinking === p.index) seat.classList.add("thinking");
        if (hand.status === "finished" && p.payout > 0) seat.classList.add("winner");
        // bet chips
        if (p.bet > 0) {
          const bx = cx + brx * Math.cos(a), by = cy + bry * Math.sin(a);
          const bet = el("div", { class: "bet", style: `left:${bx}px;top:${by}px` }, chipsEl(p.bet, hand.min_bet), el("span", { text: fmt(p.bet) }));
          if (pp && pp.bet === p.bet) bet.style.animation = "none";
          seatsRoot.appendChild(bet);
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
    // pot
    const pot = $("#pot");
    pot.innerHTML = "";
    const streetLabel = hand.status === "finished" ? "Showdown" : hand.status === "incomplete" ? "Incomplete" : hand.street.charAt(0).toUpperCase() + hand.street.slice(1);
    pot.appendChild(el("div", { class: "pot-main" }, el("span", { class: "muted", text: streetLabel + " · Pot" }), el("b", { text: fmt(hand.total_pot) })));
    if (hand.pots && hand.pots.length > 1 && hand.status === "finished") {
      hand.pots.forEach((sp, i) => pot.appendChild(el("div", { class: "pot-side", text: `${i === 0 ? "Main" : "Side " + i}: ${fmt(sp.amount)} → ${sp.winners.map((w) => byIndex[w] ? byIndex[w].name : "p" + (w + 1)).join(", ")}` })));
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

  function initSeatConfig() {
    const c = $("#seatConfig");
    c.innerHTML = "";
    [{ name: "You", agent: "human" }, { agent: "tag" }, { agent: "equity" }, { agent: "lag" }, { agent: "equity" }, { agent: "random" }]
      .forEach((d, i) => seatRow(c, i, d));
    $("#addSeat").addEventListener("click", () => { if ($$(".seat-row", c).length < 10) seatRow(c, $$(".seat-row", c).length, {}); });
    $("#removeSeat").addEventListener("click", () => { const rows = $$(".seat-row", c); if (rows.length > 2) rows[rows.length - 1].remove(); });
  }

  async function startGame() {
    const seats = readSeats($("#seatConfig"));
    const body = {
      seats, small_blind: Number($("#sb").value), big_blind: Number($("#bb").value), ante: Number($("#ante").value) || 0,
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
    $("#startGame").addEventListener("click", startGame);
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
      if (e.target.matches("input, select, textarea")) return;
      if (S.tab === "play" && !$("#actionBar").hidden) {
        if (e.key === "f" || e.key === "F") humanAct("fold");
        else if (e.key === "c" || e.key === "C") humanAct("call");
        else if (e.key === "r" || e.key === "R" || e.key === "Enter") { if (!$(".raise-group").hidden) humanAct("raise", Number($("#raiseInput").value)); }
      } else if (S.tab === "play" && S.game && S.game.active && (S.game.hand.status === "finished" || S.game.hand.status === "incomplete") && (e.key === "Enter" || e.key === " ")) {
        e.preventDefault();
        $("#nextHand").click();
      } else if (S.tab === "replay" && S.replay.data) {
        if (e.key === "ArrowRight") stepReplay(1);
        else if (e.key === "ArrowLeft") stepReplay(-1);
        else if (e.key === " ") { e.preventDefault(); togglePlay(); }
        else if (e.key === "Home") setStep(0);
        else if (e.key === "End") setStep(S.replay.data.steps.length - 1);
      }
    });
  }

  // ------------------------------------------------------------ REPLAY: loading
  async function loadRecords(body) {
    try {
      $("#handsTitle").textContent = "Loading…";
      $("#handsBox").hidden = false;
      const list = await api("/api/replay/load", body);
      S.replay.list = list;
      S.replay.data = null;
      S.replay.lens = null;
      renderHandList();
      $("#loadBox").open = false;
      toast(`Loaded ${list.count} hands from ${list.label}`);
      if (list.hands.length) openHand(list.hands[0].i);
      if (S.tab === "evaluate") setTab("evaluate");
    } catch (e) { toast(e.message, true); $("#handsTitle").textContent = "Hands"; }
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

  function setStep(k) {
    const data = S.replay.data;
    if (!data) return;
    k = Math.max(0, Math.min(k, data.steps.length - 1));
    S.replay.step = k;
    $("#tpSlider").value = k;
    $("#tpLabel").textContent = `${k} / ${data.steps.length - 1}`;
    $$("#stepList .step").forEach((e, i) => e.classList.toggle("active", i === k));
    const active = $("#stepList .step.active");
    if (active) active.scrollIntoView({ block: "nearest" });
    renderReplayStep();
  }

  function renderReplayStep() {
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
    renderTable({ hand, slots, viewer: reveal ? null : viewer, reveal, bubble, lens, payouts: S.replay.step === data.steps.length - 1 });
  }

  function stepReplay(delta) { setStep(S.replay.step + delta); }
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
        renderReplayStep();
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
    $("#tpSlider").addEventListener("input", (e) => setStep(Number(e.target.value)));
    $("#tpReveal").addEventListener("change", renderReplayStep);
    const sel = $("#lensAgent");
    sel.innerHTML = "";
    sel.appendChild(el("option", { value: "", text: "(none)" }));
    agentOptions(false).forEach((o) => sel.appendChild(o));
    sel.value = "";
    $("#lensRun").addEventListener("click", () => { if ($("#lensAgent").value) runLens(); else { S.replay.lensSpec = null; S.replay.lens = null; renderStepList(); updateLensSummary(); renderReplayStep(); } });
    $("#forkBtn").addEventListener("click", openForkModal);
    $("#forkCancel").addEventListener("click", () => { $("#forkModal").hidden = true; });
    $("#forkStart").addEventListener("click", startFork);
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
    $("#forkModal").hidden = false;
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
      $("#evalStatus").textContent = `${st.status} in ${st.result.seconds}s`;
      $("#evalExport").hidden = false;
      renderEvalResults(st.result);
    } catch (e) { toast(e.message, true); }
  }

  function tile(label, value, sub) {
    return el("div", { class: "tile" }, el("div", { class: "v", text: value }), el("div", { class: "l", text: label + (sub ? " · " + sub : "") }));
  }

  function seqColor(frac) {
    // single-hue sequential ramp (dataviz palette, blue)
    const steps = ["--seq-100", "--seq-200", "--seq-300", "--seq-400", "--seq-500", "--seq-600", "--seq-700"];
    const i = Math.min(steps.length - 1, Math.floor(frac * steps.length));
    return `var(${steps[i]})`;
  }

  function renderEvalResults(r) {
    const root = $("#evalResults");
    root.innerHTML = "";
    const box = el("div", { class: "card-box" });
    box.appendChild(el("h3", { text: `${r.agent.label || r.agent.kind} vs. recorded decisions` }));
    box.appendChild(el("div", { class: "tiles" },
      tile("decisions", String(r.decisions), `${r.hands_evaluated}/${r.hands} hands`),
      tile("same action", pct(r.accuracy), "fold / check / call / bet / raise"),
      tile("same intent", pct(r.accuracy_coarse), "fold / passive / aggressive"),
      tile("latency", r.latency_ms_mean === null ? "-" : `${r.latency_ms_mean.toFixed(0)} ms`, "per decision")));
    if (r.skipped_unknown_cards) box.appendChild(el("p", { class: "hint", text: `${r.skipped_unknown_cards} decisions skipped because the player's cards are unknown in the record. Load hands with known hole cards (Pluribus, WSOP, the iPoker hero-card sample) for full coverage.` }));
    // by street bars
    const streets = ["preflop", "flop", "turn", "river"].filter((s) => r.by_street[s]);
    const bars = el("div", { class: "bars" });
    streets.forEach((s) => {
      const v = r.by_street[s];
      bars.appendChild(el("div", { class: "bar-row" }, el("span", { text: s }), el("div", { class: "track" }, el("div", { class: "fill", style: `width:${(100 * v.accuracy).toFixed(1)}%` })), el("span", { class: "val", text: `${pct(v.accuracy)} (n=${v.n})` })));
    });
    box.appendChild(el("h4", { text: "Agreement by street" }));
    box.appendChild(bars);
    const actions = ["fold", "check", "call", "bet", "raise"];
    const bars2 = el("div", { class: "bars" });
    actions.filter((a) => r.by_action[a]).forEach((a) => {
      const v = r.by_action[a];
      bars2.appendChild(el("div", { class: "bar-row" }, el("span", { text: a }), el("div", { class: "track" }, el("div", { class: "fill", style: `width:${(100 * v.accuracy).toFixed(1)}%` })), el("span", { class: "val", text: `${pct(v.accuracy)} (n=${v.n})` })));
    });
    box.appendChild(el("h4", { text: "Agreement by the human's actual action" }));
    box.appendChild(bars2);
    // confusion matrix heatmap
    const maxv = Math.max(1, ...actions.flatMap((a) => actions.map((b) => r.confusion[a][b])));
    const table = el("table", { class: "matrix" });
    table.appendChild(el("tr", {}, el("th", { text: "actual ↓ model →" }), ...actions.map((a) => el("th", { text: a }))));
    actions.forEach((a) => {
      const row = el("tr", {}, el("th", { text: a }));
      actions.forEach((b) => {
        const v = r.confusion[a][b];
        const frac = v / maxv;
        row.appendChild(el("td", { class: a === b ? "diag" : "", style: `background:${v ? seqColor(frac) : "transparent"};color:${frac > 0.45 ? "#fff" : "var(--ink)"}`, text: v || "" }));
      });
      table.appendChild(row);
    });
    box.appendChild(el("h4", { text: "Confusion matrix" }));
    box.appendChild(el("div", { style: "overflow-x:auto" }, table));
    if (r.size_error_mean_abs_log_ratio !== null && r.size_error_mean_abs_log_ratio !== undefined) {
      box.appendChild(el("p", { class: "hint", text: `Bet sizing: when both bet/raised, the model's size differed from the human's by a factor of ${Math.exp(r.size_error_mean_abs_log_ratio).toFixed(2)} on average (${r.sized_comparisons} comparisons).` }));
    }
    root.appendChild(box);
    // mismatches
    const rows = (r.rows || []).filter((x) => !x.agree);
    const mbox = el("div", { class: "card-box" }, el("h4", { text: `Disagreements (${rows.length}${r.rows && r.rows.length < r.decisions ? ", first " + r.rows.length + " decisions" : ""}) - click to open in Replay` }));
    const ml = el("div", { class: "mismatch-list" });
    rows.slice(0, 300).forEach((x) => {
      ml.appendChild(el("div", { class: "mismatch", onclick: () => jumpToDecision(x) },
        el("span", {}, el("b", { text: x.name }), ` ${x.street} ${x.position} `, cardsInline(x.cards), ` pot ${fmt(x.pot)}${x.to_call ? ", " + fmt(x.to_call) + " to call" : ""}`),
        el("span", { class: "tag", text: "human: " + actionText({ action: x.actual, to: x.actual_to }) }),
        el("span", { class: "tag", text: "model: " + actionText({ action: x.predicted, to: x.predicted_to }) })));
    });
    mbox.appendChild(ml);
    root.appendChild(mbox);
  }

  async function jumpToDecision(row) {
    S.replay.lensSpec = S.evalResult && S.evalResult.agent ? specFromDescribe(S.evalResult.agent) : S.replay.lensSpec;
    setTab("replay");
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
      $("#simStatus").textContent = `${st.result.hands} hands in ${st.result.seconds}s`;
      $("#simExport").hidden = false;
      renderSimResults(st.result);
    } catch (e) { toast(e.message, true); }
  }

  function renderSimResults(r) {
    const root = $("#simResults");
    root.innerHTML = "";
    const box = el("div", { class: "card-box" }, el("h3", { text: "Results" }));
    const table = el("table", { class: "mini" }, el("thead", {}, el("tr", {}, el("th", { text: "Seat" }), el("th", { text: "Player" }), el("th", { text: "Type" }), el("th", { text: "Net" }), el("th", { text: "bb/100" }), el("th", { text: "± s.e." }))));
    const tb = el("tbody");
    r.seats.forEach((s) => tb.appendChild(el("tr", {}, el("td", { text: s.seat }), el("td", { text: s.name }), el("td", { text: s.agent.kind }),
      el("td", { class: s.net > 0 ? "pos" : s.net < 0 ? "neg" : "", text: (s.net > 0 ? "+" : "") + fmt(s.net) }),
      el("td", { class: s.bb_per_100 > 0 ? "pos" : s.bb_per_100 < 0 ? "neg" : "", text: s.bb_per_100.toFixed(1) }), el("td", { text: s.bb_per_100_stderr.toFixed(1) }))));
    table.appendChild(tb);
    box.appendChild(table);
    box.appendChild(el("p", { class: "hint", text: "bb/100 = big blinds won per 100 hands. Differences smaller than about two standard errors are noise - run more hands." }));
    root.appendChild(box);
    const series = r.seats.map((s, i) => ({ name: s.name, values: r.cumulative[String(s.seat)] || [], color: `var(--series-${(i % 8) + 1})` }));
    root.appendChild(lineChart(series, { title: "Cumulative net (chips) by hand", yLabel: "chips" }));
  }

  // simple SVG multi-line chart with hover crosshair
  function lineChart(series, opts) {
    const W = 380, H = 220, m = { l: 48, r: 14, t: 12, b: 26 };
    const n = Math.max(1, ...series.map((s) => s.values.length));
    let ymin = 0, ymax = 0;
    series.forEach((s) => s.values.forEach((v) => { ymin = Math.min(ymin, v); ymax = Math.max(ymax, v); }));
    if (ymin === ymax) { ymin -= 1; ymax += 1; }
    const pad = (ymax - ymin) * 0.06;
    ymin -= pad; ymax += pad;
    const x = (i) => m.l + (W - m.l - m.r) * (n > 1 ? i / (n - 1) : 0.5);
    const y = (v) => m.t + (H - m.t - m.b) * (1 - (v - ymin) / (ymax - ymin));
    const svgNS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const mk = (tag, attrs, text) => { const e = document.createElementNS(svgNS, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); if (text !== undefined) e.textContent = text; return e; };
    const ticks = 4;
    for (let t = 0; t <= ticks; t++) {
      const v = ymin + (ymax - ymin) * t / ticks;
      svg.appendChild(mk("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: "var(--line)", "stroke-width": 1 }));
      svg.appendChild(mk("text", { x: m.l - 6, y: y(v) + 4, "text-anchor": "end", fill: "var(--muted)", "font-size": 10 }, Math.round(v).toLocaleString()));
    }
    if (ymin < 0 && ymax > 0) svg.appendChild(mk("line", { x1: m.l, x2: W - m.r, y1: y(0), y2: y(0), stroke: "var(--line-2)", "stroke-width": 1.5 }));
    [0, Math.floor((n - 1) / 2), n - 1].forEach((i) => svg.appendChild(mk("text", { x: x(i), y: H - 8, "text-anchor": "middle", fill: "var(--muted)", "font-size": 10 }, `hand ${i + 1}`)));
    series.forEach((s) => {
      if (!s.values.length) return;
      const d = s.values.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
      svg.appendChild(mk("path", { d, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round" }));
      if (series.length <= 4) svg.appendChild(mk("text", { x: x(s.values.length - 1) + 3, y: y(s.values[s.values.length - 1]) + 3, fill: "var(--ink-2)", "font-size": 10 }, s.name));
    });
    const cross = mk("line", { x1: 0, x2: 0, y1: m.t, y2: H - m.b, stroke: "var(--ink-2)", "stroke-width": 1, "stroke-dasharray": "3 3", visibility: "hidden" });
    svg.appendChild(cross);
    const wrap = el("div", { class: "chart-wrap" });
    wrap.appendChild(el("h4", { text: opts.title }));
    wrap.appendChild(svg);
    const legend = el("div", { class: "legend" });
    series.forEach((s) => legend.appendChild(el("span", { style: `--c:${s.color}`, text: s.name })));
    wrap.appendChild(legend);
    const tip = el("div", { class: "chart-tip", hidden: true });
    wrap.appendChild(tip);
    svg.addEventListener("mousemove", (e) => {
      const rect = svg.getBoundingClientRect();
      const px = (e.clientX - rect.left) * W / rect.width;
      const i = Math.max(0, Math.min(n - 1, Math.round((px - m.l) / (W - m.l - m.r) * (n - 1))));
      cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("visibility", "visible");
      tip.hidden = false;
      tip.innerHTML = `<b>hand ${i + 1}</b><br>` + series.map((s) => `<span style="color:${s.color}">■</span> ${s.name}: ${fmt(s.values[i] || 0)}`).join("<br>");
      tip.style.left = Math.min(e.clientX - rect.left + 12, rect.width - 160) + "px";
      tip.style.top = (e.clientY - rect.top + 10) + "px";
    });
    svg.addEventListener("mouseleave", () => { tip.hidden = true; cross.setAttribute("visibility", "hidden"); });
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
      else if (S.tab === "replay" && S.replay.data) renderReplayStep();
    });
    setTab("play");
    await handleHash();
  }

  // Deep links: #demo (bot-only table), #replay=<sample file>&i=<hand>&step=<step>&lens=<agent spec>
  async function handleHash() {
    const h = location.hash.replace(/^#/, "");
    if (!h) return;
    const params = new URLSearchParams(h);
    if (h === "demo" || params.has("demo")) {
      $$("#seatConfig .s-agent").forEach((sel, i) => { sel.value = ["tag", "equity", "lag", "equity", "random", "maniac"][i % 6]; });
      $("#spectate").checked = true;
      $("#botDelay").value = params.get("delay") || 1200;
      await startGame();
      return;
    }
    if (params.has("replay")) {
      setTab("replay");
      const file = params.get("replay");
      if (params.get("lens")) { S.replay.lensSpec = params.get("lens"); }
      await loadRecords({ source: "sample", path: file, limit: Number($("#loadLimit").value) || null });
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
      await api("/api/simulate/start", { agents, hands: Number(params.get("hands") || 100), seed: 1 });
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
  window.PokerSim = { api, el, cardEl, cardsInline, renderTable, toast, fmt, pct, setTab, showOverlay, tile, seqColor, state: S };

  boot();
})();
