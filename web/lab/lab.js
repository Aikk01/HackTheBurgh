/* Table Lab (web/lab.html): play six-handed and read two models side by side.

   MODEL SLOTS
   Every model value on this page is drawn by a slot renderer that receives the value or null. A null
   slot prints NO_MODEL ("No model loaded"). The values come from GET /api/lab/predict; the response is
   documented at the top of pokersim/lab_api.py. When a model lands, fill its field there and the
   placeholder disappears by itself; nothing here needs deleting.
     grep -n "noModel(" web/lab/lab.js        every place a placeholder can appear
   To check the layout with values before any model exists, in the browser console:
     Lab.inject({...same shape as /api/lab/predict...})   then   Lab.clear()
   Injected values are flagged by a banner across the top.
*/
(() => {
  "use strict";

  const NO_MODEL = "No model loaded";
  const $ = (s, r = document) => r.querySelector(s);

  function el(tag, attrs, ...kids) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "text") n.textContent = v;
      else if (k === "class") n.className = v;
      else if (k === "style") n.style.cssText = v;
      else n.setAttribute(k, v === true ? "" : v);
    }
    for (const c of kids.flat()) if (c !== null && c !== undefined && c !== false) n.append(c.nodeType ? c : String(c));
    return n;
  }
  const noModel = () => el("span", { class: "nm", text: NO_MODEL });

  async function api(path, body) {
    const res = await fetch(path, body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = (res.headers.get("Content-Type") || "").includes("json") ? await res.json() : await res.text();
    if (!res.ok) throw new Error((data && data.error) || `${res.status} ${res.statusText}`);
    return data;
  }

  const L = { st: null, pred: null, predKey: null, token: 0, src: "base", sel: null, injected: false, lastCheck: null, cursor: 0, lastLog: "" };
  const SEAT_COLOURS = ["--p1", "--p2", "--p3", "--p4", "--p5", "--p1", "--p2", "--p3", "--p4"];
  const RANKS = "AKQJT98765432";
  const ACT = { fold: "Fold", check: "Check", call: "Call", bet: "Bet", raise: "Raise to", all_in: "All-in" };
  const fmt = (x) => (x === null || x === undefined ? "" : String(Math.round(Number(x) * 100) / 100));
  const pct = (p) => `${Math.round(100 * p)}%`;
  const pctHand = (p) => (p < 0.1 ? `${(100 * p).toFixed(1)}%` : pct(p));  // single hands are rarely above 10%
  const actionText = (a) => (a.amount !== null && a.amount !== undefined && ["call", "bet", "raise"].includes(a.action) ? `${ACT[a.action]} ${fmt(a.amount)}` : ACT[a.action] || a.action);
  const announce = (t) => { $("#announcer").textContent = t; };

  // ------------------------------------------------------------ table geometry helpers
  const hand = () => (L.st && L.st.active ? L.st.hand : null);
  const me = () => { const h = hand(); return h && L.st.viewer !== null && L.st.viewer !== undefined ? h.players.find((p) => p.index === L.st.viewer) : null; };
  function clockwise() {
    // players ordered clockwise starting from the viewer (or from the lowest seat when spectating)
    const h = hand();
    if (!h) return [];
    const ps = h.players.slice().sort((a, b) => a.seat - b.seat);
    const m = me();
    const start = m ? ps.indexOf(m) : 0;
    return ps.slice(start).concat(ps.slice(0, start));
  }
  const opponents = () => clockwise().filter((p) => !me() || p.index !== me().index);
  function colourOf(seat) {
    const i = opponents().findIndex((p) => p.seat === seat);
    return i < 0 ? "--line-2" : SEAT_COLOURS[i % SEAT_COLOURS.length];
  }
  const predFor = (seat) => (L.pred && L.pred.players ? L.pred.players.find((p) => p.seat === seat) : null) || null;

  // ------------------------------------------------------------ slots (model values or NO_MODEL)
  function moveSlot(pred, n) {
    if (!pred || !pred.actions || !pred.actions.length) return noModel();
    const [a, b] = pred.actions;
    return el("span", { class: "v" }, el("b", { text: actionText(a) }), el("span", { class: "pp", text: pct(a.p) }),
      n > 1 && b && b.p >= 0.01 ? el("span", { class: "p2" }, ` · ${actionText(b)} `, el("span", { class: "pp", text: pct(b.p) })) : null);
  }
  function handSlot(h, n) {
    if (!h || !h.top || !h.top.length) return noModel();
    const [a, b] = h.top;
    return el("span", { class: "v" }, el("b", { text: a.hand }), el("span", { class: "pp", text: pctHand(a.p) }),
      n > 1 && b ? el("span", { class: "p2" }, ` · ${b.hand} `, el("span", { class: "pp", text: pctHand(b.p) })) : null);
  }
  const loaded = (model, part) => !!(L.pred && L.pred.models && L.pred.models[model] && L.pred.models[model][part]);
  // your own decision: a loaded model that has nothing to say yet is waiting for your turn, not missing
  function yourSlot(value, model, n = 2) {
    if (value) return moveSlot(value, n);
    const you = L.pred && L.pred.you;
    if (loaded(model, "actions") && you && !you.your_turn) return el("span", { class: "nm", text: "Waits for your turn" });
    return noModel();
  }

  // ------------------------------------------------------------ cards
  const SUITS = { c: "♣", d: "♦", h: "♥", s: "♠" };
  function card(code, size) {
    const cls = "lc" + (size ? " " + size : "");
    if (!code) return el("div", { class: cls + " slot", "aria-hidden": "true" });
    if (code === "XX" || code === "??") return el("div", { class: cls + " back", "aria-label": "hidden card" });
    const r = code[0] === "T" ? "10" : code[0];
    const s = code[1];
    return el("div", { class: cls + (s === "h" || s === "d" ? " red" : ""), "aria-label": code },
      el("span", { class: "r", text: r }), el("span", { class: "s", text: SUITS[s] || "?" }));
  }

  // ------------------------------------------------------------ render
  function render() {
    renderBar();
    renderTable();
    renderYou();
    renderSide();
  }

  function renderBar() {
    const h = hand();
    const t = L.st && L.st.table;
    $("#handMeta").textContent = h ? `Hand ${L.st.hand_no} · blinds ${fmt(t && t.small_blind)}/${fmt(t && t.big_blind)} · ${h.status === "betting" ? h.street : "hand over"}` : "";
    const ul = $("#models");
    ul.innerHTML = "";
    const parts = { base: [["actions", "moves"], ["hands", "hands"]], rl: [["actions", "moves"]], personal: [["actions", "moves"]] };
    [["base", "Base model"], ["rl", "RL model"], ["personal", "Player models"]].forEach(([key, label]) => {
      const have = parts[key].filter(([p]) => loaded(key, p));
      const li = el("li", { class: have.length ? "on" : "" }, el("b", { text: label }));
      if (!have.length) li.append(noModel());
      else if (have.length === parts[key].length) li.append(el("span", { class: "ok", text: "loaded" }));
      else {
        li.append(el("span", { class: "ok", text: have.map(([, w]) => w).join(", ") + " ✓" }));
        li.append(el("span", { class: "what" }, ...parts[key].filter((x) => !have.includes(x)).map(([, w]) => el("span", {}, `${w}: `, noModel()))));
      }
      ul.append(li);
    });
  }

  function geometry() {
    const st = $("#stage");
    const W = st.clientWidth, H = st.clientHeight;
    const cx = W / 2, cy = H / 2 + 28;
    const rx = Math.max(160, W / 2 - 130), ry = Math.max(110, H / 2 - 76);
    return { W, H, cx, cy, rx, ry };
  }

  function renderTable() {
    const h = hand();
    const seats = $("#seats");
    seats.innerHTML = "";
    $("#center").innerHTML = "";
    const g = geometry();
    const felt = $("#felt");
    Object.assign(felt.style, { left: `${g.cx - g.rx + 18}px`, top: `${g.cy - g.ry + 10}px`, width: `${2 * (g.rx - 18)}px`, height: `${2 * (g.ry - 10)}px` });
    $("#sit").hidden = !!(h && me());
    if (!h) return;

    // centre: board, pot, last action
    const board = [0, 1, 2, 3, 4].map((i) => card(h.board[i], "lg"));
    const last = (L.st.log || []).slice(-1)[0];
    const center = $("#center");
    center.style.left = `${g.cx}px`;
    center.style.top = `${g.cy - 4}px`;
    center.append(el("div", { class: "board", "aria-label": `Board: ${h.board.filter(Boolean).join(" ") || "no cards yet"}` }, ...board),
      el("div", { class: "pot" }, "Pot ", el("b", { text: fmt(h.total_pot) })),
      el("div", { class: "last", text: last ? last.text : "" }));

    const order = clockwise();
    const n = order.length;
    const mine = me();
    order.forEach((p, k) => {
      const ang = (90 + (k * 360) / n) * (Math.PI / 180);
      const x = g.cx + g.rx * Math.cos(ang), y = g.cy + g.ry * Math.sin(ang);
      const isMe = mine && p.index === mine.index;
      const acting = h.status === "betting" && h.actor === p.index;
      const visible = isMe || p.cards.some((c) => c && c !== "XX");
      const cards = p.has_cards && !(p.folded && !visible) ? p.cards.map((c) => card(c, isMe ? null : "sm")) : [];
      const flags = [];
      if (p.is_button) flags.push("dealer");
      if (p.folded) flags.push("folded");
      else if (p.all_in) flags.push("all-in");
      const won = (h.payouts && h.payouts[p.index]) || p.payout || 0;
      const plate = el("div", { class: "plate", style: isMe ? "" : `--pc: var(${colourOf(p.seat)})` },
        el("div", { class: "hcards" }, ...cards),
        el("div", { class: "who" },
          el("span", { class: "name", text: p.name, title: p.name }),
          el("span", { class: "sub", text: `${fmt(p.stack)} · ${p.position || ""}` }),
          won > 0 && h.status !== "betting" ? el("span", { class: "flag win", text: `wins ${fmt(won)}` })
            : flags.length ? el("span", { class: "flag", text: flags.join(" · ") }) : null));
      const seat = el("div", { class: "seat" + (isMe ? " me" : "") + (acting ? " acting" : "") + (p.folded ? " folded" : ""), style: `left:${x}px;top:${y}px` });
      if (!isMe && !p.folded) seat.append(seatBox(p));
      seat.append(plate);
      seats.append(seat);
      if (p.bet > 0) seats.append(el("div", { class: "bet", style: `left:${x + (g.cx - x) * 0.4}px;top:${y + (g.cy - y) * 0.42}px`, text: fmt(p.bet) }));
    });
  }

  // the box above an opponent: what people do here (Base) and what this player does (their own model, learned at
  // this table); likely hand: the top starting hand, and the range read on this board
  function seatBox(p) {
    if (p.folded) return null;  // nothing left to predict
    const pr = predFor(p.seat);
    const b = pr && pr.base, t = pr && pr.this_player;
    // one idea per box: their next move, as people in general play (Base) and as this player plays; their likely
    // hands live in the grid on the right
    return el("div", { class: "mbox slim", "aria-label": `Next move for ${p.name}` },
      el("div", { class: "mtable two" },
        el("span", { class: "k", text: "People" }), moveSlot(b && b.action, 1),
        el("span", { class: "k", text: "This player", title: t && t.seen ? `learned from ${t.seen} of their moves` : "no moves seen yet" }),
        moveSlot(t && t.action, 1)));
  }

  function renderYou() {
    const you = L.pred && L.pred.you;
    const should = $("#should");
    should.className = "mtable two";
    should.innerHTML = "";
    should.append(el("span", { class: "k", text: "Base" }), yourSlot(you && you.should && you.should.base, "base", 1),
      el("span", { class: "k", text: "RL" }), yourSlot(you && you.should && you.should.rl, "rl", 1));

    const would = $("#would");
    would.className = "mtable two";
    would.innerHTML = "";
    const acc = you && you.accuracy;
    // your model's guess is made before you act but shown only afterwards, so it cannot steer you
    const hidden = you && you.your_turn && you.would && you.would.personal;
    would.append(el("span", { class: "k", text: "Your model" }),
      hidden ? el("span", { class: "nm", text: "Revealed after you act" }) : yourSlot(null, "personal"),
      el("span", { class: "k", text: "Right so far" }),
      acc ? el("span", { class: "v" }, el("b", { text: `${acc.right} of ${acc.total} (${pct(acc.right / Math.max(acc.total, 1))})` }),
        el("span", { class: "p2", text: ` · usual-move guess ${acc.baseline_right}` })) : noModel());
    if (L.lastCheck) {
      const c = L.lastCheck;
      would.append(el("span", { class: "k", text: "Last move" }), el("span", { class: "check" },
        `Predicted ${c.pred.toLowerCase()}, you ${c.actual} `, el("span", { class: c.hit ? "yes" : "no", text: c.hit ? "✓" : "✗" })));
    }

    // your action controls
    const h = hand();
    const m = me();
    const yourTurn = !!(h && m && h.status === "betting" && h.actor === m.index && h.legal);
    const over = !!(h && h.status !== "betting");
    $("#you").classList.toggle("turn", yourTurn);
    $("#act").classList.toggle("wait", !yourTurn);
    $("#btnNext").hidden = !over;
    $("#raise").hidden = over;
    $(".act-row").hidden = over;
    const actor = h && h.players.find((p) => p.index === h.actor);
    $("#actTitle").textContent = !h ? "Your move" : over ? "Hand over" : yourTurn ? "Your move" : `Waiting for ${actor ? actor.name : "the table"}`;
    $("#actNote").textContent = yourTurn ? "" : over ? "Press N or Enter for the next hand." : m && m.folded ? "You folded this hand." : "";
    if (yourTurn) {
      const lg = h.legal;
      $("#btnFold").disabled = !lg.fold;
      $("#btnCall").firstChild.textContent = lg.check ? "Check " : `Call ${fmt(lg.call)} `;
      const canRaise = !!lg.raise;
      ["#raiseSlider", "#raiseInput", "#btnRaise"].forEach((s) => { $(s).disabled = !canRaise; });
      if (canRaise) {
        const s = $("#raiseSlider"), i = $("#raiseInput");
        const key = `${h.hand_id}|${lg.min_raise_to}|${lg.max_raise_to}`;
        if (s.dataset.key !== key) {
          s.dataset.key = key;
          s.min = i.min = lg.min_raise_to;
          s.max = i.max = lg.max_raise_to;
          s.step = i.step = 1;
          s.value = i.value = lg.min_raise_to;
        }
        $("#btnRaise").firstChild.textContent = lg.raise_kind === "bet" ? "Bet " : "Raise ";
      }
    }
  }

  function renderSide() {
    const opps = opponents();
    // guesses exist only for players still in the hand: follow a live opponent when the selected one folds
    if (L.sel === null || !opps.some((p) => p.seat === L.sel && !p.folded)) {
      const live = opps.find((p) => !p.folded) || opps[0];
      L.sel = live ? live.seat : null;
    }
    const chips = $("#chips");
    chips.innerHTML = "";
    opps.forEach((p) => {
      const b = el("button", { type: "button", role: "radio", class: "chip" + (p.folded ? " out" : ""), "aria-checked": String(p.seat === L.sel), disabled: p.folded, style: `--pc: var(${colourOf(p.seat)})`, title: p.folded ? `${p.name} folded` : p.name },
        el("i"), el("span", { text: p.name + (p.folded ? " · folded" : "") }));
      b.addEventListener("click", () => { L.sel = p.seat; renderSide(); });
      chips.append(b);
    });

    const sel = opps.find((p) => p.seat === L.sel);
    const pc = sel ? colourOf(sel.seat) : "--line-2";
    const pr = sel ? predFor(sel.seat) : null;
    const hands = pr && pr.base ? pr.base.hands : null;
    const wrap = $("#gridWrap");
    wrap.innerHTML = "";
    wrap.style.setProperty("--pc", `var(${pc})`);
    wrap.append(grid13(hands, sel ? sel.name : ""));
    if (!hands) wrap.append(el("div", { class: "grid-over" }, noModel()));
    wrap.append(el("div", { class: "readout", text: hands ? "Hover a cell or use the arrow keys." : "" }));

    const top = $("#top2");
    top.innerHTML = "";
    top.append(el("div", { class: "row" }, el("span", { class: "k", text: "Top 2 hands" }), handSlot(hands, 2)));
    const ramp = $("#ramp");
    ramp.innerHTML = "";
    ramp.style.setProperty("--pc", `var(${pc})`);
    ramp.append(el("span", { text: "less likely" }), el("span", { class: "bar" }), el("span", { text: "more likely" }));
  }

  function grid13(hands, who) {
    const g = el("div", { class: "g13" + (hands ? "" : " empty"), role: "grid", tabindex: hands ? 0 : -1, "aria-label": hands ? `Likely hands for ${who}. Arrow keys read cells.` : `Likely hands for ${who}: ${NO_MODEL}` });
    // shade by probability per combination: offsuit cells hold 12 combinations, suited 4, pairs 6, so shading by the
    // cell total would make offsuit hands look about 3x likelier than they are
    const dens = hands ? (hands.density || hands.grid) : null;
    const vals = dens ? dens.flat() : [];
    const max = vals.length ? Math.max(...vals) : 0;
    const cells = [];
    for (let r = 0; r < 13; r++) {
      for (let c = 0; c < 13; c++) {
        const name = r === c ? RANKS[r] + RANKS[c] : r < c ? RANKS[r] + RANKS[c] + "s" : RANKS[c] + RANKS[r] + "o";
        const p = hands ? hands.grid[r][c] : 0;
        const d = dens ? dens[r][c] : 0;
        const a = d > 0 && max > 0 ? 0.1 + 0.9 * (d / max) : 0;
        const shown = hands && hands.shown === name;
        const cell = el("div", { class: "c" + (a > 0.55 ? " hi" : "") + (shown ? " shown" : ""), role: "gridcell", title: shown ? "their real hand" : "" }, el("span", { class: "f", style: `--a:${a.toFixed(3)}` }), el("span", { class: "t", text: name }));
        cell.dataset.name = name;
        cell.dataset.p = p;
        if (hands) cell.addEventListener("mouseenter", () => focusCell(r * 13 + c));
        cells.push(cell);
        g.append(cell);
      }
    }
    function focusCell(k) {
      cells[L.cursor] && cells[L.cursor].classList.remove("cursor");
      L.cursor = Math.max(0, Math.min(168, k));
      const c = cells[L.cursor];
      c.classList.add("cursor");
      const ro = $(".readout");
      if (ro) ro.textContent = `${c.dataset.name}: ${(100 * Number(c.dataset.p)).toFixed(1)}%`;
    }
    if (hands) {
      g.addEventListener("keydown", (e) => {
        const m = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: 13, ArrowUp: -13 }[e.key];
        if (m === undefined) return;
        e.preventDefault();
        focusCell(L.cursor + m);
      });
      g.addEventListener("focus", () => focusCell(L.cursor));
    }
    return g;
  }

  // ------------------------------------------------------------ data flow
  function onState(st) {
    L.st = st;
    const last = st && st.log && st.log.length ? st.log[st.log.length - 1].text : "";
    if (last && last !== L.lastLog) { L.lastLog = last; announce(last); }
    render();
    schedulePredict();
  }
  let predTimer = null;
  function schedulePredict() {
    if (L.injected) return;
    const h = hand();
    const key = h ? `${h.hand_id}|${(L.st.log || []).length}|${h.actor}|${h.status}` : "none";
    if (key === L.predKey) return;
    L.predKey = key;
    clearTimeout(predTimer);
    predTimer = setTimeout(fetchPred, 120);
  }
  async function fetchPred() {
    const tok = ++L.token;
    let p = null;
    try { p = await api("/api/lab/predict"); } catch (e) { p = null; }  // unreachable endpoint = no models
    if (tok !== L.token || L.injected) return;
    L.pred = p;
    render();
  }

  const CLASS = (a) => (a === "fold" ? "fold" : a === "check" || a === "call" ? "check/call" : "bet/raise");
  async function act(action, amount) {
    const h = hand(), m = me();
    if (!h || !m || h.status !== "betting" || h.actor !== m.index) return;
    // score the "would do" prediction made for this decision, if a personal model gave one
    const you = L.pred && L.pred.you;
    const wp = you && you.your_turn && you.would && you.would.personal;
    const actual = action === "call" ? (h.legal.check ? "check" : "call") : action;
    if (wp && wp.actions && wp.actions.length) {
      const top = wp.actions[0];
      L.lastCheck = { pred: actionText(top), actual: { fold: "folded", check: "checked", call: "called", raise: h.legal.raise_kind === "bet" ? "bet" : "raised" }[actual] || actual, hit: CLASS(top.action) === CLASS(actual === "raise" ? h.legal.raise_kind : actual) };
    }
    try { onState(await api("/api/game/act", { action, amount })); } catch (e) { $("#actNote").textContent = e.message; }
  }
  async function newTable() {
    try {
      onState(await api("/api/game/new", {
        seats: [{ name: "You", agent: "human" }, { agent: "tag" }, { agent: "pro" }, { agent: "lag" }, { agent: "equity" }, { agent: "station" }],
        small_blind: 1, big_blind: 2, ante: 0, stack: 200, bot_delay_ms: 900, spectate: false,
      }));
      L.lastCheck = null;
    } catch (e) { $("#actNote").textContent = e.message; }
  }
  async function nextHand() {
    const h = hand();
    if (!h || h.status === "betting") return;
    try { onState(await api("/api/game/next", {})); } catch (e) { $("#actNote").textContent = e.message; }
  }

  function bind() {
    $("#newTable").addEventListener("click", newTable);
    $("#sitBtn").addEventListener("click", newTable);
    $("#btnFold").addEventListener("click", () => act("fold"));
    $("#btnCall").addEventListener("click", () => act("call"));
    $("#btnRaise").addEventListener("click", () => act("raise", Number($("#raiseInput").value)));
    $("#btnNext").addEventListener("click", nextHand);
    $("#raiseSlider").addEventListener("input", (e) => { $("#raiseInput").value = e.target.value; });
    $("#raiseInput").addEventListener("input", (e) => { $("#raiseSlider").value = e.target.value; });

    $("#clearInject").addEventListener("click", () => window.Lab.clear());
    document.addEventListener("keydown", (e) => {
      if (e.target.matches("input, select, textarea") || e.metaKey || e.ctrlKey || e.altKey) return;
      const k = e.key.toLowerCase();
      if (k === "f") act("fold");
      else if (k === "c") act("call");
      else if (k === "r" && !$("#btnRaise").disabled) act("raise", Number($("#raiseInput").value));
      else if (k === "n" || (e.key === "Enter" && e.target === document.body)) nextHand();
    });
    new ResizeObserver(() => { if (L.st) renderTable(); }).observe($("#stage"));
  }

  function connect() {
    const es = new EventSource("/api/events");
    es.addEventListener("game", (e) => onState(JSON.parse(e.data).state));
  }

  window.Lab = {
    inject(pred) { L.injected = true; L.pred = pred; $("#injectBanner").hidden = false; render(); },
    clear() { L.injected = false; $("#injectBanner").hidden = true; L.predKey = null; L.pred = null; render(); schedulePredict(); },
    state: L,
  };

  async function boot() {
    bind();
    connect();
    try { onState(await api("/api/game/state")); } catch (e) { onState(null); }
  }
  boot();
})();
