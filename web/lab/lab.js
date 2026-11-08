/* Poker Mirror (web/lab.html): play six-handed and read two models side by side.

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

  const L = { st: null, pred: null, predKey: null, token: 0, src: "base", sel: null, injected: false, lastCheck: null, cursor: 0, lastLog: "", replay: false, liveSt: null,
    // motion: what each grid last showed, how much each guess moved and why, the ✓/✗ badges, when you last picked a chip
    drawn: {}, mem: {}, shift: {}, flash: {}, scale: {}, logLen: 0, picked: 0, shownPick: null };
  const MOTION = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const GROUP = (a) => (a === "fold" ? 0 : a === "check" || a === "call" ? 1 : 2);
  const groupOfText = (t) => (/\bfold/.test(t) ? 0 : /\b(check|call)/.test(t) ? 1 : 2);
  const SEAT_COLOURS = ["--p1", "--p2", "--p3", "--p4", "--p5", "--p1", "--p2", "--p3", "--p4"];
  const RANKS = "AKQJT98765432";
  const ACT = { fold: "Fold", check: "Check", call: "Call", bet: "Bet", raise: "Raise to", all_in: "All-in" };
  const fmt = (x) => (x === null || x === undefined ? "" : String(Math.round(Number(x) * 100) / 100));
  const pct = (p) => `${Math.round(100 * p)}%`;
  const pctHand = (p) => (p < 0.1 ? `${(100 * p).toFixed(1)}%` : pct(p));  // single hands are rarely above 10%
  const actionText = (a) => (a.amount !== null && a.amount !== undefined && ["call", "bet", "raise"].includes(a.action) ? `${ACT[a.action]} ${fmt(a.amount)}` :
    a.action === "raise" ? "Raise" : ACT[a.action] || a.action);
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
  // fold / check-call / bet-raise, with every raise size and all-in summed, so all models answer the same three choices
  function grouped(actions) {
    const free = actions.some((a) => a.action === "check" || a.action === "bet");
    const g = new Map();
    for (const a of actions) {
      const k = a.action === "fold" ? "fold" : a.action === "check" || a.action === "call" ? "call" : "raise";
      if (g.has(k)) g.get(k).p += a.p;
      else g.set(k, k === "raise" ? { action: free ? "bet" : "raise", amount: null, p: a.p } : { ...a });
    }
    return [...g.values()].sort((x, y) => y.p - x.p);
  }
  function moveSlot(pred, n) {
    if (!pred || !pred.actions || !pred.actions.length) return noModel();
    const [a, b] = grouped(pred.actions);
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
    if (loaded(model, "actions") && you && !you.your_turn) return el("span", { class: "nm", text: theirs() ? "Waits for their turn" : "Waits for your turn" });
    return noModel();
  }

  // a coach's advice as one bar: fold | check-call | bet-raise, its top choice lit
  function barSlot(pred, model) {
    if (!pred || !pred.actions || !pred.actions.length) return yourSlot(null, model);
    const g = grouped(pred.actions);
    const order = [0, 1, 2].map((k) => g.find((a) => GROUP(a.action) === k)).filter(Boolean);
    return el("div", { class: "cbar", role: "img", "aria-label": g.map((a) => `${actionText(a)} ${pct(a.p)}`).join(", ") },
      ...order.map((a) => el("span", { class: `g${GROUP(a.action)}` + (a === g[0] ? " top" : ""), style: `flex-grow:${Math.max(a.p, 0.001)}`, title: `${actionText(a)} ${pct(a.p)}` },
        a.p >= 0.16 ? `${actionText(a)} · ${pct(a.p)}` : "")));
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
    const all = loaded("base", "actions") && loaded("base", "hands") && loaded("rl", "actions") && loaded("personal", "actions");
    ul.hidden = all;  // the pills matter only when something is missing
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
    seats.classList.toggle("live", !!(h && h.status === "betting"));
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
      el("div", { class: "last", text: story(h) || (last ? last.text : "") }));

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
      const fl = L.flash[p.seat];
      const age = fl && fl.hid === h.hand_id ? Date.now() - fl.at : Infinity;
      if (age < 2200) plate.append(el("span", { class: "verdict " + (fl.hit ? "yes" : "no"), style: `animation-delay:-${age}ms`,
        title: "Our read of this player, made before the move", text: `${fl.verb} ${fl.hit ? "✓" : "✗"}` }));
      const seat = el("div", { class: "seat" + (isMe ? " me" : "") + (acting ? " acting" : "") + (p.folded ? " folded" : ""), style: `left:${x}px;top:${y}px` });
      const box = !isMe && !p.folded ? seatBox(p) : null;
      if (box) seat.append(box);
      seat.append(plate);
      seats.append(seat);
      if (p.bet > 0) seats.append(el("div", { class: "bet", style: `left:${x + (g.cx - x) * 0.4}px;top:${y + (g.cy - y) * 0.42}px`, text: fmt(p.bet) }));
    });
  }

  const pretty = (cs) => cs.map((c) => (c[0] === "T" ? "10" : c[0]) + (SUITS[c[1]] || "")).join("");
  // where a hand type sat in our guess: #1 = the likeliest of the 169
  function rankOf(hands, name) {
    const flat = [];
    for (let r = 0; r < 13; r++) for (let c = 0; c < 13; c++) flat.push([r === c ? RANKS[r] + RANKS[c] : r < c ? RANKS[r] + RANKS[c] + "s" : RANKS[c] + RANKS[r] + "o", hands.grid[r][c]]);
    flat.sort((x, y) => y[1] - x[1]);
    const i = flat.findIndex(([n]) => n === name);
    return i < 0 ? null : i + 1;
  }
  // the one sentence in the middle of the table: the last move, whether we read it, how much it moved our guess;
  // at the end, who won and where their cards were in our guess
  function story(h) {
    const log = (L.st && L.st.log) || [];
    const last = log[log.length - 1];
    if (h.status !== "betting") {
      const won = h.players.filter((p) => ((h.payouts && h.payouts[p.index]) || p.payout || 0) > 0);
      if (won.length !== 1) return last ? last.text : "";
      const w = won[0], amt = fmt((h.payouts && h.payouts[w.index]) || w.payout);
      const m = me();
      if (m && w.index === m.index) return `You win ${amt}`;
      const cs = w.cards.filter((c) => c && c !== "XX");
      const hs = predFor(w.seat) && predFor(w.seat).base && predFor(w.seat).base.hands;
      const rk = cs.length === 2 && hs && hs.shown ? rankOf(hs, hs.shown) : null;
      return `${w.name} wins ${amt}` + (cs.length === 2 ? ` with ${pretty(cs)}` : "") + (rk ? ` · our #${rk} guess of 169` : "");
    }
    if (!last) return "";
    const p = h.players.find((q) => q.index === last.player);
    let out = last.text;
    const fl = p && L.flash[p.seat];
    if (fl && fl.hid === h.hand_id && fl.text === last.text) out += fl.hit ? " · as we read it ✓" : ` · we read ${fl.expected} ✗`;
    const sh = p && L.shift[p.seat];
    if (sh && sh.hid === h.hand_id && sh.cause === last.text && sh.tv >= 0.1) out += ` · hand guess moved ${Math.round(100 * sh.tv)}%`;
    return out;
  }

  // the box above an opponent: what people do here (Base) and what this player does (their own model, learned at
  // this table); likely hand: the top starting hand, and the range read on this board
  function seatBox(p) {
    const h = hand();
    if (p.folded || (h && h.status !== "betting")) return null;  // nothing left to predict
    const pr = predFor(p.seat);
    const b = pr && pr.base, t = pr && pr.this_player;
    // one idea per box: their next move, as people in general play (Base) and as this player plays; their likely
    // hands live in the grid on the right
    const top = (x) => (x && x.actions && x.actions.length ? grouped(x.actions)[0] : null);
    const ga = top(b && b.action), gt = top(t && t.action);
    if (ga && gt && GROUP(ga.action) === GROUP(gt.action) && Math.abs(ga.p - gt.p) < 0.1) {
      return el("div", { class: "mbox slim", "aria-label": `Next move for ${p.name}` },
        el("div", { class: "mtable two" }, el("span", { class: "k", text: "Next move", title: "This player plays like people in general here" }), moveSlot(t.action, 1)));
    }
    return el("div", { class: "mbox slim", "aria-label": `Next move for ${p.name}` },
      el("div", { class: "mtable two" },
        el("span", { class: "k", text: "People" }), moveSlot(b && b.action, 1),
        el("span", { class: "k mine", text: "This player", title: "Plays unlike most people here: learned from their own moves" }),
        el("span", { class: "mine" }, moveSlot(t && t.action, 1))));
  }

  function renderYou() {
    const you = L.pred && L.pred.you;
    const should = $("#should");
    should.className = "mtable two";
    should.innerHTML = "";
    should.append(el("span", { class: "k", text: "Base", title: "Learned from data: what Pluribus did in spots like this, given your cards" }),
      barSlot(you && you.should && you.should.base, "base"),
      el("span", { class: "k", text: "RL", title: "Reinforcement learning: a policy trained by self-play that reads everyone's predicted hands" }),
      barSlot(you && you.should && you.should.rl, "rl"));

    const would = $("#would");
    would.className = "mtable two";
    would.innerHTML = "";
    const acc = you && you.accuracy;
    // your model's guess is made before you act but shown only afterwards, so it cannot steer you (in a replay of a
    // recorded player it is shown openly: the move is already history)
    const guess = you && you.your_turn && you.would && you.would.personal;
    would.append(el("span", { class: "k", text: theirs() ? "Their model" : "Your model" }),
      guess ? (L.replay ? moveSlot(guess, 1) : el("span", { class: "nm", text: "Revealed after you act" })) : yourSlot(null, "personal"),
      el("span", { class: "k", text: "Right so far" }),
      acc ? el("span", { class: "v" }, el("b", { text: `${acc.right} of ${acc.total} (${pct(acc.right / Math.max(acc.total, 1))})` }),
        el("span", { class: "p2", text: ` · people in general ${acc.people_right || 0}` })) :
        loaded("personal", "actions") ? el("span", { class: "nm", text: L.replay ? "Starts with their first move" : "Starts with your first move" }) : noModel());
    const yh = you && you.habits && you.habits[0];
    if (yh) would.append(el("span", { class: "k", text: theirs() ? "Their habit" : "Your habit" }),
      el("span", { class: "v" }, yh.text, el("span", { class: "p2", text: ` (${yh.detail})` })));
    const gaps = (you && you.coach_gaps) || [];
    if (gaps.length && !L.replay) {
      const g = gaps[0];
      would.append(el("span", { class: "k", text: "You vs coach" }), el("span", { class: "v" },
        `${g.where[0].toUpperCase()}${g.where.slice(1)}: you ${g.you}, coach ${g.coach}s `, el("span", { class: "p2", text: `(${g.times} of ${g.of})` })));
    }
    if (L.lastCheck) {
      const c = L.lastCheck;
      would.append(el("span", { class: "k", text: "Last move" }), el("span", { class: "check" },
        `Predicted ${c.pred.toLowerCase()}, ${theirs() ? "they" : "you"} ${c.actual} `, el("span", { class: c.hit ? "yes" : "no", text: c.hit ? "✓" : "✗" })));
    }

    // your action controls
    const h = hand();
    const m = me();
    const yourTurn = !!(h && m && h.status === "betting" && h.actor === m.index && h.legal);
    const over = !!(h && h.status !== "betting");
    $("#you").classList.toggle("turn", yourTurn);
    $("#act").classList.toggle("wait", !yourTurn);
    $("#btnNext").hidden = !over || L.replay;
    $("#raise").hidden = over || L.replay;
    $(".act-row").hidden = over || L.replay;
    $("#replayNav").hidden = !L.replay;
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
      renderSizes(h, lg, canRaise && !L.replay ? you : null);
    } else renderSizes(null);
  }

  // bet sizes as one-click presets (the sizes the models choose between), each coach's pick tagged
  function renderSizes(h, lg, you) {
    const box = $("#sizes");
    box.innerHTML = "";
    box.hidden = !you;
    if (!you) { renderBluff(null); return; }
    const m = me(), pot = Number(h.total_pot) || 0, toCall = Number(lg.call) || 0, mine = m ? Number(m.bet) || 0 : 0;
    const lo = Number(lg.min_raise_to), hi = Number(lg.max_raise_to);
    const rnd = (x) => Math.min(hi, Math.max(lo, Math.round(x)));
    const opts = [];
    [["Min", lo], ["½ pot", rnd(mine + toCall + 0.5 * (pot + toCall))], ["Pot", rnd(mine + toCall + pot + toCall)],
      ["2× pot", rnd(mine + toCall + 2 * (pot + toCall))], ["All-in", hi]].forEach(([label, amt]) => {
      const same = opts.find((o) => o.amt === amt);
      if (same) { if (amt === hi) same.label = "All-in"; } else opts.push({ label: amt === hi ? "All-in" : label, amt });
    });
    // a coach's pick: its likeliest raise size, tagged only when raising is that coach's top choice
    const pick = (pred) => {
      if (!pred || !pred.actions || !pred.actions.length || GROUP(grouped(pred.actions)[0].action) !== 2) return null;
      const r = pred.actions.filter((a) => GROUP(a.action) === 2).sort((x, y) => y.p - x.p)[0];
      const amt = r && r.amount !== null && r.amount !== undefined ? Number(r.amount) : r && r.action === "all_in" ? hi : null;
      return amt === null ? null : opts.reduce((best, o) => (Math.abs(o.amt - amt) < Math.abs(best.amt - amt) ? o : best), opts[0]);
    };
    const tags = new Map();
    [["Base", you.should && you.should.base], ["RL", you.should && you.should.rl]].forEach(([name, pred]) => {
      const o = pick(pred);
      if (o) tags.set(o, (tags.get(o) || []).concat(name));
    });
    opts.forEach((o) => {
      const b = el("button", { type: "button", class: "size" + (Number($("#raiseInput").value) === o.amt ? " on" : "") },
        el("span", { text: `${o.label} ${fmt(o.amt)}` }), tags.has(o) ? el("i", { text: tags.get(o).join(" · ") }) : null);
      b.addEventListener("click", () => { $("#raiseSlider").value = $("#raiseInput").value = o.amt; renderSizes(h, lg, you); askBluff(); });
      box.append(b);
    });
    askBluff();
  }

  // the bluff check for the amount in the raise box (GET /api/lab/bluff)
  let bluffTimer = null, bluffKey = "";
  function askBluff() {
    const h = hand();
    const key = h ? `${h.hand_id}|${(L.st.log || []).length}|${$("#raiseInput").value}` : "";
    if (key === bluffKey) return;
    bluffKey = key;
    clearTimeout(bluffTimer);
    bluffTimer = setTimeout(async () => {
      let b = null;
      try { b = await api(`/api/lab/bluff?amount=${encodeURIComponent($("#raiseInput").value)}`); } catch (e) { b = null; }
      if (key === bluffKey) renderBluff(b);
    }, 150);
  }
  function renderBluff(b) {
    const box = $("#bluff");
    box.innerHTML = "";
    box.hidden = !b || !b.players || !b.players.length;
    if (box.hidden) return;
    const h = hand(), pays = b.worth >= 0, lg = h && h.legal;
    const word = lg && lg.raise_kind === "bet" ? "bet" : "raise to";
    const worst = b.players.slice().sort((x, y) => x.fold - y.fold)[0];
    const seatOf = (i) => { const p = h && h.players.find((q) => q.index === i); return p ? p.seat : null; };
    box.append(
      el("div", { class: "bh", text: `If you ${word} ${fmt(b.amount)}` }),
      el("div", { class: "brow" }, "They all fold ", el("b", { text: pct(b.all_fold) }), " of the time ",
        el("span", { class: "p2", text: `· it needs ${pct(b.needs)}` })),
      el("div", { class: "bv " + (pays ? "ok" : "bad"), text: pays ? `Bluffing pays: about +${fmt(Math.round(b.worth * 10) / 10)} chips, even with nothing`
        : `Bluffing loses: about −${fmt(Math.round(-b.worth * 10) / 10)} chips with nothing` }),
      el("div", { class: "bw" }, ...b.players.map((p) => el("span", { style: `--pc: var(${colourOf(seatOf(p.index))})` }, el("i"), `${p.name} ${pct(p.fold)}`)),
        worst && worst.fold < 0.3 && b.players.length ? el("span", { class: "tag", text: `${worst.name} ${worst.fold < 0.1 ? "almost never folds" : "rarely folds"}` }) : null));
  }

  function renderSide() {
    const opps = opponents();
    // guesses exist only for players still in the hand: follow a live opponent when the selected one folds
    const h = hand();
    if (h && h.status !== "betting" && L.shownPick !== h.hand_id) {  // at showdown, look at a hand that was shown
      const shown = opps.find((p) => !p.folded && p.cards.some((c) => c && c !== "XX"));
      if (shown) { L.sel = shown.seat; L.shownPick = h.hand_id; }
    }
    if (L.sel === null || !opps.some((p) => p.seat === L.sel && !p.folded)) {
      const live = opps.find((p) => !p.folded) || opps[0];
      L.sel = live ? live.seat : null;
    }
    const chips = $("#chips");
    chips.innerHTML = "";
    opps.forEach((p) => {
      const b = el("button", { type: "button", role: "radio", class: "chip" + (p.folded ? " out" : ""), "aria-checked": String(p.seat === L.sel), disabled: p.folded, style: `--pc: var(${colourOf(p.seat)})`, title: p.folded ? `${p.name} folded` : p.name },
        el("i"), el("span", { text: p.name + (p.folded ? " · folded" : "") }));
      b.addEventListener("click", () => { L.sel = p.seat; L.picked = Date.now(); renderSide(); });
      chips.append(b);
    });

    const sel = opps.find((p) => p.seat === L.sel);
    const pc = sel ? colourOf(sel.seat) : "--line-2";
    const pr = sel ? predFor(sel.seat) : null;
    const hands = pr && pr.base ? pr.base.hands : null;
    const wrap = $("#gridWrap");
    wrap.innerHTML = "";
    wrap.style.setProperty("--pc", `var(${pc})`);
    wrap.append(grid13(hands, sel ? sel.name : "", sel ? sel.seat : null));
    const sh = sel && L.shift[sel.seat];
    const shiftEl = $("#shift");
    shiftEl.textContent = sh && h && sh.hid === h.hand_id ? `After “${sh.cause}”: ${Math.round(100 * sh.tv)}% of our guess moved` : "";
    if (sh && MOTION && sh.at > (shiftEl.dataset.at || 0)) {
      shiftEl.dataset.at = sh.at;
      shiftEl.animate([{ opacity: 0, transform: "translateY(-4px)" }, { opacity: 1, transform: "none" }], { duration: 400, easing: "ease-out" });
    }
    if (!hands) wrap.append(el("div", { class: "grid-over" }, noModel()));
    wrap.append(el("div", { class: "readout", text: hands ? "Hover a cell or use the arrow keys." : "" }));

    const top = $("#top2");
    top.innerHTML = "";
    top.append(el("div", { class: "row" }, el("span", { class: "k", text: "Top 2 hands" }), handSlot(hands, 2)));
    if (hands && hands.shown) {  // where the real hand sat in our guess: the likeliest X% of it held the real hand
      top.append(el("div", { class: "row real", title: "The 169 hand types ranked from our likeliest guess down. On 44,036 unseen test hands, the real hand fell inside our likeliest half 47% of the time." },
        el("span", { class: "k", text: "Real hand" }),
        el("span", { class: "v" }, el("b", { text: hands.shown }), el("span", { class: "p2", text: ` · our #${rankOf(hands, hands.shown)} guess of 169` }))));
    }
    // how good the "This player" read has been at this table, scored before each of their moves
    const acc = pr && pr.this_player && pr.this_player.accuracy;
    if (acc && acc.total) {
      top.append(el("div", { class: "row", title: acc.warm_hands ? `Read built from ${acc.warm_hands} earlier hands with ${sel.name}; scored only on moves at this table` : "" },
        el("span", { class: "k", text: "Next-move read" }),
        el("span", { class: "v" }, el("b", { text: `right ${acc.right} of ${acc.total}` }),
          el("span", { class: "p2", text: ` · people in general ${acc.people_right}` }))));
    }
    const hb = pr && pr.this_player && pr.this_player.habits && pr.this_player.habits[0];
    if (hb) top.append(habitRow("Habit", hb));
    const wd = pr && pr.this_player ? pr.this_player.width || 0 : 0;  // learned from their real hands
    if (wd >= 0.15) top.append(el("div", { class: "row habit", title: `Learned from their real hands: our guess is blended ${pct(wd)} towards plain card counting, because their hands keep turning out wider than people in general's.` },
      el("span", { class: "k", text: "Their range" }), el("span", { class: "v", text: `${wd >= 0.5 ? "Much wider" : "Wider"} than most people's: guess widened to match` })));
    const ramp = $("#ramp");
    ramp.innerHTML = "";
    ramp.style.setProperty("--pc", `var(${pc})`);
    ramp.append(el("span", { text: "less likely" }), el("span", { class: "bar" }), el("span", { text: "more likely" }));
  }

  // a learned pattern in plain words, with the counts behind it
  const habitRow = (label, h) => el("div", { class: "row habit" }, el("span", { class: "k", text: label }),
    el("span", { class: "v" }, h.text, el("span", { class: "p2", text: ` (${h.detail})` })));

  function grid13(hands, who, seat) {
    const g = el("div", { class: "g13" + (hands ? "" : " empty"), role: "grid", tabindex: hands ? 0 : -1, "aria-label": hands ? `Likely hands for ${who}. Arrow keys read cells.` : `Likely hands for ${who}: ${NO_MODEL}` });
    // shade by probability per combination: offsuit cells hold 12 combinations, suited 4, pairs 6, so shading by the
    // cell total would make offsuit hands look about 3x likelier than they are
    const dens = hands ? (hands.density || hands.grid) : null;
    const vals = dens ? dens.flat() : [];
    const hid = hand() ? hand().hand_id : null;
    // one colour scale per player and hand, so a cell only changes shade when its own odds change
    const sc = seat !== null && L.scale[seat] && L.scale[seat].hid === hid ? L.scale[seat] : { hid, max: 0 };
    sc.max = Math.max(sc.max, vals.length ? Math.max(...vals) : 0);
    if (seat !== null && hands) L.scale[seat] = sc;
    const max = sc.max;
    const cells = [];
    const before = seat !== null && L.drawn[seat] && L.drawn[seat].hid === hid ? L.drawn[seat].a : null;
    const after = [];
    for (let r = 0; r < 13; r++) {
      for (let c = 0; c < 13; c++) {
        const name = r === c ? RANKS[r] + RANKS[c] : r < c ? RANKS[r] + RANKS[c] + "s" : RANKS[c] + RANKS[r] + "o";
        const p = hands ? hands.grid[r][c] : 0;
        const d = dens ? dens[r][c] : 0;
        const a = d > 0 && max > 0 ? 0.1 + 0.9 * (d / max) : 0;
        const shown = hands && hands.shown === name;
        const f = el("span", { class: "f", style: `--a:${a.toFixed(3)}` });
        const cell = el("div", { class: "c" + (a > 0.55 ? " hi" : "") + (shown ? " shown" : ""), role: "gridcell", title: shown ? "their real hand" : "" }, f, el("span", { class: "t", text: name }));
        const k = after.push(a) - 1;
        if (before && MOTION && Math.abs(before[k] - a) > 0.02) f.animate([{ opacity: before[k] }, { opacity: a }], { duration: 700, easing: "cubic-bezier(.2,.7,.2,1)" });
        cell.dataset.name = name;
        cell.dataset.p = p;
        if (hands) cell.addEventListener("mouseenter", () => focusCell(r * 13 + c));
        cells.push(cell);
        g.append(cell);
      }
    }
    if (seat !== null && hands) L.drawn[seat] = { hid, a: after };
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
    if (L.replay) { L.liveSt = st; return; }  // the live table keeps going in the background during a replay
    noteActions(L.st, st);
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
    noteShifts(p);
    L.pred = p;
    render();
  }
  // new moves by opponents: was our read of them right (✓/✗ on their seat), and follow the action in the grid
  function noteActions(prev, st) {
    const log = (st && st.log) || [];
    const h = st && st.active ? st.hand : null;
    const fresh = prev && prev.hand && h && prev.hand.hand_id === h.hand_id ? log.slice(L.logLen) : [];
    L.logLen = log.length;
    fresh.forEach((e, i) => {
      if (e.player === null || e.player === undefined || e.player === st.viewer) return;
      const p = h.players.find((q) => q.index === e.player);
      if (!p) return;
      const pr = i === 0 ? predFor(p.seat) : null;  // the guess on screen was made for the first of these moves only
      const t = pr && pr.this_player && pr.this_player.action;
      if (t && t.actions && t.actions.length) {
        const top = grouped(t.actions)[0];
        const verb = (e.text.slice(p.name.length).trim().split(" ")[0] || "");
        L.flash[p.seat] = { hit: GROUP(top.action) === groupOfText(e.text), at: Date.now(), hid: h.hand_id, text: e.text,
          expected: actionText(top).toLowerCase(), verb: verb.charAt(0).toUpperCase() + verb.slice(1) };
      }
      if (!p.folded && Date.now() - L.picked > 4000) L.sel = p.seat;
    });
    if (fresh.length) setTimeout(() => { if (L.st) renderTable(); }, 2300);  // clear the badges
  }
  // how much each opponent's likely-hand guess moved since the last prediction, and what moved it
  function noteShifts(p) {
    const h = hand();
    const hid = h ? h.hand_id : null;
    const log = (L.st && L.st.log) || [];
    const last = log.length ? log[log.length - 1] : null;
    const by = last && h ? h.players.find((x) => x.index === last.player) : null;  // a caption only under the mover's grid
    ((p && p.players) || []).forEach((q) => {
      const g = q.base && q.base.hands && q.base.hands.grid;
      if (!g) return;
      const flat = g.flat();
      const m = L.mem[q.seat];
      if (m && m.hid === hid) {
        const tv = 0.5 * flat.reduce((s, x, i) => s + Math.abs(x - m.flat[i]), 0);
        if (tv >= 0.02 && by && by.seat === q.seat) L.shift[q.seat] = { tv, cause: last.text, at: Date.now(), hid };
      } else L.shift[q.seat] = null;
      L.mem[q.seat] = { hid, flat };
    });
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
        small_blind: 1, big_blind: 2, ante: 0, stack: 200, bot_delay_ms: 900, spectate: false, warm_reads: true, rebuy: true,
      }));
      L.lastCheck = null;
    } catch (e) { $("#actNote").textContent = e.message; }
  }
  async function nextHand() {
    const h = hand();
    if (!h || h.status === "betting") return;
    try { onState(await api("/api/game/next", {})); } catch (e) { $("#actNote").textContent = e.message; }
  }

  // ------------------------------------------------------------ how good are the guesses (GET /api/lab/scorecard)
  const sgn = (x) => `${x >= 0 ? "+" : "−"}${Math.abs(Math.round(x))}`;
  async function openScore() {
    let d = null;
    try { d = await api("/api/lab/scorecard"); } catch (e) { d = null; }
    const rows = $("#scoreRows");
    rows.innerHTML = "";
    const bars = (items) => el("div", { class: "sbars" }, ...items.map(([label, v, ours]) =>
      el("div", { class: "sb" + (ours ? " ours" : "") }, el("span", { class: "sl", text: label }),
        el("span", { class: "st" }, el("i", { style: `width:${(100 * Math.max(0, Math.min(1, v))).toFixed(1)}%` })), el("b", { text: pct(v) }))));
    const row = (title, body, note) => el("section", { class: "srow" }, el("h3", { text: title }), body, el("p", { class: "snote", text: note }));
    const m = d && d.moves, h = d && d.hands, c = d && d.coach;
    rows.append(row("Their next move", m && m.ours ? bars([["Our model of that player", m.ours, true], ["People in general", m.people], ["Their usual move", m.usual]]) : noModel(),
      m ? `${m.players ? `${m.players} real players, ${m.decisions.toLocaleString()} decisions, guessed before each move; better for ${m.helped} of ${m.players}. ` : ""}` +
        `When it says ${pct(m.says)}, it happens ${pct(m.happens)} of the time (${m.calib_n.toLocaleString()} decisions).` : ""));
    rows.append(row("Their hand", h ? bars([["Our top 10 guesses (of 169)", h.top10, true], ["Card counting alone", h.top10_counting]]) : noModel(),
      h ? `Share of ${h.n.toLocaleString()} guesses on about 2,000 unseen Pluribus hands, every card known, where the real hand was in the top 10. Our likeliest half held it ${pct(h.half)} of the time (50% would be perfect)` +
        (h.bots ? `. On this table's bots, after learning how wide each one plays: ${pct(h.bots[0])} / ${pct(h.bots[1])} / ${pct(h.bots[2])} inside our likeliest 50 / 80 / 90% (fresh hands).` : ".") : ""));
    rows.append(row("Your best move (RL coach)",
      c ? el("div", { class: "sbig" }, el("b", { text: `${sgn(c.vs_equity[0])} ± ${Math.round(c.vs_equity[1])}` }), " big blinds per 100 hands against a pure-maths bot") :
        el("div", { class: "sbig muted", text: "Tested once, on sealed tables, when training ends." }),
      c ? `On tables it never trained on or was picked on, ${c.hands.toLocaleString()} hands each, same cards for both.` +
        (c.ranges ? ` Predicted hands add ${sgn(c.ranges[0])} ± ${Math.round(c.ranges[1])}.` : "") + ` Watching the table adds ${sgn(c.styles[0])} ± ${Math.round(c.styles[1])}.` :
        "Opponents it never trained on and that were never used to pick it."));
    $("#scoreCard").hidden = false;
  }

  // ------------------------------------------------------------ replay of a real player no model has seen
  // Precomputed by scripts/build_replay.py with the same models as the live table (GET /api/lab/replay).
  const RP = { data: null, cache: {}, who: "unseen", h: 0, f: 0, timer: null };
  const theirs = () => L.replay && !(RP.data && RP.data.player.mine);
  const DONE = { fold: "folded", call: "called", raise: "raised" };
  async function openReplay() {
    if (!("unseen" in RP.cache)) {
      try { RP.cache.unseen = await api("/api/lab/replay"); } catch (e) { RP.cache.unseen = null; }
      try { RP.cache.you = await api("/api/lab/replay?who=you"); } catch (e) { RP.cache.you = null; }
      RP.who = RP.cache.you ? "you" : "unseen";
    }
    $("#replayWho").hidden = !(RP.cache.you && RP.cache.unseen);
    document.querySelectorAll("#replayWho .chip").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.who === RP.who)));
    RP.data = RP.cache[RP.who];
    if (!RP.data) { $("#actNote").textContent = "No replay built (python scripts/build_replay.py)"; return; }
    const d = RP.data, s = d.summary, mine = d.player.mine;
    $("#replayTitle").textContent = mine ? "You, learned as you play" : "A real player, learned as they play";
    $("#habitsHead").textContent = mine ? "What the model learned about you" : "What our model learned about them";
    $("#replayStep").firstChild.textContent = mine ? "Play my hands " : "Play their hands ";
    $("#replayNote").textContent = mine
      ? `${d.player.hands} of your hands at this table, studied in order. Before each of your ${d.player.decisions.toLocaleString()} moves every model guessed: ` +
        `the model of you was right ${pct(s.personal)}, people in general ${pct(s.people)}, your usual move ${pct(s.usual)}.`
      : `${d.player.hands} real hands of one online player no model was trained on, studied in order like a poker tracker would. ` +
        `We picked them out of 30 unseen players for a strong style (across all 30, our model beat people in general for 28). ` +
        `Before each of their ${d.player.decisions.toLocaleString()} moves every model guessed: ours was right ${pct(s.personal)}, people in general ${pct(s.people)}, their usual move ${pct(s.usual)}.`;
    const hl = $("#replayHabits");
    hl.innerHTML = "";
    (d.habits || []).forEach((h) => hl.append(el("li", {}, h.text, el("span", { class: "p2", text: ` (${h.detail})` }))));
    hl.hidden = !(d.habits || []).length;
    drawCurve(d.curve);
    $("#replayCard").hidden = false;
  }
  function drawCurve(rows) {
    // share right over the last 100 decisions, for each model, as the hands go by
    const W = 560, H = 150, win = 100, lo = 0.3, hi = 0.85;
    const pts = [[], [], []];
    for (let i = win; i <= rows.length; i++) {
      const w = rows.slice(i - win, i);
      for (let m = 0; m < 3; m++) pts[m].push([i, w.reduce((a, r) => a + r[1 + m], 0) / win]);
    }
    const X = (i) => 30 + ((W - 40) * (i - win)) / Math.max(1, rows.length - win);
    const Yv = (v) => 10 + (H - 30) * (1 - (Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo));
    const line = (p) => p.map(([i, v], k) => `${k ? "L" : "M"}${X(i).toFixed(1)},${Yv(v).toFixed(1)}`).join("");
    const ticks = [0.4, 0.6, 0.8].map((v) => `<line x1="30" x2="${W - 10}" y1="${Yv(v)}" y2="${Yv(v)}" class="gl"/><text x="2" y="${Yv(v) + 4}">${Math.round(v * 100)}%</text>`).join("");
    $("#replayCurve").innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Share of moves predicted right over the last 100 decisions">${ticks}` +
      `<path d="${line(pts[2])}" class="ln usu"/><path d="${line(pts[1])}" class="ln ppl"/><path d="${line(pts[0])}" class="ln you"/>` +
      `<text x="${W - 10}" y="${H - 4}" text-anchor="end">decisions studied →</text></svg>`;
  }
  function enterSteps() {
    if (!RP.data || !RP.data.hands.length) return;
    $("#replayCard").hidden = true;
    L.liveSt = L.st;
    L.replay = true;
    L.injected = true;  // no live model calls while replaying
    RP.h = 0; RP.f = 0;
    const mine = RP.data.player.mine;
    document.querySelectorAll("#you .kicker")[0].textContent = mine ? "What you should do" : "What they should do";
    document.querySelectorAll("#you .kicker")[1].textContent = mine ? "What you would do" : "What they will do";
    L.flash = {}; L.shift = {}; L.mem = {}; L.drawn = {}; L.logLen = 0;
    showFrame();
    play(true);
  }
  // the replay plays itself like a live table; pause, or step with the arrow keys
  function play(on) {
    clearInterval(RP.timer);
    RP.timer = on ? setInterval(() => stepReplay(1, true), 1500) : null;
    $("#rpPlay").textContent = on ? "❚❚ Pause" : "▶ Play";
  }
  function showFrame(forward) {
    const hnd = RP.data.hands[RP.h], fr = hnd.frames[RP.f];
    if (forward && RP.f > 0) noteActions(L.st, fr.st);  // ✓/✗ on the opponents who just moved, as at the live table
    else L.logLen = ((fr.st && fr.st.log) || []).length;
    L.st = fr.st;
    noteShifts(fr.pred);
    L.pred = fr.pred;
    const prev = RP.f > 0 ? hnd.frames[RP.f - 1].check : null;
    if (prev && forward) {  // and on the replayed player
      const m = fr.st.hand.players.find((q) => q.index === fr.st.viewer), lg = (fr.st.log || []).slice(-1)[0];
      const word = DONE[prev.actual] || prev.actual;
      if (m) L.flash[m.seat] = { hit: prev.hit, at: Date.now(), hid: fr.st.hand.hand_id, text: lg ? lg.text : "", expected: prev.pred, verb: word.charAt(0).toUpperCase() + word.slice(1) };
    }
    L.lastCheck = prev ? { pred: prev.pred, actual: DONE[prev.actual] || prev.actual, hit: prev.hit } : null;
    render();
    $("#actTitle").textContent = `Replay · hand ${RP.h + 1} of ${RP.data.hands.length} · step ${RP.f + 1}/${hnd.frames.length}`;
    $("#actNote").textContent = `Their real hand no. ${hnd.hand_no} of ${RP.data.player.hands}. Their cards are from the showdown (for the coach only); the guesses never see them.`;
  }
  function stepReplay(d, auto) {
    if (!L.replay) return;
    if (!auto) play(false);
    const hands = RP.data.hands;
    if (auto && RP.h === hands.length - 1 && RP.f === hands[RP.h].frames.length - 1) { play(false); return; }
    RP.f += d;
    if (RP.f >= hands[RP.h].frames.length) { if (RP.h < hands.length - 1) { RP.h++; RP.f = 0; } else RP.f = hands[RP.h].frames.length - 1; }
    if (RP.f < 0) { if (RP.h > 0) { RP.h--; RP.f = hands[RP.h].frames.length - 1; } else RP.f = 0; }
    showFrame(d > 0);
  }
  function exitReplay() {
    play(false);
    $("#replayCard").hidden = true;
    if (!L.replay) return;
    L.replay = false;
    L.injected = false;
    L.lastCheck = null;
    document.querySelectorAll("#you .kicker")[0].textContent = "What you should do";
    document.querySelectorAll("#you .kicker")[1].textContent = "What you would do";
    L.predKey = null;
    L.pred = null;
    onState(L.liveSt);
  }

  function bind() {
    $("#replayBtn").addEventListener("click", openReplay);
    $("#scoreBtn").addEventListener("click", openScore);
    $("#scoreExit").addEventListener("click", () => { $("#scoreCard").hidden = true; });
    $("#replayStep").addEventListener("click", enterSteps);
    $("#replayExit").addEventListener("click", exitReplay);
    $("#rpPrev").addEventListener("click", () => stepReplay(-1));
    $("#rpPlay").addEventListener("click", () => play(!RP.timer));
    document.querySelectorAll("#replayWho .chip").forEach((b) => b.addEventListener("click", () => { RP.who = b.dataset.who; openReplay(); }));
    $("#rpNext").addEventListener("click", () => stepReplay(1));
    $("#rpExit").addEventListener("click", exitReplay);
    $("#newTable").addEventListener("click", newTable);
    $("#sitBtn").addEventListener("click", newTable);
    $("#btnFold").addEventListener("click", () => act("fold"));
    $("#btnCall").addEventListener("click", () => act("call"));
    $("#btnRaise").addEventListener("click", () => act("raise", Number($("#raiseInput").value)));
    $("#btnNext").addEventListener("click", nextHand);
    $("#raiseSlider").addEventListener("input", (e) => { $("#raiseInput").value = e.target.value; askBluff(); });
    $("#raiseInput").addEventListener("input", (e) => { $("#raiseSlider").value = e.target.value; askBluff(); });

    $("#clearInject").addEventListener("click", () => window.Lab.clear());
    document.addEventListener("keydown", (e) => {
      if (e.target.matches("input, select, textarea") || e.metaKey || e.ctrlKey || e.altKey) return;
      const k = e.key.toLowerCase();
      if (!$("#scoreCard").hidden && e.key === "Escape") { $("#scoreCard").hidden = true; return; }
      if (L.replay || !$("#replayCard").hidden) {
        if (e.key === " " && L.replay) { e.preventDefault(); play(!RP.timer); }
        else if (e.key === "ArrowRight") { if (L.replay) stepReplay(1); else enterSteps(); }
        else if (e.key === "ArrowLeft") stepReplay(-1);
        else if (e.key === "Escape") exitReplay();
        return;
      }
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
