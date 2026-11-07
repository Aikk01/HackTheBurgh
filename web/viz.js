/* Shared figures and motion helpers for Poker Mirror and Evidence.
   Motion: transforms + opacity via the Web Animations API, cancellable, skipped under reduced motion. */
(function () {
  "use strict";
  const P = window.PokerSim;
  const { el, fmt, pct } = P;
  const NS = "http://www.w3.org/2000/svg";

  const reduceMotion = () => window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  function anim(node, frames, opts) {
    if (!node || reduceMotion() || !node.animate) return null;
    try { return node.animate(frames, Object.assign({ easing: "cubic-bezier(.2,.7,.2,1)", fill: "backwards" }, opts)); } catch (e) { return null; }
  }
  // Move `node` visually from a viewport rect to where it sits now (FLIP).
  function flyFrom(node, fromRect, opts) {
    if (!node || !fromRect) return null;
    const to = node.getBoundingClientRect();
    const dx = fromRect.left - to.left, dy = fromRect.top - to.top;
    if (!dx && !dy) return null;
    return anim(node, [{ transform: `translate(${dx}px, ${dy}px)` }, { transform: "none" }], Object.assign({ duration: 200 }, opts));
  }
  let announceTimer = null;
  function announce(msg) {
    const a = document.getElementById("announcer");
    if (!a) return;
    a.textContent = "";
    clearTimeout(announceTimer);
    announceTimer = setTimeout(() => { a.textContent = msg; }, 60);
  }
  const svg = (tag, attrs, text) => { const e = document.createElementNS(NS, tag); Object.entries(attrs || {}).forEach(([k, v]) => { if (v !== null && v !== undefined) e.setAttribute(k, v); }); if (text !== undefined) e.textContent = text; return e; };
  const n0 = (x) => Math.round(x).toLocaleString();
  const sgn = (x, d = 2) => (x > 0 ? "+" : x < 0 ? "−" : "") + Math.abs(x).toFixed(d);
  const SUIT_NAME = { c: "clubs", d: "diamonds", h: "hearts", s: "spades" };
  const RANK_NAME = { A: "ace", K: "king", Q: "queen", J: "jack", T: "ten" };
  const cardName = (c) => (!c || c === "XX" || c === "??") ? "hidden card" : `${RANK_NAME[c[0]] || c[0]} of ${SUIT_NAME[c[1]] || c[1]}`;
  const cardsName = (cs) => (cs || []).map(cardName).join(", ");

  // Large card face: rank top-left in serif, suit centred. Never animates on its own (identity is managed by callers).
  function card(code, size) {
    const c = P.cardEl(code, "mcard" + (size ? " " + size : ""));
    c.setAttribute("role", "img");
    c.setAttribute("aria-label", cardName(code));
    return c;
  }
  function board(codes, size) {
    return el("div", { class: "m-board", role: "group", "aria-label": "Board: " + cardsName(codes) }, (codes || []).map((c) => card(c, size)));
  }
  function dealIn(cards, stagger) {
    (cards || []).forEach((c, i) => anim(c, [{ opacity: 0, transform: "translateY(-10px) scale(.96)" }, { opacity: 1, transform: "none" }], { duration: 180, delay: (stagger || 0) * i }));
  }
  function flipIn(c) { return anim(c, [{ transform: "perspective(600px) rotateY(85deg)", opacity: 0.4 }, { transform: "none", opacity: 1 }], { duration: 200 }); }

  // ------------------------------------------------------------ filter strip (fixed full-cohort scale, excluded groups hatched in place)
  // cohorts = exhibit.two_shard.cohorts. Groups follow from the artifact: folded, continued with cards unknown, continued with cards known.
  function filterStrip(cohorts) {
    const all = cohorts.all_eligible, known = cohorts.defender_cards_known, clean = cohorts.warning_free;
    const total = all.opportunities, folded = all.folds;
    const knownCont = known.opportunities - known.folds;
    const unknownCont = total - folded - knownCont;
    const continued = total - folded;
    const partitionable = known.folds === 0 && unknownCont >= 0;
    const g = [
      { key: "folded", n: folded, label: "folded" },
      { key: "unknown", n: unknownCont, label: "continued, cards unknown" },
      { key: "known", n: knownCont, label: "continued, cards known" },
    ];
    const frac = (x) => (100 * x / total);
    const strip = el("div", { class: "fstrip" });
    const segs = {};
    let left = 0;
    g.forEach((x) => {
      const seg = el("div", { class: "fseg fseg-" + x.key, style: `left:${frac(left)}%;width:${frac(x.n)}%` }, el("div", { class: "hatch" }), el("div", { class: "fill" }));
      seg.dataset.left = frac(left);
      segs[x.key] = seg;
      strip.appendChild(seg);
      left += x.n;
    });
    // warning-free filter: the artifact records how many continued offers survive, not which group they came from,
    // so the two continued groups merge and only their aggregate retracts.
    const merged = el("div", { class: "fseg fseg-merged", style: `left:${frac(folded)}%;width:${frac(continued)}%` }, el("div", { class: "hatch" }), el("div", { class: "fill" }));
    strip.appendChild(merged);
    strip.appendChild(el("div", { class: "fstrip-total" }));
    const labels = el("div", { class: "flabels" });
    const lab = {};
    g.forEach((x) => { lab[x.key] = el("div", { class: "flabel", style: `left:${segs[x.key].dataset.left}%` }, el("b", { class: "num" }), " ", el("span")); labels.appendChild(lab[x.key]); });
    lab.merged = el("div", { class: "flabel", style: `left:${frac(folded)}%` }, el("b", { class: "num" }), " ", el("span"));
    labels.appendChild(lab.merged);
    const root = el("div", { class: "fstrip-wrap" }, strip, labels);
    const setLab = (k, num, text, show) => { lab[k].hidden = !show; lab[k].querySelector("b").textContent = num; lab[k].querySelector("span").textContent = text; };
    const scale = (seg, f) => { seg.querySelector(".fill").style.transform = `scaleX(${Math.max(0, Math.min(1, f))})`; seg.classList.toggle("excluded", f < 1); };

    function set(mode) {
      const cleanCont = clean.opportunities - clean.folds;
      strip.dataset.mode = mode;
      if (mode === "clean") {
        scale(segs.folded, folded ? clean.folds / folded : 1);
        [segs.unknown, segs.known].forEach((s) => { s.classList.add("merged-away"); });
        merged.classList.add("on");
        scale(merged, continued ? cleanCont / continued : 1);
        setLab("folded", n0(clean.folds), clean.folds === folded ? "folded, all kept" : `of ${n0(folded)} folded kept`, true);
        setLab("unknown", "", "", false); setLab("known", "", "", false);
        setLab("merged", `${n0(cleanCont)} of ${n0(continued)}`, "continued offers kept · the filter does not record which group", true);
      } else {
        [segs.unknown, segs.known].forEach((s) => s.classList.remove("merged-away"));
        merged.classList.remove("on");
        scale(merged, 1);
        const keepKnown = mode === "known";
        scale(segs.folded, keepKnown ? 0 : 1);
        scale(segs.unknown, keepKnown ? 0 : 1);
        scale(segs.known, 1);
        g.forEach((x) => setLab(x.key, n0(x.n), x.label, true));
        setLab("merged", "", "", false);
      }
      const c = mode === "clean" ? clean : mode === "known" ? known : all;
      strip.setAttribute("role", "img");
      strip.setAttribute("aria-label", `Scale of ${n0(total)} eligible offers: ${n0(folded)} folded, ${n0(unknownCont)} continued with cards unknown, ${n0(knownCont)} continued with cards known. ` +
        (mode === "all" ? "No filter: all kept." : `Filter keeps ${n0(c.opportunities)} offers, of which ${n0(c.folds)} folded.`));
    }
    return { root, set, partitionable, groups: g, total, folded };
  }

  // ------------------------------------------------------------ signed interval plot (dot + 95% interval, visible zero)
  // rows: [{label, sub, mean, lo, hi, failed, note}]
  function intervalPlot(rows, opts) {
    opts = opts || {};
    const W = Math.max(520, Math.round(opts.width || 760)), rowH = opts.rowH || 46, mt = 14, mb = 46, ml = opts.labelW || 210, mr = opts.valueW || 150;
    const H = mt + rows.length * rowH + mb;
    let lo = Math.min(0, ...rows.map((r) => r.lo)), hi = Math.max(0, ...rows.map((r) => r.hi));
    if (opts.domain) { lo = Math.min(lo, opts.domain[0]); hi = Math.max(hi, opts.domain[1]); }
    const span = hi - lo || 1;
    const step = opts.step || [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50].find((s) => span / s <= 8) || 100;
    const vf = opts.fmt || ((v) => sgn(v));
    lo = Math.floor(lo / step) * step; hi = Math.ceil(hi / step) * step;
    const x = (v) => ml + (W - ml - mr) * (v - lo) / ((hi - lo) || 1);
    // drawn at its container's width and never scaled up, so 13px labels stay 13px
    const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "iplot", role: "img", "aria-label": opts.aria || "Interval plot", style: `max-width:${W}px` });
    for (let t = lo; t <= hi + 1e-9; t += step) {
      s.appendChild(svg("line", { x1: x(t), x2: x(t), y1: mt - 4, y2: H - mb + 4, class: Math.abs(t) < 1e-9 ? "zero" : "grid" }));
      s.appendChild(svg("text", { x: x(t), y: H - mb + 20, "text-anchor": "middle", class: "tick" }, (opts.tickFmt || ((v) => sgn(v, step >= 1 ? 0 : step >= 0.1 ? 1 : 2)))(t)));
    }
    if (opts.zeroLabel) s.appendChild(svg("text", { x: x(0) + 4, y: mt + 2, class: "tick zero-label" }, opts.zeroLabel));
    s.appendChild(svg("text", { x: ml + (W - ml - mr) / 2, y: H - 8, "text-anchor": "middle", class: "axis-title" }, opts.xTitle || ""));
    const tip = el("div", { class: "chart-tip", hidden: true });
    rows.forEach((r, i) => {
      const cy = mt + i * rowH + rowH / 2;
      const g = svg("g", { class: "irow" + (r.failed ? " failed" : "") + (r.highlight ? " hl" : ""), tabindex: 0, role: "img",
        "aria-label": `${r.label}: ${vf(r.mean)} ${opts.unit || ""}, 95% interval ${vf(r.lo)} to ${vf(r.hi)}${r.failed ? ". " + (r.note || "flagged") : ""}` });
      g.appendChild(svg("rect", { x: 0, y: cy - rowH / 2, width: W, height: rowH, class: "hit" }));
      g.appendChild(svg("text", { x: ml - 14, y: cy - (r.sub ? 3 : -4), "text-anchor": "end", class: "rlabel" }, r.label));
      if (r.sub) g.appendChild(svg("text", { x: ml - 14, y: cy + 13, "text-anchor": "end", class: "rsub" }, r.sub));
      g.appendChild(svg("line", { x1: x(r.lo), x2: x(r.hi), y1: cy, y2: cy, class: "whisker" }));
      [r.lo, r.hi].forEach((v) => g.appendChild(svg("line", { x1: x(v), x2: x(v), y1: cy - 5, y2: cy + 5, class: "whisker" })));
      g.appendChild(svg("circle", { cx: x(r.mean), cy, r: 6, class: "dot" + (r.hollow ? " hollow" : "") }));
      g.appendChild(svg("text", { x: W - mr + 14, y: cy - (r.failed ? 3 : -4), class: "rval" }, `${vf(r.mean)}  [${vf(r.lo)}, ${vf(r.hi)}]`));
      if (r.failed) g.appendChild(svg("text", { x: W - mr + 14, y: cy + 13, class: "rsub flag" }, r.note || "flagged"));
      const show = (ev) => {
        tip.hidden = false;
        tip.textContent = `${r.label}: ${vf(r.mean)} ${opts.unit || ""} (95% interval ${vf(r.lo)} to ${vf(r.hi)})${r.failed ? " · " + (r.note || "") : ""}${r.extra ? " · " + r.extra : ""}`;
        const box = wrap.getBoundingClientRect(), gb = g.getBoundingClientRect();
        const px = ev && ev.clientX ? ev.clientX - box.left : gb.left - box.left + gb.width / 2;
        tip.style.left = Math.max(0, Math.min(px + 12, box.width - 280)) + "px";
        tip.style.top = (gb.top - box.top + gb.height - 4) + "px";
      };
      g.addEventListener("mousemove", show);
      g.addEventListener("focus", () => show(null));
      g.addEventListener("mouseleave", () => { tip.hidden = true; });
      g.addEventListener("blur", () => { tip.hidden = true; });
      s.appendChild(g);
    });
    const wrap = el("div", { class: "iplot-wrap" }, s, tip);
    return wrap;
  }

  // ------------------------------------------------------------ fold forecast scale: committed estimate, 90% model band, later a second mark
  function forecastScale(pred, opts) {
    opts = opts || {};
    const band = el("div", { class: "fs-band", style: `left:${100 * pred.fold_p05}%;width:${100 * (pred.fold_p95 - pred.fold_p05)}%` });
    const mark = el("div", { class: "fs-mark committed", style: `left:${100 * pred.fold_mean}%` }, el("span", { class: "fs-tag", text: pct(pred.fold_mean, 0) }));
    const next = el("div", { class: "fs-mark next", style: `left:${100 * pred.fold_mean}%`, hidden: true }, el("span", { class: "fs-tag", text: "" }));
    const track = el("div", { class: "fs-track" }, band, mark, next,
      ...[0, 0.25, 0.5, 0.75, 1].map((t) => el("div", { class: "fs-tick", style: `left:${100 * t}%` }, el("span", { text: Math.round(100 * t) + "%" }))));
    const legend = el("div", { class: "fs-legend" },
      el("span", { class: "k committed" }, el("i"), `${opts.beforeLabel || "Before your action"} `, el("b", { class: "num", text: pct(pred.fold_mean, 0) })),
      el("span", { class: "k band" }, el("i"), "90% model band ", el("b", { class: "num", text: `${pct(pred.fold_p05, 0)}–${pct(pred.fold_p95, 0)}` })),
      el("span", { class: "k next", hidden: true }, el("i"), "Next estimate ", el("b", { class: "num", text: "" })));
    const root = el("div", { class: "fscale", role: "group", "aria-label": `Fold forecast ${pct(pred.fold_mean, 0)}, 90% band ${pct(pred.fold_p05, 0)} to ${pct(pred.fold_p95, 0)}` },
      el("div", { class: "fs-title" }, opts.title || "Chance you fold"), track, legend);
    return {
      root,
      showNext(after, animate) {
        next.hidden = false;
        next.querySelector(".fs-tag").textContent = pct(after.fold_mean, 0);
        const k = legend.querySelector(".k.next");
        k.hidden = false;
        k.querySelector("b").textContent = pct(after.fold_mean, 0);
        // the committed estimate and its band stay where they were; only the new mark travels
        const go = () => { next.style.left = `${100 * after.fold_mean}%`; };
        if (animate && !reduceMotion()) requestAnimationFrame(() => requestAnimationFrame(go)); else go();
        root.setAttribute("aria-label", `Fold forecast before your action ${pct(pred.fold_mean, 0)}; next estimate ${pct(after.fold_mean, 0)}`);
      },
    };
  }

  // Three-way response split as a thin segmented bar; the observed action gets an outline, never the model's colour.
  function responseSplit(probs, actual) {
    const keys = ["fold", "call", "raise"];
    const bar = el("div", { class: "rsplit" }, keys.map((k) => el("div", { class: "rs-seg rs-" + k + (actual === k ? " actual" : ""), style: `flex:${Math.max(0.0001, probs[k])}`, title: `${k} ${pct(probs[k], 0)}` })));
    const lab = el("div", { class: "rs-labels" }, keys.map((k) => el("span", { class: actual === k ? "actual" : "" }, `${k} `, el("b", { class: "num", text: pct(probs[k], 0) }), actual === k ? el("em", { text: " · you" }) : null)));
    return el("div", { class: "rsplit-wrap", role: "img", "aria-label": "Forecast split: " + keys.map((k) => `${k} ${pct(probs[k], 0)}`).join(", ") + (actual ? `. You chose ${actual}.` : "") }, bar, lab);
  }

  // Weak / medium / strong: before the response (outline) and after it (fill).
  function bucketBars(prior, post) {
    const keys = ["weak", "medium", "strong"];
    const rows = keys.map((k) => {
      const fill = el("div", { class: "bb-post", style: `width:${100 * prior[k]}%` });
      const row = el("div", { class: "bb-row" }, el("span", { class: "bb-k", text: k }),
        el("div", { class: "bb-track" }, el("div", { class: "bb-prior", style: `width:${100 * prior[k]}%` }), fill),
        el("span", { class: "bb-v num" }, pct(post[k], 0), el("span", { class: "muted", text: ` was ${pct(prior[k], 0)}` })));
      requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = `${100 * post[k]}%`; }));
      return row;
    });
    return el("div", { class: "bbars", role: "img", "aria-label": "Estimated holdings: " + keys.map((k) => `${k} ${pct(post[k], 0)}, was ${pct(prior[k], 0)}`).join("; ") },
      el("div", { class: "bb-legend" }, el("span", { class: "k prior" }, el("i"), "before the response"), el("span", { class: "k post" }, el("i"), "after it")), ...rows);
  }

  // 13 x 13 estimated range, one focusable grid with arrow-key movement and a labelled readout.
  function rangeGrid(r) {
    const ranks = "AKQJT98765432".split("");
    const cells = [];
    const readout = el("div", { class: "rg-readout", "aria-live": "polite" });
    const grid = el("div", { class: "grid13", role: "grid", tabindex: 0, "aria-label": "Estimated 13 by 13 range. Use arrow keys to read cells." });
    let cur = 0;
    ranks.forEach((r1, i) => ranks.forEach((r2, j) => {
      const cls = i === j ? r1 + r2 : i < j ? r1 + r2 + "s" : r2 + r1 + "o";
      const w = r.grid[cls] || 0;
      const f = r.grid_max ? w / r.grid_max : 0;
      const own = r.own_cells.includes(cls);
      const cell = el("div", { class: "cell" + (own ? " own" : "") + (f > 0.55 ? " hi" : ""), role: "gridcell", style: `--w:${f.toFixed(3)}`, text: cls });
      cell.dataset.cls = cls; cell.dataset.w = w;
      cell.addEventListener("mouseenter", () => focusCell(i * 13 + j, false));
      cells.push(cell);
      grid.appendChild(cell);
    }));
    const kind = (c) => (c.length === 2 ? "pair" : c[2] === "s" ? "suited" : "offsuit");
    function focusCell(k, kb) {
      cells[cur].classList.remove("cursor");
      cur = Math.max(0, Math.min(168, k));
      cells[cur].classList.add("cursor");
      const c = cells[cur].dataset.cls, w = Number(cells[cur].dataset.w);
      readout.textContent = `${c} (${kind(c)}): ${(100 * w).toFixed(2)}% of the estimated range` + (r.own_cells.includes(c) ? " · your hand" : "");
      if (kb) grid.setAttribute("aria-activedescendant", "");
    }
    grid.addEventListener("keydown", (e) => {
      const m = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: 13, ArrowUp: -13, Home: -cur, End: 168 - cur }[e.key];
      if (m === undefined) return;
      e.preventDefault();
      focusCell(cur + m, true);
    });
    grid.addEventListener("focus", () => focusCell(cur, true));
    const top = Object.entries(r.grid).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([k]) => k).join(", ");
    readout.textContent = `Most weight: ${top}. Hover or use arrow keys to read a cell.`;
    return el("div", { class: "rgrid" }, grid, readout);
  }

  // Two comparable chip values on one 0-100 scale (opening pot = 100 chips).
  function evCompare(a) {
    const rows = [
      { k: "Check back", v: a.ev_check * 100, band: null, rec: a.recommend === "check" },
      { k: "Bet 50", v: a.ev_bet * 100, band: a.ev_bet_band ? a.ev_bet_band.map((x) => x * 100) : null, rec: a.recommend === "bet" },
    ];
    const max = Math.max(100, ...rows.map((r) => (r.band ? r.band[1] : r.v)));
    const X = (v) => `${Math.max(0, Math.min(100, 100 * v / max))}%`;
    return el("div", { class: "evc", role: "img", "aria-label": rows.map((r) => `${r.k}: ${n0(r.v)} chips${r.band ? `, band ${n0(r.band[0])} to ${n0(r.band[1])}` : ""}`).join("; ") },
      ...rows.map((r) => el("div", { class: "evc-row" + (r.rec ? " rec" : "") },
        el("span", { class: "evc-k", text: r.k }),
        el("div", { class: "evc-track" },
          r.band ? el("div", { class: "evc-band", style: `left:${X(r.band[0])};width:calc(${X(r.band[1])} - ${X(r.band[0])})` }) : null,
          el("div", { class: "evc-dot", style: `left:${X(r.v)}` })),
        el("span", { class: "evc-v num" }, `${n0(r.v)}`, el("span", { class: "muted", text: " chips" })))),
      el("div", { class: "evc-axis" }, el("span"), el("div", { class: "evc-ticks" }, el("span", { text: "0" }), el("span", { text: n0(max / 2) }), el("span", { text: n0(max) })), el("span", { text: "chips" })));
  }

  window.Viz = { reduceMotion, anim, flyFrom, announce, svg, card, board, dealIn, flipIn, filterStrip, intervalPlot, forecastScale, responseSplit, bucketBars, rangeGrid, evCompare, cardName, cardsName, sgn, n0 };
})();
