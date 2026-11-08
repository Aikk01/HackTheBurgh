"""Build the Poker Mirror deck: docs/slides/PokerMirror.pptx (and an HTML preview for checking the layout).

    python docs/slides/build_deck.py [--preview DIR]     # needs python-pptx (pip install python-pptx)

One layout description per slide, in inches and points; the same description is written to PowerPoint and, with
--preview, to one HTML page per slide (for a browser screenshot), so what is checked is what ships.  Content and
numbers come from docs/slides/SLIDES.md and SOURCES.md.
"""
from __future__ import annotations

import argparse
import html
import os

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 13.333, 7.5

# the demo's palette (web/lab/lab.css)
BG, CARD, CARD2, LINE = "0E302B", "123A33", "164239", "33665B"
INK, INK2, MUTED, AMBER, GOOD = "FFF8E7", "D8D2BF", "B6C5BC", "FFD58C", "7FCB8A"
P1, P2, P3, P4 = "3987E5", "D95926", "199E70", "9085E9"
HEAD, BODY = "Georgia", "Arial"


def card(x, y, w, h, fill=CARD, line=None):
    return {"k": "rect", "x": x, "y": y, "w": w, "h": h, "fill": fill, "line": line}


def text(x, y, w, h, paras, size=14, color=INK, font=BODY, bold=False, align="l", valign="t", space=6):
    """paras: a string, or a list of paragraphs; a paragraph is a string or a list of runs (str or (str, opts))."""
    if isinstance(paras, str):
        paras = [paras]
    return {"k": "text", "x": x, "y": y, "w": w, "h": h, "paras": paras, "size": size, "color": color, "font": font,
            "bold": bold, "align": align, "valign": valign, "space": space}


def image(path, x, y, w, h, alt):
    return {"k": "img", "path": path, "x": x, "y": y, "w": w, "h": h, "alt": alt}


def arrow(x, y, w=0.28, h=0.22, color=AMBER):
    return {"k": "arrow", "x": x, "y": y, "w": w, "h": h, "color": color}


def title(t, sub=None):
    els = [text(0.6, 0.42, 12.1, 0.75, t, size=34, font=HEAD, color=INK)]
    if sub:
        els.append(text(0.6, 1.17, 12.1, 0.4, sub, size=15, color=MUTED))
    return els


def stat(x, y, w, h, big, label, color=AMBER, big_size=34):
    return [card(x, y, w, h),
            text(x + 0.3, y + 0.25, w - 0.6, 0.85, big, size=big_size, font=HEAD, color=color),
            text(x + 0.3, y + 1.15, w - 0.6, h - 1.35, label, size=14, color=INK2)]


SLIDES = []

# 1 · introduction and fit -------------------------------------------------------------------------------------------
s = [text(0.6, 0.55, 6.4, 0.3, "HackTheBurgh 2026 · HRT · G-Research · Make the invisible visible", size=12, color=MUTED),
     text(0.6, 0.95, 6.4, 1.0, "Poker Mirror", size=54, font=HEAD, color=INK),
     text(0.6, 2.0, 6.2, 1.1, "A poker training simulator that makes the hidden visible, and measures every prediction "
          "on data it never saw.", size=19, color=INK2)]
rows = [("HRT · data for predictions and decisions",
         "Every move is predicted before it happens and checked; a bluff check prices the decision; an RL coach is "
         "tested once on sealed tables."),
        ("Make the invisible visible",
         "Hidden cards as a live range, habits in plain words, a hole-card cam no model sees, a Shark whose read of "
         "you is on screen."),
        ("G-Research · real time", "Every model updates move by move as the hand is played.")]
for i, (k, v) in enumerate(rows):
    y = 3.2 + i * 1.3
    s += [card(0.6, y, 6.2, 1.15),
          text(0.85, y + 0.17, 5.7, 0.3, k, size=14, color=AMBER, bold=True),
          text(0.85, y + 0.5, 5.7, 0.6, v, size=13, color=INK2)]
s += [card(7.15, 1.45, 5.6, 3.65, fill=CARD2),
      image("img/table_holecard_cam.png", 7.25, 1.55, 5.4, 3.375, "The Poker Mirror table with the hole-card cam on"),
      text(7.15, 5.25, 5.6, 0.6, "The live table: each player's next move above their seat, their likely hands on the "
           "right, and every card shown to the audience, tagged \"hidden from the models\".", size=12, color=MUTED)]
SLIDES.append({"els": s, "notes":
    "Poker is the purest problem of deciding under hidden information against people. Poker Mirror shows you what is "
    "hidden: their likely cards, their habits, what your bluff is worth, and what a coach trained by reinforcement "
    "learning would do. Every number on screen was measured on data no model saw. For HRT: predictions before every "
    "move, a decision you can act on, and a coach tested once on sealed tables. For Make the invisible visible: ranges, "
    "habits, a hole-card cam the models never see. For G-Research: it all updates live, move by move."})

# 2 · motivation and the data analysis -------------------------------------------------------------------------------
s = title("The data told us where the edge is",
          "Our first analysis: heads-up river decisions in 21.6 million real hands")
stats = [("0 of 4,156", "hands where the defender's cards were shown that were folds: the data hides every folded hand"),
         ("10×", "how much more the situation (bet size, board, betting) predicts than the person's own history"),
         ("+8.0 vs −2.8", "chips per spot for a bot tuned to one opponent: against a frequent folder vs an average player")]
for i, (big, lab) in enumerate(stats):
    s += stat(0.6 + i * 4.1, 1.85, 3.85, 2.35, big, lab)
s.append(text(0.6, 4.5, 12.1, 0.35, "So we built it in four steps", size=16, color=INK, bold=True))
steps = [("1", "Model the situation", "people in general"), ("2", "Tilt it per person", "a Bayesian update"),
         ("3", "Show what is hidden", "ranges, habits, leaks"), ("4", "Test on unseen data", "held-out and sealed")]
for i, (n, a, b) in enumerate(steps):
    x = 0.6 + i * 3.1
    s += [card(x, 5.0, 2.7, 1.2, fill=CARD2),
          text(x + 0.2, 5.13, 0.45, 0.55, n, size=28, font=HEAD, color=AMBER),
          text(x + 0.7, 5.2, 1.9, 0.4, a, size=13, color=INK, bold=True),
          text(x + 0.7, 5.62, 1.9, 0.45, b, size=13, color=INK2)]
    if i < 3:
        s.append(arrow(x + 2.76, 5.49))
s.append(text(0.6, 6.5, 12.1, 0.55, "The person still adds a small, real edge (+0.0028 nats, 95% CI [+0.0002, +0.0060]); "
              "player types add nothing once you know the person. The edge is largest against players with a leak, "
              "so we built a mirror for leaks: yours and your opponents'.", size=12, color=MUTED))
SLIDES.append({"els": s, "notes":
    "Our first finding was humbling. In heads-up river decisions, the situation explains about ten times more than "
    "the person. Folds never reveal cards, so the data hides exactly the hands that fold, which is the first invisible "
    "thing. But the person adds a small, real edge, and a bot tuned to one opponent wins a lot against a distinctive "
    "player and loses a little against an average one. So: model the situation first, tilt it per person, show what is "
    "hidden, and test everything on unseen data."})

# 3 · method and sources ---------------------------------------------------------------------------------------------
s = title("Real data, strict splits, four models", "Each model answers one question, and each is tested on players and hands it never saw")
s += [card(0.6, 1.85, 3.7, 4.35),
      text(0.9, 2.05, 3.2, 0.3, "Data", size=14, color=AMBER, bold=True)]
for i, (big, lab) in enumerate([("21.6M", "real online hands from five sites, 5–6 players, no-limit"),
                                ("10,000", "hands Pluribus played against professionals, every card known"),
                                ("31", "real players kept out of everything: 30 to test, 1 for the replay")]):
    y = 2.45 + i * 1.22
    s += [text(0.9, y, 3.2, 0.5, big, size=26, font=HEAD, color=INK),
          text(0.9, y + 0.5, 3.2, 0.65, lab, size=12, color=INK2)]
models = [("Next move", "What will this player do next? Learned from every decision: \"people in general\".", P1),
          ("Hand model", "What are they holding? 169 hand types from public actions; tested on Pluribus hands.", P2),
          ("Personal model", "What does this person do differently? A Bayesian update from their own moves, live.", P3),
          ("RL coach", "What should you do? PPO over millions of simulated hands against human-like players.", P4)]
for i, (k, v, c) in enumerate(models):
    x, y = 4.6 + (i % 2) * 4.1, 1.85 + (i // 2) * 1.55
    s += [card(x, y, 3.85, 1.35),
          text(x + 0.25, y + 0.18, 3.35, 0.3, k, size=15, color=c, bold=True),
          text(x + 0.25, y + 0.55, 3.35, 0.75, v, size=13, color=INK2)]
s += [card(4.6, 5.0, 7.95, 1.2, fill=CARD2),
      text(4.85, 5.13, 7.5, 0.3, "The players at the table", size=14, color=AMBER, bold=True),
      text(4.85, 5.45, 7.5, 0.7, "Pro copies the Pluribus pros; Shark, Gambler, Sticky and Rock are the four classic types, "
           "mined from 10,213 online regulars and fitted until their stats match (Sticky: VPIP 39%, PFR 10% vs real 42%, 11%).",
           size=12, color=INK2),
      text(0.6, 6.5, 12.1, 0.4, "Compute: the University of Edinburgh's Eddie cluster (32-core nodes) and one RTX 5090 PC.",
           size=12, color=MUTED)]
SLIDES.append({"els": s, "notes":
    "The data: 21.6 million online hands plus the ten thousand hands Pluribus played against professionals, where every "
    "card is known, so they are the fair test for guessing hands. We split by player and keep 31 real players out of "
    "everything. Four models, each answering one question. The bots at the table are the four classic player types, "
    "mined from ten thousand real regulars and fitted until their statistics match."})

# 4 · technical detail and the training / evaluation flow -------------------------------------------------------------
s = title("How it learns, and how we tested it", "Standard methods, combined so each one feeds the next, and tested once on sealed tables")
flow = ["Data split by player and file", "Supervised models + calibration checks", "Imitation start: copies Pluribus (88%)",
        "PPO vs human-like bots, rule bots, itself", "Held-out tables every 50 updates", "Sealed test: registered first, run once"]
bw = 1.78
for i, f in enumerate(flow):
    x = 0.6 + i * (bw + 0.284)
    s += [card(x, 1.8, bw, 1.05, fill=CARD2, line=(AMBER if i == 5 else None)),
          text(x + 0.15, 1.88, bw - 0.3, 0.9, f, size=12, color=INK, align="c", valign="m")]
    if i < 5:
        s.append(arrow(x + bw + 0.02, 2.215, w=0.24, h=0.2))
cols = [("RL coach · PPO",
         ["193 inputs, including each opponent's predicted range", "512-512-256 network, 7 actions, reward = big blinds won",
          "Clip 0.2, GAE λ 0.95; 30% human-like opponents", "Why PPO: stable on noisy rewards, learns a mixed strategy on CPUs "
          "(Pluribus needed ~12,400 core-hours)"]),
        ("Personal model · Bayes",
         ["Starts as people in general", "Each situation (street × facing a bet) is pulled toward the player's own fold / "
          "call / raise counts: a Dirichlet update with a prior worth 8 moves", "Public actions only, never hidden cards",
          "Method of Bayes' Bluff (Southey et al., 2005)"]),
        ("Bluff check and the Shark",
         ["Copy the hand, add your bet, ask each player's own model how likely they fold, multiply",
          "A 2-number correction, fitted on one session, scored on another",
          "The Shark asks the same about you and shifts its raising (×2.5, capped): one level, so no loops"])]
for i, (k, items) in enumerate(cols):
    x = 0.6 + i * 4.1
    s += [card(x, 3.15, 3.85, 3.55),
          text(x + 0.25, 3.32, 3.35, 0.3, k, size=15, color=AMBER, bold=True),
          text(x + 0.25, 3.75, 3.35, 2.85, [[("• ", {"color": AMBER}), it] for it in items], size=12, color=INK2, space=7)]
SLIDES.append({"els": s, "notes":
    "The training and evaluation flow, left to right: split the data by player, train the supervised models and check "
    "their calibration, start the coach as a copy of Pluribus, improve it with PPO against human-like players, rule bots "
    "and earlier copies of itself, keep each run's best version on held-out tables, and finally run a sealed test that "
    "was written down before any result and run once. The coach reads the hand model's predictions; the bluff check "
    "reads the personal models; the Shark reads them too, one level only, so there are no loops."})

# 5 · findings and generalisation -------------------------------------------------------------------------------------
s = title("What we found, and where it transfers",
          "Everything on data or opponents no model was trained on; the coach on sealed tables, 30,000 hands each")
s += [card(0.6, 1.85, 6.1, 3.85),
      {"k": "chart", "x": 0.75, "y": 1.95, "w": 5.8, "h": 3.15,
       "title": "Sealed test: big blinds won per 100 hands vs a maths bot",
       "cats": ["Copy of Pluribus, no RL", "Leash 0.1", "Best from scratch", "Shipped coach", "Same method, 2nd seed"],
       "vals": [2, 79, 115, 156, 202], "hi": 3},
      text(0.85, 5.15, 5.7, 0.45, "Shipped coach +156 ± 15; RL adds +154 ± 14 over the copy; our hand predictions add +82 ± 14.",
           size=11, color=MUTED)]
for i, (big, lab) in enumerate([("75% → 76%", "next move: says 75%, happens 76% (670,615 decisions)"),
                                ("30% vs 28%", "bluff check: everyone folds, said vs happened (871 real bets)"),
                                ("26% → 52%", "the Shark's bets into a leaky player by hand 50 (31% when not reading)")]):
    y = 1.85 + i * 1.32
    s += [card(7.0, y, 5.75, 1.18),
          text(7.25, y + 0.2, 2.6, 0.75, big, size=26, font=HEAD, color=AMBER, valign="m"),
          text(9.95, y + 0.17, 2.6, 0.85, lab, size=12, color=INK2, valign="m")]
s.append(text(0.6, 5.95, 12.1, 0.3, "The recipe transfers wherever decisions depend on reading people under hidden information",
              size=14, color=INK, bold=True))
for i, t_ in enumerate(["Trading and market making", "Negotiation and auctions", "Security and fraud", "Training tools for people"]):
    x = 0.6 + i * 3.1
    s += [card(x, 6.38, 2.85, 0.55, fill=CARD2), text(x, 6.38, 2.85, 0.55, t_, size=13, color=INK, align="c", valign="m")]
SLIDES.append({"els": s, "notes":
    "Findings. The next-move model is calibrated: when it says 75%, it happens 76% of the time. The bluff check said "
    "everyone folds 30% of the time and they did 28%. The coach wins 156 big blinds per 100 hands against a maths bot on "
    "sealed tables, plus or minus 15; reinforcement learning adds 154 over the same network copying Pluribus, and our hand "
    "predictions are worth 82 of that. What surprised us: left alone, RL over-fits its training opponents after about "
    "100 updates; a leash toward Pluribus stops the drift but wins less; opponent-style inputs added nothing measurable. "
    "The recipe: a population model, a Bayesian per-person tilt, a calibrated decision check, and a policy started from "
    "expert imitation and tested on sealed counterparties. It transfers to market making, negotiation, security and any "
    "tool that trains human decisions. Caveat: the coach is tested against bots built from real players, not live people, "
    "and this is a training and review tool, not for real-money play."})


# ---------------------------------------------------------------------------------------------------------------------
def build_pptx(path):
    from pptx import Presentation
    from pptx.chart.data import CategoryChartData
    from pptx.dml.color import RGBColor
    from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
    from pptx.util import Inches, Pt

    rgb = RGBColor.from_string
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    for sl in SLIDES:
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = rgb(BG)
        for e in sl["els"]:
            x, y, w, h = Inches(e["x"]), Inches(e["y"]), Inches(e["w"]), Inches(e["h"])
            if e["k"] == "rect":
                sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
                sh.adjustments[0] = min(0.12, 0.12 / max(e["h"], 0.1))
                sh.fill.solid()
                sh.fill.fore_color.rgb = rgb(e["fill"])
                if e["line"]:
                    sh.line.color.rgb = rgb(e["line"])
                    sh.line.width = Pt(1.5)
                else:
                    sh.line.fill.background()
                sh.shadow.inherit = False
            elif e["k"] == "arrow":
                sh = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x, y, w, h)
                sh.fill.solid()
                sh.fill.fore_color.rgb = rgb(e["color"])
                sh.line.fill.background()
                sh.shadow.inherit = False
            elif e["k"] == "img":
                pic = slide.shapes.add_picture(os.path.join(HERE, e["path"]), x, y, w, h)
                pic._element.nvPicPr.cNvPr.set("descr", e["alt"])
            elif e["k"] == "text":
                tb = slide.shapes.add_textbox(x, y, w, h)
                tf = tb.text_frame
                tf.word_wrap = True
                tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
                tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE}[e["valign"]]
                for i, para in enumerate(e["paras"]):
                    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                    p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER}[e["align"]]
                    p.space_after = Pt(e["space"])
                    for r_ in (para if isinstance(para, list) else [para]):
                        t_, o = (r_, {}) if isinstance(r_, str) else r_
                        run = p.add_run()
                        run.text = t_
                        f = run.font
                        f.name, f.size, f.bold = o.get("font", e["font"]), Pt(o.get("size", e["size"])), o.get("bold", e["bold"])
                        f.color.rgb = rgb(o.get("color", e["color"]))
            elif e["k"] == "chart":
                cd = CategoryChartData()
                cd.categories = e["cats"]
                cd.add_series("bb/100", e["vals"])
                ch = slide.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, x, y, w, h, cd).chart
                ch.has_legend = False
                ch.has_title = True
                ch.chart_title.text_frame.text = e["title"]
                tp = ch.chart_title.text_frame.paragraphs[0].runs[0].font
                tp.size, tp.bold, tp.name = Pt(13), True, BODY
                tp.color.rgb = rgb(INK)
                ch.font.size, ch.font.name = Pt(12), BODY
                ch.font.color.rgb = rgb(INK2)
                plot = ch.plots[0]
                plot.gap_width = 60
                plot.has_data_labels = True
                dl = plot.data_labels
                dl.position = XL_LABEL_POSITION.OUTSIDE_END
                dl.number_format, dl.number_format_is_linked = '+0;-0;0', False
                dl.font.size, dl.font.bold = Pt(12), True
                dl.font.color.rgb = rgb(INK)
                ser = plot.series[0]
                for i, _ in enumerate(e["vals"]):
                    pt = ser.points[i]
                    pt.format.fill.solid()
                    pt.format.fill.fore_color.rgb = rgb(AMBER if i == e["hi"] else LINE)
                va = ch.value_axis
                va.has_major_gridlines = False
                va.visible = False
                va.maximum_scale = 240
                ca = ch.category_axis
                ca.reverse_order = True  # bar charts draw the first category at the bottom; read top to bottom instead
                ca.format.line.color.rgb = rgb(LINE)
                ca.tick_labels.font.size = Pt(12)
                ca.tick_labels.font.color.rgb = rgb(INK2)
        slide.notes_slide.notes_text_frame.text = sl["notes"]
    prs.core_properties.title = "Poker Mirror"
    prs.save(path)


def build_preview(out_dir):
    """One HTML page per slide at 96 px per inch (1280 x 720), the same coordinates and point sizes as the deck."""
    os.makedirs(out_dir, exist_ok=True)
    px = lambda v: f"{v * 96:.1f}px"  # noqa: E731
    fam = {HEAD: "Georgia, 'DejaVu Serif', serif", BODY: "Arial, 'Liberation Sans', sans-serif"}
    for n, sl in enumerate(SLIDES, 1):
        parts = []
        for e in sl["els"]:
            box = f"position:absolute;left:{px(e['x'])};top:{px(e['y'])};width:{px(e['w'])};height:{px(e['h'])};"
            if e["k"] == "rect":
                bd = f"border:2px solid #{e['line']};box-sizing:border-box;" if e["line"] else ""
                parts.append(f'<div style="{box}background:#{e["fill"]};border-radius:10px;{bd}"></div>')
            elif e["k"] == "arrow":
                parts.append(f'<div style="{box}background:#{e["color"]};clip-path:polygon(0 30%,60% 30%,60% 0,100% 50%,60% 100%,60% 70%,0 70%)"></div>')
            elif e["k"] == "img":
                parts.append(f'<img src="{os.path.join(HERE, e["path"])}" style="{box}">')
            elif e["k"] == "text":
                ps = []
                for para in e["paras"]:
                    runs = []
                    for r_ in (para if isinstance(para, list) else [para]):
                        t_, o = (r_, {}) if isinstance(r_, str) else r_
                        runs.append(f'<span style="color:#{o.get("color", e["color"])};font-weight:{700 if o.get("bold", e["bold"]) else 400}">{html.escape(t_)}</span>')
                    ps.append(f'<p style="margin:0 0 {e["space"]}pt 0">{"".join(runs)}</p>')
                jc = {"t": "flex-start", "m": "center"}[e["valign"]]
                parts.append(f'<div style="{box}display:flex;flex-direction:column;justify-content:{jc};font-family:{fam[e["font"]]};'
                             f'font-size:{e["size"]}pt;line-height:1.2;text-align:{"center" if e["align"] == "c" else "left"};outline:1px dashed rgba(255,0,0,.25)">{"".join(ps)}</div>')
            elif e["k"] == "chart":
                bars = "".join(f'<div style="display:flex;align-items:center;margin:10px 0"><span style="width:45%;text-align:right;padding-right:8px">{html.escape(c)}</span>'
                               f'<span style="height:24px;width:{40 * v / 240:.1f}%;background:#{AMBER if i == e["hi"] else LINE}"></span><b style="padding-left:6px;color:#{INK}">+{v}</b></div>'
                               for i, (c, v) in enumerate(zip(e["cats"], e["vals"])))
                parts.append(f'<div style="{box}font-family:Arial;font-size:12pt;color:#{INK2}"><div style="text-align:center;font-weight:700;color:#{INK};font-size:13pt;margin:6px 0 10px">{html.escape(e["title"])}</div>{bars}</div>')
        page = (f'<!doctype html><html><head><meta charset="utf-8"><style>body{{margin:0}}</style></head><body>'
                f'<div style="position:relative;width:1280px;height:720px;background:#{BG};overflow:hidden">{"".join(parts)}</div></body></html>')
        with open(os.path.join(out_dir, f"slide{n}.html"), "w", encoding="utf-8") as f:
            f.write(page)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "PokerMirror.pptx"))
    ap.add_argument("--preview", default=None)
    a = ap.parse_args()
    build_pptx(a.out)
    if a.preview:
        build_preview(a.preview)
    print(f"{len(SLIDES)} slides -> {a.out}")
