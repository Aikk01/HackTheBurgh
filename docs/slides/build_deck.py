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

# white slides, large type: dark green text, a dark amber accent that stays readable on white
BG, CARD, CARD2, LINE = "FFFFFF", "F1F6F3", "E4EEE9", "C5D6CF"
INK, INK2, MUTED, AMBER, GOOD = "0E302B", "2E3B37", "5E6E68", "B86A00", "1F7A5E"
P1, P2, P3, P4 = "2F6FC4", "C24E1C", "177E5A", "6F63C8"
HEAD, BODY = "Georgia", "Arial"


def card(x, y, w, h, fill=CARD, line=None):
    return {"k": "rect", "x": x, "y": y, "w": w, "h": h, "fill": fill, "line": line}


def text(x, y, w, h, paras, size=16, color=INK2, font=BODY, bold=False, align="l", valign="t", space=6):
    """paras: a string, or a list of paragraphs; a paragraph is a string or a list of runs (str or (str, opts))."""
    if isinstance(paras, str):
        paras = [paras]
    return {"k": "text", "x": x, "y": y, "w": w, "h": h, "paras": paras, "size": size, "color": color, "font": font,
            "bold": bold, "align": align, "valign": valign, "space": space}


def image(path, x, y, w, h, alt):
    return {"k": "img", "path": path, "x": x, "y": y, "w": w, "h": h, "alt": alt}


def arrow(x, y, w=0.3, h=0.26, color=AMBER):
    return {"k": "arrow", "x": x, "y": y, "w": w, "h": h, "color": color}


def title(t, sub=None):
    els = [text(0.6, 0.38, 12.1, 0.8, t, size=40, font=HEAD, color=INK)]
    if sub:
        els.append(text(0.6, 1.2, 12.1, 0.45, sub, size=18, color=MUTED))
    return els


def stat(x, y, w, h, big, label, big_size=44):
    return [card(x, y, w, h),
            text(x + 0.3, y + 0.22, w - 0.6, 0.85, big, size=big_size, font=HEAD, color=AMBER),
            text(x + 0.3, y + 1.15, w - 0.6, h - 1.3, label, size=16, color=INK2)]


SLIDES = []

# 1 · introduction and fit -------------------------------------------------------------------------------------------
s = [text(0.6, 0.5, 6.6, 0.35, "HackTheBurgh 2026 · HRT · G-Research · Make the invisible visible", size=14, color=MUTED),
     text(0.6, 0.9, 6.6, 1.1, "Poker Mirror", size=60, font=HEAD, color=INK),
     text(0.6, 2.05, 6.3, 0.9, "Poker training that makes the hidden visible, measured on data it never saw.",
          size=22, color=INK2)]
rows = [("HRT · predictions and decisions", "Every move predicted and checked; a bluff check prices the decision."),
        ("Make the invisible visible", "Hidden cards as a live range, habits in plain words, a hole-card cam."),
        ("G-Research · real time", "Every model updates as the hand is played.")]
for i, (k, v) in enumerate(rows):
    y = 3.15 + i * 1.37
    s += [card(0.6, y, 6.3, 1.22),
          text(0.85, y + 0.14, 5.8, 0.35, k, size=17, color=GOOD, bold=True),
          text(0.85, y + 0.52, 5.8, 0.65, v, size=16, color=INK2)]
s += [card(7.2, 1.5, 5.55, 3.6, fill=CARD2),
      image("img/table_holecard_cam.png", 7.3, 1.6, 5.35, 3.344, "The Poker Mirror table with the hole-card cam on"),
      text(7.2, 5.25, 5.55, 0.9, "The live table: next moves above each seat, likely hands on the right, every card shown "
           "to the audience and hidden from the models.", size=14, color=MUTED)]
SLIDES.append({"els": s, "notes":
    "Poker is the purest problem of deciding under hidden information against people. Poker Mirror shows you what is "
    "hidden: their likely cards, their habits, what your bluff is worth, and what a coach trained by reinforcement "
    "learning would do. Every number on screen was measured on data no model saw. For HRT: predictions before every "
    "move and checked, a bluff check that prices the decision, and an RL coach tested once on sealed tables. For Make "
    "the invisible visible: ranges, habits, a hole-card cam the models never see, and a Shark whose read of you is on "
    "screen. For G-Research: it all updates live, move by move."})

# 2 · motivation and the data analysis -------------------------------------------------------------------------------
s = title("The data told us where the edge is", "Our first analysis: heads-up river decisions in 21.6 million real hands")
stats = [("0 of 4,156", "shown hands that were folds: the data hides every folded hand"),
         ("10×", "how much more the situation predicts than the person"),
         ("+8.0 / −2.8", "chips per spot exploiting a frequent folder / an average player")]
for i, (big, lab) in enumerate(stats):
    s += stat(0.6 + i * 4.1, 1.85, 3.85, 2.3, big, lab, big_size=38)
s.append(text(0.6, 4.35, 12.1, 0.4, "So we built it in four steps", size=18, color=INK, bold=True))
steps = [("1", "Model the situation", "people in general"), ("2", "Tilt it per person", "a Bayesian update"),
         ("3", "Show what is hidden", "ranges, habits, leaks"), ("4", "Test on unseen data", "held-out and sealed")]
for i, (n, a, b) in enumerate(steps):
    x = 0.6 + i * 3.12
    s += [card(x, 4.85, 2.78, 1.55, fill=CARD2),
          text(x + 0.25, 4.93, 0.6, 0.6, n, size=32, font=HEAD, color=AMBER),
          text(x + 0.25, 5.5, 2.4, 0.4, a, size=17, color=INK, bold=True),
          text(x + 0.25, 5.9, 2.4, 0.4, b, size=15, color=INK2)]
    if i < 3:
        s.append(arrow(x + 2.81, 5.5))
s.append(text(0.6, 6.6, 12.1, 0.6, "The person still adds a small, real edge (+0.0028 nats, 95% CI [+0.0002, +0.0060]), "
              "largest against players with a leak: so we built a mirror for leaks.", size=14, color=MUTED))
SLIDES.append({"els": s, "notes":
    "Our first finding was humbling. In heads-up river decisions from 21.6 million hands, the situation (bet size, board, "
    "betting) explains about ten times more than the person's own history. Folds never reveal cards: of 4,156 hands where "
    "the defender's cards were shown, none were folds, so the data hides exactly the hands that fold. But the person adds "
    "a small, real edge, and a bot tuned to one opponent wins 8 chips per spot against a frequent folder while losing 2.8 "
    "against an average player. Player types add nothing once you know the person. So: model the situation first, tilt "
    "it per person, show what is hidden, and test everything on unseen data."})

# 3 · method and sources ---------------------------------------------------------------------------------------------
s = title("Real data, strict splits, four models", "Each model answers one question and is tested on data it never saw")
s += [card(0.6, 1.85, 3.75, 4.55),
      text(0.9, 2.02, 3.2, 0.35, "Data", size=18, color=GOOD, bold=True)]
for i, (big, lab) in enumerate([("21.6M", "real online hands, 5–6 players"), ("10,000", "Pluribus hands, every card known"),
                                ("31", "real players kept out, to test")]):
    y = 2.5 + i * 1.28
    s += [text(0.9, y, 3.2, 0.6, big, size=32, font=HEAD, color=INK),
          text(0.9, y + 0.62, 3.2, 0.5, lab, size=15, color=INK2)]
models = [("Next move", "What will they do next? Learned from every decision.", P1),
          ("Hand model", "What are they holding? 169 hand types, from public actions.", P2),
          ("Personal model", "What does this person do differently? Bayesian, live.", P3),
          ("RL coach", "What should you do? PPO, millions of simulated hands.", P4)]
for i, (k, v, c) in enumerate(models):
    x, y = 4.6 + (i % 2) * 4.12, 1.85 + (i // 2) * 1.6
    s += [card(x, y, 3.95, 1.42),
          text(x + 0.25, y + 0.15, 3.45, 0.35, k, size=18, color=c, bold=True),
          text(x + 0.25, y + 0.56, 3.45, 0.8, v, size=15, color=INK2)]
s += [card(4.6, 5.05, 8.07, 1.35, fill=CARD2),
      text(4.85, 5.17, 7.6, 0.35, "The players at the table", size=17, color=GOOD, bold=True),
      text(4.85, 5.55, 7.6, 0.8, "Pro copies the Pluribus pros. Shark, Gambler, Sticky and Rock are the four classic types, "
           "mined from 10,213 real regulars.", size=15, color=INK2),
      text(0.6, 6.62, 12.1, 0.4, "Compute: the University of Edinburgh's Eddie cluster and one RTX 5090 PC.", size=14, color=MUTED)]
SLIDES.append({"els": s, "notes":
    "The data: 21.6 million online hands from five sites, plus the ten thousand hands Pluribus played against "
    "professionals, where every card is known, so they are the fair test for guessing hands. We split by player and keep "
    "31 real players out of everything: 30 to test the personal model, one for the replay. Four models, each answering "
    "one question. The bots at the table are the four classic player types, mined from 10,213 real regulars (the most "
    "extreme 15% of each type) and fitted until their VPIP, PFR and AF match: Sticky plays 39% of hands and raises 10%, "
    "against 42% and 11% for the real players."})

# 4 · technical detail and the training / evaluation flow -------------------------------------------------------------
s = title("How it learns, and how we tested it", "Standard methods, combined, then tested once on sealed tables")
flow = ["Split data by player", "Train and calibrate models", "Copy Pluribus first (88%)", "PPO vs human-like bots",
        "Held-out check every 50 updates", "Sealed test, run once"]
bw = 1.8
for i, f in enumerate(flow):
    x = 0.6 + i * (bw + 0.26)
    s += [card(x, 1.85, bw, 1.25, fill=CARD2, line=(AMBER if i == 5 else None)),
          text(x + 0.12, 1.9, bw - 0.24, 1.15, f, size=15, color=INK, align="c", valign="m", bold=(i == 5))]
    if i < 5:
        s.append(arrow(x + bw + 0.01, 2.36, w=0.24, h=0.24))
cols = [("RL coach · PPO", ["193 inputs, incl. each opponent's predicted range", "512-512-256 network, 7 actions",
                            "Stable on noisy rewards; a mixed strategy on CPUs"]),
        ("Personal model · Bayes", ["Starts as people in general", "Pulled toward the player's own counts in each situation",
                                    "Public actions only (Bayes' Bluff, 2005)"]),
        ("Bluff check and Shark", ["Each player's own model gives P(fold); multiply", "Calibrated on one session, tested on another",
                                   "The Shark asks the same about you: one level, no loops"])]
for i, (k, items) in enumerate(cols):
    x = 0.6 + i * 4.1
    s += [card(x, 3.4, 3.88, 3.5),
          text(x + 0.25, 3.56, 3.4, 0.4, k, size=18, color=GOOD, bold=True),
          text(x + 0.25, 4.05, 3.4, 2.75, [[("• ", {"color": AMBER, "bold": True}), it] for it in items], size=16,
               color=INK2, space=10)]
SLIDES.append({"els": s, "notes":
    "Left to right: split the data by player; train the supervised models and check their calibration on held-out "
    "players; start the coach as a copy of the Pluribus model (88% agreement); improve it with PPO against human-like "
    "players, rule bots and earlier copies of itself; keep each run's best version on held-out tables every 50 updates; "
    "and finally a sealed test, written down before any result and run once. PPO details: 193 inputs, a 512-512-256 "
    "network, 7 actions, clip 0.2, GAE lambda 0.95; we chose PPO because Pluribus's own method needed about 12,400 "
    "core-hours, and PPO is stable on noisy rewards and learns a mixed strategy. The personal model starts as people in "
    "general and is pulled toward the player's own fold, call and raise counts in each situation, a Dirichlet update with "
    "a prior worth 8 moves. The bluff check asks each player's own model how likely they fold and multiplies; a "
    "two-number correction is fitted on one session and scored on another. The Shark asks the same question about you "
    "and shifts its raising, one level only, so there are no loops."})

# 5 · findings and generalisation -------------------------------------------------------------------------------------
s = title("What we found, and where it transfers", "On data and opponents no model was trained on")
s += [card(0.6, 1.85, 6.15, 4.0),
      {"k": "chart", "x": 0.75, "y": 1.95, "w": 5.85, "h": 3.15,
       "title": "Sealed test: bb/100 vs a maths bot",
       "cats": ["Copy of Pluribus, no RL", "Leash 0.1", "Best from scratch", "Shipped coach", "2nd seed"],
       "vals": [2, 79, 115, 156, 202], "hi": 3},
      text(0.85, 5.12, 5.7, 0.65, ["Shipped coach +156 ± 15 · RL adds +154 ± 14", "Our hand predictions add +82 ± 14"], size=14,
           color=MUTED, space=2)]
for i, (big, lab) in enumerate([("75% → 76%", "next move: said vs happened"), ("30% vs 28%", "bluff check: said vs happened"),
                                ("26% → 52%", "the Shark's bets into a leaky player")]):
    y = 1.85 + i * 1.37
    s += [card(7.0, y, 5.75, 1.22),
          text(7.25, y + 0.12, 3.0, 1.0, big, size=32, font=HEAD, color=AMBER, valign="m"),
          text(10.3, y + 0.12, 2.3, 1.0, lab, size=15, color=INK2, valign="m")]
s.append(text(0.6, 6.05, 12.1, 0.4, "Transfers to decisions against people you cannot fully see", size=17, color=INK, bold=True))
for i, t_ in enumerate(["Trading", "Negotiation", "Security and fraud", "Training people"]):
    x = 0.6 + i * 3.1
    s += [card(x, 6.5, 2.85, 0.6, fill=CARD2), text(x, 6.5, 2.85, 0.6, t_, size=16, color=INK, align="c", valign="m")]
SLIDES.append({"els": s, "notes":
    "Findings, all on data or opponents no model was trained on. The next-move model is calibrated: when it says 75%, it "
    "happens 76% of the time over 670,615 decisions. The bluff check said everyone folds 30% of the time and they did 28%, "
    "over 871 real bets. The coach wins 156 big blinds per 100 hands against a maths bot on sealed tables, plus or minus "
    "15; reinforcement learning adds 154 over the same network copying Pluribus, and our hand predictions are worth 82. "
    "The Shark doubles its bets into a leaky player within 50 hands, 26% to 52%, against 31% when it does not read. What "
    "surprised us: left alone, RL over-fits its training opponents after about 100 updates; a leash toward Pluribus stops "
    "the drift but wins less; opponent-style inputs added nothing measurable. The recipe is a population model, a "
    "Bayesian per-person tilt, a calibrated decision check and a policy started from expert imitation, tested on sealed "
    "counterparties: it transfers to market making, negotiation, security and tools that train human decisions. Caveat: "
    "the coach is tested against bots built from real players, not live people, and this is a training and review tool, "
    "not for real-money play."})


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
                tp.size, tp.bold, tp.name = Pt(16), True, BODY
                tp.color.rgb = rgb(INK)
                ch.font.size, ch.font.name = Pt(15), BODY
                ch.font.color.rgb = rgb(INK2)
                plot = ch.plots[0]
                plot.gap_width = 60
                plot.has_data_labels = True
                dl = plot.data_labels
                dl.position = XL_LABEL_POSITION.OUTSIDE_END
                dl.number_format, dl.number_format_is_linked = '+0;-0;0', False
                dl.font.size, dl.font.bold = Pt(15), True
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
                ca.tick_labels.font.size = Pt(15)
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
                parts.append(f'<div style="{box}font-family:Arial;font-size:15pt;color:#{INK2}"><div style="text-align:center;font-weight:700;color:#{INK};font-size:16pt;margin:6px 0 10px">{html.escape(e["title"])}</div>{bars}</div>')
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
