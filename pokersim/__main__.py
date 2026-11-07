"""Command line entry point: ``python -m pokersim <command>``."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

from . import dataset, phh
from .agents import available_agents, make_agent
from .evaluate import shadow_evaluate
from .replay import replay
from .simulate import run_match


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import serve
    serve(args.host, args.port, open_browser=args.open)
    return 0


def _load(path: str, limit: int | None) -> List[phh.HandRecord]:
    if os.path.exists(path):
        return phh.load_file(path, limit=limit)
    return dataset.fetch_hands(path, limit=limit)


def cmd_replay(args: argparse.Namespace) -> int:
    recs = _load(args.file, None)
    rec = recs[args.index]
    res = replay(rec)
    print(f"{rec.label}: {rec.variant}, {rec.n} players, blinds {rec.blinds}")
    for s in res.steps:
        st = s.state
        pot = st.get("total_pot")
        print(f"{st.get('street', ''):8s} pot {pot!s:>10}  {s.label}")
    print("finishing stacks:", [p.stack for p in res.hand.players])
    if res.stacks_match is not None:
        print("matches record:", res.stacks_match)
    for w in res.warnings:
        print("warning:", w)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    recs: List[phh.HandRecord] = []
    for f in args.files:
        recs += _load(f, args.limit)
    agent = make_agent(args.agent, seed=args.seed)

    def progress(done: int, total: int, report) -> None:
        if done % 25 == 0 or done == total:
            n = len(report.rows)
            acc = sum(r.agree for r in report.rows) / n if n else 0
            print(f"\r{done}/{total} hands, {n} decisions, accuracy {acc:.3f}", end="", file=sys.stderr)

    report = shadow_evaluate(recs, agent, max_decisions=args.max_decisions, progress=progress,
                             only_player=args.player)
    print(file=sys.stderr)
    summary = report.summary()
    if args.json:
        print(json.dumps(report.to_dict(include_rows=args.rows), indent=1))
        return 0
    print(f"agent: {summary['agent']}")
    print(f"hands: {summary['hands']} (evaluated {summary['hands_evaluated']}), decisions: {summary['decisions']}, "
          f"skipped (unknown cards): {summary['skipped_unknown_cards']}")
    if summary["decisions"]:
        print(f"accuracy: {summary['accuracy']:.3f}   coarse (fold/passive/aggressive): {summary['accuracy_coarse']:.3f}")
        print("by street: " + ", ".join(f"{k} {v['accuracy']:.3f} (n={v['n']})" for k, v in summary["by_street"].items()))
        print("by actual action: " + ", ".join(f"{k} {v['accuracy']:.3f} (n={v['n']})" for k, v in summary["by_action"].items()))
        if summary["size_error_mean_abs_log_ratio"] is not None:
            print(f"bet sizing: mean |log(pred/actual)| = {summary['size_error_mean_abs_log_ratio']:.3f} "
                  f"over {summary['sized_comparisons']} sized comparisons")
        print("confusion (rows = actual, cols = predicted):")
        acts = ["fold", "check", "call", "bet", "raise"]
        print("        " + "".join(f"{a:>7s}" for a in acts))
        for a in acts:
            print(f"{a:>7s} " + "".join(f"{summary['confusion'][a][b]:7d}" for b in acts))
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    specs = [s.strip() for s in args.agents.split(",") if s.strip()]
    res = run_match(specs, hands=args.hands, stack=args.stack, small_blind=args.sb, big_blind=args.bb,
                    ante=args.ante, seed=args.seed, reset_stacks=not args.carry_stacks,
                    progress=lambda d, t: print(f"\r{d}/{t}", end="", file=sys.stderr))
    print(file=sys.stderr)
    print(f"{res.hands} hands in {res.seconds:.1f}s")
    for s in res.seats:
        print(f"  {s['name']:<18s} {s['agent']['kind']:<8s} net {s['net']:>10.2f}  "
              f"{s['bb_per_100']:>7.1f} bb/100 (+-{s['bb_per_100_stderr']:.1f})")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(phh.dump_hands(res.records))
        print(f"wrote {len(res.records)} hands to {args.out}")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    if args.list is not None:
        for item in dataset.list_dir(args.list):
            size = f"{item['size']:>12}" if item["size"] else " " * 12
            print(f"{item['type']:<9s} {size}  {item['path']}")
        return 0
    for p in args.paths:
        local = dataset.download(p)
        print(local)
    return 0


def cmd_agents(_args: argparse.Namespace) -> int:
    for a in available_agents():
        print(f"{a['id']:<8s} {a['label']:<16s} {a['description']}")
    print("\nSpecs: 'equity:aggression=0.8,tightness=0.3'  'http:http://127.0.0.1:5001/act'  'claude:claude-opus-5-5,effort=low'")
    return 0


def main(argv: List[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # suit symbols on Windows consoles
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(prog="pokersim", description="Interactive, data-driven poker simulator")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the web UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--open", action="store_true", help="open a browser")
    s.set_defaults(fn=cmd_serve)

    r = sub.add_parser("replay", help="print a hand step by step")
    r.add_argument("file", help="local .phh/.phhs file or a path inside the Hugging Face dataset")
    r.add_argument("--index", type=int, default=0)
    r.set_defaults(fn=cmd_replay)

    e = sub.add_parser("evaluate", help="shadow-evaluate a model against recorded decisions")
    e.add_argument("files", nargs="+")
    e.add_argument("--agent", default="equity")
    e.add_argument("--limit", type=int, default=None, help="hands per file")
    e.add_argument("--max-decisions", type=int, default=None)
    e.add_argument("--player", default=None, help="only evaluate decisions of this player name")
    e.add_argument("--seed", type=int, default=1)
    e.add_argument("--json", action="store_true")
    e.add_argument("--rows", action="store_true", help="include per-decision rows in --json output")
    e.set_defaults(fn=cmd_evaluate)

    m = sub.add_parser("simulate", help="self-play between agents")
    m.add_argument("--agents", default="equity,tag,lag,random", help="comma separated agent specs")
    m.add_argument("--hands", type=int, default=200)
    m.add_argument("--stack", type=float, default=200)
    m.add_argument("--sb", type=float, default=1)
    m.add_argument("--bb", type=float, default=2)
    m.add_argument("--ante", type=float, default=0)
    m.add_argument("--seed", type=int, default=None)
    m.add_argument("--carry-stacks", action="store_true", help="do not reset stacks after each hand")
    m.add_argument("--out", default=None, help="write the hands as a .phhs file")
    m.set_defaults(fn=cmd_simulate)

    f = sub.add_parser("fetch", help="download dataset files (or --list a folder)")
    f.add_argument("paths", nargs="*")
    f.add_argument("--list", default=None, metavar="FOLDER")
    f.set_defaults(fn=cmd_fetch)

    a = sub.add_parser("agents", help="list built-in agents")
    a.set_defaults(fn=cmd_agents)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
