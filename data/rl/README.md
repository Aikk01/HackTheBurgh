# Model files

The app loads everything from this folder.

| Path | What | Made by |
|---|---|---|
| `hand_range.npz/.json` | Hand model: each opponent's likely starting hand (169 types) | `scripts/train_hand_range.py` |
| `next_move.npz/.json` | Next-move model: the Base advice, "People", the personal models' prior, the Pro/Machine bots | `scripts/train_next_move.py` |
| `policy.npz/.json` | **The live RL coach**: a copy of one folder in `coaches/` (its json names the run and update) | copied by hand |
| `coaches/<run>-upd<N>/` | Every RL coach version we have put live or kept, with its json | copied from a run's `export_heldout/` |
| `runs/` | Local training runs (checkpoints are not committed) | `scripts/rl_supervise.py` |
| `remote/` | Mirrors of the Eddie runs, without checkpoints | `logs/mirror_remote.sh` (local) |

To change the live coach: copy `coaches/<name>/policy.*` here, run `python scripts/build_replay.py` (the replay's coach
advice comes from `policy.npz`), restart the server.

`coaches/v1-upd50-old-inputs/` uses an older input format and does not load with the current code; it is kept for the
record only.
