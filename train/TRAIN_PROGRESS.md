# Training progress (live)

The current state of every training run. Update it at each evaluation or event; the plan is in [README.md](README.md).

Last update: 2026-11-08 02:55.

## Runs

| Run | Machine | Started | Ends (auto) | Update | Speed | Status |
|---|---|---|---|---|---|---|
| **v3-ranges** (predicted-hand inputs on) | 5090 PC, 14 workers | 02:05 | ~07:25 | 26 | ~780 steps/s from update 26 (1,640 before) | Healthy; past copies joined at update 25 (expected value-loss bump, entropy dip) |
| **v3-noranges** (ablation, inputs off) | 2080 PC, 14 workers | 02:06 | ~07:25 | 30 | 1,630 steps/s from update 26 (3,300 before) | Healthy; same transition at update 26 |
| **v2-eddie** (insurance, old opponents) | Eddie, 8 cores, `--cpu` | 02:07 | ~07:25 | 63 (resumed from 56) | ~830 steps/s | Healthy |
| **v3-ranges-s2** (second seed, inputs on) | Eddie, 32 cores, `--cpu`, 30 workers | 02:44 | ~07:20 | 11 | ~2,630 steps/s before past copies join | Healthy; the fastest run |

**Projected updates by the end:**
- v3-ranges: about 250 (≈80 s per update; the hand simulation runs for the hero and for every past copy).
- v3-noranges: about 480.
- v2-eddie: about 300.

The two v3 runs are compared at equal update numbers, so the slower one sets the comparison range.

**Evaluations:** every 50 updates. The first comes at v3-ranges update 50, around 03:20; v3-noranges gets there
earlier.

## Events so far

| Time | Run | Event | What we did |
|---|---|---|---|
| 02:05 | v3-noranges | First 30 s reported to the borrowed PC's own wandb account | Restarted with wandb offline; logs are mirrored to the 5090 and sent from there |
| 02:16–02:17 | v3-noranges | ALERT entropy dip at updates 27–28 | Expected: past copies joined at update 25. KL 0.02–0.03 and clip fraction 0.19–0.25, both inside the limits. No action |
| 02:23 | v3-ranges | ALERT entropy dip at update 26 | Same transition; no action |
| 02:32 | v3-noranges | First evaluation (update 50): held-out −32 bb/100 vs the equity bot: +46 ± 25 against Pluribus clones, −164 ± 64 against the rule bots | Exported as best; the rule bots are the weak spot |
| 02:34 | v3-ranges | ALERT speed 621 steps/s | Our 30-player evaluation was sharing the 5090; moved it to Eddie |
| 02:44 | — | Added v3-ranges-s2 on Eddie (32 cores) | Second seed, faster than the 5090 |

## Where things are

- **Live curves:** wandb project `poker-mirror-rl`, runs `v3-ranges`, `v3-noranges`, `v2-eddie`.
- **Per run:**
  - `data/rl/runs/<run>/supervise.log`: events;
  - `eval.json`: evaluations;
  - `export_heldout/`: the best held-out model so far;
  - `best_heldout_history/`: every best;
  - `keep/`: every 10-minute checkpoint.
- **Remote runs** are mirrored to `data/rl/remote/<run>/` every 5 minutes.
- **Stopping a run:** `touch data/rl/runs/<run>/STOP` on its machine. It saves and exits; for Eddie, use `qdel`.

## Base models (finished)

| Model | Result | File |
|---|---|---|
| Hand model | +0.31 nats over card counting on the unbiased Pluribus test | `data/rl/hand_range.npz` |
| Next-move model | +0.12 nats over frequency baselines (online, public); +0.40 (with cards, top-1 81%) | `data/rl/next_move.npz` |
| Fallback RL policy | v2 update 50: beat the equity bot on all 5 old test tables | `data/rl/v2_u50/` |

## Reserved demo player

- **Who:** PartyPoker player `KsdS3CqVKoZ9FsoxzScmwg`, 666 hands at 5–6-player tables. No hand of theirs reached any
  model's training or evaluation data. Their hands are in `data/demo/reserved_hands.phhs`, and both extraction
  scripts now skip them.
- **Why not the first candidate:** it was an iPoker player with 969 hands, but iPoker records stacks as infinite, so
  those hands can't be replayed.
