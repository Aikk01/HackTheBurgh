# Training progress (live)

The current state of every training run. Update it at each evaluation or event; the plan is in [README.md](README.md).

Last update: 2026-11-08 03:49.

## Runs

Four runs are on the University of Edinburgh's Eddie cluster since 02:53; a fifth (v3-noranges-s2) runs on the 5090
PC from 03:48. The 5090 PC is also the hub (monitor, wandb mirror, demo server); the 2080 PC is no longer used.

**Reading the evaluations:** single evaluations swing a lot (v3-ranges went −74.5 → +93.9 in 50 updates; its rule-bot
score −293 → +229). Picking the highest of noisy evaluations flatters the winner, which is why the final verdict is
the sealed test (06), with two seeds per setting.

| Run | Eddie job | Started | Ends (auto) | Update | Speed | Status |
|---|---|---|---|---|---|---|
| **v3-ranges** (predicted-hand inputs on) | 61797926, 32 cores, 30 workers | 02:05 (moved 02:53) | ~07:20 | 64 | ~2,230 steps/s (was ~730 on the 5090) | Healthy |
| **v3-noranges** (ablation, inputs off) | 61797927, 32 cores, 30 workers | 02:06 (moved 02:53) | ~07:20 | 106 | ~2,480 steps/s (was ~1,630 on the 2080) | Healthy; best held-out so far |
| **v3-ranges-s2** (second seed, inputs on) | 32 cores, 30 workers | 02:44 | ~07:20 | 38 | ~1,720 steps/s after past copies joined | Healthy |
| **v3-noranges-s2** (second seed, inputs off) | 5090 PC, 14 workers | 03:48 | ~07:20 | 1 | ~3,500 steps/s before past copies join | Healthy; makes the ranges comparison two seeds against two |
| **v2-eddie** (insurance, old opponents) | 8 cores | 02:07 | ~07:25 | 93 | ~780 steps/s | Healthy |

**Evaluations** run every 50 updates; the two v3 runs are compared at equal update numbers.

| Run | Update | Held-out vs equity bot (bb/100) | Pluribus clones | Rule bots | Mixed | In-distribution |
|---|---|---|---|---|---|---|
| v3-noranges | 50 | −32.4 | +46 ± 25 | −164 ± 64 | +21 ± 40 | +847 |
| v3-noranges | 100 | **+28.0** | +24 ± 31 | +13 ± 57 | +47 ± 47 | +749 |
| v3-noranges | 150 | +25.1 | −56 ± 41 | +40 ± 60 | +91 ± 43 | +908 |
| v3-noranges | 200 | **+82.2** | −12 ± 34 | +159 ± 52 | +99 ± 50 | +695 |
| v3-ranges | 50 | −64.5 | +63 ± 26 | −274 ± 69 | +18 ± 44 | +694 |
| v3-ranges | 100 | −74.5 | +62 ± 26 | −293 ± 74 | +8 ± 45 | +944 |
| v3-ranges | 150 | **+93.9** | +51 ± 26 | +229 ± 71 | +2 ± 61 | +645 |
| v3-ranges-s2 | 50 | +57.7 | +10 ± 34 | +154 ± 55 | +8 ± 53 | +1,233 |
| v2-eddie | 100 | −34.1 | +33 ± 25 | −207 ± 71 | +73 ± 44 | +1,206 |

At update 50, zeroing v3-ranges' range inputs costs 150 bb/100. That measures how much the policy leans on them,
not their value; the comparison with v3-noranges at equal updates answers that.

## Events so far

| Time | Run | Event | What we did |
|---|---|---|---|
| 02:05 | v3-noranges | First 30 s reported to the borrowed PC's own wandb account | Restarted with wandb offline; logs are mirrored to the 5090 and sent from there |
| 02:16–02:17 | v3-noranges | ALERT entropy dip at updates 27–28 | Expected: past copies joined at update 25. KL 0.02–0.03 and clip fraction 0.19–0.25, both inside the limits. No action |
| 02:23 | v3-ranges | ALERT entropy dip at update 26 | Same transition; no action |
| 02:32 | v3-noranges | First evaluation (update 50): held-out −32 bb/100 vs the equity bot: +46 ± 25 against Pluribus clones, −164 ± 64 against the rule bots | Exported as best; the rule bots are the weak spot |
| 02:34 | v3-ranges | ALERT speed 621 steps/s | Our 30-player evaluation was sharing the 5090; moved it to Eddie |
| 02:44 | — | Added v3-ranges-s2 on Eddie (32 cores) | Second seed, faster than the 5090 |
| 02:52–02:53 | v3-ranges, v3-noranges | Moved to Eddie (32 cores each) for speed | Graceful stop (saves `latest.pt`), copied, resumed with `--first-resume`; the 2080 is freed |
| 02:57 | v3-ranges | First evaluation (update 50): held-out −64.5 | Exported as best. Better than v3-noranges against Pluribus clones, worse against rule bots |
| 03:01 | v3-noranges | Evaluation at update 100: held-out **+28.0**, rule bots now +13 | New best; the rule-bot weakness is closing |

## Where things are

- **Live curves:** wandb project `poker-mirror-rl`, runs `v3-ranges`, `v3-noranges`, `v3-ranges-s2`, `v2-eddie`.
  The Eddie runs log offline; the 5090 mirrors their folders every 5 minutes and sends the metrics.
- **Per run:**
  - `data/rl/runs/<run>/supervise.log`: events;
  - `eval.json`: evaluations;
  - `export_heldout/`: the best held-out model so far;
  - `best_heldout_history/`: every best;
  - `keep/`: every 10-minute checkpoint.
- **Remote runs** are mirrored to `data/rl/remote/<run>/` every 5 minutes.
- **Stopping a run:** `touch <run dir>/STOP` on Eddie. The run saves and exits; `qdel` only as a last resort.

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
