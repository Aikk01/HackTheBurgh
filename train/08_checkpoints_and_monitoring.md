# 08 · Saving models, monitoring, resets

Every run is started by `scripts/rl_supervise.py`. It launches `rl-train`, watches it every 20 s, sends metrics to
wandb (`poker-mirror-rl`; offline on Eddie), and writes one line per event to `<run>/supervise.log`. The 5090 PC
follows all three logs (local and over ssh), so every event reaches one stream.

## What is kept

| What | Where |
|---|---|
| Every 10-minute checkpoint | `<run>/keep/updNNNNN.pt`, for rollback and resume |
| Best held-out model | `<run>/export_heldout/policy.npz` + `best_heldout.pt` |
| Every best held-out model since the start | `<run>/best_heldout_history/policy_updNNNNN.npz` |
| Best in-distribution model | `<run>/export/` + `best.pt` |
| All evaluations | `<run>/eval.json`, `final_eval.json`, `log.csv` |

## Events and automatic actions

| Event | Rule | Action |
|---|---|---|
| EVAL | Every evaluation | Report held-out score per table, in-distribution mean, range-input reliance, new best |
| ALERT | Value loss above 2.5× its recent median and above 30; KL above 0.06; clip fraction above 0.35; entropy down 0.15 within 5 updates; speed below 40% | Report only. The value-loss bump when past copies join (update 26) is expected |
| STALL | No new best held-out in 4 evaluations | Report; decide whether to stop the run |
| RESET | NaN/inf in any metric, or entropy below 0.2 | Kill the trainer; resume from the last kept checkpoint **before** the problem with the learning rate halved; at most 2 per run |
| RESTART | Trainer exits without `final_eval.json` | Resume from `latest.pt` for the remaining time; at most 3 per run |
| DONE / DEAD | Finished, or out of resets or restarts | Report |

- **Manual stop:** `touch <run>/STOP`. The trainer finishes its current update, saves `latest.pt` and exits.
- **SIGTERM** to the trainer does the same.

Never change the hand model, the next-move model or the inputs during a run: a change means a new run.
