# 07 · Action plan (6 hours of training from launch, T)

| When | 5090 PC | 2080 PC | Eddie | Screen / personal model |
|---|---|---|---|---|
| Done | Corpus extraction; hand model; next-move model; RL code with held-out tables and paired evaluation; supervisor | uv env | uv env | — |
| T | Start **v3-ranges** under the supervisor | Start **v3-noranges** under the supervisor | Submit **v2-eddie** | Decision log, personal model, `/advice` additions |
| T to T+5:30 | All runs watched from the 5090 hub; resets automatic (08); every evaluation reported | — | Pulled every 15 minutes | Screen; rehearsal hands |
| T+5:30 | Stop (`STOP` file). Final evaluation on fresh deals. Copy the best held-out export to `data/rl/policy.npz`. Commit and push | Stop; copy run folder back | `qdel`; copy back | Pull; rehearsal with models frozen |

## Commands

```bash
# 5090 PC
uv run python scripts/rl_supervise.py --run-dir data/rl/runs/v3-ranges --hours 5.5 -- --mix v3 --seed 1 --workers 14

# 2080 PC (inside the project folder)
uv run python scripts/rl_supervise.py --run-dir data/rl/runs/v3-noranges --hours 5.5 -- --mix v3 --seed 1 --workers 14 --no-range-inputs

# Eddie (eddie/rl_cpu.sh: -pe sharedmem 8, -l h_vmem=4G, -l h_rt=06:00:00; WANDB_MODE=offline)
qsub -w v eddie/rl_cpu.sh && qsub eddie/rl_cpu.sh
```

## Go/no-go

- **T+0:30:** both v3 runs pass their first updates with no alerts beyond the expected value-loss bump when past
  copies join (update 26). Otherwise fix and restart; at this point that costs little.
- **Each evaluation:** the held-out score is tracked. Two evaluations of falling held-out with rising
  in-distribution means overfitting: keep the best export and consider stopping that run.
- **T+5:30:** ship the best held-out export across runs; v2 update 50 if nothing beats it on the held-out tables.

## Committed

**Yes:**
- the code;
- `train/`;
- the model files and their json;
- per run: `eval.json`, `final_eval.json`, `log.csv`, `supervise.log`, `export_heldout/policy.*`.

**No:**
- `.pt` checkpoints;
- `keep/`;
- `data/cache/`;
- `data/players/`;
- `logs/`;
- `wandb/`.
