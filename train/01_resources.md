# 01 · Machines

| Machine | Hardware | Runs |
|---|---|---|
| **5090 PC** | RTX 5090 32 GB, 16 threads, 62 GB RAM | Data extraction and base-model training (done); RL run **v3-ranges**; the monitoring hub; building the screen |
| **2080 PC** (a teammate's, borrowed) | RTX 2080 SUPER 8 GB, 16 threads, 62 GB RAM | RL run **v3-noranges** |
| **Eddie** (university Grid Engine cluster) | Processor queue with thousands of free slots | RL run **v2-eddie** (insurance), 8 cores, `--cpu`, lowest priority |
| **Mac M3 Pro** | Laptop | Runs the demo (NumPy only); no training |

## Why processor cores matter more than GPUs

PPO self-play spends almost all its time playing hands in Python worker processes. The network update takes a few
seconds on any GPU, or on 16 processor cores. Each free 16-thread machine therefore adds about one run's worth of
throughput.

Measured on the 5090 PC with 14 workers:

| Run | Speed | Note |
|---|---|---|
| v1 (146 inputs) | 3,400 → 1,800 steps/s | The drop came when past copies joined as opponents |
| v2 (193 inputs, hand inputs on) | 1,700 → 780 steps/s | The hand-range simulation costs about half the speed |

## Setup (identical everywhere): uv + pyproject.toml

```bash
git clone -b train-v3 https://github.com/Aikk01/HackTheBurgh.git && cd HackTheBurgh
uv sync --extra train --extra prep      # Python 3.12 (.python-version); torch from the CUDA 12.8 index on Linux
```

- On the borrowed PC, uv, its cache and its Python live inside one project folder: `UV_INSTALL_DIR`, `UV_CACHE_DIR`
  and `UV_PYTHON_INSTALL_DIR` all point into it. Deleting that folder removes everything. Nothing outside it is
  touched.
- On Eddie the venv lives on scratch. Compute nodes may have no internet, so wandb runs offline there and is synced
  from the login node.
- `rl-train` forces one BLAS thread per worker. Without that, each worker starts a thread per core and the run halves
  in speed.
