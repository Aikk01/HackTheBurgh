#!/bin/bash
# Eddie job: RL that starts from imitation of the Base model (scripts/bc_init.py), then PPO under the supervisor.
# Runs from its own checkout so the running jobs' code is untouched; the run folder is given as an absolute path.
#   qsub -N rl_v3_bc -pe sharedmem 32 -l h_vmem=3G -l h_rt=03:40:00 eddie/rl_bc.sh <run dir> <end HH:MM>
#$ -cwd
#$ -o logs/
#$ -j y
set -euo pipefail
export WANDB_MODE=offline OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN=$1; END=$2
echo "host $(hostname) · $(nproc) cores · $(date)"
mkdir -p "$RUN"
[ -f "$RUN/bc_init.pt" ] || .venv/bin/python -u scripts/bc_init.py --out "$RUN/bc_init.pt" --workers 30 --decisions 300000
HOURS=$(.venv/bin/python -c "
import datetime as d; n = d.datetime.now(); h, m = map(int, '$END'.split(':')); print(round((n.replace(hour=h, minute=m, second=0) - n).total_seconds() / 3600, 2))")
.venv/bin/python -u scripts/rl_supervise.py --run-dir "$RUN" --hours "$HOURS" --name v3-bc --first-resume "$RUN/bc_init.pt" \
  -- --mix v3 --seed 3 --cpu --workers 30
