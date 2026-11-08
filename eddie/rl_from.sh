#!/bin/bash
# Eddie job: PPO under the supervisor, starting from a given checkpoint (e.g. an imitation start), from this checkout.
#   qsub -N <name> -pe sharedmem 32 -l h_vmem=3G -l h_rt=02:45:00 eddie/rl_from.sh <run dir> <end HH:MM> <start .pt> \
#        [supervisor flags] -- [trainer flags]
#$ -cwd
#$ -o logs/
#$ -j y
set -euo pipefail
export WANDB_MODE=offline OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN=$1; END=$2; INIT=$3; shift 3
echo "host $(hostname) · $(nproc) cores · $(date)"
mkdir -p "$RUN"
HOURS=$(.venv/bin/python -c "
import datetime as d; n = d.datetime.now(); h, m = map(int, '$END'.split(':')); print(round((n.replace(hour=h, minute=m, second=0) - n).total_seconds() / 3600, 2))")
.venv/bin/python -u scripts/rl_supervise.py --run-dir "$RUN" --hours "$HOURS" --name "$(basename "$RUN")" --first-resume "$INIT" "$@"
