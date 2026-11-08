#!/bin/bash
# Eddie (Grid Engine) processor job: RL under the supervisor, PyTorch on the CPU.
# Submit from the repo folder on scratch:  qsub -w v eddie/rl_cpu.sh && qsub eddie/rl_cpu.sh
# Arguments after the script name are passed to scripts/rl_supervise.py (see train/07_action_plan.md).
#$ -N rl_v2_eddie
#$ -cwd
#$ -pe sharedmem 8
#$ -l h_vmem=4G
#$ -l h_rt=06:00:00
#$ -o logs/
#$ -j y
set -euo pipefail
export WANDB_MODE=offline OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "host $(hostname) · $(nproc) cores · $(date)"
.venv/bin/python -u scripts/rl_supervise.py "$@"
