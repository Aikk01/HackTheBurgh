#!/bin/bash
# Eddie processor job for analysis scripts: qsub eddie/cpu_job.sh scripts/<script>.py [args]
#$ -N poker_cpu
#$ -cwd
#$ -pe sharedmem 8
#$ -l h_vmem=4G
#$ -l h_rt=02:00:00
#$ -o logs/
#$ -j y
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "host $(hostname) · $(nproc) cores · $(date)"
.venv/bin/python -u "$@"
