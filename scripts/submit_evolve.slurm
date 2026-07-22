#!/bin/bash
#SBATCH --job-name=nhqge-evolve
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=nhqge_evolve_%j.out
#SBATCH --error=nhqge_evolve_%j.err

# ── Bridges-2 H100 job: NHQGE with evolving mean temperature ──

module purge
module load cuda/12.4

source activate base

export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export PYTHONPATH=$SLURM_SUBMIT_DIR

cd $SLURM_SUBMIT_DIR

echo "=== NHQGE evolve_mean ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo ""

# Ra=100, Nx=256, Nz=128, evolve_mean, hyper-4
python scripts/run_evolve.py 256 128 100

echo ""
echo "Completed: $(date)"
