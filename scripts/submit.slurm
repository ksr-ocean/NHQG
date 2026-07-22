#!/bin/bash
#SBATCH --job-name=nhqge
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=nhqge_%j.out
#SBATCH --error=nhqge_%j.err

# ── Bridges-2 GPU job script for NHQGE solver ──
# For production on H100: change to --gres=gpu:h100-80:1

module purge
module load cuda/12.4

# Activate conda environment
source activate base

# Enable float64 for JAX
export JAX_ENABLE_X64=1

# Pre-allocate 90% of GPU memory
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

cd $SLURM_SUBMIT_DIR

echo "=== NHQGE Solver ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo ""

# --- Choose run mode ---
# Rubio reproduction:
# python scripts/run_rubio.py 256 32

# Parametric sweep:
# python scripts/run_sweep.py 256

# Benchmark:
python scripts/benchmark.py

echo ""
echo "Completed: $(date)"
