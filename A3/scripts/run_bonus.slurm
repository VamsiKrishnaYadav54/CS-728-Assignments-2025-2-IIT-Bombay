#!/bin/bash
#SBATCH --job-name=PA3_Bonus
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/bonus_%j.out
#SBATCH --error=logs/bonus_%j.err

# ============================================================================
# BONUS: Test different K values (10, 20, 30)
# This runs Part 3 three times with different head counts
# Estimated time: 8-12 hours
# ============================================================================

SCRIPT_DIR="/home/$(whoami)/CS728_PA3"

mkdir -p ${SCRIPT_DIR}/logs
mkdir -p ${SCRIPT_DIR}/results

cd ${SCRIPT_DIR}

# Load modules
# module load python/3.10
# module load cuda/11.8

export HF_HOME="${HOME}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=============================================="
echo "Running BONUS: K Comparison (10, 20, 30)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "=============================================="

nvidia-smi

# Run bonus experiments
python run_bonus.py

echo "=============================================="
echo "Bonus experiments completed at: $(date)"
echo "=============================================="
