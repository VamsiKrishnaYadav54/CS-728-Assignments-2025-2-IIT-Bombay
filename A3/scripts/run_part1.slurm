#!/bin/bash
#SBATCH --job-name=PA3_Part1
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/part1_%j.out
#SBATCH --error=logs/part1_%j.err

# ============================================================================
# Part 1: Classical Retrieval (BM25, MiniLM, UAE)
# Estimated time: ~30-60 minutes
# ============================================================================

# Get the directory where this script is located
SCRIPT_DIR="/home/$(whoami)/CS728_PA3"

# Create log directory
mkdir -p ${SCRIPT_DIR}/logs
mkdir -p ${SCRIPT_DIR}/results

# Change to project directory
cd ${SCRIPT_DIR}

# Load modules (adjust based on your cluster's module system)
# module load python/3.10
# module load cuda/11.8

# Activate conda environment (if using conda)
# source activate pa3_env
# OR use system Python with pip packages

# Set cache directory for HuggingFace (models should be pre-downloaded)
export HF_HOME="${HOME}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_OFFLINE=1  # Prevent download attempts

echo "=============================================="
echo "Running Part 1: Classical Retrieval"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "=============================================="

# Run Part 1
python part1_retrieval.py

echo "=============================================="
echo "Part 1 completed at: $(date)"
echo "=============================================="
