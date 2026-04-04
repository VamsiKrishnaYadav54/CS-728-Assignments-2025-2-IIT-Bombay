#!/bin/bash
#SBATCH --job-name=PA3_Part3
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/part3_%j.out
#SBATCH --error=logs/part3_%j.err

# ============================================================================
# Part 3: Retrieval Heads (Head Selection + Testing)
# This takes longer because it analyzes ALL heads on training data
# Estimated time: 4-6 hours
# ============================================================================

# Get the directory where this script is located
SCRIPT_DIR="/home/$(whoami)/CS728_PA3"

# Create directories
mkdir -p ${SCRIPT_DIR}/logs
mkdir -p ${SCRIPT_DIR}/results

# Change to project directory
cd ${SCRIPT_DIR}

# Load modules
# module load python/3.10
# module load cuda/11.8

# Set HuggingFace cache
export HF_HOME="${HOME}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_OFFLINE=1

# Reduce memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=============================================="
echo "Running Part 3: Retrieval Heads"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start time: $(date)"
echo "=============================================="

nvidia-smi

# Run Part 3 with default K=20
python run3.py --max_heads 20

echo "=============================================="
echo "Part 3 completed at: $(date)"
echo "=============================================="
