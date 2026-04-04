#!/bin/bash
#SBATCH --job-name=PA3_Part2
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/part2_%j.out
#SBATCH --error=logs/part2_%j.err

# ============================================================================
# Part 2: Attention-Based Retrieval
# Requires ~40GB GPU memory for Llama-2-7B with attention extraction
# Estimated time: 2-4 hours depending on number of queries
# ============================================================================

# Get the directory where this script is located
SCRIPT_DIR="/home/$(whoami)/CS728_PA3"

# Create directories
mkdir -p ${SCRIPT_DIR}/logs
mkdir -p ${SCRIPT_DIR}/results
mkdir -p ${SCRIPT_DIR}/plot2

# Change to project directory
cd ${SCRIPT_DIR}

# Load modules
# module load python/3.10
# module load cuda/11.8

# Set HuggingFace cache (models pre-downloaded on login node)
export HF_HOME="${HOME}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_OFFLINE=1

# Reduce memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=============================================="
echo "Running Part 2: Attention-Based Retrieval"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start time: $(date)"
echo "=============================================="

# Check GPU memory
nvidia-smi

# Run Part 2
python run2.py

echo "=============================================="
echo "Part 2 completed at: $(date)"
echo "=============================================="

# Check if plot was generated
if [ -f "plot2/gold_attention_plot.png" ]; then
    echo "Plot generated successfully!"
else
    echo "WARNING: Plot not found!"
fi
