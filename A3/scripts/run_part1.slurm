#!/bin/bash
#SBATCH --job-name=PA3_Part1
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --qos=a40
#SBATCH --output=logs/part1_%j.out
#SBATCH --error=logs/part1_%j.err

# Your project directory
SCRIPT_DIR="/home/cccp/25m2125/VAMSI/CS728/CS-728-Assignments-2025-2-IIT-Bombay/A3"

mkdir -p ${SCRIPT_DIR}/logs
cd ${SCRIPT_DIR}

# Activate conda
source /lustre-flash/apps/spack/share/spack/setup-env.sh
source ~/miniconda3/bin/activate CS728

# Use models from project directory
export HF_HOME="${SCRIPT_DIR}/models/Llama-3.2-1B-local"
export TRANSFORMERS_CACHE="${SCRIPT_DIR}/models/Llama-3.2-1B-local"
export SENTENCE_TRANSFORMERS_HOME="${SCRIPT_DIR}/models/sentence_transformers"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "=============================================="
echo "Running Part 1: Classical Retrieval"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Model dir: ${HF_HOME}"
echo "Start time: $(date)"
echo "=============================================="

python run1.py

echo "Part 1 completed at: $(date)"
