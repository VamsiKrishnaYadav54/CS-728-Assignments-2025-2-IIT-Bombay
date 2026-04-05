#!/bin/bash
#SBATCH --job-name=PA3_Bonus_Strategies
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --qos=a40
#SBATCH --output=logs/bonus_strategies_%j.out
#SBATCH --error=logs/bonus_strategies_%j.err

# ============================================================================
# BONUS: Test different head selection strategies
# ============================================================================

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
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "=============================================="
echo "BONUS: Testing Different Head Selection Strategies"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "=============================================="

python run_bonus_strategies.py \
    --model /home/cccp/25m2125/VAMSI/CS728/CS-728-Assignments-2025-2-IIT-Bombay/A3/models/Llama-3.2-1B-local \
    --seed 64 \
    --max_heads 10 \
    --train_samples 200

echo "=============================================="
echo "BONUS completed at: $(date)"
echo "=============================================="
