#!/bin/bash
# ============================================================================
# download_models.sh
# Run this on LOGIN NODE (has internet) BEFORE submitting SLURM jobs
# Downloads models to ./models directory in your project
# ============================================================================

echo "=============================================="
echo "CS728 PA3 - Model Download Script"
echo "=============================================="

# Get the directory where this script is located
SCRIPT_DIR="/home/cccp/25m2125/VAMSI/CS728/CS-728-Assignments-2025-2-IIT-Bombay/A3/scripts"
PROJECT_DIR="/home/cccp/25m2125/VAMSI/CS728/CS-728-Assignments-2025-2-IIT-Bombay/A3"

# Set model directory inside project
MODEL_DIR="${PROJECT_DIR}/models"
mkdir -p "$MODEL_DIR"

# Set HuggingFace to use project directory
export HF_HOME="${MODEL_DIR}/hf_cache"
export TRANSFORMERS_CACHE="${MODEL_DIR}/hf_cache"
export SENTENCE_TRANSFORMERS_HOME="${MODEL_DIR}/sentence_transformers"

mkdir -p "$HF_HOME"
mkdir -p "$SENTENCE_TRANSFORMERS_HOME"

echo "Project directory: $PROJECT_DIR"
echo "Models will be saved to: $MODEL_DIR"
echo ""

# Download Part 2 & 3: LLaMA 3.2 1B Instruct
echo ""
echo "=== Downloading Llama-3.2-1B-Instruct ==="
python -c '
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN", None)
if hf_token is None:
    raise ValueError("HUGGINGFACE_HUB_TOKEN not set in environment!")

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    use_auth_token=hf_token
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    use_auth_token=hf_token
)

print("Models loaded successfully!")
'

# # Download Part 1: Sentence Transformers
# echo ""
# echo "=== Downloading msmarco-MiniLM-L-6-v3 ==="
# python -c "
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L-6-v3')
# print('msmarco-MiniLM downloaded successfully!')
# "

# echo ""
# echo "=== Downloading UAE-Large-V1 ==="
# python -c "
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('WhereIsAI/UAE-Large-V1')
# print('UAE-Large-V1 downloaded successfully!')
# "

echo ""
echo "=============================================="
echo "All models downloaded to: $MODEL_DIR"
echo "=============================================="
