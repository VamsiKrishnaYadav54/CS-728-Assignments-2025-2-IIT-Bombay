#!/bin/bash
# =============================================================================
# CS728 PA3 Setup Script
# Run this on the LOGIN NODE (with internet access)
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "CS728 PA3 Setup"
echo "=============================================="

# Get project directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Project directory: $SCRIPT_DIR"

# Step 1: Clone the assignment repository (if not already done)
if [ ! -d "data" ]; then
    echo ""
    echo "Step 1: Cloning assignment repository..."
    git clone https://github.com/deekshakoul/CS728_PA3 temp_repo
    
    # Copy data files
    if [ -d "temp_repo/data" ]; then
        cp -r temp_repo/data ./
        echo "Data copied successfully!"
    fi
    
    # Check for any template files we should use
    if [ -f "temp_repo/run2.py" ]; then
        echo "NOTE: The repo has template files. Check if you need to use them."
        echo "Template files found: $(ls temp_repo/*.py 2>/dev/null || echo 'none')"
    fi
    
    rm -rf temp_repo
else
    echo "Step 1: Data directory already exists, skipping clone."
fi

# Step 2: Install Python packages
echo ""
echo "Step 2: Installing Python packages..."
pip install -r requirements.txt --user

# Step 3: Download NLTK data
echo ""
echo "Step 3: Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# Step 4: Download models
echo ""
echo "Step 4: Downloading models (this may take a while)..."

# Set HuggingFace cache
export HF_HOME="${HOME}/.cache/huggingface"
mkdir -p "$HF_HOME"

# Download sentence transformers
python -c "
from sentence_transformers import SentenceTransformer
print('Downloading msmarco-MiniLM...')
SentenceTransformer('sentence-transformers/msmarco-MiniLM-L-6-v2')
print('Downloading UAE-Large-V1...')
SentenceTransformer('WhereIsAI/UAE-Large-V1')
print('Done!')
"

# Download LLM tokenizer (full model will be downloaded on first use)
# Check if you have HuggingFace access token for Llama-2
echo ""
echo "For Llama-2, you need a HuggingFace token with access."
echo "Run: huggingface-cli login"
echo ""

python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = 'meta-llama/Llama-2-7b-hf'

try:
    print(f'Downloading tokenizer for {MODEL_NAME}...')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print('Tokenizer downloaded!')
    
    print(f'Downloading model {MODEL_NAME}...')
    print('This will take 15-30 minutes...')
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    del model
    print('Model downloaded!')
except Exception as e:
    print(f'Error: {e}')
    print('You may need to:')
    print('1. Run: huggingface-cli login')
    print('2. Accept Llama-2 license at https://huggingface.co/meta-llama/Llama-2-7b-hf')
"

# Step 5: Create directories
echo ""
echo "Step 5: Creating directories..."
mkdir -p logs results plot2

# Summary
echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Directory structure:"
ls -la
echo ""
echo "Next steps:"
echo "1. Verify data files exist: ls data/"
echo "2. Submit Part 1: sbatch scripts/run_part1.slurm"
echo "3. Submit Part 2: sbatch scripts/run_part2.slurm"
echo "4. Submit Part 3: sbatch scripts/run_part3.slurm"
echo "5. (Optional) Run bonus: sbatch scripts/run_bonus.slurm"
echo ""
echo "Check job status: squeue -u \$USER"
echo "View logs: tail -f logs/*.out"
