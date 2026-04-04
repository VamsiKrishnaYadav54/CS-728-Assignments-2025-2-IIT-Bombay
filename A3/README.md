# CS728 Programming Assignment 3
## Retrieval, Attention, and LLM

This repository contains the code for PA3, which studies how language models select tools given natural language queries.

---

## Quick Start (HPC Cluster with SLURM)

### Step 1: Setup (Run on Login Node - Has Internet)
```bash
cd CS728_PA3
chmod +x setup.sh
./setup.sh
```

This will:
- Clone the assignment data
- Install Python packages
- Download all required models (this takes ~30 min)

### Step 2: Login to HuggingFace (for Llama-2 access)
```bash
huggingface-cli login
# Enter your token from https://huggingface.co/settings/tokens
```

Make sure you've accepted the Llama-2 license at:
https://huggingface.co/meta-llama/Llama-2-7b-hf

### Step 3: Submit Jobs
```bash
# Part 1: Classical Retrieval (~1 hour)
sbatch scripts/run_part1.slurm

# Part 2: Attention-Based Retrieval (~4 hours)
sbatch scripts/run_part2.slurm

# Part 3: Retrieval Heads (~6 hours)
sbatch scripts/run_part3.slurm

# Bonus: Test K=10,20,30 (~10 hours)
sbatch scripts/run_bonus.slurm
```

### Step 4: Check Status
```bash
squeue -u $USER          # Check job queue
tail -f logs/part1_*.out # Watch output in real-time
```

---

## Project Structure

```
CS728_PA3/
├── data/                    # Assignment data (from repo)
│   ├── tools.json           # ~100 tools with descriptions
│   ├── train_queries.json   # Training queries
│   └── test_queries.json    # Test queries
├── part1_retrieval.py       # Part 1: BM25, MiniLM, UAE
├── code2.py                 # Part 2: Core functions
├── run2.py                  # Part 2: Runner
├── code3.py                 # Part 3: Core functions
├── run3.py                  # Part 3: Runner
├── run_bonus.py             # Bonus: K comparison
├── scripts/                 # SLURM job scripts
│   ├── run_part1.slurm
│   ├── run_part2.slurm
│   ├── run_part3.slurm
│   └── run_bonus.slurm
├── results/                 # Output JSON files
├── plot2/                   # Generated plots
├── logs/                    # SLURM job logs
├── requirements.txt
├── setup.sh
└── README.md
```

---

## Part Descriptions

### Part 1: Classical Retrieval
- **BM25**: Sparse retrieval using term frequency
- **msmarco-MiniLM**: Dense retrieval with sentence embeddings
- **UAE-Large-V1**: Another dense retrieval model

**Output**: `results/part1_results.json`

### Part 2: Attention-Based Retrieval
- Put all tools + query in a single prompt
- Extract attention matrices from Llama-2
- Score tools based on query→tool attention
- Analyze "lost-in-the-middle" effect

**Output**: 
- `results/part2_results.json`
- `plot2/gold_attention_plot.png`

### Part 3: Retrieval Heads
- **Phase 1**: Select K=20 heads that best retrieve correct tools
- **Phase 2**: Use only those heads for retrieval
- Compare with Parts 1 & 2

**Output**: 
- `results/part3_results_k20.json`
- `results/part3_selected_heads_k20.json`

### Bonus
- Test K = 10, 20, 30 heads
- Compare recall performance

**Output**: `results/bonus_k_comparison.json`

---

## Expected Results Format

### Part 1
```json
{
  "BM25": {"Recall@1": 0.XX, "Recall@5": 0.XX},
  "msmarco-MiniLM": {"Recall@1": 0.XX, "Recall@5": 0.XX},
  "UAE-Large-V1": {"Recall@1": 0.XX, "Recall@5": 0.XX}
}
```

### Part 2
```json
{
  "Recall@1": 0.XX,
  "Recall@5": 0.XX
}
```

### Part 3
```json
{
  "max_heads": 20,
  "Recall@1": 0.XX,
  "Recall@5": 0.XX,
  "selected_heads": [[layer, head], ...]
}
```

---

## Troubleshooting

### OOM Error
If you get GPU out-of-memory errors:
1. Reduce batch size (if applicable)
2. Use `--no_compile` flag
3. Request more GPU memory in SLURM script

### Model Download Fails
- Check internet connectivity (login node only)
- Verify HuggingFace token: `huggingface-cli whoami`
- Accept model license on HuggingFace website

### Compute Node Can't Find Model
```bash
# Set offline mode in your script
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HOME}/.cache/huggingface"
```

---

## Report Deliverables

1. **Part 1**: Table with Recall@1 and Recall@5 for BM25, MiniLM, UAE
2. **Part 2.1**: Table with Recall@1 and Recall@5
3. **Part 2.2**: Plot showing position effects (`plot2/gold_attention_plot.png`)
4. **Part 3.1**: Head selection strategy + selected heads list
5. **Part 3.2**: Table with Recall@1 and Recall@5
6. **Part 3.3**: Comparison of all three parts
7. **Bonus**: Table showing K=10,20,30 comparison

---

## Authors
T Vamsi Krishna Yadav 25M2125

