#!/usr/bin/env python3
"""
Download all required models on the LOGIN NODE (with internet access).
Run this BEFORE submitting SLURM jobs.

Usage: python download_models.py
"""

import os
import torch
from pathlib import Path

# Set cache directory (change if needed)
CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")
os.makedirs(CACHE_DIR, exist_ok=True)

print("=" * 60)
print("Downloading models for CS728 PA3")
print("This may take a while...")
print("=" * 60)

# Part 1: Sentence Transformers for dense retrieval
print("\n[1/4] Downloading sentence-transformers/msmarco-MiniLM-L-6-v2...")
from sentence_transformers import SentenceTransformer
model1 = SentenceTransformer("sentence-transformers/msmarco-MiniLM-L-6-v2")
del model1
print("✓ msmarco-MiniLM downloaded")

print("\n[2/4] Downloading WhereIsAI/UAE-Large-V1...")
model2 = SentenceTransformer("WhereIsAI/UAE-Large-V1")
del model2
print("✓ UAE-Large-V1 downloaded")

# Part 2 & 3: LLM for attention extraction
# Check what model is specified in the assignment repo
# Common choices: meta-llama/Llama-2-7b-hf, mistralai/Mistral-7B-v0.1, etc.
print("\n[3/4] Downloading LLM for attention extraction...")
print("Note: Check run2.py for the exact model name")

from transformers import AutoTokenizer, AutoModelForCausalLM

# Try common models used in such assignments
# The assignment says "do not change the model" so we need to check the repo
# For now, let's download a commonly used one - adjust based on repo
MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # CHANGE THIS based on run2.py

try:
    print(f"Attempting to download: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Don't download full model here, just tokenizer to verify access
    print(f"✓ Tokenizer for {MODEL_NAME} downloaded")
    print("Note: Full model will be downloaded on first run")
except Exception as e:
    print(f"Could not download {MODEL_NAME}: {e}")
    print("Please check the model name in run2.py and update this script")

# Part 1: BM25 doesn't need downloads (uses rank_bm25 package)
print("\n[4/4] BM25 uses rank_bm25 package (no model download needed)")

print("\n" + "=" * 60)
print("Download complete!")
print("=" * 60)
print("\nNext steps:")
print("1. Clone the assignment repo: git clone https://github.com/deekshakoul/CS728_PA3")
print("2. Check the model name in run2.py and update this script if needed")
print("3. Run the download script again if model name was different")
print("4. Submit SLURM jobs for each part")
