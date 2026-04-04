#!/usr/bin/env python3
"""
Part 3: Retrieval Heads Runner

Phase 1: Select retrieval-relevant attention heads using training data
Phase 2: Use selected heads for retrieval on test data
"""

import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import numpy as np

from code2 import PromptUtils
from code3 import (
    get_query_span,
    select_retrieval_heads,
    query_to_docs_attention_heads,
    attention_based_ranking,
    compute_recall
)

# Set seeds for reproducibility (DO NOT CHANGE)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Model name (DO NOT CHANGE)
MODEL_NAME = "meta-llama/Llama-2-7b-hf"


def load_data(data_dir: str = "data"):
    """Load tools and queries."""
    data_path = Path(data_dir)
    
    with open(data_path / "tools.json", "r") as f:
        tools = json.load(f)
    
    with open(data_path / "train_queries.json", "r") as f:
        train_queries = json.load(f)
    
    with open(data_path / "test_queries.json", "r") as f:
        test_queries = json.load(f)
    
    return tools, train_queries, test_queries


def run_retrieval_with_selected_heads(
    model,
    tokenizer,
    prompt_utils: PromptUtils,
    queries: list,
    selected_heads: list,
    device: str,
    max_length: int = 4096
):
    """Run retrieval using only selected heads."""
    rankings = []
    gold_indices = []
    
    for query_data in tqdm(queries, desc="Testing with selected heads"):
        query_text = query_data["query"]
        gold_tool = query_data["tool"]
        
        # Find gold index
        if isinstance(gold_tool, int):
            gold_idx = gold_tool
        else:
            for i, tool in enumerate(prompt_utils.tools):
                if tool["name"] == gold_tool:
                    gold_idx = i
                    break
        
        gold_indices.append(gold_idx)
        
        # Tool order
        tool_order = list(range(len(prompt_utils.tools)))
        
        # Construct prompt
        prompt = prompt_utils.construct_prompt(query_text, tool_order)
        
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(device)
        
        # Get doc spans
        doc_spans = prompt_utils.get_doc_spans(tokenizer, prompt, tool_order)
        
        # Run model
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_attentions=True,
                return_dict=True
            )
        
        attentions = outputs.attentions
        
        # Get query span
        query_span = get_query_span(tokenizer, prompt, inputs.input_ids)
        
        # Compute scores using ONLY selected heads
        tool_scores = query_to_docs_attention_heads(
            attentions, query_span, doc_spans, selected_heads
        )
        
        # Rank
        ranking = attention_based_ranking(tool_scores)
        rankings.append(ranking)
        
        # Clear memory
        del outputs, attentions
        torch.cuda.empty_cache()
    
    return rankings, gold_indices


def main(max_heads: int = 20):
    """
    Main function for Part 3.
    
    Args:
        max_heads: Number of heads to select (for bonus experiments)
    """
    print("=" * 60)
    print(f"Part 3: Retrieval Heads (K={max_heads})")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    tools, train_queries, test_queries = load_data()
    print(f"Loaded {len(tools)} tools")
    print(f"Loaded {len(train_queries)} training queries")
    print(f"Loaded {len(test_queries)} test queries")
    
    # Initialize prompt utils
    prompt_utils = PromptUtils(tools)
    
    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        output_attentions=True
    )
    model.eval()
    
    # ==========================================================================
    # Phase 1: Head Selection using training data
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Phase 1: Selecting Retrieval Heads")
    print("=" * 60)
    
    selected_heads = select_retrieval_heads(
        model, tokenizer, prompt_utils, 
        train_queries, device,
        max_heads=max_heads
    )
    
    # Save selected heads
    heads_info = {
        "max_heads": max_heads,
        "selected_heads": [(l, h) for l, h in selected_heads],
        "selection_strategy": "Mean Reciprocal Rank (MRR) of gold tool ranking"
    }
    
    Path("results").mkdir(exist_ok=True)
    with open(f"results/part3_selected_heads_k{max_heads}.json", "w") as f:
        json.dump(heads_info, f, indent=2)
    
    # ==========================================================================
    # Phase 2: Retrieval using selected heads
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Phase 2: Testing with Selected Heads")
    print("=" * 60)
    
    rankings, gold_indices = run_retrieval_with_selected_heads(
        model, tokenizer, prompt_utils,
        test_queries, selected_heads, device
    )
    
    # Compute metrics
    recall_1 = compute_recall(rankings, gold_indices, k=1)
    recall_5 = compute_recall(rankings, gold_indices, k=5)
    
    print("\n" + "=" * 60)
    print(f"Part 3 Results (K={max_heads} heads)")
    print("=" * 60)
    print(f"Recall@1: {recall_1:.4f}")
    print(f"Recall@5: {recall_5:.4f}")
    
    # Save results
    results = {
        "max_heads": max_heads,
        "Recall@1": recall_1,
        "Recall@5": recall_5,
        "selected_heads": [(l, h) for l, h in selected_heads]
    }
    
    with open(f"results/part3_results_k{max_heads}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to results/part3_results_k{max_heads}.json")
    print(f"Selected heads saved to results/part3_selected_heads_k{max_heads}.json")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_heads", type=int, default=20,
                        help="Number of heads to select (default: 20)")
    args = parser.parse_args()
    
    main(max_heads=args.max_heads)
