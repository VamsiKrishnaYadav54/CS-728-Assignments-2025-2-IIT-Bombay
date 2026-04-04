#!/usr/bin/env python3
"""
Part 3: Retrieval Heads

This module contains functions for:
1. Identifying retrieval-relevant attention heads (Phase 1)
2. Computing attention scores using only selected heads (Phase 2)
"""

import json
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


def get_query_span(
    tokenizer,
    prompt: str,
    input_ids: torch.Tensor
) -> Tuple[int, int]:
    """
    Identify the token span corresponding to the query in the prompt.
    Same as in code2.py
    """
    query_marker = "Query: "
    end_marker = "\nWhich tool should be used for this query?"
    
    query_start_char = prompt.find(query_marker)
    query_end_char = prompt.find(end_marker)
    
    if query_start_char == -1 or query_end_char == -1:
        raise ValueError("Could not find query markers in prompt")
    
    text_before_query = prompt[:query_start_char + len(query_marker)]
    text_including_query = prompt[:query_end_char]
    
    tokens_before = tokenizer.encode(text_before_query, add_special_tokens=False)
    tokens_including = tokenizer.encode(text_including_query, add_special_tokens=False)
    
    start_idx = len(tokens_before)
    end_idx = len(tokens_including)
    
    return (start_idx, end_idx)


def select_retrieval_heads(
    model,
    tokenizer,
    prompt_utils,
    train_queries: List[Dict],
    device: str,
    max_heads: int = 20,
    max_length: int = 4096
) -> List[Tuple[int, int]]:
    """
    Select the top-K attention heads that are most useful for retrieval.
    
    Strategy: For each head, compute how often it ranks the correct tool 
    at the top (or gives it high attention). Heads that consistently 
    assign high attention to the correct tool are selected.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt_utils: PromptUtils instance
        train_queries: Training queries for head selection
        device: Device to use
        max_heads: Number of heads to select (K)
        max_length: Maximum sequence length
    
    Returns:
        List of (layer_id, head_id) tuples for selected heads
    
    TODO: Implement this function
    """
    # Get model configuration
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    
    print(f"Model has {num_layers} layers and {num_heads} heads per layer")
    print(f"Total heads: {num_layers * num_heads}")
    
    # Track head performance: how well each head ranks the gold tool
    # We'll use "gold tool rank" as the metric (lower is better)
    head_ranks = defaultdict(list)  # (layer, head) -> list of gold ranks
    
    # Also track attention score to gold
    head_gold_attention = defaultdict(list)  # (layer, head) -> list of attention scores
    
    for query_data in tqdm(train_queries, desc="Analyzing heads"):
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
        
        # Use natural tool order
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
        query_start, query_end = query_span
        
        # Analyze each head
        for layer_idx, layer_attention in enumerate(attentions):
            # layer_attention shape: [batch, num_heads, seq_len, seq_len]
            layer_attention = layer_attention.squeeze(0)  # [num_heads, seq_len, seq_len]
            
            for head_idx in range(num_heads):
                head_attention = layer_attention[head_idx]  # [seq_len, seq_len]
                
                # Attention from query to all tokens
                query_attention = head_attention[query_start:query_end, :]  # [query_len, seq_len]
                query_attention_agg = query_attention.mean(dim=0)  # [seq_len]
                
                # Compute score for each tool
                tool_scores = {}
                for tool_idx, doc_start, doc_end in doc_spans:
                    score = query_attention_agg[doc_start:doc_end].sum().item()
                    tool_scores[tool_idx] = score
                
                # Rank tools
                sorted_tools = sorted(tool_scores.items(), key=lambda x: x[1], reverse=True)
                ranking = [t[0] for t in sorted_tools]
                
                # Find gold rank (1-indexed)
                gold_rank = ranking.index(gold_idx) + 1 if gold_idx in ranking else len(ranking)
                head_ranks[(layer_idx, head_idx)].append(gold_rank)
                
                # Track gold attention
                gold_attention = tool_scores.get(gold_idx, 0.0)
                head_gold_attention[(layer_idx, head_idx)].append(gold_attention)
        
        # Clear memory
        del outputs, attentions
        torch.cuda.empty_cache()
    
    # Compute head scores
    # Strategy: Use Mean Reciprocal Rank (MRR) as the selection criterion
    # MRR = average of 1/rank for gold tool
    head_scores = {}
    for (layer_idx, head_idx), ranks in head_ranks.items():
        mrr = np.mean([1.0 / r for r in ranks])
        head_scores[(layer_idx, head_idx)] = mrr
    
    # Sort heads by MRR (descending) and select top K
    sorted_heads = sorted(head_scores.items(), key=lambda x: x[1], reverse=True)
    selected_heads = [head for head, score in sorted_heads[:max_heads]]
    
    # Print selected heads
    print(f"\nSelected {len(selected_heads)} heads (by MRR):")
    print("-" * 50)
    for i, (layer, head) in enumerate(selected_heads):
        mrr = head_scores[(layer, head)]
        mean_rank = np.mean(head_ranks[(layer, head)])
        print(f"  {i+1}. Layer {layer}, Head {head}: MRR={mrr:.4f}, Mean Rank={mean_rank:.2f}")
    
    return selected_heads


def query_to_docs_attention_heads(
    attentions: Tuple[torch.Tensor, ...],
    query_span: Tuple[int, int],
    doc_spans: List[Tuple[int, int, int]],
    selected_heads: List[Tuple[int, int]],
    aggregation: str = "mean"
) -> Dict[int, float]:
    """
    Compute attention-based scores using ONLY selected heads.
    
    Args:
        attentions: Tuple of attention tensors from the model
        query_span: (start, end) token indices for the query
        doc_spans: List of (tool_idx, start, end) for each tool
        selected_heads: List of (layer_id, head_id) to use
        aggregation: How to aggregate ("mean", "max", "sum")
    
    Returns:
        Dictionary mapping tool_idx to attention score
    
    TODO: Implement this function
    """
    query_start, query_end = query_span
    
    # Collect attention from selected heads only
    head_attentions = []
    
    for layer_idx, head_idx in selected_heads:
        # Get attention for this layer and head
        layer_attention = attentions[layer_idx]  # [batch, heads, seq, seq]
        head_attention = layer_attention[0, head_idx, :, :]  # [seq, seq]
        head_attentions.append(head_attention)
    
    # Stack and average: [num_selected_heads, seq, seq] -> [seq, seq]
    stacked = torch.stack(head_attentions, dim=0)
    avg_attention = stacked.mean(dim=0)  # [seq_len, seq_len]
    
    # Extract attention FROM query tokens
    query_attention = avg_attention[query_start:query_end, :]  # [query_len, seq_len]
    
    # Aggregate across query tokens
    if aggregation == "mean":
        query_attention_agg = query_attention.mean(dim=0)
    elif aggregation == "max":
        query_attention_agg = query_attention.max(dim=0)[0]
    elif aggregation == "sum":
        query_attention_agg = query_attention.sum(dim=0)
    else:
        query_attention_agg = query_attention.mean(dim=0)
    
    # Compute score for each tool
    tool_scores = {}
    for tool_idx, doc_start, doc_end in doc_spans:
        score = query_attention_agg[doc_start:doc_end].sum().item()
        tool_scores[tool_idx] = score
    
    return tool_scores


def attention_based_ranking(tool_scores: Dict[int, float]) -> List[int]:
    """Rank tools by attention score (descending)."""
    sorted_tools = sorted(tool_scores.items(), key=lambda x: x[1], reverse=True)
    return [tool_idx for tool_idx, score in sorted_tools]


def compute_recall(
    rankings: List[List[int]], 
    gold_indices: List[int], 
    k: int
) -> float:
    """Compute Recall@k."""
    hits = 0
    for ranking, gold_idx in zip(rankings, gold_indices):
        if gold_idx in ranking[:k]:
            hits += 1
    return hits / len(gold_indices)
