'''
BONUS: Different Head Selection Strategies for Part 3

Strategies implemented:
1. Recall@1 based (original) - heads that rank gold tool first most often
2. Average Rank based - heads with lowest average rank for gold tool
3. Attention Magnitude based - heads that give highest attention to gold tool
4. Combined Score - weighted combination of recall and avg rank
5. Top-5 Recall based - heads that put gold tool in top 5 most often
'''

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import json
import random
import numpy as np
import torch
from tqdm import tqdm

from utils import load_model_tokenizer, PromptUtils, get_queries_and_items


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_head_statistics(train_queries, model, tokenizer, tools, device, num_samples=200):
    """
    Collect statistics for each head across training queries.
    Returns metrics that can be used for different selection strategies.
    """
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    
    # Statistics to collect
    head_recall_at_1 = torch.zeros(num_layers, num_heads, device=device)  # Times gold ranked 1st
    head_recall_at_5 = torch.zeros(num_layers, num_heads, device=device)  # Times gold in top 5
    head_total_rank = torch.zeros(num_layers, num_heads, device=device)   # Sum of gold ranks
    head_total_attention = torch.zeros(num_layers, num_heads, device=device)  # Sum of attention to gold
    head_count = torch.zeros(num_layers, num_heads, device=device)
    
    print(f"Collecting head statistics from {num_samples} training queries...")
    
    for qix in tqdm(range(min(num_samples, len(train_queries))), desc="Analyzing heads"):
        sample = train_queries[qix]
        question = sample["text"]
        gold_tool_name = sample["gold_tool_name"]
        
        tool_ids = list(tools.keys())
        random.shuffle(tool_ids)
        
        putils = PromptUtils(
            tokenizer=tokenizer,
            doc_ids=tool_ids,
            dict_all_docs=tools,
        )
        item_spans = putils.doc_spans
        map_docname_id = putils.dict_doc_name_id
        gold_tool_idx = map_docname_id[gold_tool_name]
        
        prompt = putils.create_prompt(query=question)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        input_ids = inputs.input_ids[0]
        
        with torch.no_grad():
            attentions = model(**inputs).attentions
        
        # Get query span
        last_doc_end = putils.doc_spans[-1][1]
        query_start = last_doc_end + 1 + putils.add_text1_length + 1
        query_end = input_ids.shape[0] - putils.prompt_suffix_length
        
        # Analyze each head
        for layer_idx in range(num_layers):
            layer_attention = attentions[layer_idx].squeeze(0)
            
            for head_idx in range(num_heads):
                head_attention = layer_attention[head_idx]
                query_attention = head_attention[query_start:query_end, :]
                
                # Score each document
                doc_scores = torch.zeros(len(item_spans), device=device)
                for doc_idx, (doc_start, doc_end) in enumerate(item_spans):
                    doc_scores[doc_idx] = query_attention[:, doc_start:doc_end].sum()
                
                # Get rank of gold tool
                sorted_indices = torch.argsort(doc_scores, descending=True)
                gold_rank = (sorted_indices == gold_tool_idx).nonzero(as_tuple=True)[0].item()
                
                # Update statistics
                if gold_rank == 0:
                    head_recall_at_1[layer_idx, head_idx] += 1
                if gold_rank < 5:
                    head_recall_at_5[layer_idx, head_idx] += 1
                head_total_rank[layer_idx, head_idx] += gold_rank
                head_total_attention[layer_idx, head_idx] += doc_scores[gold_tool_idx].item()
                head_count[layer_idx, head_idx] += 1
        
        del attentions
        if qix % 50 == 0:
            torch.cuda.empty_cache()
    
    return {
        'recall_at_1': head_recall_at_1,
        'recall_at_5': head_recall_at_5,
        'total_rank': head_total_rank,
        'total_attention': head_total_attention,
        'count': head_count,
        'num_layers': num_layers,
        'num_heads': num_heads
    }


def select_heads_strategy1_recall(stats, max_heads):
    """Strategy 1: Select heads with highest Recall@1 (original approach)"""
    scores = stats['recall_at_1']
    return select_top_k(scores, max_heads, stats['num_heads'])


def select_heads_strategy2_avg_rank(stats, max_heads):
    """Strategy 2: Select heads with lowest average rank (inverted for selection)"""
    avg_rank = stats['total_rank'] / stats['count'].clamp(min=1)
    # Invert: lower rank is better, so we want heads with smallest avg_rank
    # Use negative so that topk gives us the smallest
    scores = -avg_rank
    return select_top_k(scores, max_heads, stats['num_heads'])


def select_heads_strategy3_attention(stats, max_heads):
    """Strategy 3: Select heads with highest average attention to gold tool"""
    avg_attention = stats['total_attention'] / stats['count'].clamp(min=1)
    return select_top_k(avg_attention, max_heads, stats['num_heads'])


def select_heads_strategy4_combined(stats, max_heads):
    """Strategy 4: Combined score = normalized recall + 0.3 * normalized inverse rank"""
    recall = stats['recall_at_1']
    avg_rank = stats['total_rank'] / stats['count'].clamp(min=1)
    
    # Normalize recall to [0, 1]
    max_recall = recall.max()
    norm_recall = recall / max_recall if max_recall > 0 else recall
    
    # Normalize inverse rank to [0, 1]
    max_rank = avg_rank.max()
    norm_inv_rank = 1 - (avg_rank / max_rank) if max_rank > 0 else torch.ones_like(avg_rank)
    
    scores = norm_recall + 0.3 * norm_inv_rank
    return select_top_k(scores, max_heads, stats['num_heads'])


def select_heads_strategy5_recall_at_5(stats, max_heads):
    """Strategy 5: Select heads with highest Recall@5"""
    scores = stats['recall_at_5']
    return select_top_k(scores, max_heads, stats['num_heads'])


def select_top_k(scores, k, num_heads):
    """Helper to select top-k heads from score matrix"""
    flat_scores = scores.flatten()
    top_k_indices = torch.topk(flat_scores, k=k).indices
    
    selected_heads = []
    for flat_idx in top_k_indices:
        layer_idx = flat_idx.item() // num_heads
        head_idx = flat_idx.item() % num_heads
        selected_heads.append((layer_idx, head_idx))
    
    return selected_heads


def evaluate_heads(test_queries, model, tokenizer, tools, device, selected_heads):
    """Evaluate selected heads on test set"""
    correct_at_1 = 0
    correct_at_5 = 0
    total = 0
    
    for qix in tqdm(range(len(test_queries)), desc="Evaluating"):
        sample = test_queries[qix]
        question = sample["text"]
        gold_tool_name = sample["gold_tool_name"]
        
        shuffled_keys = list(tools.keys())
        random.shuffle(shuffled_keys)
        
        putils = PromptUtils(
            tokenizer=tokenizer,
            doc_ids=shuffled_keys,
            dict_all_docs=tools,
        )
        
        item_spans = putils.doc_spans
        map_docname_id = putils.dict_doc_name_id
        gold_tool_id = map_docname_id[gold_tool_name]
        
        prompt = putils.create_prompt(query=question)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        
        with torch.no_grad():
            attentions = model(**inputs).attentions
        
        # Get query span
        last_doc_end = putils.doc_spans[-1][1]
        query_start = last_doc_end + 1 + putils.add_text1_length + 1
        query_end = inputs.input_ids.shape[1] - putils.prompt_suffix_length
        
        # Score documents using selected heads
        doc_scores = torch.zeros(len(item_spans), device=device)
        for layer_idx, head_idx in selected_heads:
            head_attention = attentions[layer_idx][0, head_idx]
            query_attention = head_attention[query_start:query_end, :]
            
            for doc_idx, (doc_start, doc_end) in enumerate(item_spans):
                doc_scores[doc_idx] += query_attention[:, doc_start:doc_end].sum()
        
        doc_scores = doc_scores / len(selected_heads)
        
        # Get rank
        ranked_docs = torch.argsort(doc_scores, descending=True)
        gold_rank = (ranked_docs == gold_tool_id).nonzero(as_tuple=True)[0].item()
        
        if gold_rank == 0:
            correct_at_1 += 1
        if gold_rank < 5:
            correct_at_5 += 1
        total += 1
        
        del attentions
        if qix % 100 == 0:
            torch.cuda.empty_cache()
    
    return correct_at_1 / total, correct_at_5 / total


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=64)
parser.add_argument('--model', type=str, default="meta-llama/Llama-3.2-1B-Instruct")
parser.add_argument('--max_heads', type=int, default=20)
parser.add_argument('--train_samples', type=int, default=200)
args = parser.parse_args()


if __name__ == '__main__':
    seed_all(args.seed)
    model_name = args.model
    device = "cuda:0"
    
    tokenizer, model = load_model_tokenizer(model_name=model_name, device=device, dtype=torch.float16)
    train_queries, test_queries, tools = get_queries_and_items()
    
    print("="*70)
    print("BONUS: Comparing Different Head Selection Strategies")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Max heads: {args.max_heads}")
    print(f"Training samples: {args.train_samples}")
    print("="*70)
    
    # Collect statistics once (used by all strategies)
    print("\n[Step 1] Collecting head statistics...")
    stats = collect_head_statistics(
        train_queries[:args.train_samples],
        model, tokenizer, tools, device,
        num_samples=args.train_samples
    )
    
    # Define strategies
    strategies = {
        "Strategy 1: Recall@1": select_heads_strategy1_recall,
        "Strategy 2: Avg Rank": select_heads_strategy2_avg_rank,
        "Strategy 3: Attention Magnitude": select_heads_strategy3_attention,
        "Strategy 4: Combined (Recall + Rank)": select_heads_strategy4_combined,
        "Strategy 5: Recall@5": select_heads_strategy5_recall_at_5,
    }
    
    results = {}
    
    print("\n[Step 2] Evaluating each strategy...")
    
    for strategy_name, strategy_fn in strategies.items():
        print(f"\n{'='*60}")
        print(f"Testing: {strategy_name}")
        print("="*60)
        
        # Reset seed for fair comparison
        seed_all(args.seed)
        
        # Select heads using this strategy
        selected_heads = strategy_fn(stats, args.max_heads)
        print(f"Selected heads: {selected_heads[:5]}... (showing first 5)")
        
        # Evaluate
        recall_1, recall_5 = evaluate_heads(
            test_queries, model, tokenizer, tools, device, selected_heads
        )
        
        results[strategy_name] = {
            "recall_at_1": recall_1,
            "recall_at_5": recall_5,
            "selected_heads": selected_heads
        }
        
        print(f"Recall@1: {recall_1:.4f}")
        print(f"Recall@5: {recall_5:.4f}")
    
    # Print summary table
    print("\n" + "="*70)
    print("BONUS RESULTS SUMMARY - Head Selection Strategies")
    print("="*70)
    print(f"{'Strategy':<40} {'Recall@1':<12} {'Recall@5':<12}")
    print("-"*64)
    for strategy_name, metrics in results.items():
        print(f"{strategy_name:<40} {metrics['recall_at_1']:.4f}       {metrics['recall_at_5']:.4f}")
    print("="*70)
    
    # Find best strategy
    best_strategy = max(results.keys(), key=lambda x: results[x]['recall_at_1'])
    print(f"\nBest Strategy (by Recall@1): {best_strategy}")
    print(f"  Recall@1: {results[best_strategy]['recall_at_1']:.4f}")
    print(f"  Recall@5: {results[best_strategy]['recall_at_5']:.4f}")
    
    # Save results
    with open("results_bonus_strategies.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results_bonus_strategies.json")
