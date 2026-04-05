'''
Part 2: are we lost in the middle?

Goal:
    - visualize the attention from the query to gold document based on the distance between them
    - use attention as a metric to rank documents for a query 
'''
import gc
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1" # remove this line when downloading fresh
import argparse
import json 
import time
import pandas as pd
from tqdm import tqdm
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_model_tokenizer, PromptUtils, get_queries_and_items

# -------------------------
# Do NOT change
# -------------------------
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def query_to_docs_attention(attentions, query_span, doc_spans):
    """
    attentions: tuple(num_layers) of [1, heads, N, N]
    query_span: (start, end)
    doc_spans: list of (start, end)
    """
    doc_scores = torch.zeros(len(doc_spans), device=attentions[0].device)
    
    # TODO 1: implement to get final query to doc attention stored in doc_scores
    query_start, query_end = query_span
    
    # Stack all attention matrices and average over layers and heads
    all_attention = torch.stack([attn.squeeze(0) for attn in attentions], dim=0)
    avg_attention = all_attention.mean(dim=(0, 1))  # [seq_len, seq_len]
    
    # Extract attention from query tokens to all positions
    query_attention = avg_attention[query_start:query_end, :]  # [query_len, seq_len]
    
    # For each document, sum attention from query tokens to doc tokens
    for doc_idx, (doc_start, doc_end) in enumerate(doc_spans):
        doc_scores[doc_idx] = query_attention[:, doc_start:doc_end].sum()
    
    return doc_scores


def analyze_gold_attention(results, save_path="plot2/gold_attention_plot.png"):
    # TODO 2: visualize graph
    """
    input -> result: list of dicts with keys:
                        - gold_position
                        - gold_score
                        - gold_rank
    GOAL: Using the results data, generate a visualization that shows how attention to the gold tool varies with its position in the prompt.

    Requirements:
        - The plot should clearly illustrate whether position affects attention.
        - Save the plot as an image file under folder plot2.
        - You are free to choose how to aggregate and visualize the data.
    """
    os.makedirs("plot2", exist_ok=True)
    
    positions = [r["gold_position"] for r in results]
    scores = [r["gold_score"] for r in results]
    ranks = [r["gold_rank"] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Position vs Attention Score
    ax1 = axes[0, 0]
    ax1.scatter(positions, scores, alpha=0.3, s=10, c='blue')
    # Binned average
    num_bins = 20
    bins = np.linspace(0, max(positions), num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = []
    for i in range(num_bins):
        mask = (np.array(positions) >= bins[i]) & (np.array(positions) < bins[i+1])
        bin_means.append(np.mean(np.array(scores)[mask]) if mask.sum() > 0 else np.nan)
    ax1.plot(bin_centers, bin_means, 'r-', linewidth=2, label='Binned Mean')
    ax1.set_xlabel('Gold Tool Position')
    ax1.set_ylabel('Attention Score')
    ax1.set_title('Attention to Gold Tool vs Position')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Position vs Rank
    ax2 = axes[0, 1]
    ax2.scatter(positions, ranks, alpha=0.3, s=10, c='green')
    bin_rank_means = []
    for i in range(num_bins):
        mask = (np.array(positions) >= bins[i]) & (np.array(positions) < bins[i+1])
        bin_rank_means.append(np.mean(np.array(ranks)[mask]) if mask.sum() > 0 else np.nan)
    ax2.plot(bin_centers, bin_rank_means, 'orange', linewidth=2, label='Binned Mean')
    ax2.set_xlabel('Gold Tool Position')
    ax2.set_ylabel('Rank of Gold Tool')
    ax2.set_title('Rank vs Position')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Histogram of positions
    ax3 = axes[1, 0]
    ax3.hist(positions, bins=20, edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Gold Tool Position')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Gold Tool Positions')
    
    # Plot 4: Recall by position tercile
    ax4 = axes[1, 1]
    num_tools = max(positions) + 1
    tercile_bounds = [0, num_tools//3, 2*num_tools//3, num_tools]
    tercile_recall_1, tercile_recall_5 = [], []
    for i in range(3):
        mask = [(tercile_bounds[i] <= p < tercile_bounds[i+1]) for p in positions]
        tercile_ranks = [r for r, m in zip(ranks, mask) if m]
        if len(tercile_ranks) > 0:
            tercile_recall_1.append(sum([1 for r in tercile_ranks if r == 0]) / len(tercile_ranks))
            tercile_recall_5.append(sum([1 for r in tercile_ranks if r < 5]) / len(tercile_ranks))
        else:
            tercile_recall_1.append(0)
            tercile_recall_5.append(0)
    
    x = np.arange(3)
    width = 0.35
    ax4.bar(x - width/2, tercile_recall_1, width, label='Recall@1', color='steelblue')
    ax4.bar(x + width/2, tercile_recall_5, width, label='Recall@5', color='coral')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Beginning', 'Middle', 'End'])
    ax4.set_ylabel('Recall')
    ax4.set_title('Recall by Position (Lost-in-the-Middle)')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")


def get_query_span(putils, total_length):
    # TODO 3: Query span
    """
    Identify the token span corresponding to the query.
    Note: you are free to add/remove args in this function
    """
    last_doc_end = putils.doc_spans[-1][1]
    query_start = last_doc_end + 1 + putils.add_text1_length + 1
    query_end = total_length - putils.prompt_suffix_length
    return (query_start, query_end)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=64)
parser.add_argument('--model', type=str, default="meta-llama/Llama-3.2-1B-Instruct")
parser.add_argument('--top_heads', type=int, default=20)
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
args = parser.parse_args()


if __name__ == '__main__':
    seed_all(seed=args.seed)
    model_name = args.model
    device = "cuda:0"
    
    tokenizer, model = load_model_tokenizer(model_name=model_name, device=device, dtype=torch.float16)
    num_heads = model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers
    d = getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)
    num_key_value_groups = num_heads//model.config.num_key_value_heads
    softmax_scaling=d**-0.5
    train_queries, test_queries, tools = get_queries_and_items()
 

    print("---- debug print start ----")
    print(f"seed: {args.seed}, model: {model_name}")
    print("model.config._attn_implementation: ", model.config._attn_implementation)

    dict_head_freq = {}
    df_data = []
    avg_latency = []
    count = 0
    start_time = time.time()
    results = []
    
    # Metrics
    correct_at_1 = 0
    correct_at_5 = 0
    total = 0
    
    for qix in tqdm(range(len(test_queries))):
        sample =  test_queries[qix]
        qid = sample["qid"]
        question = sample["text"]
        gold_tool_name = sample["gold_tool_name"]

        # --------------------
        # Do Not change the shuffling here
        # --------------------
        num_dbs = len(tools)
        shuffled_keys = list(tools.keys())
        random.shuffle(shuffled_keys)

        putils = PromptUtils(
            tokenizer=tokenizer, 
            doc_ids=shuffled_keys, 
            dict_all_docs=tools,
            )
        item_spans = putils.doc_spans
        doc_lengths = putils.doc_lengths
        map_docname_id = putils.dict_doc_name_id
        map_id_docname = {v:k for k, v in map_docname_id.items()}
        db_lengths_pt = torch.tensor(doc_lengths, device=device)
        
        gold_tool_id = map_docname_id[gold_tool_name]

        prompt = putils.create_prompt(query=question)
        inputs = tokenizer(prompt, return_tensors = "pt", add_special_tokens = False).to(device)

        if args.debug and qix < 5:
            ip_ids = inputs.input_ids[0].cpu()
            print("-------"*5)
            print(prompt)
            print("-------"*5)
            print("---- doc1 ----")
            print(tokenizer.decode(ip_ids[item_spans[0][0]: item_spans[0][1]]))
            print("---- lastdoc ----")
            print(tokenizer.decode(ip_ids[item_spans[-1][0]: item_spans[-1][1]]))
            print("-------"*5)


        with torch.no_grad():
            attentions = model(**inputs).attentions
            '''
                attentions - tuple of length = # layers
                attentions[0].shape - [1, h, N, N] : first layer's attention matrix for h heads
            '''
        
        query_span = get_query_span(putils, inputs.input_ids.shape[1])

        doc_scores = query_to_docs_attention(attentions, query_span, item_spans)

        # TODO: find gold_rank- rank of gold tool in doc_scores
        # TODO: find gold_score - score of gold tool
        sorted_indices = torch.argsort(doc_scores, descending=True)
        gold_rank = (sorted_indices == gold_tool_id).nonzero(as_tuple=True)[0].item()
        gold_score = doc_scores[gold_tool_id].item()
        
        results.append({
            "qid": qid,
            "gold_position": gold_tool_id,
            "gold_score": gold_score,
            "gold_rank": gold_rank
        })

        # TODO: calucalte recall@1, recall@5 metric and print at end of loop
        if gold_rank == 0:
            correct_at_1 += 1
        if gold_rank < 5:
            correct_at_5 += 1
        total += 1
        
        # Memory cleanup
        del attentions
        if qix % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    # Print final metrics
    recall_at_1 = correct_at_1 / total
    recall_at_5 = correct_at_5 / total
    
    print("\n" + "="*60)
    print("PART 2 RESULTS - Attention-Based Retrieval")
    print("="*60)
    print(f"Recall@1: {recall_at_1:.4f}")
    print(f"Recall@5: {recall_at_5:.4f}")
    print(f"Total queries: {total}")
    
    # Save results
    with open("results_part2.json", "w") as f:
        json.dump({
            "recall_at_1": recall_at_1,
            "recall_at_5": recall_at_5,
            "total_queries": total
        }, f, indent=2)
    print("Results saved to results_part2.json")

    analyze_gold_attention(results)
