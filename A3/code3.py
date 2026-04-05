import torch
from tqdm import tqdm
from utils import PromptUtils
import random 

def select_retrieval_heads(train_queries, model, tokenizer, tools, device, max_heads=20):
    # TODO 3: Head selection
    """
    Identify a subset of attention heads that are most useful for retrieving the correct tool.
    Requirements:
    - Use the same prompt structure as Part-2
    - Use attention patterns(query -> tool) to score heads
    - Aggregate signals across training queries
    - Return "max_heads" heads as (layer, head)
    Notes:
    - You must construct prompts and extract attentions inside this function
    - Avoid hardcoding specific queries or tools
    """
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    
    # accumulate scores per head
    head_scores = torch.zeros(num_layers, num_heads, device=device)
    head_counts = torch.zeros(num_layers, num_heads, device=device)
    
    print(f"Selecting heads using {len(train_queries)} training queries...")
    
    for qix in tqdm(range(len(train_queries)), desc="Head Selection"):
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
        
        # For each head, check if it ranks gold tool first
        for layer_idx in range(num_layers):
            layer_attention = attentions[layer_idx].squeeze(0)  # [num_heads, seq_len, seq_len]
            
            for head_idx in range(num_heads):
                head_attention = layer_attention[head_idx]  # [seq_len, seq_len]
                query_attention = head_attention[query_start:query_end, :]
                
                # Score each document
                doc_scores = torch.zeros(len(item_spans), device=device)
                for doc_idx, (doc_start, doc_end) in enumerate(item_spans):
                    doc_scores[doc_idx] = query_attention[:, doc_start:doc_end].sum()
                
                # Check if gold is ranked first
                sorted_indices = torch.argsort(doc_scores, descending=True)
                gold_rank = (sorted_indices == gold_tool_idx).nonzero(as_tuple=True)[0].item()
                
                if gold_rank == 0:
                    head_scores[layer_idx, head_idx] += 1
                head_counts[layer_idx, head_idx] += 1
        
        del attentions
        if qix % 50 == 0:
            torch.cuda.empty_cache()
    
    # Select top-k heads
    flat_scores = head_scores.flatten()
    top_k_indices = torch.topk(flat_scores, k=max_heads).indices
    
    selected_heads = []
    for flat_idx in top_k_indices:
        layer_idx = flat_idx.item() // num_heads
        head_idx = flat_idx.item() % num_heads
        selected_heads.append((layer_idx, head_idx))
    
    print(f"\nSelected {len(selected_heads)} heads")
    
    assert len(selected_heads) == max_heads
    return selected_heads
