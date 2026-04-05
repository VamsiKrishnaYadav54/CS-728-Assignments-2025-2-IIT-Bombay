'''
Part 1: Classical Retrieval Methods

Goal:
    - Implement BM25, msmarco-MiniLM, UAE-Large-V1 retrievers
    - Report Recall@1 and Recall@5 for each method
'''
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from utils import get_queries_and_items


class BM25Retriever:
    """BM25 sparse retrieval baseline"""
    
    def __init__(self, tools):
        self.tool_names = list(tools.keys())
        self.tool_descriptions = [tools[name] for name in self.tool_names]
        
        # Tokenize tool descriptions (simple whitespace tokenization)
        tokenized_docs = [desc.lower().split() for desc in self.tool_descriptions]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        self.name_to_idx = {name: idx for idx, name in enumerate(self.tool_names)}
    
    def retrieve(self, query, top_k=100):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return ranked_indices.tolist()
    
    def get_tool_idx(self, tool_name):
        return self.name_to_idx[tool_name]


class DenseRetriever:
    """Dense retrieval using sentence transformers"""
    
    def __init__(self, tools, model_name):
        self.tool_names = list(tools.keys())
        self.tool_descriptions = [tools[name] for name in self.tool_names]
        
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        print("Encoding tool descriptions...")
        self.tool_embeddings = self.model.encode(
            self.tool_descriptions, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        # Normalize for cosine similarity
        self.tool_embeddings = self.tool_embeddings / np.linalg.norm(
            self.tool_embeddings, axis=1, keepdims=True
        )
        
        self.name_to_idx = {name: idx for idx, name in enumerate(self.tool_names)}
    
    def retrieve(self, query, top_k=100):
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        similarities = np.dot(self.tool_embeddings, query_embedding.T).squeeze()
        ranked_indices = np.argsort(similarities)[::-1][:top_k]
        return ranked_indices.tolist()
    
    def get_tool_idx(self, tool_name):
        return self.name_to_idx[tool_name]


def evaluate(retriever, test_queries, name):
    recall_1, recall_5, total = 0, 0, 0
    
    for sample in tqdm(test_queries, desc=f"Evaluating {name}"):
        query = sample["text"]
        gold_tool_name = sample["gold_tool_name"]
        gold_idx = retriever.get_tool_idx(gold_tool_name)
        
        ranked = retriever.retrieve(query, top_k=100)
        
        if gold_idx in ranked[:1]:
            recall_1 += 1
        if gold_idx in ranked[:5]:
            recall_5 += 1
        total += 1
    
    return recall_1 / total, recall_5 / total


if __name__ == "__main__":
    print("Loading data...")
    train_queries, test_queries, tools = get_queries_and_items()
    print(f"Test queries: {len(test_queries)}, Tools: {len(tools)}")
    
    results = {}
    
    # BM25
    print("\n" + "="*50)
    print("Running BM25...")
    bm25 = BM25Retriever(tools)
    r1, r5 = evaluate(bm25, test_queries, "BM25")
    results['BM25'] = {"Recall@1": r1, "Recall@5": r5}
    print(f"BM25 - Recall@1: {r1:.4f}, Recall@5: {r5:.4f}")
    
    # msmarco-MiniLM
    print("\n" + "="*50)
    print("Running msmarco-MiniLM...")
    msmarco = DenseRetriever(tools, "sentence-transformers/msmarco-MiniLM-L-6-v3")
    r1, r5 = evaluate(msmarco, test_queries, "msmarco-MiniLM")
    results['msmarco-MiniLM'] = {"Recall@1": r1, "Recall@5": r5}
    print(f"msmarco-MiniLM - Recall@1: {r1:.4f}, Recall@5: {r5:.4f}")
    
    # UAE-Large-V1
    print("\n" + "="*50)
    print("Running UAE-Large-V1...")
    uae = DenseRetriever(tools, "WhereIsAI/UAE-Large-V1")
    r1, r5 = evaluate(uae, test_queries, "UAE-Large-V1")
    results['UAE-Large-V1'] = {"Recall@1": r1, "Recall@5": r5}
    print(f"UAE-Large-V1 - Recall@1: {r1:.4f}, Recall@5: {r5:.4f}")
    
    # Print summary
    print("\n" + "="*60)
    print("PART 1 RESULTS - Classical Retrieval")
    print("="*60)
    print(f"{'Method':<20} {'Recall@1':<15} {'Recall@5':<15}")
    print("-"*50)
    for method, metrics in results.items():
        print(f"{method:<20} {metrics['Recall@1']:.4f}          {metrics['Recall@5']:.4f}")
    
    with open("results_part1.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results_part1.json")
