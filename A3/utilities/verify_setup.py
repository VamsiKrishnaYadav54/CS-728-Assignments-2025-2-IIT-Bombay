#!/usr/bin/env python3
"""
Quick sanity check script to verify everything is set up correctly.
Run this before submitting SLURM jobs to catch issues early.
"""

import sys
from pathlib import Path

def check_imports():
    """Check all required packages can be imported."""
    print("Checking imports...")
    
    packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("sentence_transformers", "Sentence-Transformers"),
        ("rank_bm25", "BM25"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("nltk", "NLTK"),
        ("tqdm", "tqdm"),
    ]
    
    all_ok = True
    for package, name in packages:
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name}: {e}")
            all_ok = False
    
    return all_ok


def check_data():
    """Check data files exist."""
    print("\nChecking data files...")
    
    data_dir = Path("data")
    required_files = ["tools.json", "train_queries.json", "test_queries.json"]
    
    all_ok = True
    for fname in required_files:
        fpath = data_dir / fname
        if fpath.exists():
            import json
            with open(fpath) as f:
                data = json.load(f)
            print(f"  ✓ {fname} ({len(data)} items)")
        else:
            print(f"  ✗ {fname} NOT FOUND")
            all_ok = False
    
    return all_ok


def check_gpu():
    """Check GPU availability."""
    print("\nChecking GPU...")
    
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  ✓ GPU: {gpu_name}")
        print(f"  ✓ Memory: {gpu_mem:.1f} GB")
        return True
    else:
        print("  ✗ No GPU available (will use CPU - very slow!)")
        return False


def check_models():
    """Check if models are cached."""
    print("\nChecking model cache...")
    
    import os
    cache_dir = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    
    # Check for sentence transformers
    st_models = [
        "sentence-transformers--msmarco-MiniLM-L-6-v2",
        "WhereIsAI--UAE-Large-V1"
    ]
    
    for model in st_models:
        model_path = cache_dir / f"models--{model}"
        if model_path.exists():
            print(f"  ✓ {model}")
        else:
            # Try alternate path format
            alt_path = cache_dir / model.replace("--", "/")
            if alt_path.exists():
                print(f"  ✓ {model}")
            else:
                print(f"  ? {model} (may need to download)")
    
    # Check for LLM
    llm_model = "meta-llama--Llama-2-7b-hf"
    model_path = cache_dir / f"models--{llm_model}"
    if model_path.exists():
        print(f"  ✓ {llm_model}")
    else:
        print(f"  ? {llm_model} (may need to download)")
    
    return True


def quick_test():
    """Run a quick functional test."""
    print("\nRunning quick functional test...")
    
    try:
        # Test BM25
        from rank_bm25 import BM25Okapi
        corpus = [["hello", "world"], ["test", "document"]]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(["hello"])
        print("  ✓ BM25 works")
        
        # Test sentence transformer
        from sentence_transformers import SentenceTransformer
        # Don't actually load - just check import
        print("  ✓ SentenceTransformer importable")
        
        # Test local code
        from code2 import PromptUtils, get_query_span, query_to_docs_attention
        print("  ✓ code2.py imports work")
        
        from code3 import select_retrieval_heads, query_to_docs_attention_heads
        print("  ✓ code3.py imports work")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False


def main():
    print("="*60)
    print("CS728 PA3 - Setup Verification")
    print("="*60)
    
    results = []
    
    results.append(("Imports", check_imports()))
    results.append(("Data", check_data()))
    results.append(("GPU", check_gpu()))
    results.append(("Models", check_models()))
    results.append(("Quick Test", quick_test()))
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    all_pass = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed and name not in ["GPU", "Models"]:  # GPU/Models warnings are OK
            all_pass = False
    
    if all_pass:
        print("\n✓ All critical checks passed! Ready to run experiments.")
    else:
        print("\n✗ Some checks failed. Fix issues before running experiments.")
        sys.exit(1)


if __name__ == "__main__":
    main()
