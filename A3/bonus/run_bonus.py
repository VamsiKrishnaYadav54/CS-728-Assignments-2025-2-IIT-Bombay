#!/usr/bin/env python3
"""
Bonus: Test different numbers of selected heads (K = 10, 20, 30)
"""

import json
from pathlib import Path
from run3 import main as run_part3

def run_bonus_experiments():
    """Run Part 3 with different K values."""
    
    print("=" * 60)
    print("BONUS: Testing different K values")
    print("=" * 60)
    
    k_values = [10, 20, 30]
    all_results = {}
    
    for k in k_values:
        print(f"\n{'=' * 60}")
        print(f"Running with K = {k}")
        print("=" * 60)
        
        results = run_part3(max_heads=k)
        all_results[k] = results
    
    # Summary table
    print("\n" + "=" * 60)
    print("BONUS: Summary - Effect of K on Recall")
    print("=" * 60)
    print(f"{'K':<10} {'Recall@1':<15} {'Recall@5':<15}")
    print("-" * 40)
    for k in k_values:
        r1 = all_results[k]["Recall@1"]
        r5 = all_results[k]["Recall@5"]
        print(f"{k:<10} {r1:<15.4f} {r5:<15.4f}")
    
    # Save combined results
    with open("results/bonus_k_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\nBonus results saved to results/bonus_k_comparison.json")


if __name__ == "__main__":
    run_bonus_experiments()
