#!/usr/bin/env python3
"""
Generate a summary of all results for the report.
Run after all parts are complete.
"""

import json
from pathlib import Path

def load_json(path):
    """Load JSON file if exists."""
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return None

def main():
    print("=" * 70)
    print("CS728 PA3 - Results Summary for Report")
    print("=" * 70)
    
    results_dir = Path("results")
    
    # Part 1
    print("\n" + "=" * 70)
    print("PART 1: Classical Retrieval")
    print("=" * 70)
    
    part1 = load_json(results_dir / "part1_results.json")
    if part1:
        print(f"\n{'Method':<20} {'Recall@1':<12} {'Recall@5':<12}")
        print("-" * 44)
        for method, scores in part1.items():
            print(f"{method:<20} {scores['Recall@1']:<12.4f} {scores['Recall@5']:<12.4f}")
    else:
        print("Part 1 results not found!")
    
    # Part 2
    print("\n" + "=" * 70)
    print("PART 2: Attention-Based Retrieval")
    print("=" * 70)
    
    part2 = load_json(results_dir / "part2_results.json")
    if part2:
        print(f"\nRecall@1: {part2['Recall@1']:.4f}")
        print(f"Recall@5: {part2['Recall@5']:.4f}")
        
        if Path("plot2/gold_attention_plot.png").exists():
            print("\n✓ Position analysis plot: plot2/gold_attention_plot.png")
        else:
            print("\n✗ Position analysis plot NOT FOUND")
    else:
        print("Part 2 results not found!")
    
    # Part 3
    print("\n" + "=" * 70)
    print("PART 3: Retrieval Heads")
    print("=" * 70)
    
    part3 = load_json(results_dir / "part3_results_k20.json")
    heads_info = load_json(results_dir / "part3_selected_heads_k20.json")
    
    if part3:
        print(f"\nRecall@1: {part3['Recall@1']:.4f}")
        print(f"Recall@5: {part3['Recall@5']:.4f}")
        
        if heads_info and 'selected_heads' in heads_info:
            print(f"\nSelected {len(heads_info['selected_heads'])} heads:")
            for i, (layer, head) in enumerate(heads_info['selected_heads'][:10]):
                print(f"  {i+1}. Layer {layer}, Head {head}")
            if len(heads_info['selected_heads']) > 10:
                print(f"  ... and {len(heads_info['selected_heads']) - 10} more")
            
            if 'selection_strategy' in heads_info:
                print(f"\nSelection Strategy: {heads_info['selection_strategy']}")
    else:
        print("Part 3 results not found!")
    
    # Comparison Table
    print("\n" + "=" * 70)
    print("COMPARISON: All Methods")
    print("=" * 70)
    
    print(f"\n{'Method':<30} {'Recall@1':<12} {'Recall@5':<12}")
    print("-" * 54)
    
    if part1:
        for method, scores in part1.items():
            print(f"{method:<30} {scores['Recall@1']:<12.4f} {scores['Recall@5']:<12.4f}")
    
    if part2:
        print(f"{'Attention (All Heads)':<30} {part2['Recall@1']:<12.4f} {part2['Recall@5']:<12.4f}")
    
    if part3:
        print(f"{'Attention (Selected Heads)':<30} {part3['Recall@1']:<12.4f} {part3['Recall@5']:<12.4f}")
    
    # Bonus
    bonus = load_json(results_dir / "bonus_k_comparison.json")
    if bonus:
        print("\n" + "=" * 70)
        print("BONUS: Effect of K on Retrieval Heads")
        print("=" * 70)
        
        print(f"\n{'K':<10} {'Recall@1':<12} {'Recall@5':<12}")
        print("-" * 34)
        for k, results in sorted(bonus.items(), key=lambda x: int(x[0])):
            print(f"{k:<10} {results['Recall@1']:<12.4f} {results['Recall@5']:<12.4f}")
    
    print("\n" + "=" * 70)
    print("END OF SUMMARY")
    print("=" * 70)


if __name__ == "__main__":
    main()
