#!/usr/bin/env python3
"""
CS728 PA3 - Master Runner

This script can run all parts of the assignment sequentially or individually.
Usage:
    python run_all.py --part 1      # Run only Part 1
    python run_all.py --part 2      # Run only Part 2
    python run_all.py --part 3      # Run only Part 3
    python run_all.py --part all    # Run all parts
    python run_all.py --part bonus  # Run bonus experiments
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_part1():
    """Run Part 1: Classical Retrieval"""
    print("\n" + "="*70)
    print("RUNNING PART 1: Classical Retrieval")
    print("="*70 + "\n")
    
    result = subprocess.run([sys.executable, "part1_retrieval.py"], 
                          capture_output=False)
    return result.returncode == 0


def run_part2():
    """Run Part 2: Attention-Based Retrieval"""
    print("\n" + "="*70)
    print("RUNNING PART 2: Attention-Based Retrieval")
    print("="*70 + "\n")
    
    result = subprocess.run([sys.executable, "run2.py"], 
                          capture_output=False)
    return result.returncode == 0


def run_part3(max_heads=20):
    """Run Part 3: Retrieval Heads"""
    print("\n" + "="*70)
    print(f"RUNNING PART 3: Retrieval Heads (K={max_heads})")
    print("="*70 + "\n")
    
    result = subprocess.run([sys.executable, "run3.py", 
                           f"--max_heads={max_heads}"], 
                          capture_output=False)
    return result.returncode == 0


def run_bonus():
    """Run Bonus: Different K values"""
    print("\n" + "="*70)
    print("RUNNING BONUS: K Comparison (10, 20, 30)")
    print("="*70 + "\n")
    
    result = subprocess.run([sys.executable, "run_bonus.py"], 
                          capture_output=False)
    return result.returncode == 0


def generate_summary():
    """Generate final summary"""
    print("\n" + "="*70)
    print("GENERATING SUMMARY")
    print("="*70 + "\n")
    
    result = subprocess.run([sys.executable, "generate_summary.py"], 
                          capture_output=False)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="CS728 PA3 Runner")
    parser.add_argument("--part", type=str, default="all",
                       choices=["1", "2", "3", "all", "bonus", "summary"],
                       help="Which part to run")
    parser.add_argument("--max_heads", type=int, default=20,
                       help="Number of heads for Part 3 (default: 20)")
    args = parser.parse_args()
    
    # Create necessary directories
    Path("results").mkdir(exist_ok=True)
    Path("plot2").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    success = True
    
    if args.part == "1":
        success = run_part1()
    elif args.part == "2":
        success = run_part2()
    elif args.part == "3":
        success = run_part3(args.max_heads)
    elif args.part == "bonus":
        success = run_bonus()
    elif args.part == "summary":
        success = generate_summary()
    elif args.part == "all":
        # Run all parts sequentially
        success = run_part1()
        if success:
            success = run_part2()
        if success:
            success = run_part3(args.max_heads)
        if success:
            success = generate_summary()
    
    if success:
        print("\n✓ Completed successfully!")
    else:
        print("\n✗ Failed! Check logs for errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
