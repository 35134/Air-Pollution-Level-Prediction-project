"""
Simple verification script for the cleaned project
"""

import os
import sys

def main():
    print("=== CLEAN PROJECT VERIFICATION ===")
    
    # Check essential files
    essential_files = [
        "main.py",
        "requirements.txt", 
        "src/config.py",
        "src/models.py",
        "src/__init__.py",
        "data/metadata.json",
        "data/cleaned/train_cleaned.csv",
        "data/cleaned/val_cleaned.csv", 
        "data/cleaned/test_cleaned.csv",
        "data/cleaned/train_engineered_optimized.csv",
        "data/cleaned/val_engineered_optimized.csv",
        "data/cleaned/test_engineered_optimized.csv",
        "results/baseline_model.pkl",
        "results/optimized_model.pkl",
        "results/model_comparison.json"
    ]
    
    all_present = True
    for filepath in essential_files:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"OK {filepath}: {size:,} bytes")
        else:
            print(f"MISSING {filepath}: MISSING")
            all_present = False
    
    print(f"\nSummary: {sum(1 for f in essential_files if os.path.exists(f))}/{len(essential_files)} files present")
    
    if all_present:
        print("\nCLEAN PROJECT VERIFICATION: SUCCESS!")
        print("The project is ready for execution!")
        print("\nTo run: python main.py")
        return True
    else:
        print("\nCLEAN PROJECT VERIFICATION: FAILED!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
# Final Project Verification

# Final Project Verification Complete

