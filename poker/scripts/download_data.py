"""
Data downloading script for PokerBench dataset from HuggingFace.

This script downloads the PokerBench dataset which contains:
- Preflop scenarios (60k train, 1k test)
- Postflop scenarios (500k train, 10k test)

Both JSON (prompts + labels) and CSV (structured data) formats.
"""

import os
import argparse
from pathlib import Path
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm


def download_pokerbench(output_dir: str = "data/raw/pokerbench"):
    """
    Download PokerBench dataset from HuggingFace.
    
    Args:
        output_dir: Directory to save downloaded data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Downloading PokerBench dataset from HuggingFace...")
    
    try:
        # Load the full dataset
        dataset = load_dataset("RZ412/PokerBench")
        
        print(f"Dataset loaded successfully!")
        print(f"Dataset structure: {dataset}")
        
        # Save each split
        for split_name, split_data in dataset.items():
            print(f"\nProcessing split: {split_name}")
            
            # Convert to pandas DataFrame
            df = split_data.to_pandas()
            
            # Save as CSV
            csv_path = output_path / f"{split_name}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved CSV: {csv_path} ({len(df)} rows)")
            
            # Save as JSON (preserving structure)
            json_path = output_path / f"{split_name}.json"
            df.to_json(json_path, orient='records', lines=True)
            print(f"Saved JSON: {json_path}")
            
            # Print sample
            if len(df) > 0:
                print(f"\nSample from {split_name}:")
                print(df.head(2))
                print(f"\nColumns: {list(df.columns)}")
        
        print(f"\n✓ Dataset downloaded successfully to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False


def download_from_github(output_dir: str = "data/raw/poker_transformers"):
    """
    Instructions for downloading Poker_Transformers dataset from GitHub.
    
    Args:
        output_dir: Directory to clone the repository
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\nPoker_Transformers GitHub Repository:")
    print("To download this dataset, clone the repository:")
    print(f"  git clone https://github.com/SoelMgd/Poker_Transformers {output_dir}")
    print("\nOr download specific files manually from:")
    print("  https://github.com/SoelMgd/Poker_Transformers")


def create_data_placeholders(base_dir: str = "data"):
    """Create placeholder .gitkeep files in data directories."""
    base_path = Path(base_dir)
    
    for subdir in ["raw", "processed", "models"]:
        dir_path = base_path / subdir
        dir_path.mkdir(parents=True, exist_ok=True)
        gitkeep = dir_path / ".gitkeep"
        gitkeep.touch()
    
    print(f"✓ Created data directory structure in {base_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download poker datasets for training prediction models"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Base directory for downloaded data"
    )
    parser.add_argument(
        "--skip-pokerbench",
        action="store_true",
        help="Skip downloading PokerBench dataset"
    )
    
    args = parser.parse_args()
    
    # Create directory structure
    create_data_placeholders("data")
    
    # Download PokerBench
    if not args.skip_pokerbench:
        pokerbench_dir = os.path.join(args.output_dir, "pokerbench")
        success = download_pokerbench(pokerbench_dir)
        if not success:
            print("Warning: PokerBench download failed. Check your internet connection.")
    
    # GitHub instructions
    github_dir = os.path.join(args.output_dir, "poker_transformers")
    download_from_github(github_dir)
    
    print("\n" + "="*60)
    print("Data download process completed!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. Check downloaded data in: {args.output_dir}")
    print(f"2. Run preprocessing: python src/data/preprocess.py")
    print(f"3. Explore data: jupyter notebook notebooks/01_data_exploration.ipynb")


if __name__ == "__main__":
    main()
