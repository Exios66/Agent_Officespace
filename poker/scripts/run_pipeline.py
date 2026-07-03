#!/usr/bin/env python3
"""
Complete pipeline script: Download -> Preprocess -> Engineer -> Train -> Evaluate

Usage:
    python scripts/run_pipeline.py --model-type xgboost
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with error code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run complete poker prediction pipeline"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="xgboost",
        choices=['xgboost', 'lightgbm', 'random_forest', 'mlp', 'lstm'],
        help="Model type to train"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip data download step"
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocessing step"
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip feature engineering step"
    )
    
    args = parser.parse_args()
    
    # Track success
    success = True
    
    # Step 1: Download data
    if not args.skip_download:
        success = run_command(
            ["python", "scripts/download_data.py"],
            "Step 1: Downloading data"
        ) and success
    
    # Step 2: Preprocess
    if not args.skip_preprocess and success:
        success = run_command(
            ["python", "src/data/preprocess.py",
             "--raw-dir", "data/raw/pokerbench",
             "--output-dir", "data/processed"],
            "Step 2: Preprocessing data"
        ) and success
    
    # Step 3: Feature engineering
    if not args.skip_features and success:
        success = run_command(
            ["python", "src/features/engineering.py",
             "--input-dir", "data/processed",
             "--output-dir", "data/processed"],
            "Step 3: Engineering features"
        ) and success
    
    # Step 4: Train model
    if success:
        if args.model_type in ['xgboost', 'lightgbm', 'random_forest']:
            success = run_command(
                ["python", "src/models/train_ml.py",
                 "--data-dir", "data/processed",
                 "--output-dir", "data/models",
                 "--model-type", args.model_type],
                f"Step 4: Training {args.model_type} model"
            ) and success
        elif args.model_type in ['mlp', 'lstm']:
            success = run_command(
                ["python", "src/models/train_nn.py",
                 "--data-dir", "data/processed",
                 "--output-dir", "data/models",
                 "--model-type", args.model_type],
                f"Step 4: Training {args.model_type} model"
            ) and success
    
    # Step 5: Evaluate
    if success:
        model_ext = "pkl" if args.model_type in ['xgboost', 'lightgbm', 'random_forest'] else "pth"
        model_type = "ml" if args.model_type in ['xgboost', 'lightgbm', 'random_forest'] else "nn"
        
        success = run_command(
            ["python", "src/evaluation/evaluate.py",
             "--model-path", f"data/models/{args.model_type}_model.{model_ext}",
             "--model-type", model_type,
             "--data-path", "data/processed/test_features.parquet",
             "--output-dir", "data/evaluation"],
            "Step 5: Evaluating model"
        ) and success
    
    # Summary
    print(f"\n{'='*60}")
    if success:
        print("✓ Pipeline completed successfully!")
        print(f"\nModel saved to: data/models/{args.model_type}_model.*")
        print(f"Results saved to: data/evaluation/")
    else:
        print("✗ Pipeline failed at one or more steps")
        sys.exit(1)
    print('='*60)


if __name__ == "__main__":
    main()
