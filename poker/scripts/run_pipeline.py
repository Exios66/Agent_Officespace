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


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_command(cmd: list, description: str) -> bool:
    """Run a command and handle errors."""
    print(f"\n{'=' * 60}")
    print(description)
    print('=' * 60)

    try:
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"✗ {description} failed with error code {exc.returncode}")
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
        help="Model type to train",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip data download step",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocessing step",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip feature engineering step",
    )

    args = parser.parse_args()
    success = True

    if not args.skip_download:
        success = run_command(
            [sys.executable, "scripts/download_data.py"],
            "Step 1: Downloading data",
        ) and success

    if not args.skip_preprocess and success:
        success = run_command(
            [
                sys.executable,
                "src/data/preprocess.py",
                "--raw-dir",
                "data/raw/pokerbench",
                "--output-dir",
                "data/processed",
            ],
            "Step 2: Preprocessing data",
        ) and success

    if not args.skip_features and success:
        success = run_command(
            [
                sys.executable,
                "src/features/engineering.py",
                "--input-dir",
                "data/processed",
                "--output-dir",
                "data/processed",
            ],
            "Step 3: Engineering features",
        ) and success

    if success:
        if args.model_type in ['xgboost', 'lightgbm', 'random_forest']:
            success = run_command(
                [
                    sys.executable,
                    "src/models/train_ml.py",
                    "--data-dir",
                    "data/processed",
                    "--output-dir",
                    "data/models",
                    "--model-type",
                    args.model_type,
                ],
                f"Step 4: Training {args.model_type} model",
            ) and success
        elif args.model_type in ['mlp', 'lstm']:
            success = run_command(
                [
                    sys.executable,
                    "src/models/train_nn.py",
                    "--data-dir",
                    "data/processed",
                    "--output-dir",
                    "data/models",
                    "--model-type",
                    args.model_type,
                    "--epochs",
                    "5",
                ],
                f"Step 4: Training {args.model_type} model",
            ) and success

    if success:
        model_ext = (
            "pkl"
            if args.model_type in ['xgboost', 'lightgbm', 'random_forest']
            else "pth"
        )
        model_type = (
            "ml"
            if args.model_type in ['xgboost', 'lightgbm', 'random_forest']
            else "nn"
        )

        success = run_command(
            [
                sys.executable,
                "src/evaluation/evaluate.py",
                "--model-path",
                f"data/models/{args.model_type}_model.{model_ext}",
                "--model-type",
                model_type,
                "--data-path",
                "data/processed/test_features.parquet",
                "--output-dir",
                "data/evaluation",
            ],
            "Step 5: Evaluating model",
        ) and success

    print(f"\n{'=' * 60}")
    if success:
        print("✓ Pipeline completed successfully!")
        print(f"\nModel saved to: data/models/{args.model_type}_model.*")
        print("Results saved to: data/evaluation/")
    else:
        print("✗ Pipeline failed at one or more steps")
        sys.exit(1)
    print('=' * 60)


if __name__ == "__main__":
    main()
