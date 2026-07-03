"""
Data downloading script for the PokerBench dataset on Hugging Face Hub.

The Hub repo (``RZ412/PokerBench``) ships two coordinated file formats for each
split:

* ``preflop_*_game_scenario_information.csv`` — structured columns
  (``hero_pos``, ``hero_holding``, ``prev_line``, ``num_players``,
  ``num_bets``, ``available_moves``, ``pot_size``, ``correct_decision``).
  Consumed by the classical ML pipeline (`preprocess.py` -> `engineering.py`
  -> `train_ml.py`).
* ``preflop_*_prompt_and_label.json`` — LLM prompt/response pairs consumed by
  the LLM fine-tune track.

We download both. The structured CSVs are additionally aliased to
``train.csv`` / ``test.csv`` (the filenames the preprocessor / notebooks
expect) so downstream code has a stable entry point.

Previously this script used ``datasets.load_dataset('RZ412/PokerBench')``,
which surfaces only the LLM prompt/label view and therefore left the
downstream classical pipeline unable to find its expected columns. Using
``hf_hub_download`` avoids that mismatch.
"""

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


POKERBENCH_REPO = "RZ412/PokerBench"

# Structured CSVs — required by the classical / notebook pipeline.
PREFLOP_CSV_FILES = {
    "train": "preflop_60k_train_set_game_scenario_information.csv",
    "test": "preflop_1k_test_set_game_scenario_information.csv",
}

# LLM prompt / label JSONs — required by the LLM fine-tune track.
PREFLOP_JSON_FILES = {
    "train": "preflop_60k_train_set_prompt_and_label.json",
    "test": "preflop_1k_test_set_prompt_and_label.json",
}


def _copy_to(src: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def download_pokerbench(output_dir: str = "data/raw/pokerbench") -> bool:
    """Download PokerBench preflop CSV + JSON files and expose the CSVs as
    ``train.csv`` / ``test.csv`` for the downstream preprocessor."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Downloading PokerBench preflop CSV + JSON files from Hugging Face Hub...")

    try:
        for split, filename in PREFLOP_CSV_FILES.items():
            local_path = hf_hub_download(
                repo_id=POKERBENCH_REPO,
                filename=filename,
                repo_type="dataset",
            )

            csv_dst = output_path / filename
            _copy_to(local_path, csv_dst)

            # Alias to the simple `train.csv` / `test.csv` name expected by
            # `src/data/preprocess.py` and the notebooks. Some historical CSV
            # dumps carry an unnamed pandas index column, so we strip it here.
            df = pd.read_csv(csv_dst)
            df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
            alias_dst = output_path / f"{split}.csv"
            df.to_csv(alias_dst, index=False)

            print(f"  {split}: {csv_dst.name} -> {alias_dst.name} ({len(df)} rows)")
            print(f"    columns: {list(df.columns)}")

        for split, filename in PREFLOP_JSON_FILES.items():
            local_path = hf_hub_download(
                repo_id=POKERBENCH_REPO,
                filename=filename,
                repo_type="dataset",
            )
            json_dst = output_path / filename
            _copy_to(local_path, json_dst)
            print(f"  {split} (LLM JSON): {json_dst.name}")

        print(f"\n\u2713 PokerBench downloaded to {output_path}")
        return True

    except Exception as e:  # pragma: no cover - network / auth issues surface here
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
