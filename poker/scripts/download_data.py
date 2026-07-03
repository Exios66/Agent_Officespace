"""
Data downloading script for PokerBench dataset from HuggingFace.

Downloads structured preflop/postflop CSV files from the PokerBench repository.
The default HuggingFace dataset split only contains instruction/output pairs and
is not suitable for the structured ML pipeline without these CSV files.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


PREFLOP_FILES = {
    "train": "preflop_60k_train_set_game_scenario_information.csv",
    "test": "preflop_1k_test_set_game_scenario_information.csv",
}

POSTFLOP_FILES = {
    "train": "postflop_500k_train_set_game_scenario_information.csv",
    "test": "postflop_10k_test_set_game_scenario_information.csv",
}


def download_pokerbench(
    output_dir: str = "data/raw/pokerbench",
    include_postflop: bool = False,
) -> bool:
    """
    Download structured PokerBench CSV files from HuggingFace.

    Args:
        output_dir: Directory to save downloaded data.
        include_postflop: Whether to also download postflop CSV files.

    Returns:
        True if preflop files were downloaded successfully.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Downloading structured PokerBench CSV files from HuggingFace...")

    try:
        for split, filename in PREFLOP_FILES.items():
            local_path = hf_hub_download(
                repo_id="RZ412/PokerBench",
                filename=filename,
                repo_type="dataset",
            )
            df = pd.read_csv(local_path)
            if "Unnamed: 0" in df.columns:
                df = df.drop(columns=["Unnamed: 0"])

            csv_path = output_path / f"{split}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved preflop {split}: {csv_path} ({len(df)} rows)")
            print(f"  Columns: {list(df.columns)}")

        if include_postflop:
            postflop_dir = output_path / "postflop"
            postflop_dir.mkdir(parents=True, exist_ok=True)
            for split, filename in POSTFLOP_FILES.items():
                local_path = hf_hub_download(
                    repo_id="RZ412/PokerBench",
                    filename=filename,
                    repo_type="dataset",
                )
                df = pd.read_csv(local_path)
                if "Unnamed: 0" in df.columns:
                    df = df.drop(columns=["Unnamed: 0"])

                csv_path = postflop_dir / f"{split}.csv"
                df.to_csv(csv_path, index=False)
                print(f"Saved postflop {split}: {csv_path} ({len(df)} rows)")

        print(f"\n✓ Structured PokerBench data saved to {output_path}")
        return True

    except Exception as exc:
        print(f"Error downloading dataset: {exc}")
        return False


def download_instruction_pairs(output_dir: str = "data/raw/pokerbench/prompts") -> bool:
    """Download instruction/output JSON files for LLM fine-tuning."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    prompt_files = {
        "train": "preflop_60k_train_set_prompt_and_label.json",
        "test": "preflop_1k_test_set_prompt_and_label.json",
    }

    try:
        for split, filename in prompt_files.items():
            local_path = hf_hub_download(
                repo_id="RZ412/PokerBench",
                filename=filename,
                repo_type="dataset",
            )
            destination = output_path / f"{split}.json"
            destination.write_bytes(Path(local_path).read_bytes())
            print(f"Saved prompt data: {destination}")
        return True
    except Exception as exc:
        print(f"Warning: Could not download prompt files: {exc}")
        return False


def download_from_github(output_dir: str = "data/raw/poker_transformers"):
    """Print instructions for the Poker_Transformers GitHub repository."""
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

    for subdir in ["raw", "processed", "models", "evaluation"]:
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
        help="Base directory for downloaded data",
    )
    parser.add_argument(
        "--skip-pokerbench",
        action="store_true",
        help="Skip downloading PokerBench dataset",
    )
    parser.add_argument(
        "--include-postflop",
        action="store_true",
        help="Also download postflop structured CSV files",
    )
    parser.add_argument(
        "--include-prompts",
        action="store_true",
        help="Also download instruction/output JSON prompt files",
    )

    args = parser.parse_args()

    create_data_placeholders("data")

    if not args.skip_pokerbench:
        pokerbench_dir = os.path.join(args.output_dir, "pokerbench")
        success = download_pokerbench(
            pokerbench_dir,
            include_postflop=args.include_postflop,
        )
        if not success:
            print("Warning: PokerBench download failed. Check your internet connection.")

        if args.include_prompts:
            prompts_dir = os.path.join(pokerbench_dir, "prompts")
            download_instruction_pairs(prompts_dir)

    download_from_github(os.path.join(args.output_dir, "poker_transformers"))

    print("\n" + "=" * 60)
    print("Data download process completed!")
    print("=" * 60)
    print("\nNext steps:")
    print(f"1. Check downloaded data in: {args.output_dir}/pokerbench")
    print("2. Run preprocessing: python src/data/preprocess.py")
    print("3. Explore data: jupyter notebook notebooks/01_quickstart.ipynb")


if __name__ == "__main__":
    main()
