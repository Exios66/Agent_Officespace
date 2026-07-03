# Poker Prediction Algorithm - Getting Started

This guide will walk you through setting up and training your first poker prediction model.

## Installation

1. **Clone the repository** (if applicable):
```bash
cd poker
```

2. **Install dependencies**:

The old default installs the full stack:

```bash
pip install -r requirements.txt
```

Or pick a feature-tailored layer (see
[`../requirements/README.md`](../requirements/README.md) for the full
menu):

```bash
pip install -r requirements/ml.txt   # XGBoost / LightGBM path
pip install -r requirements/nn.txt   # PyTorch MLP / LSTM path
pip install -r requirements/llm.txt  # LoRA fine-tuning path
```

## Quick Start

### Step 1: Download Data

Download the PokerBench dataset from HuggingFace:

```bash
python scripts/download_data.py
```

This will download:
- Preflop data: 60k training samples, 1k test samples
- Postflop data: 500k training samples, 10k test samples

Data will be saved to `data/raw/pokerbench/`

### Step 2: Preprocess Data

Parse and clean the raw data:

```bash
python src/data/preprocess.py --raw-dir data/raw/pokerbench --output-dir data/processed
```

This creates:
- `data/processed/train_processed.parquet`
- `data/processed/test_processed.parquet`

### Step 3: Feature Engineering

Extract poker-specific features:

```bash
python src/features/engineering.py --input-dir data/processed --output-dir data/processed
```

This creates:
- `data/processed/train_features.parquet`
- `data/processed/test_features.parquet`

### Step 4: Train a Model

#### Option A: Traditional ML (XGBoost)

```bash
python src/models/train_ml.py \
    --data-dir data/processed \
    --output-dir data/models \
    --model-type xgboost \
    --val-split 0.2
```

#### Option B: Neural Network (MLP)

```bash
python src/models/train_nn.py \
    --data-dir data/processed \
    --output-dir data/models \
    --model-type mlp \
    --batch-size 256 \
    --epochs 50 \
    --lr 0.001
```

#### Option C: Fine-tune LLM

First, prepare the data in instruction format:

```bash
python src/llm/train_llm.py \
    --data-dir data/processed \
    --prepare-only
```

Then train (requires GPU):

```bash
python src/llm/train_llm.py \
    --data-dir data/processed \
    --output-dir data/models/llm_poker \
    --model-name mistralai/Mistral-7B-v0.1 \
    --use-lora \
    --epochs 3 \
    --batch-size 4
```

### Step 5: Evaluate

Evaluate your trained model:

```bash
# For ML models
python src/evaluation/evaluate.py \
    --model-path data/models/xgboost_model.pkl \
    --model-type ml \
    --data-path data/processed/test_features.parquet \
    --output-dir data/evaluation

# For NN models
python src/evaluation/evaluate.py \
    --model-path data/models/mlp_model.pth \
    --model-type nn \
    --data-path data/processed/test_features.parquet \
    --output-dir data/evaluation
```

## Project Structure

```
poker/
├── data/
│   ├── raw/              # Original datasets
│   ├── processed/        # Processed data
│   └── models/           # Trained models
├── src/
│   ├── data/             # Data loading and preprocessing
│   ├── features/         # Feature engineering
│   ├── models/           # Model training (ML & NN)
│   ├── llm/              # LLM fine-tuning
│   └── evaluation/       # Evaluation and inference
├── configs/              # Configuration files
├── scripts/              # Utility scripts
└── notebooks/            # Jupyter notebooks
```

## Next Steps

1. **Explore the data**: Check out the notebooks for data exploration
2. **Experiment with features**: Modify `src/features/engineering.py` to add new features
3. **Tune hyperparameters**: Edit config files in `configs/`
4. **Compare models**: Train multiple models and compare results
5. **Add postflop data**: Extend to postflop decision-making

## Common Issues

### Out of Memory

If you encounter OOM errors:
- Reduce batch size
- Use gradient accumulation
- Enable 8-bit quantization for LLMs

### Slow Training

- Use GPU if available
- Reduce dataset size for testing
- Use LoRA for efficient LLM training

### Missing Dependencies

Make sure all packages are installed:
```bash
pip install -r requirements.txt --upgrade
```

## Resources

- [PokerBench Dataset](https://huggingface.co/datasets/RZ412/PokerBench)
- [Poker Hand Evaluation](https://en.wikipedia.org/wiki/Poker_probability)
- [Game Theory Optimal Poker](https://en.wikipedia.org/wiki/Game_theory#Poker)

## Contact

For questions or issues, please open an issue on the repository.
