# Poker Prediction Algorithm

A comprehensive poker prediction system for No-Limit Texas Hold'em, focusing on preflop decision-making based on game theory optimal (GTO) solver data.

## Project Overview

This project aims to build predictive models for poker decision-making using:

- **Traditional ML approaches**: Random Forests, XGBoost, Neural Networks
- **Deep Learning**: Custom architectures for poker state representation
- **LLM Fine-tuning**: Transformer-based models trained on poker scenarios

### Key Features

- **Preflop Prediction**: Predict optimal actions based on:
  - Hero's hole cards
  - Stack sizes (effective stacks, pot odds)
  - Table position (UTG, HJ, CO, BTN, SB, BB)
  - Previous action sequence
  - Number of players in hand
  - Pot size and bet sizing

- **Feature Engineering**: Advanced poker-specific features including:
  - Hand strength calculations
  - Position relative strength
  - Pot odds and implied odds
  - Action sequence embeddings
  - Stack-to-pot ratio (SPR)

- **Multiple Model Types**:
  - Gradient Boosting (XGBoost/LightGBM)
  - Neural Networks (MLP, LSTM for action sequences)
  - Transformer-based LLMs (fine-tuned)

## Data Sources

### Primary Dataset: PokerBench

- **Source**: [RZ412/PokerBench](https://huggingface.co/datasets/RZ412/PokerBench)
- **Size**:
  - Preflop: 60k training samples, 1k test samples
  - Postflop: 500k training samples, 10k test samples
- **Format**: JSON (prompts + labels) and CSV (structured game data)
- **Quality**: Solver-generated optimal decisions (GTO play)

### Secondary Dataset: Poker Transformers

- **Source**: [SoelMgd/Poker_Transformers](https://github.com/SoelMgd/Poker_Transformers)
- **Focus**: LLM training for poker scenarios

## Project Structure

```
poker/
├── data/
│   ├── raw/              # Original downloaded datasets
│   ├── processed/        # Processed and feature-engineered data
│   └── models/           # Trained model checkpoints
├── src/
│   ├── data/             # Data downloading and preprocessing
│   ├── features/         # Feature engineering modules
│   ├── models/           # Model architectures and training
│   ├── evaluation/       # Evaluation metrics and utilities
│   └── llm/              # LLM fine-tuning scripts
├── notebooks/            # Jupyter notebooks for exploration
├── configs/              # Configuration files
├── tests/                # Unit and integration tests
├── scripts/              # Utility scripts
└── README.md             # This file
```

## Getting Started

### Installation

Old default (installs the full stack — every training path, every
notebook dependency, every tracker):

```bash
pip install -r requirements.txt
```

Feature-tailored (recommended). Pick only what you need — full menu in
[`requirements/README.md`](requirements/README.md):

```bash
pip install -r requirements/base.txt      # preprocess + features only
pip install -r requirements/ml.txt        # + XGBoost / LightGBM
pip install -r requirements/nn.txt        # + PyTorch MLP / LSTM
pip install -r requirements/llm.txt       # + LoRA fine-tuning
pip install -r requirements/viz.txt       # + matplotlib / seaborn / plotly
pip install -r requirements/tracking.txt  # + wandb / tensorboard
pip install -r requirements/dev.txt       # + jupyter / pytest / black
pip install -r requirements/all.txt       # everything (same as requirements.txt)
```

### Download Data

```bash
python scripts/download_data.py
```

### Train Models

```bash
# Traditional ML model
python src/models/train_ml.py --config configs/xgboost_config.yaml

# Neural network model
python src/models/train_nn.py --config configs/nn_config.yaml

# Fine-tune LLM
python src/llm/train_llm.py --config configs/llm_config.yaml
```

### Evaluate

```bash
python src/evaluation/evaluate.py --model_path data/models/best_model.pkl
```

## Model Performance

Measured by [`notebooks/02_prediction_success_evaluation.ipynb`](notebooks/02_prediction_success_evaluation.ipynb)
against the 1 000-hand PokerBench preflop test split (canonicalised to
`{fold, check, call, raise}`; 4-class balanced 250 rows/class):

| Model | Test Accuracy | Macro-F1 | Log-Loss | Top-2 | Fit (s) |
|-------|--------------:|---------:|---------:|------:|--------:|
| LightGBM        | **0.969** | 0.969 | 0.085 | 1.000 | 3.2 |
| HistGradientBoosting | 0.961 | 0.961 | 0.104 | 1.000 | 3.4 |
| XGBoost         | 0.960 | 0.960 | 0.100 | 0.999 | 1.9 |
| RandomForest    | 0.925 | 0.925 | 0.196 | 0.999 | 1.0 |
| MLP (sklearn)   | 0.922 | 0.922 | 0.192 | 0.997 | 2.5 |
| LogisticRegression | 0.862 | 0.862 | 0.334 | 0.993 | 4.1 |

Reproduce with:

```bash
jupyter nbconvert --to notebook --execute notebooks/02_prediction_success_evaluation.ipynb \
    --output 02_prediction_success_evaluation.ipynb
```

Results are persisted to `data/evaluation/multi_algo_results.json`.

## Notebooks

| notebook | what it does |
|----------|--------------|
| [`notebooks/01_quickstart.ipynb`](notebooks/01_quickstart.ipynb) | Auto-downloads PokerBench, runs preprocess → feature engineering → XGBoost training → evaluation → feature importance → single-hand prediction → model save. Doubles as an integration test for `scripts/run_pipeline.py`. |
| [`notebooks/02_prediction_success_evaluation.ipynb`](notebooks/02_prediction_success_evaluation.ipynb) | Head-to-head **prediction-of-success evaluation** across 6 algorithms (logistic / random forest / hist-gradient-boosting / XGBoost / LightGBM / sklearn MLP) on identical features. Reports the leaderboard above plus per-class F1/recall tables, confusion matrices, and a shared calibration curve. |

## Future Enhancements

- [ ] Postflop decision-making
- [ ] Multi-street planning
- [ ] Opponent modeling
- [ ] Real-time inference API
- [ ] Integration with poker hand history parsers
- [ ] Exploitative play adjustments
- [ ] Multi-table tournament (MTT) adaptations
- [ ] Cash game vs tournament strategy differentiation

## Contributing

This is a private research project. For questions or suggestions, please contact the project maintainer.

## License

Private - All Rights Reserved

## Acknowledgments

- PokerBench dataset creators
- GTO solver community
- Hugging Face for dataset hosting
