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

```bash
pip install -r requirements.txt
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

| Model Type | Test Accuracy | Notes |
|------------|---------------|-------|
| XGBoost | TBD | Baseline model |
| Neural Network | TBD | Sequential action processing |
| Fine-tuned LLM | TBD | Natural language understanding |

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
