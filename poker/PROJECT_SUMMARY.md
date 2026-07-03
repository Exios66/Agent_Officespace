# Poker Prediction Algorithm Project Summary

## Overview

This project provides a comprehensive framework for building poker prediction algorithms using machine learning and deep learning techniques. The system focuses on preflop decision-making in No-Limit Texas Hold'em, with support for both traditional ML models and LLM fine-tuning.

## Project Structure

```
poker/
├── data/
│   ├── raw/              # Original datasets from HuggingFace
│   ├── processed/        # Preprocessed and feature-engineered data
│   └── models/           # Trained model checkpoints
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocess.py           # Data preprocessing pipeline
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineering.py          # Poker-specific feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_ml.py             # Traditional ML training (XGBoost, etc.)
│   │   └── train_nn.py             # Neural network training (MLP, LSTM)
│   ├── llm/
│   │   ├── __init__.py
│   │   └── train_llm.py            # LLM fine-tuning (LoRA, PEFT)
│   └── evaluation/
│       ├── __init__.py
│       └── evaluate.py             # Model evaluation and inference
├── configs/
│   ├── xgboost_config.yaml        # XGBoost configuration
│   ├── nn_config.yaml             # Neural network configuration
│   └── llm_config.yaml            # LLM fine-tuning configuration
├── scripts/
│   ├── download_data.py           # Download PokerBench dataset
│   └── run_pipeline.py            # End-to-end pipeline script
├── notebooks/
│   └── 01_quickstart.ipynb        # Quick start tutorial
├── docs/
│   ├── GETTING_STARTED.md         # Installation and setup guide
│   ├── USAGE.md                   # Advanced usage patterns
│   └── ROADMAP.md                 # Future development plans
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # Project overview
```

## Key Features

### 1. Data Processing
- **Automated downloading** from HuggingFace (PokerBench dataset)
- **Preprocessing pipeline** for parsing poker-specific fields
- **Feature engineering** with advanced poker knowledge:
  - Hand strength evaluation (Sklansky-Chubukov rankings)
  - Position-based features
  - Stack-to-pot ratio (SPR) calculations
  - Pot odds and implied odds
  - Action sequence encoding
  - Aggression factors

### 2. Multiple Model Types

#### Traditional ML
- XGBoost (gradient boosting)
- LightGBM (gradient boosting)
- Random Forest
- Logistic Regression (baseline)

#### Neural Networks
- Multi-Layer Perceptron (MLP)
- LSTM for sequential action modeling
- Custom poker-specific architectures

#### Large Language Models
- Fine-tuning support for Mistral, Llama, Mixtral
- LoRA/PEFT for efficient training
- Instruction format preparation
- Alpaca and ChatML templates

### 3. Comprehensive Evaluation
- Accuracy, precision, recall, F1 metrics
- Confusion matrices
- Per-class performance analysis
- Feature importance analysis
- Model comparison utilities

### 4. Inference & Deployment
- Trained model loading and inference
- Probability predictions
- Batch processing support
- REST API examples (Flask, FastAPI)

## Dataset

### Primary: PokerBench
- **Source**: [RZ412/PokerBench](https://huggingface.co/datasets/RZ412/PokerBench)
- **Size**: 
  - Preflop: 60,000 training + 1,000 test samples
  - Postflop: 500,000 training + 10,000 test samples
- **Quality**: GTO solver-generated optimal decisions
- **Format**: Both JSON (instruction format) and CSV (structured data)

### Secondary: Poker_Transformers
- **Source**: [SoelMgd/Poker_Transformers](https://github.com/SoelMgd/Poker_Transformers)
- **Focus**: LLM training for poker scenarios

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run complete pipeline
python scripts/run_pipeline.py --model-type xgboost

# Or step by step:

# Download data
python scripts/download_data.py

# Preprocess
python src/data/preprocess.py

# Engineer features
python src/features/engineering.py

# Train model
python src/models/train_ml.py --model-type xgboost

# Evaluate
python src/evaluation/evaluate.py \
    --model-path data/models/xgboost_model.pkl \
    --model-type ml \
    --data-path data/processed/test_features.parquet
```

## Model Performance Targets

| Model Type | Target Accuracy | Inference Speed |
|------------|----------------|-----------------|
| XGBoost | >80% | <10ms |
| Neural Network | >82% | <20ms |
| Fine-tuned LLM | >85% | <100ms |

## Key Technologies

- **ML/DL**: scikit-learn, XGBoost, LightGBM, PyTorch
- **LLM**: Hugging Face Transformers, PEFT, LoRA
- **Data**: pandas, datasets, pyarrow
- **Poker**: treys (hand evaluation), custom features
- **Visualization**: matplotlib, seaborn, plotly

## Use Cases

1. **GTO Training Tool**: Learn optimal preflop play
2. **Hand Analysis**: Analyze specific poker scenarios
3. **Decision Support**: Real-time recommendations during play
4. **Strategy Research**: Explore poker theory with ML
5. **Bot Development**: Foundation for poker AI agents

## Configuration

All models are configurable via YAML files in `configs/`:
- Hyperparameters
- Training settings
- Feature selection
- Evaluation metrics

## Extensibility

The modular design allows easy extension:
- Add new feature extractors in `src/features/`
- Implement new model types in `src/models/`
- Create custom evaluation metrics in `src/evaluation/`
- Add new data sources in `src/data/`

## Documentation

- **GETTING_STARTED.md**: Step-by-step setup guide
- **USAGE.md**: Advanced usage, API integration, best practices
- **ROADMAP.md**: Future features and development plans
- **Notebooks**: Interactive tutorials and examples

## Future Enhancements

### Near-term
- Postflop decision-making
- Opponent modeling
- Tournament ICM calculations
- Real-time API

### Long-term
- Multi-street planning
- Reinforcement learning
- Self-play training
- Explainable AI

## Performance Optimization

- **Data**: Parquet format for efficient I/O
- **Training**: GPU acceleration, LoRA for LLMs
- **Inference**: Model quantization, batching
- **Caching**: Feature caching for repeated scenarios

## Best Practices

1. **Data**: Use preprocessed parquet files for training
2. **Features**: Engineer domain-specific features before training
3. **Models**: Start with XGBoost baseline, then try neural networks
4. **LLMs**: Use LoRA for efficient fine-tuning
5. **Evaluation**: Always test on held-out test set

## Dependencies

Core requirements:
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- XGBoost 2.0+
- scikit-learn 1.3+

See `requirements.txt` for complete list.

## Contributing

Areas for contribution:
1. Data collection and curation
2. Feature engineering improvements
3. New model architectures
4. Infrastructure and deployment
5. Documentation and tutorials

## License

Private - All Rights Reserved

## Acknowledgments

- PokerBench dataset creators
- HuggingFace for dataset hosting and tools
- GTO solver community
- Open source ML/DL community

## Contact

For questions or collaboration, please create an issue on the repository.

---

**Status**: ✅ Complete MVP  
**Version**: 1.0.0  
**Last Updated**: 2026-07-03
