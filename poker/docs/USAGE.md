# Poker Prediction Algorithm - Usage Guide

This guide covers common use cases and advanced usage patterns.

## Table of Contents

1. [Making Predictions](#making-predictions)
2. [Custom Features](#custom-features)
3. [Model Comparison](#model-comparison)
4. [Hyperparameter Tuning](#hyperparameter-tuning)
5. [Real-time Inference](#real-time-inference)
6. [Integration](#integration)

## Making Predictions

### Load a Trained Model

```python
from src.evaluation.evaluate import PokerInference

# Load model
inference = PokerInference(
    model_path="data/models/xgboost_model.pkl",
    model_type="ml"  # or 'nn' or 'llm'
)

# Make prediction
features = {
    'hero_position_idx': 5,  # BTN
    'hand_strength_score': 0.95,
    'is_premium': 1,
    'pot_size': 15.0,
    'spr': 10.0,
    'action_count': 2,
    # ... other features
}

prediction = inference.predict(features)
print(f"Recommended action: {prediction}")

# Get probabilities
result = inference.predict(features, return_proba=True)
print(f"Prediction: {result['prediction']}")
print(f"Probabilities: {result['probabilities']}")
```

### Batch Predictions

```python
import pandas as pd

# Load test data
df = pd.read_parquet("data/processed/test_features.parquet")

# Make predictions for all samples
predictions = []
for idx, row in df.iterrows():
    features = row[inference.feature_names].to_dict()
    pred = inference.predict(features)
    predictions.append(pred)

df['predictions'] = predictions
```

### LLM Inference

```python
# Load LLM
inference_llm = PokerInference(
    model_path="data/models/llm_poker",
    model_type="llm"
)

# Create scenario
scenario = {
    'position': 'BTN',
    'hand': 'AhKh',
    'num_players': 4,
    'pot_size': 12.5,
    'action_sequence': 'UTG/3.0bb/CO/call'
}

# Get prediction
decision = inference_llm.predict(scenario)
print(f"LLM Decision: {decision}")
```

## Custom Features

### Add New Features

Edit `src/features/engineering.py`:

```python
class CustomFeatureExtractor:
    """Add your custom features here."""
    
    def extract_vpip_features(self, action_sequence):
        """Calculate VPIP-related features."""
        voluntary_actions = ['call', 'raise', 'bet']
        vpip_actions = sum(
            1 for action in action_sequence 
            if action['action'] in voluntary_actions
        )
        return {
            'vpip_count': vpip_actions,
            'vpip_ratio': vpip_actions / len(action_sequence) if action_sequence else 0
        }

# Add to PokerFeatureEngineer.engineer_features()
vpip_features = df['action_sequence'].apply(
    self.custom_extractor.extract_vpip_features
)
```

### Feature Selection

```python
from sklearn.feature_selection import SelectKBest, f_classif

# Load data
X, y = trainer.prepare_features(df_train)

# Select top K features
selector = SelectKBest(f_classif, k=50)
X_selected = selector.fit_transform(X, y)

# Get selected feature names
selected_features = [
    feature for feature, selected 
    in zip(X.columns, selector.get_support()) 
    if selected
]
print(f"Selected features: {selected_features}")
```

## Model Comparison

### Compare Multiple Models

```python
from src.evaluation.evaluate import ModelEvaluator
import json

# Load results for multiple models
evaluator = ModelEvaluator()

models = ['xgboost', 'lightgbm', 'random_forest', 'mlp']
results = {}

for model in models:
    with open(f"data/models/{model}_results.json", 'r') as f:
        results[model] = json.load(f)['test']

# Compare
comparison_df = evaluator.compare_models(results)
print(comparison_df)

# Save comparison
comparison_df.to_csv("data/evaluation/model_comparison.csv", index=False)
```

### Ensemble Models

```python
from sklearn.ensemble import VotingClassifier

# Load multiple models
model1 = pickle.load(open("data/models/xgboost_model.pkl", 'rb'))['model']
model2 = pickle.load(open("data/models/lightgbm_model.pkl", 'rb'))['model']
model3 = pickle.load(open("data/models/random_forest_model.pkl", 'rb'))['model']

# Create ensemble
ensemble = VotingClassifier(
    estimators=[
        ('xgb', model1),
        ('lgb', model2),
        ('rf', model3)
    ],
    voting='soft'
)

# The ensemble will use the already-fitted models
ensemble.fit(X_train, y_train)  # This step can be skipped for pre-trained models
```

## Hyperparameter Tuning

### Grid Search

```python
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# Define parameter grid
param_grid = {
    'max_depth': [6, 8, 10],
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 0.9, 1.0]
}

# Create model
model = xgb.XGBClassifier(random_state=42)

# Grid search
grid_search = GridSearchCV(
    model, param_grid,
    cv=5, scoring='accuracy',
    n_jobs=-1, verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

### Optuna Optimization

```python
import optuna
import xgboost as xgb

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
    }
    
    model = xgb.XGBClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    return accuracy

# Optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"Best params: {study.best_params}")
print(f"Best accuracy: {study.best_value:.4f}")
```

## Real-time Inference

### Create REST API

```python
from flask import Flask, request, jsonify
from src.evaluation.evaluate import PokerInference

app = Flask(__name__)

# Load model at startup
model = PokerInference(
    model_path="data/models/xgboost_model.pkl",
    model_type="ml"
)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    try:
        prediction = model.predict(data, return_proba=True)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### FastAPI Version

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict

app = FastAPI()

class PredictionRequest(BaseModel):
    features: Dict[str, float]

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        prediction = model.predict(request.features, return_proba=True)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

## Integration

### Hand History Parser Integration

```python
def parse_hand_history(hand_text: str) -> Dict:
    """Parse poker hand history into features."""
    # Your parsing logic here
    features = {
        'position': extract_position(hand_text),
        'hand': extract_hand(hand_text),
        'pot_size': extract_pot_size(hand_text),
        # ... more features
    }
    return features

# Use with inference
hand_history = "..."  # Raw hand history text
features = parse_hand_history(hand_history)
prediction = inference.predict(features)
```

### PokerStars Integration

```python
from pypokerengine.api.game import setup_config, start_poker

def ai_decision_callback(state, player_info):
    """Callback for AI decision."""
    # Extract features from state
    features = extract_features_from_state(state, player_info)
    
    # Get prediction
    decision = inference.predict(features)
    
    # Convert to PokerStars action
    action = convert_to_poker_action(decision, state)
    
    return action
```

## Best Practices

1. **Feature Normalization**: Always normalize features before inference
2. **Model Versioning**: Keep track of model versions and their performance
3. **Monitoring**: Log predictions and actual outcomes for model improvement
4. **A/B Testing**: Test new models against baseline in production
5. **Retraining**: Regularly retrain on new data to maintain performance

## Troubleshooting

### Prediction Errors

If you get errors during prediction:
1. Check feature names match training features
2. Ensure no missing values in features
3. Verify feature dtypes are correct

### Performance Issues

For slow inference:
1. Batch predictions instead of one-by-one
2. Use model quantization for neural networks
3. Consider model distillation for faster inference

## Further Reading

- [Feature Engineering for Poker](docs/feature_engineering.md)
- [Model Architecture Details](docs/model_architectures.md)
- [API Reference](docs/api_reference.md)
