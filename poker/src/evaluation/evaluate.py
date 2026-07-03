"""
Evaluation and inference utilities for poker prediction models.

Supports:
- Model evaluation metrics
- Inference on new hands
- Comparison across models
"""

import sys

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import json
import pickle
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_utils import normalize_labels


class ModelEvaluator:
    """Evaluate poker prediction models."""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                           labels: Optional[List[str]] = None) -> Dict[str, any]:
        """
        Evaluate predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Label names
            
        Returns:
            Dictionary of metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred, normalize=True)

        label_values = labels if labels is not None else sorted(set(y_true) | set(y_pred))

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=label_values, average=None, zero_division=0
        )

        # Macro/Weighted averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=label_values, average='macro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=label_values, average='weighted', zero_division=0
        )

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=label_values)
        
        results = {
            'accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted),
            'confusion_matrix': cm.tolist()
        }
        
        # Per-class results
        results['per_class'] = {}
        for i, label in enumerate(label_values):
            results['per_class'][label] = {
                'precision': float(precision[i]) if i < len(precision) else 0.0,
                'recall': float(recall[i]) if i < len(recall) else 0.0,
                'f1': float(f1[i]) if i < len(f1) else 0.0,
                'support': int(support[i]) if i < len(support) else 0
            }
        
        return results
    
    def plot_confusion_matrix(self, cm: np.ndarray, labels: List[str],
                            save_path: Optional[str] = None):
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            labels: Class labels
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def compare_models(self, results_dict: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            results_dict: Dictionary of model_name -> results
            
        Returns:
            Comparison DataFrame
        """
        comparison_data = []
        
        for model_name, results in results_dict.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results.get('accuracy', 0),
                'Precision (Macro)': results.get('precision_macro', 0),
                'Recall (Macro)': results.get('recall_macro', 0),
                'F1 (Macro)': results.get('f1_macro', 0),
                'Precision (Weighted)': results.get('precision_weighted', 0),
                'Recall (Weighted)': results.get('recall_weighted', 0),
                'F1 (Weighted)': results.get('f1_weighted', 0)
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Accuracy', ascending=False)
        
        return df
    
    def print_evaluation_summary(self, results: Dict[str, any], model_name: str = "Model"):
        """
        Print evaluation summary.
        
        Args:
            results: Evaluation results
            model_name: Name of model
        """
        print(f"\n{'='*60}")
        print(f"Evaluation Results: {model_name}")
        print('='*60)
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:          {results['accuracy']:.4f}")
        print(f"  Precision (Macro): {results['precision_macro']:.4f}")
        print(f"  Recall (Macro):    {results['recall_macro']:.4f}")
        print(f"  F1 (Macro):        {results['f1_macro']:.4f}")
        
        if 'per_class' in results:
            print(f"\nPer-Class Metrics:")
            for label, metrics in results['per_class'].items():
                print(f"  {label}:")
                print(f"    Precision: {metrics['precision']:.4f}")
                print(f"    Recall:    {metrics['recall']:.4f}")
                print(f"    F1:        {metrics['f1']:.4f}")
                print(f"    Support:   {metrics['support']}")


class PokerInference:
    """Inference class for poker prediction models."""
    
    def __init__(self, model_path: str, model_type: str):
        """
        Initialize inference.
        
        Args:
            model_path: Path to saved model
            model_type: Type of model ('ml', 'nn', 'llm')
        """
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.label_encoder = None
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model."""
        print(f"Loading {self.model_type} model from {self.model_path}")
        
        if self.model_type == 'ml':
            # Load pickle model
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            self.model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.feature_names = model_data['feature_names']
        
        elif self.model_type == 'nn':
            # Load PyTorch model
            from src.models.train_nn import PokerNNTrainer
            trainer = PokerNNTrainer.load_model(self.model_path)
            self.model = trainer.model
            self.label_encoder = trainer.label_encoder
            self.feature_names = trainer.feature_names
            self.device = trainer.device
        
        elif self.model_type == 'llm':
            # Load LLM
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            self.model.eval()
        
        print("Model loaded successfully!")
    
    def predict(self, features: Union[pd.DataFrame, Dict],
               return_proba: bool = False) -> Union[str, Dict]:
        """
        Make prediction.
        
        Args:
            features: Features for prediction
            return_proba: Whether to return probabilities
            
        Returns:
            Prediction or prediction with probabilities
        """
        if self.model_type in ['ml', 'nn']:
            # Convert to DataFrame if dict
            if isinstance(features, dict):
                features = pd.DataFrame([features])
            
            # Ensure correct feature order and fill missing columns
            features = features.reindex(columns=self.feature_names, fill_value=0)
            features = features.fillna(0).replace([np.inf, -np.inf], 0)
            
            if self.model_type == 'ml':
                pred_encoded = self.model.predict(features)
                pred = self.label_encoder.inverse_transform(pred_encoded)[0]
                
                if return_proba:
                    proba = self.model.predict_proba(features)[0]
                    proba_dict = {
                        label: float(prob) 
                        for label, prob in zip(self.label_encoder.classes_, proba)
                    }
                    return {'prediction': pred, 'probabilities': proba_dict}
                return pred
            
            elif self.model_type == 'nn':
                self.model.eval()
                with torch.no_grad():
                    features_tensor = torch.FloatTensor(features.values).to(self.device)
                    outputs = self.model(features_tensor)
                    
                    if return_proba:
                        proba = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                        pred_idx = proba.argmax()
                        pred = self.label_encoder.inverse_transform([pred_idx])[0]
                        proba_dict = {
                            label: float(prob)
                            for label, prob in zip(self.label_encoder.classes_, proba)
                        }
                        return {'prediction': pred, 'probabilities': proba_dict}
                    else:
                        pred_idx = outputs.argmax(dim=1).cpu().numpy()[0]
                        pred = self.label_encoder.inverse_transform([pred_idx])[0]
                        return pred
        
        elif self.model_type == 'llm':
            # Format prompt
            if isinstance(features, dict):
                prompt = self.format_prompt(features)
            else:
                prompt = features
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract decision from response
            decision = self.extract_decision(response)
            
            return decision
    
    def format_prompt(self, scenario: Dict) -> str:
        """
        Format poker scenario into prompt for LLM.
        
        Args:
            scenario: Dictionary with scenario information
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an expert No-Limit Texas Hold'em poker player. 

Game Scenario:
- Format: 6-handed No-Limit Texas Hold'em
- Your Position: {scenario.get('position', 'Unknown')}
- Your Hand: {scenario.get('hand', 'Unknown')}
- Number of Players: {scenario.get('num_players', 6)}
- Pot Size: {scenario.get('pot_size', 0)} BB
- Previous Actions: {scenario.get('action_sequence', 'No action yet')}

What is the optimal decision in this situation?

### Response:"""
        
        return prompt
    
    def extract_decision(self, response: str) -> str:
        """
        Extract decision from LLM response.
        
        Args:
            response: LLM response text
            
        Returns:
            Extracted decision
        """
        # Simple extraction - look for keywords
        response_lower = response.lower()
        
        decisions = ['fold', 'check', 'call', 'bet', 'raise', 'allin']
        for decision in decisions:
            if decision in response_lower:
                return decision
        
        return "unknown"


def main():
    """Main evaluation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate poker models")
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to saved model"
    )
    parser.add_argument(
        "--model-type", type=str, required=True,
        choices=['ml', 'nn', 'llm'],
        help="Model type"
    )
    parser.add_argument(
        "--data-path", type=str,
        help="Path to test data"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/evaluation",
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize inference
    inference = PokerInference(args.model_path, args.model_type)
    
    # Load test data if provided
    if args.data_path:
        print(f"\nLoading test data from {args.data_path}")
        
        if args.data_path.endswith('.parquet'):
            df_test = pd.read_parquet(args.data_path)
        else:
            df_test = pd.read_csv(args.data_path)
        
        print(f"Loaded {len(df_test)} test samples")
        
        # Make predictions
        print("\nMaking predictions...")
        predictions = []
        
        for idx, row in df_test.iterrows():
            if args.model_type in ['ml', 'nn']:
                # Use numeric features
                features = row[inference.feature_names].to_dict()
                pred = inference.predict(features)
            else:
                # Use scenario dict for LLM
                scenario = {
                    'position': row.get('hero_pos'),
                    'hand': row.get('hero_holding'),
                    'num_players': row.get('num_players'),
                    'pot_size': row.get('pot_size'),
                    'action_sequence': row.get('prev_line', '')
                }
                pred = inference.predict(scenario)
            
            predictions.append(pred)
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(df_test)} samples")
        
        # Evaluate
        evaluator = ModelEvaluator()
        
        # Get true labels
        target_col = 'decision_type' if 'decision_type' in df_test.columns else 'correct_decision'
        y_true = normalize_labels(df_test[target_col]).values
        y_pred = np.array(predictions)

        # Get unique labels
        labels = sorted(set(list(y_true) + list(y_pred)))
        
        # Evaluate
        results = evaluator.evaluate_predictions(y_true, y_pred, labels)
        evaluator.print_evaluation_summary(results, model_name=args.model_type)
        
        # Save results
        results_path = output_path / f"{args.model_type}_evaluation.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")
        
        # Plot confusion matrix
        cm = np.array(results['confusion_matrix'])
        cm_path = output_path / f"{args.model_type}_confusion_matrix.png"
        evaluator.plot_confusion_matrix(cm, labels, save_path=str(cm_path))
    
    else:
        print("\nNo test data provided. Model loaded and ready for inference.")
        print("Example usage:")
        print("  features = {'position': 'BTN', 'hand': 'AhKh', ...}")
        print("  prediction = inference.predict(features)")
    
    print(f"\n{'='*60}")
    print("Evaluation completed!")
    print('='*60)


if __name__ == "__main__":
    main()
