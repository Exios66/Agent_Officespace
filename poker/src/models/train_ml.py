"""
Traditional ML models for poker decision prediction.

Includes:
- XGBoost
- LightGBM
- Random Forest
- Logistic Regression (baseline)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import xgboost as xgb
import lightgbm as lgb

import warnings
import sys
from pathlib import Path

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_utils import prepare_features as build_features


class PokerMLTrainer:
    """Trainer for traditional ML models."""
    
    def __init__(self, model_type: str = 'xgboost', random_state: int = 42):
        """
        Initialize trainer.
        
        Args:
            model_type: Type of model ('xgboost', 'lightgbm', 'random_forest', 'logistic')
            random_state: Random seed
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = None
        self.config = {}
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        fit: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for training or inference.

        Args:
            df: DataFrame with engineered features
            fit: Whether to infer feature columns and fit label encoder

        Returns:
            Tuple of (X, y)
        """
        X, y, feature_names, label_encoder = build_features(
            df,
            feature_names=self.feature_names,
            label_encoder=self.label_encoder,
            fit=fit,
        )
        self.feature_names = feature_names
        self.label_encoder = label_encoder

        print(f"Prepared {len(feature_names)} features")
        print(f"Target distribution:\n{y.value_counts()}")

        return X, y
    
    def create_model(self, n_classes: int) -> any:
        """
        Create model based on type.
        
        Args:
            n_classes: Number of classes
            
        Returns:
            Model instance
        """
        if self.model_type == 'xgboost':
            if n_classes == 2:
                self.config = {
                    'objective': 'binary:logistic',
                    'max_depth': 8,
                    'learning_rate': 0.1,
                    'n_estimators': 200,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': self.random_state,
                    'n_jobs': -1
                }
            else:
                self.config = {
                    'objective': 'multi:softmax',
                    'num_class': n_classes,
                    'max_depth': 8,
                    'learning_rate': 0.1,
                    'n_estimators': 200,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': self.random_state,
                    'n_jobs': -1
                }
            model = xgb.XGBClassifier(**self.config)
        
        elif self.model_type == 'lightgbm':
            self.config = {
                'objective': 'multiclass' if n_classes > 2 else 'binary',
                'num_class': n_classes if n_classes > 2 else None,
                'max_depth': 8,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'n_jobs': -1,
                'verbose': -1
            }
            # Remove None values
            self.config = {k: v for k, v in self.config.items() if v is not None}
            model = lgb.LGBMClassifier(**self.config)
        
        elif self.model_type == 'random_forest':
            self.config = {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'random_state': self.random_state,
                'n_jobs': -1
            }
            model = RandomForestClassifier(**self.config)
        
        elif self.model_type == 'logistic':
            self.config = {
                'max_iter': 1000,
                'random_state': self.random_state,
                'n_jobs': -1
            }
            model = LogisticRegression(**self.config)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None, 
              y_val: Optional[pd.Series] = None) -> Dict[str, any]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training results dictionary
        """
        print(f"\nTraining {self.model_type} model...")
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        # Create model
        n_classes = len(self.label_encoder.classes_)
        self.model = self.create_model(n_classes)
        
        # Train
        if self.model_type in ['xgboost', 'lightgbm'] and X_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            
            if self.model_type == 'xgboost':
                self.model.fit(
                    X_train, y_train_encoded,
                    eval_set=[(X_val, y_val_encoded)],
                    verbose=False
                )
            else:  # lightgbm
                self.model.fit(
                    X_train, y_train_encoded,
                    eval_set=[(X_val, y_val_encoded)],
                    callbacks=[lgb.log_evaluation(period=0)]
                )
        else:
            self.model.fit(X_train, y_train_encoded)
        
        # Evaluate on training set
        train_preds = self.model.predict(X_train)
        train_acc = accuracy_score(y_train_encoded, train_preds)
        
        results = {
            'train_accuracy': train_acc,
            'n_features': X_train.shape[1],
            'n_samples': len(X_train),
            'n_classes': n_classes,
            'classes': self.label_encoder.classes_.tolist()
        }
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            val_preds = self.model.predict(X_val)
            val_acc = accuracy_score(y_val_encoded, val_preds)
            results['val_accuracy'] = val_acc
            
            print(f"Training accuracy: {train_acc:.4f}")
            print(f"Validation accuracy: {val_acc:.4f}")
        else:
            print(f"Training accuracy: {train_acc:.4f}")
        
        return results
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, any]:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation results
        """
        print("\nEvaluating model...")

        known_mask = y_test.isin(self.label_encoder.classes_)
        if not known_mask.all():
            dropped = int((~known_mask).sum())
            print(f"Warning: dropping {dropped} test samples with unseen labels")
            X_test = X_test.loc[known_mask].reset_index(drop=True)
            y_test = y_test.loc[known_mask].reset_index(drop=True)

        # Encode labels
        y_test_encoded = self.label_encoder.transform(y_test)
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test_encoded, y_pred)

        labels = sorted(set(self.label_encoder.classes_))
        report = classification_report(
            y_test_encoded,
            y_pred,
            labels=range(len(labels)),
            target_names=labels,
            output_dict=True,
            zero_division=0,
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test_encoded, y_pred)
        
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(
            y_test_encoded,
            y_pred,
            labels=range(len(labels)),
            target_names=labels,
            zero_division=0,
        ))
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
        
        return results
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            })
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            print(f"\nTop {top_n} Most Important Features:")
            print(importance_df.head(top_n))
            
            return importance_df.head(top_n)
        else:
            print("Model does not support feature importance")
            return pd.DataFrame()
    
    def save_model(self, output_path: str):
        """
        Save trained model.
        
        Args:
            output_path: Path to save model
        """
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'config': self.config
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nModel saved to {output_path}")
    
    @classmethod
    def load_model(cls, model_path: str) -> 'PokerMLTrainer':
        """
        Load trained model.
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Loaded trainer instance
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        trainer = cls(model_type=model_data['model_type'])
        trainer.model = model_data['model']
        trainer.label_encoder = model_data['label_encoder']
        trainer.feature_names = model_data['feature_names']
        trainer.config = model_data['config']
        
        return trainer


def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train ML models for poker prediction")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory with processed data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/models",
        help="Directory to save models"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="xgboost",
        choices=['xgboost', 'lightgbm', 'random_forest', 'logistic'],
        help="Model type"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load training data
    print("Loading training data...")
    train_path = Path(args.data_dir) / "train_features.parquet"
    df_train = pd.read_parquet(train_path)
    print(f"Loaded {len(df_train)} training samples")
    
    # Initialize trainer
    trainer = PokerMLTrainer(model_type=args.model_type)
    
    # Prepare features
    X, y = trainer.prepare_features(df_train, fit=True)
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_split, random_state=42, stratify=y
    )
    
    # Train model
    train_results = trainer.train(X_train, y_train, X_val, y_val)
    
    # Feature importance
    trainer.get_feature_importance(top_n=20)
    
    # Load test data and evaluate
    test_path = Path(args.data_dir) / "test_features.parquet"
    if test_path.exists():
        print("\nLoading test data...")
        df_test = pd.read_parquet(test_path)
        X_test, y_test = trainer.prepare_features(df_test, fit=False)
        
        # Evaluate
        test_results = trainer.evaluate(X_test, y_test)
        
        # Save results
        results_path = output_path / f"{args.model_type}_results.json"
        all_results = {
            'train': train_results,
            'test': test_results
        }
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {results_path}")
    
    # Save model
    model_path = output_path / f"{args.model_type}_model.pkl"
    trainer.save_model(str(model_path))
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print('='*60)


if __name__ == "__main__":
    main()
