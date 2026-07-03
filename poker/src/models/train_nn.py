"""
Neural network models for poker decision prediction.

Includes:
- MLP (Multi-Layer Perceptron)
- LSTM for action sequence modeling
- Attention-based models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import sys
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_utils import prepare_features as build_features


class PokerDataset(Dataset):
    """PyTorch dataset for poker data."""
    
    def __init__(self, X: pd.DataFrame, y: pd.Series, label_encoder: LabelEncoder):
        """
        Initialize dataset.
        
        Args:
            X: Features DataFrame
            y: Labels Series
            label_encoder: Label encoder for targets
        """
        self.X = torch.FloatTensor(X.values)
        self.y = torch.LongTensor(label_encoder.transform(y))
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PokerMLP(nn.Module):
    """Multi-Layer Perceptron for poker prediction."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 output_dim: int, dropout: float = 0.3):
        """
        Initialize MLP.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (number of classes)
            dropout: Dropout probability
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class PokerLSTM(nn.Module):
    """LSTM-based model for action sequence modeling."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 output_dim: int, dropout: float = 0.3):
        """
        Initialize LSTM model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            output_dim: Output dimension (number of classes)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        # x shape: (batch, features)
        # Reshape for LSTM: (batch, seq_len=1, features)
        x = x.unsqueeze(1)
        
        lstm_out, _ = self.lstm(x)
        
        # Take last output
        last_out = lstm_out[:, -1, :]
        
        return self.fc(last_out)


class PokerNNTrainer:
    """Trainer for neural network models."""
    
    def __init__(self, model_type: str = 'mlp', device: str = None, 
                 random_state: int = 42):
        """
        Initialize trainer.
        
        Args:
            model_type: Type of model ('mlp', 'lstm')
            device: Device for training ('cuda' or 'cpu')
            random_state: Random seed
        """
        self.model_type = model_type
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.random_state = random_state
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        print(f"Using device: {self.device}")
    
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
    
    def create_model(self, input_dim: int, output_dim: int) -> nn.Module:
        """
        Create model based on type.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension (number of classes)
            
        Returns:
            Model instance
        """
        if self.model_type == 'mlp':
            model = PokerMLP(
                input_dim=input_dim,
                hidden_dims=[512, 256, 128],
                output_dim=output_dim,
                dropout=0.3
            )
        elif self.model_type == 'lstm':
            model = PokerLSTM(
                input_dim=input_dim,
                hidden_dim=256,
                num_layers=2,
                output_dim=output_dim,
                dropout=0.3
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model.to(self.device)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None,
              batch_size: int = 256,
              epochs: int = 50,
              lr: float = 0.001) -> Dict[str, any]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            batch_size: Batch size
            epochs: Number of epochs
            lr: Learning rate
            
        Returns:
            Training history
        """
        print(f"\nTraining {self.model_type} model...")
        
        # Encode labels
        y_train_encoded = y_train.copy()
        self.label_encoder.fit(y_train)
        
        # Create datasets
        train_dataset = PokerDataset(X_train, y_train, self.label_encoder)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        if X_val is not None and y_val is not None:
            val_dataset = PokerDataset(X_val, y_val, self.label_encoder)
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )
        else:
            val_loader = None
        
        # Create model
        input_dim = X_train.shape[1]
        output_dim = len(self.label_encoder.classes_)
        self.model = self.create_model(input_dim, output_dim)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        
        # Training loop
        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += batch_y.size(0)
                train_correct += predicted.eq(batch_y).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validate
            if val_loader is not None:
                val_loss, val_acc = self.evaluate_loader(val_loader, criterion)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                scheduler.step(val_acc)
                
                print(f"Epoch {epoch+1}: "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.best_model_state = self.model.state_dict().copy()
            else:
                print(f"Epoch {epoch+1}: "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # Load best model
        if val_loader is not None and hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
            print(f"\nBest validation accuracy: {best_val_acc:.4f}")
        
        return history
    
    def evaluate_loader(self, data_loader: DataLoader, 
                       criterion: nn.Module) -> Tuple[float, float]:
        """
        Evaluate model on a data loader.
        
        Args:
            data_loader: DataLoader
            criterion: Loss function
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
        
        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series,
                batch_size: int = 256) -> Dict[str, any]:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            batch_size: Batch size
            
        Returns:
            Evaluation results
        """
        print("\nEvaluating model...")
        
        test_dataset = PokerDataset(X_test, y_test, self.label_encoder)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc = self.evaluate_loader(test_loader, criterion)
        
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        
        return {
            'loss': test_loss,
            'accuracy': test_acc
        }
    
    def save_model(self, output_path: str):
        """
        Save trained model.
        
        Args:
            output_path: Path to save model
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'input_dim': len(self.feature_names),
            'output_dim': len(self.label_encoder.classes_)
        }
        
        torch.save(checkpoint, output_path)
        print(f"\nModel saved to {output_path}")
    
    @classmethod
    def load_model(cls, model_path: str, device: str = None) -> 'PokerNNTrainer':
        """
        Load trained model.
        
        Args:
            model_path: Path to saved model
            device: Device to load model on
            
        Returns:
            Loaded trainer instance
        """
        checkpoint = torch.load(model_path, map_location='cpu')
        
        trainer = cls(model_type=checkpoint['model_type'], device=device)
        trainer.label_encoder = checkpoint['label_encoder']
        trainer.feature_names = checkpoint['feature_names']
        
        trainer.model = trainer.create_model(
            checkpoint['input_dim'], checkpoint['output_dim']
        )
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.model.eval()
        
        return trainer


def main():
    """Main training script."""
    import argparse
    from sklearn.model_selection import train_test_split
    
    parser = argparse.ArgumentParser(description="Train neural network models")
    parser.add_argument(
        "--data-dir", type=str, default="data/processed",
        help="Directory with processed data"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/models",
        help="Directory to save models"
    )
    parser.add_argument(
        "--model-type", type=str, default="mlp",
        choices=['mlp', 'lstm'],
        help="Model type"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.2,
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
    trainer = PokerNNTrainer(model_type=args.model_type)
    
    # Prepare features
    X, y = trainer.prepare_features(df_train, fit=True)
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_split, random_state=42, stratify=y
    )
    
    # Train model
    history = trainer.train(
        X_train, y_train, X_val, y_val,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr
    )
    
    # Load test data and evaluate
    test_path = Path(args.data_dir) / "test_features.parquet"
    if test_path.exists():
        print("\nLoading test data...")
        df_test = pd.read_parquet(test_path)
        X_test, y_test = trainer.prepare_features(df_test, fit=False)
        
        # Evaluate
        test_results = trainer.evaluate(X_test, y_test, batch_size=args.batch_size)
        
        # Save results
        results_path = output_path / f"{args.model_type}_results.json"
        all_results = {
            'history': history,
            'test': test_results
        }
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {results_path}")
    
    # Save model
    model_path = output_path / f"{args.model_type}_model.pth"
    trainer.save_model(str(model_path))
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print('='*60)


if __name__ == "__main__":
    main()
