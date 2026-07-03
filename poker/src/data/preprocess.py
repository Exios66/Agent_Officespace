"""
Data preprocessing module for poker datasets.

Handles:
- Loading raw CSV/JSON data
- Parsing poker-specific fields (positions, actions, cards)
- Basic data cleaning and validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import re


class PokerDataPreprocessor:
    """Preprocessor for PokerBench dataset."""
    
    # Position mappings
    POSITIONS = ['UTG', 'HJ', 'CO', 'BTN', 'SB', 'BB']
    POSITION_TO_IDX = {pos: idx for idx, pos in enumerate(POSITIONS)}
    
    # Action types
    ACTIONS = ['fold', 'check', 'call', 'bet', 'raise', 'allin']
    ACTION_TO_IDX = {act: idx for idx, act in enumerate(ACTIONS)}
    
    def __init__(self, raw_data_dir: str = "data/raw/pokerbench"):
        self.raw_data_dir = Path(raw_data_dir)
        self.data = {}
    
    def load_data(self, split: str = "train") -> pd.DataFrame:
        """
        Load data for a specific split.
        
        Args:
            split: Data split name (e.g., 'train', 'test')
            
        Returns:
            DataFrame with loaded data
        """
        csv_path = self.raw_data_dir / f"{split}.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        print(f"Loaded {split} split: {len(df)} samples")
        
        self.data[split] = df
        return df
    
    def parse_card(self, card_str: str) -> Tuple[str, str]:
        """
        Parse a card string like 'Kd' into rank and suit.
        
        Args:
            card_str: Card string (e.g., 'Kd', 'Ah', '2c')
            
        Returns:
            Tuple of (rank, suit)
        """
        if len(card_str) != 2:
            return ('?', '?')
        return (card_str[0], card_str[1])
    
    def parse_hand(self, hand_str: str) -> List[Tuple[str, str]]:
        """
        Parse a hand string like 'KdKc' into list of cards.
        
        Args:
            hand_str: Hand string (e.g., 'KdKc', 'AhQh')
            
        Returns:
            List of (rank, suit) tuples
        """
        if pd.isna(hand_str) or len(hand_str) != 4:
            return [('?', '?'), ('?', '?')]
        
        card1 = self.parse_card(hand_str[:2])
        card2 = self.parse_card(hand_str[2:])
        return [card1, card2]
    
    def parse_action_sequence(self, action_str: str) -> List[Dict[str, any]]:
        """
        Parse action sequence like 'UTG/2.0bb/BTN/call/SB/13.0bb'.
        
        Args:
            action_str: Action sequence string
            
        Returns:
            List of action dictionaries
        """
        if pd.isna(action_str):
            return []
        
        actions = []
        parts = action_str.split('/')

        # Track whether any bet/raise has already appeared in the sequence.
        # Previously we (incorrectly) used `actions[-1]['action'] == 'fold'` to
        # detect the "no bet yet" case, which caused any sized action following
        # a fold to be classified as a `bet` instead of a `raise`, e.g.
        #   UTG/2.0bb/HJ/fold/CO/6.0bb  ->  CO action mis-labeled as `bet`.
        bet_or_raise_seen = False

        i = 0
        while i < len(parts):
            if i + 1 < len(parts):
                position = parts[i]
                action = parts[i + 1]

                # Check if it's a bet/raise with amount
                amount = None
                if 'bb' in action:
                    amount_str = action.replace('bb', '').strip()
                    try:
                        amount = float(amount_str)
                    except ValueError:
                        amount = None
                    action_type = 'raise' if bet_or_raise_seen else 'bet'
                    bet_or_raise_seen = True
                else:
                    action_type = action

                actions.append({
                    'position': position,
                    'action': action_type,
                    'amount': amount
                })

                i += 2
            else:
                i += 1

        return actions
    
    # PokerBench encodes "raise to X" as a bare bet-size string (e.g. ``3.0bb``)
    # instead of a leading verb. Any such token collapses to ``raise`` so we
    # keep the label space at a stable {fold, check, call, raise, allin}.
    _BET_SIZING_RE = re.compile(r"^\d+(?:\.\d+)?\s*bb$")

    @classmethod
    def _canonical_decision(cls, raw) -> str:
        """Canonicalise a raw ``correct_decision`` string to one of
        ``{fold, check, call, raise, allin, unknown}``.

        PokerBench decisions come in many flavours: bare verbs (``fold``,
        ``check``), verb+size (``bet 24``, ``Raise 3.0bb``), size-only
        (``3.0bb``), and all-in (``all-in``, ``allin``). Squashing them to
        the 5 canonical actions is required for the multiclass classifier to
        train on stratified splits — otherwise singleton size classes crash
        both the split and ``classification_report``.
        """
        if not isinstance(raw, str):
            return 'unknown'
        r = raw.strip().lower()
        if not r:
            return 'unknown'
        if r.startswith('fold'):
            return 'fold'
        if r.startswith('check'):
            return 'check'
        if r.startswith('call'):
            return 'call'
        if r.startswith('allin') or r.startswith('all-in') or r.startswith('all in'):
            return 'allin'
        if r.startswith('bet') or r.startswith('raise'):
            return 'raise'
        if cls._BET_SIZING_RE.match(r):
            return 'raise'
        return 'unknown'

    def extract_preflop_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from preflop data.
        
        Args:
            df: DataFrame with preflop data
            
        Returns:
            DataFrame with extracted features
        """
        print("Extracting preflop features...")
        
        # Create a copy to avoid modifying original
        features_df = df.copy()
        
        # Parse hero position
        if 'hero_pos' in df.columns:
            features_df['hero_position_idx'] = features_df['hero_pos'].map(
                self.POSITION_TO_IDX
            ).fillna(-1).astype(int)
        
        # Parse hero hand
        if 'hero_holding' in df.columns:
            features_df['hero_hand'] = features_df['hero_holding'].apply(self.parse_hand)
            
            # Extract individual cards
            features_df['hero_card1_rank'] = features_df['hero_hand'].apply(
                lambda x: x[0][0] if len(x) > 0 else '?'
            )
            features_df['hero_card1_suit'] = features_df['hero_hand'].apply(
                lambda x: x[0][1] if len(x) > 0 else '?'
            )
            features_df['hero_card2_rank'] = features_df['hero_hand'].apply(
                lambda x: x[1][0] if len(x) > 1 else '?'
            )
            features_df['hero_card2_suit'] = features_df['hero_hand'].apply(
                lambda x: x[1][1] if len(x) > 1 else '?'
            )
        
        # Parse action sequence
        if 'prev_line' in df.columns:
            features_df['action_sequence'] = features_df['prev_line'].apply(
                self.parse_action_sequence
            )
            
            # Extract action sequence length
            features_df['num_actions'] = features_df['action_sequence'].apply(len)
            
            # Count action types
            for action_type in ['fold', 'call', 'bet', 'raise', 'allin']:
                features_df[f'num_{action_type}'] = features_df['action_sequence'].apply(
                    lambda seq: sum(1 for act in seq if action_type in act['action'].lower())
                )
        
        # Parse decision
        if 'correct_decision' in df.columns:
            features_df['decision_type'] = features_df['correct_decision'].apply(
                self._canonical_decision
            )
        
        # Convert numeric columns
        numeric_cols = ['num_players', 'num_bets', 'pot_size']
        for col in numeric_cols:
            if col in features_df.columns:
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
        
        print(f"Extracted features shape: {features_df.shape}")
        return features_df
    
    def save_processed_data(self, df: pd.DataFrame, split: str, 
                          output_dir: str = "data/processed"):
        """
        Save processed data.
        
        Args:
            df: Processed DataFrame
            split: Split name
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        csv_path = output_path / f"{split}_processed.csv"
        
        # Convert complex columns to strings for CSV
        df_save = df.copy()
        for col in df_save.columns:
            if df_save[col].dtype == 'object':
                # Convert lists/dicts to JSON strings
                df_save[col] = df_save[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x
                )
        
        df_save.to_csv(csv_path, index=False)
        print(f"Saved processed data to {csv_path}")
        
        # Save as parquet (better for preserving types)
        parquet_path = output_path / f"{split}_processed.parquet"
        df.to_parquet(parquet_path, index=False)
        print(f"Saved processed data to {parquet_path}")


def main():
    """Main preprocessing pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess poker datasets")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw/pokerbench",
        help="Directory containing raw data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory for processed data"
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs='+',
        default=['train', 'test'],
        help="Data splits to process"
    )
    
    args = parser.parse_args()
    
    preprocessor = PokerDataPreprocessor(args.raw_dir)
    
    for split in args.splits:
        try:
            print(f"\n{'='*60}")
            print(f"Processing {split} split")
            print('='*60)
            
            # Load data
            df = preprocessor.load_data(split)
            
            # Extract features
            df_processed = preprocessor.extract_preflop_features(df)
            
            # Save processed data
            preprocessor.save_processed_data(df_processed, split, args.output_dir)
            
            print(f"\n✓ Completed processing {split} split")
            
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print(f"Skipping {split} split")
        except Exception as e:
            print(f"Error processing {split} split: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("Preprocessing completed!")
    print('='*60)


if __name__ == "__main__":
    main()
