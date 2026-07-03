"""
Feature engineering for poker prediction.

This module contains advanced poker-specific feature engineering including:
- Hand strength evaluation
- Position-based features
- Stack and pot odds calculations
- Action sequence embeddings
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings


class HandStrengthEvaluator:
    """Evaluate poker hand strength for preflop scenarios."""
    
    # Card rank values
    RANK_VALUES = {
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
        '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
    }
    
    # Preflop hand strength groups (Sklansky-Chubukov rankings simplified)
    HAND_GROUPS = {
        1: [('AA', False), ('KK', False), ('QQ', False), ('JJ', False), ('AK', True)],
        2: [('TT', False), ('AQ', True), ('AJ', True), ('KQ', True), ('AK', False)],
        3: [('99', False), ('AQ', False), ('AJ', False), ('KQ', False), ('KJ', True), ('AT', True), ('A9', True)],
        4: [('88', False), ('KJ', False), ('QJ', True), ('JT', True), ('AT', False), ('KT', True)],
        5: [('77', False), ('66', False), ('A9', False), ('A8', True), ('A7', True), ('KT', False), ('QT', True), ('JT', False)],
        6: [('55', False), ('44', False), ('A8', False), ('A7', False), ('A6', True), ('A5', True), ('A4', True), ('A3', True), ('A2', True)],
        7: [('33', False), ('22', False), ('K9', True), ('Q9', True), ('J9', True), ('T9', True), ('98', True)],
        8: [('K9', False), ('Q9', False), ('J9', False), ('T9', False), ('98', False), ('87', True), ('76', True), ('65', True)]
    }
    
    def __init__(self):
        # Build reverse lookup
        self.hand_to_group = {}
        for group, hands in self.HAND_GROUPS.items():
            for hand, suited in hands:
                self.hand_to_group[(hand, suited)] = group
    
    def parse_hand_notation(self, card1_rank: str, card2_rank: str, 
                          card1_suit: str, card2_suit: str) -> Tuple[str, bool, bool]:
        """
        Parse hand into standard notation.
        
        Args:
            card1_rank, card2_rank: Rank of cards
            card1_suit, card2_suit: Suit of cards
            
        Returns:
            Tuple of (hand_notation, is_suited, is_pair)
        """
        rank1_val = self.RANK_VALUES.get(card1_rank, 0)
        rank2_val = self.RANK_VALUES.get(card2_rank, 0)
        
        # Order by rank (higher first)
        if rank1_val >= rank2_val:
            hand = f"{card1_rank}{card2_rank}"
        else:
            hand = f"{card2_rank}{card1_rank}"
        
        is_suited = (card1_suit == card2_suit)
        is_pair = (card1_rank == card2_rank)
        
        return hand, is_suited, is_pair
    
    def get_hand_strength(self, card1_rank: str, card2_rank: str,
                         card1_suit: str, card2_suit: str) -> Dict[str, any]:
        """
        Calculate hand strength features.
        
        Args:
            card1_rank, card2_rank: Rank of cards
            card1_suit, card2_suit: Suit of cards
            
        Returns:
            Dictionary of hand strength features
        """
        hand, is_suited, is_pair = self.parse_hand_notation(
            card1_rank, card2_rank, card1_suit, card2_suit
        )
        
        # Find hand group
        hand_group = 9  # Default: weak hand
        for group_num in range(1, 9):
            if (hand, is_suited) in [(h[0], h[1]) for h in self.HAND_GROUPS.get(group_num, [])]:
                hand_group = group_num
                break
        
        # Calculate additional features
        rank1_val = self.RANK_VALUES.get(card1_rank, 0)
        rank2_val = self.RANK_VALUES.get(card2_rank, 0)
        
        features = {
            'hand_notation': hand,
            'is_suited': int(is_suited),
            'is_pair': int(is_pair),
            'hand_group': hand_group,
            'hand_strength_score': 1.0 / hand_group if hand_group > 0 else 0,  # Normalized strength
            'high_card_value': max(rank1_val, rank2_val),
            'low_card_value': min(rank1_val, rank2_val),
            'rank_gap': abs(rank1_val - rank2_val),
            'has_ace': int('A' in [card1_rank, card2_rank]),
            'has_face': int(any(r in ['J', 'Q', 'K', 'A'] for r in [card1_rank, card2_rank])),
            'is_premium': int(hand_group <= 2),  # Top 2 groups
            'is_playable': int(hand_group <= 5)  # Top 5 groups
        }
        
        return features


class PositionFeatureExtractor:
    """Extract position-based features."""
    
    POSITIONS = ['UTG', 'HJ', 'CO', 'BTN', 'SB', 'BB']
    
    def __init__(self):
        self.position_to_idx = {pos: idx for idx, pos in enumerate(self.POSITIONS)}
    
    def get_position_features(self, position: str, num_players: int = 6) -> Dict[str, any]:
        """
        Calculate position-based features.
        
        Args:
            position: Player position
            num_players: Number of players at table
            
        Returns:
            Dictionary of position features
        """
        pos_idx = self.position_to_idx.get(position, -1)
        
        features = {
            'position_idx': pos_idx,
            'is_early_position': int(position in ['UTG', 'HJ']),
            'is_middle_position': int(position in ['CO']),
            'is_late_position': int(position in ['BTN']),
            'is_blind': int(position in ['SB', 'BB']),
            'is_button': int(position == 'BTN'),
            'is_cutoff': int(position == 'CO'),
            'position_strength': (pos_idx + 1) / len(self.POSITIONS),  # Normalized
            'relative_position': pos_idx / num_players if num_players > 0 else 0
        }
        
        return features


class PotOddsCalculator:
    """Calculate pot odds and related features."""
    
    @staticmethod
    def calculate_pot_odds(pot_size: float, bet_to_call: float) -> float:
        """
        Calculate pot odds.
        
        Args:
            pot_size: Current pot size
            bet_to_call: Amount to call
            
        Returns:
            Pot odds ratio
        """
        if bet_to_call <= 0:
            return 0.0
        return pot_size / (pot_size + bet_to_call)
    
    @staticmethod
    def calculate_spr(effective_stack: float, pot_size: float) -> float:
        """
        Calculate Stack-to-Pot Ratio.
        
        Args:
            effective_stack: Effective stack size
            pot_size: Current pot size
            
        Returns:
            SPR value
        """
        if pot_size <= 0:
            return 999.0  # Infinite SPR
        return effective_stack / pot_size
    
    def get_pot_features(self, pot_size: float, effective_stack: float,
                        last_bet_size: Optional[float] = None) -> Dict[str, any]:
        """
        Calculate pot-related features.
        
        Args:
            pot_size: Current pot size
            effective_stack: Effective stack size
            last_bet_size: Last bet/raise amount
            
        Returns:
            Dictionary of pot features
        """
        spr = self.calculate_spr(effective_stack, pot_size)
        
        features = {
            'pot_size': pot_size,
            'effective_stack': effective_stack,
            'spr': min(spr, 50.0),  # Cap at 50 for numerical stability
            'pot_to_stack_ratio': pot_size / effective_stack if effective_stack > 0 else 0,
            'is_low_spr': int(spr < 3),
            'is_medium_spr': int(3 <= spr <= 8),
            'is_high_spr': int(spr > 8)
        }
        
        if last_bet_size is not None:
            features['last_bet_size'] = last_bet_size
            features['last_bet_to_pot'] = last_bet_size / pot_size if pot_size > 0 else 0
            features['pot_odds'] = self.calculate_pot_odds(pot_size, last_bet_size)
        
        return features


class ActionSequenceEncoder:
    """Encode action sequences for model input."""
    
    ACTION_TYPES = ['fold', 'check', 'call', 'bet', 'raise', 'allin']
    POSITIONS = ['UTG', 'HJ', 'CO', 'BTN', 'SB', 'BB']
    
    def __init__(self):
        self.action_to_idx = {act: idx for idx, act in enumerate(self.ACTION_TYPES)}
        self.position_to_idx = {pos: idx for idx, pos in enumerate(self.POSITIONS)}
    
    def encode_action_sequence(self, actions: List[Dict[str, any]], 
                              max_length: int = 20) -> Dict[str, any]:
        """
        Encode action sequence into numerical features.
        
        Args:
            actions: List of action dictionaries
            max_length: Maximum sequence length
            
        Returns:
            Dictionary of encoded features
        """
        if not actions:
            return {
                'action_count': 0,
                'fold_count': 0,
                'call_count': 0,
                'raise_count': 0,
                'aggression_factor': 0.0,
                'last_action_type': -1,
                'last_action_amount': 0.0
            }
        
        # Count action types
        action_counts = {'fold': 0, 'call': 0, 'bet': 0, 'raise': 0, 'allin': 0}
        for action in actions:
            action_type = action['action'].lower()
            for key in action_counts:
                if key in action_type:
                    action_counts[key] += 1
                    break
        
        # Aggression factor: (bets + raises) / calls
        aggressive_actions = action_counts['bet'] + action_counts['raise'] + action_counts['allin']
        passive_actions = action_counts['call']
        aggression_factor = aggressive_actions / passive_actions if passive_actions > 0 else aggressive_actions
        
        # Last action
        last_action = actions[-1]
        last_action_type = self.action_to_idx.get(last_action['action'].lower(), -1)
        last_action_amount = last_action.get('amount', 0.0) or 0.0
        
        features = {
            'action_count': len(actions),
            'fold_count': action_counts['fold'],
            'call_count': action_counts['call'],
            'raise_count': action_counts['bet'] + action_counts['raise'],
            'allin_count': action_counts['allin'],
            'aggression_factor': min(aggression_factor, 10.0),  # Cap for stability
            'last_action_type': last_action_type,
            'last_action_amount': last_action_amount
        }
        
        return features


class PokerFeatureEngineer:
    """Main feature engineering pipeline."""
    
    def __init__(self):
        self.hand_evaluator = HandStrengthEvaluator()
        self.position_extractor = PositionFeatureExtractor()
        self.pot_calculator = PotOddsCalculator()
        self.action_encoder = ActionSequenceEncoder()
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply full feature engineering pipeline.
        
        Args:
            df: DataFrame with preprocessed data
            
        Returns:
            DataFrame with engineered features
        """
        print("Engineering poker features...")
        
        df_features = df.copy()
        
        # Hand strength features
        if all(col in df.columns for col in ['hero_card1_rank', 'hero_card2_rank', 
                                               'hero_card1_suit', 'hero_card2_suit']):
            print("  - Hand strength features")
            hand_features = df.apply(
                lambda row: self.hand_evaluator.get_hand_strength(
                    row['hero_card1_rank'], row['hero_card2_rank'],
                    row['hero_card1_suit'], row['hero_card2_suit']
                ), axis=1
            )
            hand_features_df = pd.DataFrame(hand_features.tolist())
            df_features = pd.concat([df_features, hand_features_df], axis=1)
        
        # Position features
        if 'hero_pos' in df.columns:
            print("  - Position features")
            position_features = df['hero_pos'].apply(
                lambda pos: self.position_extractor.get_position_features(pos)
            )
            position_features_df = pd.DataFrame(position_features.tolist())
            df_features = pd.concat([df_features, position_features_df], axis=1)
        
        # Pot odds features
        if 'pot_size' in df.columns:
            print("  - Pot odds features")
            # Estimate effective stack (can be improved with actual stack data)
            df_features['estimated_stack'] = df_features['pot_size'] * 5  # Rough estimate
            
            pot_features = df_features.apply(
                lambda row: self.pot_calculator.get_pot_features(
                    row['pot_size'], row['estimated_stack']
                ), axis=1
            )
            pot_features_df = pd.DataFrame(pot_features.tolist())
            df_features = pd.concat([df_features, pot_features_df], axis=1)
        
        # Action sequence features
        if 'action_sequence' in df.columns:
            print("  - Action sequence features")
            action_features = df['action_sequence'].apply(
                self.action_encoder.encode_action_sequence
            )
            action_features_df = pd.DataFrame(action_features.tolist())
            df_features = pd.concat([df_features, action_features_df], axis=1)
        
        print(f"Final feature shape: {df_features.shape}")
        return df_features


def main():
    """Test feature engineering pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Engineer poker features")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/processed",
        help="Directory with preprocessed data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory for output"
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs='+',
        default=['train', 'test'],
        help="Data splits to process"
    )
    
    args = parser.parse_args()
    
    engineer = PokerFeatureEngineer()
    
    for split in args.splits:
        try:
            print(f"\n{'='*60}")
            print(f"Engineering features for {split} split")
            print('='*60)
            
            # Load preprocessed data
            input_path = f"{args.input_dir}/{split}_processed.parquet"
            df = pd.read_parquet(input_path)
            print(f"Loaded {len(df)} samples")
            
            # Engineer features
            df_engineered = engineer.engineer_features(df)
            
            # Save
            output_path = f"{args.output_dir}/{split}_features.parquet"
            df_engineered.to_parquet(output_path, index=False)
            print(f"Saved to {output_path}")
            
            # Print feature summary
            print(f"\nFeature columns ({len(df_engineered.columns)}):")
            print(df_engineered.columns.tolist())
            
        except Exception as e:
            print(f"Error processing {split}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("Feature engineering completed!")
    print('='*60)


if __name__ == "__main__":
    main()
