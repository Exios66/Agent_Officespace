"""
LLM fine-tuning preparation and training scripts.

Supports:
- Data formatting for LLM fine-tuning
- Hugging Face Transformers training
- LoRA/PEFT efficient fine-tuning
- Instruction format preparation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch


class PokerLLMDataPreparator:
    """Prepare poker data for LLM fine-tuning."""
    
    def __init__(self, instruction_template: str = "alpaca"):
        """
        Initialize data preparator.
        
        Args:
            instruction_template: Template format ('alpaca', 'chatml', 'custom')
        """
        self.instruction_template = instruction_template
    
    def format_preflop_instruction(self, row: pd.Series) -> Dict[str, str]:
        """
        Format a preflop scenario into instruction format.
        
        Args:
            row: DataFrame row with poker data
            
        Returns:
            Dictionary with instruction and output
        """
        # Extract key information
        position = row.get('hero_pos', 'Unknown')
        hand = row.get('hero_holding', 'Unknown')
        action_seq = row.get('prev_line', '')
        pot_size = row.get('pot_size', 0)
        num_players = row.get('num_players', 6)
        decision = row.get('correct_decision', 'Unknown')
        
        # Create instruction
        instruction = f"""You are an expert No-Limit Texas Hold'em poker player. 
        
Game Scenario:
- Format: 6-handed No-Limit Texas Hold'em
- Your Position: {position}
- Your Hand: {hand}
- Number of Players: {num_players}
- Pot Size: {pot_size} BB
- Previous Actions: {action_seq if action_seq else 'No action yet'}

What is the optimal decision in this situation?"""
        
        output = decision
        
        if self.instruction_template == "alpaca":
            return {
                "instruction": instruction.strip(),
                "input": "",
                "output": output
            }
        elif self.instruction_template == "chatml":
            messages = [
                {"role": "system", "content": "You are an expert poker player specializing in Game Theory Optimal (GTO) play."},
                {"role": "user", "content": instruction.strip()},
                {"role": "assistant", "content": output}
            ]
            return {"messages": messages}
        else:
            return {
                "text": f"### Instruction:\n{instruction.strip()}\n\n### Response:\n{output}"
            }
    
    def prepare_dataset(self, df: pd.DataFrame, output_path: Optional[str] = None) -> Dataset:
        """
        Prepare dataset for fine-tuning.
        
        Args:
            df: DataFrame with poker data
            output_path: Optional path to save formatted data
            
        Returns:
            Hugging Face Dataset
        """
        print(f"Preparing dataset with {len(df)} samples...")
        
        # Format all rows
        formatted_data = []
        for idx, row in df.iterrows():
            try:
                formatted = self.format_preflop_instruction(row)
                formatted_data.append(formatted)
            except Exception as e:
                print(f"Error formatting row {idx}: {e}")
                continue
        
        print(f"Formatted {len(formatted_data)} samples")
        
        # Create dataset
        dataset = Dataset.from_list(formatted_data)
        
        # Save if requested
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as JSON
            with open(output_path, 'w') as f:
                json.dump(formatted_data, f, indent=2)
            print(f"Saved formatted data to {output_path}")
        
        return dataset


class PokerLLMTrainer:
    """Trainer for LLM fine-tuning on poker data."""
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-v0.1",
                 use_lora: bool = True, device: str = None):
        """
        Initialize LLM trainer.
        
        Args:
            model_name: HuggingFace model name
            use_lora: Whether to use LoRA for efficient fine-tuning
            device: Device for training
        """
        self.model_name = model_name
        self.use_lora = use_lora
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        print(f"Model: {model_name}")
        print(f"LoRA: {use_lora}")
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer."""
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Loading model...")
        
        # Load model with appropriate settings
        if self.use_lora:
            # Load in 8-bit for LoRA
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                load_in_8bit=True if self.device == "cuda" else False,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            # Prepare for k-bit training
            if self.device == "cuda":
                model = prepare_model_for_kbit_training(model)
            
            # Add LoRA adapters
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],  # Adjust based on model
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        else:
            # Full fine-tuning
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
        
        self.model = model
        print("Model loaded successfully!")
    
    def tokenize_function(self, examples):
        """Tokenize examples for training."""
        if "text" in examples:
            texts = examples["text"]
        elif "messages" in examples:
            texts = []
            for messages in examples["messages"]:
                parts = []
                for message in messages:
                    role = message["role"].capitalize()
                    parts.append(f"{role}: {message['content']}")
                texts.append("\n".join(parts))
        elif "instruction" in examples:
            texts = []
            for inst, inp, out in zip(
                examples["instruction"],
                examples["input"],
                examples["output"],
            ):
                if inp:
                    text = (
                        f"### Instruction:\n{inst}\n\n"
                        f"### Input:\n{inp}\n\n"
                        f"### Response:\n{out}"
                    )
                else:
                    text = f"### Instruction:\n{inst}\n\n### Response:\n{out}"
                texts.append(text)
        else:
            raise ValueError("Unknown data format")

        return self.tokenizer(
            texts,
            truncation=True,
            max_length=512,
            padding="max_length",
        )
    
    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None,
              output_dir: str = "data/models/llm_poker",
              num_epochs: int = 3,
              batch_size: int = 4,
              learning_rate: float = 2e-4,
              gradient_accumulation_steps: int = 4):
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            output_dir: Directory to save model
            num_epochs: Number of training epochs
            batch_size: Batch size per device
            learning_rate: Learning rate
            gradient_accumulation_steps: Gradient accumulation steps
        """
        print("\nPreparing training...")
        
        # Tokenize datasets
        print("Tokenizing datasets...")
        train_tokenized = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        if eval_dataset:
            eval_tokenized = eval_dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=eval_dataset.column_names
            )
        else:
            eval_tokenized = None
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        import inspect

        training_kwargs = dict(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            save_total_limit=3,
            load_best_model_at_end=bool(eval_tokenized),
            fp16=self.device == "cuda",
            report_to="none",
            remove_unused_columns=True,
        )

        if eval_tokenized is not None:
            training_kwargs["eval_steps"] = 500
            init_params = inspect.signature(TrainingArguments.__init__).parameters
            if "eval_strategy" in init_params:
                training_kwargs["eval_strategy"] = "steps"
            else:
                training_kwargs["evaluation_strategy"] = "steps"

        training_args = TrainingArguments(**training_kwargs)
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=eval_tokenized,
            data_collator=data_collator,
        )
        
        # Train
        print("\nStarting training...")
        trainer.train()
        
        # Save final model
        print(f"\nSaving model to {output_dir}")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print("\nTraining completed!")
        
        return trainer


def main():
    """Main script for LLM fine-tuning."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune LLM on poker data")
    parser.add_argument(
        "--data-dir", type=str, default="data/processed",
        help="Directory with processed data"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/models/llm_poker",
        help="Directory to save model"
    )
    parser.add_argument(
        "--model-name", type=str, default="mistralai/Mistral-7B-v0.1",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Use LoRA for efficient fine-tuning",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA and run full fine-tuning",
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size per device"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--prepare-only", action="store_true",
        help="Only prepare data without training"
    )
    
    args = parser.parse_args()
    
    # Load data
    print("Loading processed data...")
    train_path = Path(args.data_dir) / "train_processed.parquet"
    test_path = Path(args.data_dir) / "test_processed.parquet"
    
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path) if test_path.exists() else None
    
    print(f"Loaded {len(df_train)} training samples")
    if df_test is not None:
        print(f"Loaded {len(df_test)} test samples")
    
    # Prepare data
    preparator = PokerLLMDataPreparator(instruction_template="alpaca")
    
    formatted_train_path = Path(args.data_dir) / "train_llm_format.json"
    train_dataset = preparator.prepare_dataset(df_train, formatted_train_path)
    
    eval_dataset = None
    if df_test is not None:
        formatted_test_path = Path(args.data_dir) / "test_llm_format.json"
        eval_dataset = preparator.prepare_dataset(df_test, formatted_test_path)
    
    if args.prepare_only:
        print("\nData preparation completed!")
        print(f"Training data: {formatted_train_path}")
        if eval_dataset:
            print(f"Test data: {formatted_test_path}")
        return
    
    use_lora = args.use_lora and not args.no_lora

    # Train model
    trainer_obj = PokerLLMTrainer(
        model_name=args.model_name,
        use_lora=use_lora,
    )
    
    trainer_obj.setup_model_and_tokenizer()
    
    trainer_obj.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    print(f"\n{'='*60}")
    print("LLM fine-tuning completed!")
    print(f"Model saved to: {args.output_dir}")
    print('='*60)


if __name__ == "__main__":
    main()
