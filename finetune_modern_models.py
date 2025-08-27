#!/usr/bin/env python3
"""
Modern Model Fine-tuning: Llama 3 or Mistral 7B Only
Apple Silicon Compatible
"""

import os
import json
import logging
import torch
from typing import List, Dict
import warnings

# Suppress SSL warning
warnings.filterwarnings("ignore", category=UserWarning)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

def setup_model_and_tokenizer():
    """Load ONLY Llama 3 or Mistral 7B models"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        
        logger.info("Loading Modern Models: Llama 3 or Mistral 7B Only")
        
        # Set Hugging Face token
        token = "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"
        
        # ONLY modern models - no fallbacks
        model_options = [
            "meta-llama/Meta-Llama-3.1-1B-Instruct",  # Llama 3.1 1B (smaller, fits in memory)
            "meta-llama/Meta-Llama-3.1-8B-Instruct",  # Llama 3.1 8B
            "meta-llama/Meta-Llama-3.1-8B",           # Llama 3.1 8B Base
            "mistralai/Mistral-7B-Instruct-v0.2",     # Mistral 7B Instruct
            "mistralai/Mistral-7B-v0.1",              # Mistral 7B Base
        ]
        
        model = None
        tokenizer = None
        selected_model = None
        
        for model_name in model_options:
            try:
                logger.info(f"Trying modern model: {model_name}")
                
                # Load tokenizer with token
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    token=token,
                    trust_remote_code=True
                )
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = "right"
                
                # Load model without quantization for Apple Silicon
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    token=token,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    device_map=None,  # Don't use device_map to avoid meta tensors
                    low_cpu_mem_usage=False  # Disable to avoid meta tensors
                )
                
                selected_model = model_name
                logger.info(f"✅ SUCCESS! Loaded modern model: {model_name}")
                break
                
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue
        
        if model is None:
            logger.error("❌ FAILED: Could not load ANY modern model (Llama 3 or Mistral 7B)")
            logger.error("Please check your Hugging Face authentication or internet connection")
            logger.error("Available models tried:")
            for model_name in model_options:
                logger.error(f"  - {model_name}")
            raise Exception("No modern models could be loaded. Please check authentication.")
        
        # Add LoRA adapters
        logger.info("Adding LoRA adapters...")
        
        # Determine target modules based on model type
        if "llama" in selected_model.lower():
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "mistral" in selected_model.lower():
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        else:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)

        # Ensure model is in training mode and parameters require gradients
        model.train()
        for param in model.parameters():
            if param.requires_grad:
                param.requires_grad_(True)

        logger.info("✅ Model and tokenizer loaded successfully!")
        logger.info(f"Using modern model: {selected_model}")
        return model, tokenizer, selected_model
        
    except Exception as e:
        logger.error(f"Failed to load modern model: {e}")
        raise

def generate_training_data(n_samples: int = 500) -> List[Dict]:
    """Generate synthetic financial text-to-SQL training pairs"""
    import random
    
    logger.info(f"Generating {n_samples} training samples...")
    
    tickers = ['SPY', 'AAPL', 'NFLX', 'GOOGL', 'AMZN', 'META', 'MSFT', 'TSLA', 'ABNB']
    
    # Template patterns for financial queries
    templates = [
        ("Show me {ticker}'s closing prices for the last 30 days",
         "SELECT Date, Close FROM stock_data WHERE ticker = '{ticker}' AND Date >= date('now', '-30 days') ORDER BY Date;"),
        
        ("What was {ticker}'s trading volume yesterday?",
         "SELECT Date, Volume FROM stock_data WHERE ticker = '{ticker}' ORDER BY Date DESC LIMIT 1;"),
        
        ("Calculate {ticker}'s volatility over the past month",
         "SELECT ticker, SQRT(AVG(move * move)) as volatility FROM stock_data WHERE ticker = '{ticker}' AND Date >= date('now', '-30 days');"),
        
        ("Show me {ticker}'s best performing days this year",
         "SELECT Date, Close, move FROM stock_data WHERE ticker = '{ticker}' AND Date >= '2025-01-01' ORDER BY move DESC LIMIT 10;"),
        
        ("Find all days when {ticker} moved more than 5%",
         "SELECT Date, Close, move FROM stock_data WHERE ticker = '{ticker}' AND ABS(move) > 5 ORDER BY Date;"),
        
        ("Compare {ticker1} and {ticker2}'s average returns this year",
         "SELECT ticker, AVG(move) as avg_return FROM stock_data WHERE ticker IN ('{ticker1}', '{ticker2}') AND Date >= '2025-01-01' GROUP BY ticker;"),
        
        ("What were the top 3 gainers yesterday?",
         "SELECT ticker, move FROM stock_data WHERE Date = (SELECT MAX(Date) FROM stock_data) ORDER BY move DESC LIMIT 3;"),
        
        ("Show me all stocks that gapped up more than 3%",
         "SELECT ticker, Date, Open, Close, move FROM stock_data WHERE move > 3 ORDER BY move DESC;"),
    ]
    
    training_pairs = []
    
    for i in range(n_samples):
        template_text, template_sql = random.choice(templates)
        
        # Fill in placeholders
        if "{ticker1}" in template_text and "{ticker2}" in template_sql:
            ticker1, ticker2 = random.sample(tickers, 2)
            text = template_text.format(ticker1=ticker1, ticker2=ticker2)
            sql = template_sql.format(ticker1=ticker1, ticker2=ticker2)
        else:
            ticker = random.choice(tickers)
            text = template_text.format(ticker=ticker)
            sql = template_sql.format(ticker=ticker)
        
        # Create training text
        full_text = f"Convert to SQL: {text}\n\nSQL: {sql}"
        
        training_pairs.append({
            "text": full_text
        })
    
    logger.info(f"✅ Generated {len(training_pairs)} training pairs")
    return training_pairs

def create_custom_dataset(train_data, tokenizer):
    """Create a custom dataset that properly handles tokenization"""
    from torch.utils.data import Dataset

    class FinancialSQLDataset(Dataset):
        def __init__(self, data, tokenizer, max_length=512):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            text = self.data[idx]["text"]

            # Tokenize the text
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )

            # Remove the batch dimension
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": encoding["input_ids"].squeeze(0).clone()
            }

    return FinancialSQLDataset(train_data, tokenizer)

# Import Trainer for custom class
from transformers import Trainer

class AppleSiliconTrainer(Trainer):
    """Custom trainer for Apple Silicon with meta tensor handling"""
    
    def _move_model_to_device(self, model, device):
        """Override to handle meta tensors properly"""
        if hasattr(model, 'is_loaded_in_8bit') and model.is_loaded_in_8bit:
            # For 8-bit models, don't move to device
            return model
        elif hasattr(model, 'is_loaded_in_4bit') and model.is_loaded_in_4bit:
            # For 4-bit models, don't move to device
            return model
        else:
            # For regular models, use to_empty for meta tensors
            try:
                return model.to(device)
            except NotImplementedError as e:
                if "meta tensor" in str(e):
                    return model.to_empty(device=device)
                else:
                    raise e

def main():
    logger.info("=" * 60)
    logger.info("Modern Model Fine-tuning: Llama 3 or Mistral 7B Only")
    logger.info("Apple Silicon Compatible")
    logger.info("=" * 60)
    
    try:
        # Check for MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            logger.info("✅ Apple Silicon detected - MPS backend available")
        
        # Generate or load training data
        training_file = "modern_training_data.json"
        if os.path.exists(training_file):
            logger.info("Loading existing training data...")
            with open(training_file, 'r') as f:
                train_data = json.load(f)
        else:
            logger.info("Generating new training data...")
            train_data = generate_training_data(500)
            with open(training_file, 'w') as f:
                json.dump(train_data, f, indent=2)
            logger.info(f"✅ Training data saved to {training_file}")
        
        # Load model and tokenizer
        logger.info("Loading modern model and tokenizer...")
        model, tokenizer, model_name = setup_model_and_tokenizer()
        
        # Create custom dataset
        logger.info("Creating custom dataset...")
        dataset = create_custom_dataset(train_data, tokenizer)
        
        # Training configuration
        from transformers import TrainingArguments, Trainer
        
        logger.info("Starting training...")
        
        training_args = TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            max_steps=200,  # More steps for better training
            learning_rate=1e-4,
            fp16=False,
            bf16=False,
            logging_steps=10,
            optim="adamw_torch",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="./modern_model_outputs",
            save_strategy="steps",
            save_steps=50,
            report_to="none",
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            gradient_checkpointing=True,  # Enable to save memory
            dataloader_num_workers=0,  # Reduce memory usage
            use_cpu=True,  # Use CPU instead of MPS to avoid memory issues
        )
        
        trainer = AppleSiliconTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )
        
        # Train the model
        trainer.train()
        
        # Save the final model
        output_dir = "./modern_financial_model"
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save model info
        model_info = {
            "base_model": model_name,
            "training_steps": 200,
            "learning_rate": 1e-4,
            "dataset_size": len(train_data)
        }
        
        with open(os.path.join(output_dir, "model_info.json"), 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"✅ Training completed! Model saved to {output_dir}")
        logger.info(f"Modern model used: {model_name}")
        logger.info("🎉 Ready to convert financial queries to SQL!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error("Please check your Hugging Face authentication or try again later.")
        raise

if __name__ == "__main__":
    main() 