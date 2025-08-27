#!/usr/bin/env python3
"""
Enhanced Financial Text-to-SQL Fine-tuning
More data, better prompts, longer training
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
    """Load model without quantization for Apple Silicon"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        
        logger.info("Loading model without quantization for Apple Silicon...")
        
        # Use a smaller model that works well on Apple Silicon
        model_name = "microsoft/DialoGPT-medium"
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        # Load model without quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Add LoRA adapters
        logger.info("Adding LoRA adapters...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["c_attn", "c_proj"],  # DialoGPT uses different module names
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        
        logger.info("✅ Model and tokenizer loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def generate_enhanced_training_data(n_samples: int = 1000) -> List[Dict]:
    """Generate enhanced synthetic financial text-to-SQL training pairs"""
    import random
    
    logger.info(f"Generating {n_samples} enhanced training samples...")
    
    tickers = ['SPY', 'AAPL', 'NFLX', 'GOOGL', 'AMZN', 'META', 'MSFT', 'TSLA', 'ABNB']
    
    # Enhanced template patterns for financial queries
    templates = [
        # Basic queries
        ("Show me {ticker}'s closing prices for the last 30 days",
         "SELECT Date, Close FROM stock_data WHERE ticker = '{ticker}' AND Date >= date('now', '-30 days') ORDER BY Date;"),
        
        ("What was {ticker}'s trading volume yesterday?",
         "SELECT Date, Volume FROM stock_data WHERE ticker = '{ticker}' ORDER BY Date DESC LIMIT 1;"),
        
        ("Get {ticker}'s latest price",
         "SELECT Date, Close FROM stock_data WHERE ticker = '{ticker}' ORDER BY Date DESC LIMIT 1;"),
        
        # Financial analysis
        ("Calculate {ticker}'s volatility over the past month",
         "SELECT ticker, SQRT(AVG(move * move)) as volatility FROM stock_data WHERE ticker = '{ticker}' AND Date >= date('now', '-30 days');"),
        
        ("Show me {ticker}'s best performing days this year",
         "SELECT Date, Close, move FROM stock_data WHERE ticker = '{ticker}' AND Date >= '2025-01-01' ORDER BY move DESC LIMIT 10;"),
        
        ("Find all days when {ticker} moved more than 5%",
         "SELECT Date, Close, move FROM stock_data WHERE ticker = '{ticker}' AND ABS(move) > 5 ORDER BY Date;"),
        
        ("What is {ticker}'s average daily return?",
         "SELECT AVG(move) as avg_return FROM stock_data WHERE ticker = '{ticker}';"),
        
        # Comparative analysis
        ("Compare {ticker1} and {ticker2}'s average returns this year",
         "SELECT ticker, AVG(move) as avg_return FROM stock_data WHERE ticker IN ('{ticker1}', '{ticker2}') AND Date >= '2025-01-01' GROUP BY ticker;"),
        
        ("Which stock performed better, {ticker1} or {ticker2}?",
         "SELECT ticker, AVG(move) as performance FROM stock_data WHERE ticker IN ('{ticker1}', '{ticker2}') GROUP BY ticker ORDER BY performance DESC;"),
        
        # Market analysis
        ("What were the top 3 gainers yesterday?",
         "SELECT ticker, move FROM stock_data WHERE Date = (SELECT MAX(Date) FROM stock_data) ORDER BY move DESC LIMIT 3;"),
        
        ("Show me all stocks that gapped up more than 3%",
         "SELECT ticker, Date, Open, Close, move FROM stock_data WHERE move > 3 ORDER BY move DESC;"),
        
        # Time-based queries
        ("Get {ticker}'s monthly returns for 2024",
         "SELECT strftime('%Y-%m', Date) as month, ((MAX(Close) - MIN(Close)) / MIN(Close) * 100) as monthly_return FROM stock_data WHERE ticker = '{ticker}' AND strftime('%Y', Date) = '2024' GROUP BY strftime('%Y-%m', Date) ORDER BY month;"),
        
        ("Show me {ticker}'s price range this week",
         "SELECT MIN(Low) as min_price, MAX(High) as max_price FROM stock_data WHERE ticker = '{ticker}' AND Date >= date('now', '-7 days');"),
        
        # Volume analysis
        ("Which day had the highest volume for {ticker}?",
         "SELECT Date, Volume FROM stock_data WHERE ticker = '{ticker}' ORDER BY Volume DESC LIMIT 1;"),
        
        ("Calculate {ticker}'s average volume over the last month",
         "SELECT AVG(Volume) as avg_volume FROM stock_data WHERE ticker = '{ticker}' AND Date >= date('now', '-30 days');"),
    ]
    
    training_pairs = []
    
    for i in range(n_samples):
        template_text, template_sql = random.choice(templates)
        
        # Fill in placeholders
        if "{ticker1}" in template_text and "{ticker2}" in template_text:
            ticker1, ticker2 = random.sample(tickers, 2)
            text = template_text.format(ticker1=ticker1, ticker2=ticker2)
            sql = template_sql.format(ticker1=ticker1, ticker2=ticker2)
        else:
            ticker = random.choice(tickers)
            text = template_text.format(ticker=ticker)
            sql = template_sql.format(ticker=ticker)
        
        # Create enhanced text format with better prompting
        full_text = f"### Financial Query to SQL Conversion\n\nQuery: {text}\n\nSQL Query:\n{sql}"
        
        training_pairs.append({
            "text": full_text
        })
    
    logger.info(f"✅ Generated {len(training_pairs)} enhanced training pairs")
    return training_pairs

def create_custom_dataset(train_data, tokenizer):
    """Create a custom dataset that properly handles tokenization"""
    from torch.utils.data import Dataset
    
    class FinancialSQLDataset(Dataset):
        def __init__(self, data, tokenizer, max_length=512):  # Increased max_length
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

def main():
    logger.info("=" * 60)
    logger.info("Enhanced Financial Text-to-SQL Fine-tuning")
    logger.info("More Data, Better Prompts, Longer Training")
    logger.info("=" * 60)
    
    try:
        # Check for MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            logger.info("✅ Apple Silicon detected - MPS backend available")
        
        # Generate or load training data
        training_file = "enhanced_training_data.json"
        if os.path.exists(training_file):
            logger.info("Loading existing training data...")
            with open(training_file, 'r') as f:
                train_data = json.load(f)
        else:
            logger.info("Generating new enhanced training data...")
            train_data = generate_enhanced_training_data(1000)  # More training data
            with open(training_file, 'w') as f:
                json.dump(train_data, f, indent=2)
            logger.info(f"✅ Training data saved to {training_file}")
        
        # Load model and tokenizer
        logger.info("Loading model and tokenizer...")
        model, tokenizer = setup_model_and_tokenizer()
        
        # Create custom dataset
        logger.info("Creating custom dataset...")
        dataset = create_custom_dataset(train_data, tokenizer)
        
        # Training configuration
        from transformers import TrainingArguments, Trainer
        
        logger.info("Starting enhanced training...")
        
        training_args = TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,  # More gradient accumulation
            warmup_steps=20,  # More warmup
            max_steps=200,  # More training steps
            learning_rate=5e-5,  # Lower learning rate
            fp16=False,
            bf16=False,
            logging_steps=10,
            optim="adamw_torch",
            weight_decay=0.01,
            lr_scheduler_type="cosine",  # Better scheduler
            seed=3407,
            output_dir="./enhanced_outputs",
            save_strategy="steps",
            save_steps=50,
            report_to="none",
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            gradient_checkpointing=False,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )
        
        # Train the model
        trainer.train()
        
        # Save the final model
        output_dir = "./enhanced_financial_model"
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        logger.info(f"✅ Enhanced training completed! Model saved to {output_dir}")
        logger.info("🎉 Ready to convert financial queries to SQL!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main() 