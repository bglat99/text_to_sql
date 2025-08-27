#!/usr/bin/env python3
"""
Financial Text-to-SQL Fine-tuning for Apple Silicon
Uses a model compatible with MPS backend
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
    """Load model compatible with Apple Silicon"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        
        logger.info("Loading model compatible with Apple Silicon...")
        
        # Use a model that works well on Apple Silicon
        model_name = "microsoft/DialoGPT-medium"  # Smaller model for testing
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        # Configure quantization for Apple Silicon
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Prepare model for training
        model = prepare_model_for_kbit_training(model)
        
        # Add LoRA adapters
        logger.info("Adding LoRA adapters...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
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

def generate_training_data(n_samples: int = 2000) -> List[Dict]:
    """Generate synthetic financial text-to-SQL training pairs"""
    import random
    
    logger.info(f"Generating {n_samples} training samples...")
    
    tickers = ['SPY', 'AAPL', 'NFLX', 'GOOGL', 'AMZN', 'META', 'MSFT', 'TSLA', 'ABNB']
    
    schema_info = """
Table: stock_data
Columns:
- Date (TEXT): Trading date in YYYY-MM-DD format
- Open (REAL): Opening price
- High (REAL): Highest price of the day  
- Low (REAL): Lowest price of the day
- Close (REAL): Closing price
- Adj_Close (REAL): Adjusted closing price
- Volume (INTEGER): Trading volume
- ticker (TEXT): Stock ticker symbol
- move (REAL): Daily percentage change from previous day

Available tickers: SPY, AAPL, NFLX, GOOGL, AMZN, META, MSFT, TSLA, ABNB
"""
    
    # Template patterns for financial queries
    templates = [
        # Basic queries
        ("Show me {ticker}'s closing prices for the last 30 days",
         "SELECT Date, Close FROM stock_data WHERE ticker = '{ticker}' AND Date >= date('now', '-30 days') ORDER BY Date;"),
        
        ("What was {ticker}'s trading volume yesterday?",
         "SELECT Date, Volume FROM stock_data WHERE ticker = '{ticker}' ORDER BY Date DESC LIMIT 1;"),
        
        # Financial analysis
        ("Calculate {ticker}'s volatility over the past month",
         "SELECT ticker, SQRT(AVG(move * move)) as volatility FROM stock_data WHERE ticker = '{ticker}' AND Date >= date('now', '-30 days');"),
        
        ("Show me {ticker}'s best performing days this year",
         "SELECT Date, Close, move FROM stock_data WHERE ticker = '{ticker}' AND Date >= '2025-01-01' ORDER BY move DESC LIMIT 10;"),
        
        ("Find all days when {ticker} moved more than 5%",
         "SELECT Date, Close, move FROM stock_data WHERE ticker = '{ticker}' AND ABS(move) > 5 ORDER BY Date;"),
        
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
    ]
    
    training_pairs = []
    
    for i in range(n_samples):
        template_text, template_sql = random.choice(templates)
        
        # Fill in placeholders
        ticker = random.choice(tickers)
        ticker1, ticker2 = random.sample(tickers, 2)
        
        text = template_text.format(ticker=ticker, ticker1=ticker1, ticker2=ticker2)
        sql = template_sql.format(ticker=ticker, ticker1=ticker1, ticker2=ticker2)
        
        training_pairs.append({
            "instruction": "Convert this financial query to SQL:",
            "input": f"Schema:\n{schema_info}\n\nQuery: {text}",
            "output": sql
        })
    
    logger.info(f"✅ Generated {len(training_pairs)} training pairs")
    return training_pairs

def format_training_data(examples):
    """Format data for training"""
    texts = []
    for i in range(len(examples["instruction"])):
        text = f"""### Instruction:
{examples["instruction"][i]}

### Input:
{examples["input"][i]}

### Response:
{examples["output"][i]}"""
        texts.append(text)
    return {"text": texts}

def tokenize_function(examples):
    """Tokenize the training examples"""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=1024,  # Shorter for DialoGPT
        return_tensors=None
    )

def main():
    logger.info("=" * 60)
    logger.info("Financial Text-to-SQL Fine-tuning")
    logger.info("Optimized for Apple Silicon")
    logger.info("=" * 60)
    
    try:
        # Check for MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            logger.info("✅ Apple Silicon detected - MPS backend available")
        
        # Generate or load training data
        training_file = "financial_training_data.json"
        if os.path.exists(training_file):
            logger.info("Loading existing training data...")
            with open(training_file, 'r') as f:
                train_data = json.load(f)
        else:
            logger.info("Generating new training data...")
            train_data = generate_training_data(2000)
            with open(training_file, 'w') as f:
                json.dump(train_data, f, indent=2)
            logger.info(f"✅ Training data saved to {training_file}")
        
        # Load model and tokenizer
        logger.info("Loading model and tokenizer...")
        model, tokenizer = setup_model_and_tokenizer()
        
        # Prepare dataset
        from datasets import Dataset
        dataset = Dataset.from_list(train_data)
        dataset = dataset.map(format_training_data, batched=True)
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Training configuration
        from transformers import TrainingArguments, Trainer
        
        logger.info("Starting training...")
        
        training_args = TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            max_steps=500,
            learning_rate=2e-4,
            fp16=False,  # Use bf16 for Apple Silicon
            bf16=True,   # Better for Apple Silicon
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="./outputs",
            save_strategy="steps",
            save_steps=100,
            report_to="none",  # Disable wandb
            dataloader_pin_memory=False,  # Disable for Apple Silicon
            remove_unused_columns=False,
            gradient_checkpointing=True,
        )
        
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=tokenized_dataset,
            args=training_args,
        )
        
        # Train the model
        trainer.train()
        
        # Save the final model
        output_dir = "./financial_sql_model"
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        logger.info(f"✅ Training completed! Model saved to {output_dir}")
        logger.info("🎉 Ready to convert financial queries to SQL!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main() 