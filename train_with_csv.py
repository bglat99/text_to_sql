#!/usr/bin/env python3
"""
Train Llama 3.1 with Custom CSV Dataset
Supports loading your own financial text-to-SQL data for improved training
"""

import os
import pandas as pd
import torch
import logging
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class CustomFinancialDataset:
    """Custom dataset for financial text-to-SQL data"""
    
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0).clone()
        }

def load_csv_data(csv_file):
    """Load and validate CSV training data"""
    logger.info(f"Loading training data from {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
        
        # Validate required columns
        required_columns = ['query', 'sql']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        logger.info(f"Loaded {len(df)} training examples")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Format data for training
        training_data = []
        for idx, row in df.iterrows():
            query = str(row['query']).strip()
            sql = str(row['sql']).strip()
            
            # Format as instruction-following prompt
            formatted_text = f"Convert to SQL: {query}\nSQL: {sql}"
            training_data.append(formatted_text)
        
        logger.info("Sample training examples:")
        for i, example in enumerate(training_data[:3]):
            logger.info(f"  Example {i+1}: {example[:100]}...")
        
        return training_data
        
    except Exception as e:
        logger.error(f"Error loading CSV data: {e}")
        raise

def setup_model_and_tokenizer():
    """Load Llama 3.1 model with LoRA configuration"""
    logger.info("Loading Llama 3.1 8B model...")
    
    token = "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"
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=None,
            low_cpu_mem_usage=True
        )
        
        # Move to CPU for memory efficiency
        model = model.to('cpu')
        
        # LoRA configuration for financial domain
        lora_config = LoraConfig(
            r=16,  # Increased rank for better performance
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.train()
        
        # Check trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def train_with_custom_data(csv_file, output_dir="./custom_financial_model", max_steps=1000, learning_rate=1e-4):
    """Train model with custom CSV data"""
    
    logger.info("=" * 60)
    logger.info("CUSTOM FINANCIAL MODEL TRAINING")
    logger.info("=" * 60)
    
    # Load training data
    training_data = load_csv_data(csv_file)
    
    # Setup model
    model, tokenizer = setup_model_and_tokenizer()
    
    # Create dataset
    dataset = CustomFinancialDataset(training_data, tokenizer)
    
    # Training arguments optimized for custom data
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=max(50, max_steps // 20),  # 5% warmup
        max_steps=max_steps,
        learning_rate=learning_rate,
        fp16=False,
        bf16=False,
        logging_steps=max(10, max_steps // 40),  # Log 40 times during training
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        output_dir=output_dir,
        save_strategy="steps",
        save_steps=max(100, max_steps // 10),  # Save 10 checkpoints
        report_to="none",
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        gradient_checkpointing=False,
        dataloader_num_workers=0,
        use_cpu=True,  # Force CPU training for memory efficiency
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Start training
    logger.info(f"Starting training with {len(training_data)} examples...")
    logger.info(f"Max steps: {max_steps}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Output directory: {output_dir}")
    
    # Train
    trainer.train()
    
    # Save final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training info
    training_info = {
        "training_data_size": len(training_data),
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "training_date": datetime.now().isoformat(),
        "source_csv": csv_file
    }
    
    import json
    with open(f"{output_dir}/training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info(f"Model saved to: {output_dir}")
    logger.info(f"Training examples processed: {len(training_data)}")
    logger.info("=" * 60)
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(description="Train Llama 3.1 with custom CSV dataset")
    parser.add_argument("--csv_file", required=True, help="Path to CSV file with query,sql columns")
    parser.add_argument("--output_dir", default="./custom_financial_model", help="Output directory for trained model")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum training steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    # Validate CSV file exists
    if not os.path.exists(args.csv_file):
        logger.error(f"CSV file not found: {args.csv_file}")
        return
    
    logger.info(f"Training configuration:")
    logger.info(f"  CSV file: {args.csv_file}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Max steps: {args.max_steps}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    
    try:
        output_dir = train_with_custom_data(
            csv_file=args.csv_file,
            output_dir=args.output_dir,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate
        )
        
        logger.info(f"\n✅ SUCCESS! Your custom model is ready at: {output_dir}")
        logger.info(f"\n🧪 Test it with:")
        logger.info(f"python3 test_custom_model.py --model_path {output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()