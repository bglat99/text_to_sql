#!/usr/bin/env python3
"""
Memory-Optimized Training for Large Financial Datasets
Handles 6000+ examples efficiently on MacBook Air M3
"""

import os
import pandas as pd
import torch
import logging
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datetime import datetime
import gc

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class MemoryEfficientDataset:
    """Memory-efficient dataset that processes data in chunks"""
    
    def __init__(self, data, tokenizer, max_length=256):  # Reduced max_length
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # Tokenize with reduced max_length for memory efficiency
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,  # Much smaller for memory
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0).clone()
        }

def load_and_sample_data(csv_file, sample_size=1000):
    """Load data and sample for memory efficiency"""
    logger.info(f"Loading training data from {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
        
        # Validate required columns
        required_columns = ['query', 'sql']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        logger.info(f"Loaded {len(df)} total examples")
        
        # Sample data for memory efficiency
        if len(df) > sample_size:
            logger.info(f"Sampling {sample_size} examples for memory efficiency")
            df = df.sample(n=sample_size, random_state=42)
        
        # Format data for training
        training_data = []
        for idx, row in df.iterrows():
            query = str(row['query']).strip()
            sql = str(row['sql']).strip()
            
            # Shorter format for memory efficiency
            formatted_text = f"Q: {query}\nA: {sql}"
            training_data.append(formatted_text)
        
        logger.info(f"Using {len(training_data)} training examples")
        logger.info("Sample training examples:")
        for i, example in enumerate(training_data[:2]):
            logger.info(f"  Example {i+1}: {example[:100]}...")
        
        return training_data
        
    except Exception as e:
        logger.error(f"Error loading CSV data: {e}")
        raise

def setup_lightweight_model():
    """Load model with maximum memory optimization"""
    logger.info("Loading Llama 3.1 8B model with memory optimization...")
    
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
        
        # Load model with maximum memory optimization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=None,
            low_cpu_mem_usage=True,
            offload_folder="./offload"  # Offload to disk
        )
        
        # Force CPU usage
        model = model.to('cpu')
        
        # Minimal LoRA configuration for memory efficiency
        lora_config = LoraConfig(
            r=8,  # Reduced rank
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],  # Fewer target modules
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

def train_large_dataset(csv_file, output_dir="./large_financial_model", max_steps=500, learning_rate=2e-4, sample_size=1000):
    """Train model with large dataset using memory optimization"""
    
    logger.info("=" * 60)
    logger.info("LARGE DATASET FINANCIAL MODEL TRAINING")
    logger.info("=" * 60)
    
    # Load training data (sampled for memory)
    training_data = load_and_sample_data(csv_file, sample_size)
    
    # Setup model
    model, tokenizer = setup_lightweight_model()
    
    # Create dataset
    dataset = MemoryEfficientDataset(training_data, tokenizer)
    
    # Training arguments optimized for large datasets
    training_args = TrainingArguments(
        per_device_train_batch_size=1,  # Minimal batch size
        gradient_accumulation_steps=8,  # Higher accumulation
        warmup_steps=50,
        max_steps=max_steps,
        learning_rate=learning_rate,
        fp16=False,
        bf16=False,
        logging_steps=25,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        output_dir=output_dir,
        save_strategy="steps",
        save_steps=100,
        report_to="none",
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        gradient_checkpointing=True,  # Enable for memory saving
        dataloader_num_workers=0,
        use_cpu=True,  # Force CPU training
        max_grad_norm=1.0,  # Gradient clipping
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
    
    try:
        # Train
        trainer.train()
        
        # Save final model
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save training info
        training_info = {
            "training_data_size": len(training_data),
            "total_dataset_size": 6222,  # Your full dataset size
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "training_date": datetime.now().isoformat(),
            "source_csv": csv_file,
            "sample_size": sample_size
        }
        
        import json
        with open(f"{output_dir}/training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info(f"Model saved to: {output_dir}")
        logger.info(f"Training examples processed: {len(training_data)}")
        logger.info(f"Total dataset available: 6,222 examples")
        logger.info("=" * 60)
        
        return output_dir
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Clean up memory
        del model, tokenizer, dataset
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

def main():
    parser = argparse.ArgumentParser(description="Train Llama 3.1 with large financial dataset")
    parser.add_argument("--csv_file", required=True, help="Path to CSV file with query,sql columns")
    parser.add_argument("--output_dir", default="./large_financial_model", help="Output directory for trained model")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum training steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--sample_size", type=int, default=1000, help="Number of examples to sample for training")
    
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
    logger.info(f"  Sample size: {args.sample_size}")
    
    try:
        output_dir = train_large_dataset(
            csv_file=args.csv_file,
            output_dir=args.output_dir,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            sample_size=args.sample_size
        )
        
        logger.info(f"\n✅ SUCCESS! Your large dataset model is ready at: {output_dir}")
        logger.info(f"\n🧪 Test it with:")
        logger.info(f"python3 test_custom_model.py --model_path {output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()