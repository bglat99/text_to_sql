#!/usr/bin/env python3
"""
Ultra-Lightweight Training for MacBook Air M3
Uses smaller model or extreme memory optimization
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

class UltraLightDataset:
    """Ultra-lightweight dataset with minimal memory footprint"""
    
    def __init__(self, data, tokenizer, max_length=128):  # Very small max_length
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # Minimal tokenization
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

def load_minimal_data(csv_file, sample_size=200):
    """Load minimal data for ultra-lightweight training"""
    logger.info(f"Loading minimal training data from {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
        
        # Sample very small subset
        if len(df) > sample_size:
            logger.info(f"Sampling {sample_size} examples for ultra-lightweight training")
            df = df.sample(n=sample_size, random_state=42)
        
        # Format data with minimal text
        training_data = []
        for idx, row in df.iterrows():
            query = str(row['query']).strip()
            sql = str(row['sql']).strip()
            
            # Ultra-short format
            formatted_text = f"{query} -> {sql}"
            training_data.append(formatted_text)
        
        logger.info(f"Using {len(training_data)} ultra-lightweight examples")
        return training_data
        
    except Exception as e:
        logger.error(f"Error loading CSV data: {e}")
        raise

def setup_ultra_light_model():
    """Try smaller models first, fallback to Llama with extreme optimization"""
    
    # Try smaller models first
    small_models = [
        "microsoft/DialoGPT-medium",  # Much smaller
        "distilgpt2",  # Very small
        "gpt2",  # Small
    ]
    
    token = "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"
    
    for model_name in small_models:
        try:
            logger.info(f"Trying smaller model: {model_name}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=None,
                low_cpu_mem_usage=True
            )
            
            # Move to CPU
            model = model.to('cpu')
            
            # Minimal LoRA
            lora_config = LoraConfig(
                r=4,  # Very small rank
                lora_alpha=8,
                target_modules=["c_attn", "c_proj"] if "gpt" in model_name else ["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            # Apply LoRA
            model = get_peft_model(model, lora_config)
            model.train()
            
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            
            logger.info(f"✅ Successfully loaded {model_name}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
            
            return model, tokenizer, model_name
            
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}")
            continue
    
    # Fallback: Try Llama with extreme optimization
    try:
        logger.info("Falling back to Llama 3.1 with extreme memory optimization...")
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        # Extreme memory optimization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=None,
            low_cpu_mem_usage=True,
            offload_folder="./offload",
            max_memory={0: "2GB", "cpu": "6GB"}  # Limit memory usage
        )
        
        model = model.to('cpu')
        
        # Minimal LoRA
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        model.train()
        
        return model, tokenizer, model_name
        
    except Exception as e:
        logger.error(f"All models failed to load: {e}")
        raise

def train_ultra_light(csv_file, output_dir="./ultra_light_model", max_steps=100, learning_rate=5e-4, sample_size=200):
    """Ultra-lightweight training"""
    
    logger.info("=" * 60)
    logger.info("ULTRA-LIGHTWEIGHT FINANCIAL MODEL TRAINING")
    logger.info("=" * 60)
    
    # Load minimal data
    training_data = load_minimal_data(csv_file, sample_size)
    
    # Setup model
    model, tokenizer, model_name = setup_ultra_light_model()
    
    # Create dataset
    dataset = UltraLightDataset(training_data, tokenizer)
    
    # Ultra-lightweight training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,  # Very high accumulation
        warmup_steps=10,
        max_steps=max_steps,
        learning_rate=learning_rate,
        fp16=False,
        bf16=False,
        logging_steps=10,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir=output_dir,
        save_strategy="steps",
        save_steps=50,
        report_to="none",
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        use_cpu=True,
        max_grad_norm=1.0,
        save_total_limit=1,  # Only keep 1 checkpoint
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
    logger.info(f"Starting ultra-lightweight training...")
    logger.info(f"Model: {model_name}")
    logger.info(f"Training examples: {len(training_data)}")
    logger.info(f"Max steps: {max_steps}")
    logger.info(f"Learning rate: {learning_rate}")
    
    try:
        # Train
        trainer.train()
        
        # Save model
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save training info
        training_info = {
            "model_name": model_name,
            "training_data_size": len(training_data),
            "total_dataset_size": 6222,
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
        logger.info("ULTRA-LIGHTWEIGHT TRAINING COMPLETED!")
        logger.info(f"Model saved to: {output_dir}")
        logger.info(f"Model used: {model_name}")
        logger.info("=" * 60)
        
        return output_dir
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Clean up
        del model, tokenizer, dataset
        gc.collect()

def main():
    parser = argparse.ArgumentParser(description="Ultra-lightweight training for MacBook Air M3")
    parser.add_argument("--csv_file", required=True, help="Path to CSV file")
    parser.add_argument("--output_dir", default="./ultra_light_model", help="Output directory")
    parser.add_argument("--max_steps", type=int, default=100, help="Max training steps")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--sample_size", type=int, default=200, help="Sample size")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        logger.error(f"CSV file not found: {args.csv_file}")
        return
    
    try:
        output_dir = train_ultra_light(
            csv_file=args.csv_file,
            output_dir=args.output_dir,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            sample_size=args.sample_size
        )
        
        logger.info(f"\n✅ SUCCESS! Ultra-lightweight model ready at: {output_dir}")
        logger.info(f"\n🧪 Test it with:")
        logger.info(f"python3 test_custom_model.py --model_path {output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")

if __name__ == "__main__":
    main()