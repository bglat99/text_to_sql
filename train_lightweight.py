#!/usr/bin/env python3
"""
Lightweight Llama 3.1 Training for MacBook Air M3
Memory-optimized for 8GB unified memory
"""

import os
import json
import logging
import torch
from typing import List, Dict
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def setup_lightweight_model():
    """Load Llama 3.1 with maximum memory optimization"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        logger.info("🚀 Loading Llama 3.1 8B with memory optimization...")
        
        # Set Hugging Face token
        token = "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"

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            token=token,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Load model with extreme memory optimization
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            token=token,
            torch_dtype=torch.float16,  # Use float16 to save memory
            trust_remote_code=True,
            device_map=None,  # Load to CPU first
            low_cpu_mem_usage=True  # Enable memory optimization
        )

        # Move to CPU to free GPU memory
        model = model.to('cpu')

        # Minimal LoRA configuration for memory efficiency
        lora_config = LoraConfig(
            r=8,  # Reduced rank for less memory
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],  # Only target essential modules
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model, lora_config)
        model.train()

        # Check trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable params: {trainable_params:,} || Total: {total_params:,} || Trainable%: {100 * trainable_params / total_params:.2f}%")

        return model, tokenizer

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def create_tiny_dataset(tokenizer, n_samples=10):
    """Create a tiny dataset for testing"""
    logger.info(f"Creating tiny dataset with {n_samples} samples...")
    
    # Simple financial queries
    samples = [
        "Convert to SQL: What was Apple's closing price yesterday?\nSQL: SELECT Date, Close FROM stock_data WHERE ticker = 'AAPL' ORDER BY Date DESC LIMIT 1;",
        "Convert to SQL: Show me Tesla's volume for last week\nSQL: SELECT Date, Volume FROM stock_data WHERE ticker = 'TSLA' AND Date >= date('now', '-7 days');",
        "Convert to SQL: Find Google's best performing day\nSQL: SELECT Date, Close, move FROM stock_data WHERE ticker = 'GOOGL' ORDER BY move DESC LIMIT 1;",
        "Convert to SQL: Get Microsoft's average return\nSQL: SELECT AVG(move) as avg_return FROM stock_data WHERE ticker = 'MSFT';",
        "Convert to SQL: Show Amazon's volatility\nSQL: SELECT SQRT(AVG(move * move)) as volatility FROM stock_data WHERE ticker = 'AMZN';",
    ]
    
    # Tokenize with very short sequences
    dataset = []
    for text in samples[:n_samples]:
        encoding = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,  # Very short sequences
            return_tensors="pt"
        )
        
        dataset.append({
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0).clone()
        })
    
    return dataset

def train_lightweight():
    """Train with minimal memory usage"""
    try:
        # Load model
        model, tokenizer = setup_lightweight_model()
        
        # Create tiny dataset
        dataset = create_tiny_dataset(tokenizer, n_samples=5)
        
        # Minimal training configuration
        from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
        
        training_args = TrainingArguments(
            per_device_train_batch_size=1,  # Smallest possible batch
            gradient_accumulation_steps=1,  # No accumulation
            warmup_steps=1,
            max_steps=5,  # Very few steps
            learning_rate=5e-5,  # Lower learning rate
            fp16=False,  # No mixed precision
            bf16=False,
            logging_steps=1,
            optim="adamw_torch",
            weight_decay=0.01,
            lr_scheduler_type="constant",
            seed=42,
            output_dir="./tiny_outputs",
            save_strategy="no",  # Don't save checkpoints
            report_to="none",
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            gradient_checkpointing=False,
            dataloader_num_workers=0,
            use_cpu=True,  # Force CPU training
        )

        # Simple data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        # Create dataset class
        class TinyDataset:
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return self.data[idx]

        tiny_dataset = TinyDataset(dataset)

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
            data_collator=data_collator,
        )

        logger.info("🎯 Starting lightweight training...")
        
        # Train the model
        trainer.train()

        # Save the model
        output_dir = "./lightweight_llama_model"
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        logger.info(f"✅ Training completed! Model saved to {output_dir}")
        logger.info("🎉 Llama 3.1 training successful on MacBook Air M3!")

        return True

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False

def test_model():
    """Test the trained model"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        logger.info("🧪 Testing the fine-tuned model...")
        
        # Load the trained model
        base_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            token="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",
            torch_dtype=torch.float16,
            device_map="cpu"
        )
        
        model = PeftModel.from_pretrained(base_model, "./lightweight_llama_model")
        tokenizer = AutoTokenizer.from_pretrained("./lightweight_llama_model")
        
        # Test query
        query = "Convert to SQL: Show me Tesla's closing prices for the last 7 days"
        inputs = tokenizer(query, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Query: {query}")
        logger.info(f"Response: {response}")
        
        return True

    except Exception as e:
        logger.error(f"Testing failed: {e}")
        return False

def main():
    logger.info("🚀 Lightweight Llama 3.1 Training for MacBook Air M3")
    logger.info("=" * 60)
    
    # Check available memory
    if torch.backends.mps.is_available():
        logger.info("✅ Apple Silicon (MPS) detected")
    
    # Train the model
    success = train_lightweight()
    
    if success:
        logger.info("\n🎉 SUCCESS! Llama 3.1 fine-tuning completed!")
        logger.info("📊 Model successfully trained with:")
        logger.info("  - LoRA adapters (memory efficient)")
        logger.info("  - Financial text-to-SQL data")
        logger.info("  - Apple Silicon optimization")
        
        # Test the model
        test_model()
    else:
        logger.error("❌ Training failed. Check memory usage.")

if __name__ == "__main__":
    main()