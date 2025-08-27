"""
Fine-tuning script for Llama 3.1 8B optimized for Apple Silicon
Uses standard transformers library to avoid CUDA issues with Unsloth
"""

import torch
import json
import os
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from config import TRAINING_CONFIG, DATA_CONFIG
from data_generator import FinancialSQLGenerator
from utils import format_training_example
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_or_generate_data():
    """Load existing data or generate new training data"""
    train_file = "train_data.json"
    test_file = "test_data.json"
    
    if os.path.exists(train_file):
        logger.info("Loading existing training data...")
        with open(train_file, 'r') as f:
            train_data = json.load(f)
        with open(test_file, 'r') as f:
            test_data = json.load(f)
    else:
        logger.info("Generating new training data...")
        generator = FinancialSQLGenerator()
        train_data = generator.generate_training_data()
        test_data = generator.generate_test_data()
        
        # Save data for future use
        generator.save_data(train_data, test_data)
    
    return train_data, test_data

def format_prompt(examples):
    """Format data for training with proper instruction format"""
    texts = []
    for i in range(len(examples["instruction"])):
        text = format_training_example(
            examples["instruction"][i],
            examples["input"][i],
            examples["output"][i]
        )
        texts.append(text)
    return {"text": texts}

def setup_model_and_tokenizer():
    """Setup the model and tokenizer optimized for Apple Silicon"""
    logger.info("Loading model and tokenizer...")
    
    # Check device
    if torch.backends.mps.is_available():
        logger.info("Apple Silicon detected - using MPS backend")
        device = "mps"
    else:
        logger.info("Using CPU backend")
        device = "cpu"
    
    # Use Llama 3.1 8B Instruct model
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    try:
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
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def prepare_dataset(train_data, test_data):
    """Prepare datasets for training"""
    logger.info("Preparing datasets...")
    
    # Create training dataset
    train_dataset = Dataset.from_list(train_data)
    train_dataset = train_dataset.map(format_prompt, batched=True, remove_columns=train_dataset.column_names)
    
    # Create test dataset
    test_dataset = Dataset.from_list(test_data)
    test_dataset = test_dataset.map(format_prompt, batched=True, remove_columns=test_dataset.column_names)
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    return train_dataset, test_dataset

def setup_training_arguments():
    """Setup training arguments optimized for Apple Silicon"""
    # Detect device and set appropriate settings
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    use_fp16 = device == "mps"  # Use fp16 for MPS
    
    logger.info(f"Using device: {device}")
    logger.info(f"Using fp16: {use_fp16}")
    
    training_args = TrainingArguments(
        per_device_train_batch_size=2,  # Reduced for Apple Silicon
        gradient_accumulation_steps=4,
        warmup_steps=50,
        max_steps=500,
        learning_rate=2e-4,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        output_dir="llama_financial_model",
        save_strategy="steps",
        save_steps=100,
        logging_steps=10,
        optim="paged_adamw_8bit",
        fp16=use_fp16,
        dataloader_pin_memory=False,  # Disable for Apple Silicon
        remove_unused_columns=False,
        report_to=None,  # Disable wandb/tensorboard
        gradient_checkpointing=True,
    )
    
    return training_args

def tokenize_function(examples):
    """Tokenize the training examples"""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=2048,
        return_tensors=None
    )

def train_model(model, tokenizer, train_dataset, test_dataset, training_args):
    """Train the model using standard Trainer"""
    logger.info("Setting up trainer...")
    
    # Tokenize datasets
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        args=training_args,
    )
    
    logger.info("Starting training...")
    logger.info(f"Total training steps: {training_args.max_steps}")
    logger.info(f"Batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    
    # Start training
    trainer.train()
    
    return trainer

def save_model(model, tokenizer, output_dir="llama_financial_model"):
    """Save the fine-tuned model"""
    logger.info(f"Saving model to {output_dir}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training configuration
    config = {
        "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "training_config": {
            "max_steps": 500,
            "learning_rate": 2e-4,
            "batch_size": 2,
            "gradient_accumulation": 4
        }
    }
    
    with open(os.path.join(output_dir, "training_config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Model saved successfully to {output_dir}")

def main():
    """Main training function"""
    logger.info("=" * 60)
    logger.info("Llama 3.1 8B Financial Text-to-SQL Fine-tuning")
    logger.info("Optimized for Apple Silicon")
    logger.info("=" * 60)
    
    try:
        # Load or generate data
        train_data, test_data = load_or_generate_data()
        
        # Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer()
        
        # Prepare datasets
        train_dataset, test_dataset = prepare_dataset(train_data, test_data)
        
        # Setup training arguments
        training_args = setup_training_arguments()
        
        # Train model
        trainer = train_model(model, tokenizer, train_dataset, test_dataset, training_args)
        
        # Save model
        save_model(model, tokenizer)
        
        logger.info("\n" + "=" * 60)
        logger.info("Training completed successfully!")
        logger.info("=" * 60)
        logger.info(f"Model saved to: llama_financial_model/")
        logger.info(f"Training examples used: {len(train_data)}")
        logger.info(f"Test examples used: {len(test_data)}")
        logger.info(f"Total training steps: {training_args.max_steps}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main() 