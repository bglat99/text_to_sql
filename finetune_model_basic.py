"""
Basic fine-tuning script using standard transformers training
"""

import torch
import json
import os
from datasets import Dataset
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM, Trainer
from config import TRAINING_CONFIG, DATA_CONFIG
from data_generator import FinancialSQLGenerator
from utils import format_training_example

def load_or_generate_data():
    """
    Load existing data or generate new training data
    """
    train_file = "train_data.json"
    test_file = "test_data.json"
    
    if os.path.exists(train_file):
        print("Loading existing training data...")
        with open(train_file, 'r') as f:
            train_data = json.load(f)
        with open(test_file, 'r') as f:
            test_data = json.load(f)
    else:
        print("Generating new training data...")
        generator = FinancialSQLGenerator()
        train_data = generator.generate_training_data()
        test_data = generator.generate_test_data()
        
        # Save data for future use
        generator.save_data(train_data, test_data)
    
    return train_data, test_data

def format_prompt(examples):
    """
    Format data for training with proper instruction format
    """
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
    """
    Setup the model and tokenizer for Apple Silicon
    """
    print("Loading model and tokenizer...")
    
    # Use a smaller model that's more compatible with Apple Silicon
    model_name = "microsoft/DialoGPT-medium"  # Smaller model for testing
    
    print(f"Using model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto" if torch.backends.mps.is_available() else None
    )
    
    return model, tokenizer

def prepare_dataset(train_data, test_data, tokenizer):
    """
    Prepare datasets for training
    """
    print("Preparing datasets...")
    
    # Create training dataset
    train_dataset = Dataset.from_list(train_data)
    train_dataset = train_dataset.map(format_prompt, batched=True, remove_columns=train_dataset.column_names)
    
    # Create test dataset
    test_dataset = Dataset.from_list(test_data)
    test_dataset = test_dataset.map(format_prompt, batched=True, remove_columns=test_dataset.column_names)
    
    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=512,
            return_tensors=None
        )
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    return train_dataset, test_dataset

def setup_training_arguments():
    """
    Setup training arguments optimized for Apple Silicon
    """
    # Detect device and set appropriate settings
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    use_fp16 = False  # Disable fp16 for MPS compatibility
    
    print(f"Using device: {device}")
    print(f"Using fp16: {use_fp16}")
    
    # Reduced training parameters for faster testing
    training_args = TrainingArguments(
        per_device_train_batch_size=1,  # Reduced for memory
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,  # Increased to compensate
        warmup_steps=5,
        max_steps=100,  # Reduced for faster training
        learning_rate=5e-5,  # Slightly higher for smaller model
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        save_strategy="steps",
        save_steps=50,
        logging_steps=1,
        optim="adamw_torch",  # Use torch optimizer
        fp16=use_fp16,
        bf16=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=None,
    )
    
    return training_args

def train_model(model, tokenizer, train_dataset, test_dataset, training_args):
    """
    Train the model using standard Trainer
    """
    print("Setting up trainer...")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )
    
    print("Starting training...")
    print(f"Total training steps: {training_args.max_steps}")
    print(f"Batch size: {training_args.per_device_train_batch_size}")
    print(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    
    # Start training
    trainer.train()
    
    return trainer

def save_model(model, tokenizer, output_dir="financial_sql_model_basic"):
    """
    Save the fine-tuned model
    """
    print(f"Saving model to {output_dir}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training configuration
    config = {
        "model_name": "microsoft/DialoGPT-medium",
        "training_config": TRAINING_CONFIG,
        "data_config": DATA_CONFIG,
    }
    
    with open(os.path.join(output_dir, "training_config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Model saved successfully to {output_dir}")

def main():
    """
    Main training function
    """
    print("=" * 60)
    print("Financial Text-to-SQL Fine-tuning (Basic)")
    print("=" * 60)
    
    # Load or generate data
    train_data, test_data = load_or_generate_data()
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Prepare datasets
    train_dataset, test_dataset = prepare_dataset(train_data, test_data, tokenizer)
    
    # Setup training arguments
    training_args = setup_training_arguments()
    
    # Train model
    trainer = train_model(model, tokenizer, train_dataset, test_dataset, training_args)
    
    # Save model
    save_model(model, tokenizer)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print(f"Model saved to: financial_sql_model_basic/")
    print(f"Training examples used: {len(train_data)}")
    print(f"Test examples used: {len(test_data)}")
    print(f"Total training steps: {training_args.max_steps}")
    
    # Show sample predictions
    print("\nSample predictions:")
    sample_queries = [
        "Show me Apple's closing prices for the last 30 days",
        "Calculate Tesla's volatility over the past month",
        "What was the best performing stock this week?"
    ]
    
    for query in sample_queries:
        prompt = f"""### Instruction:
Convert this financial query to SQL:

### Input:
Schema: stock_data table with Date, Open, High, Low, Close, Volume, ticker, move columns

Query: {query}

### Response:
"""
        
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=256, temperature=0.1, do_sample=True)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        sql = response.split("### Response:")[1].strip() if "### Response:" in response else response
        
        print(f"\nQuery: {query}")
        print(f"Generated SQL: {sql}")

if __name__ == "__main__":
    main() 