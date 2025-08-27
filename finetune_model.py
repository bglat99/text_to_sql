"""
Fine-tuning script for Llama 3.1 8B using QLoRA and Unsloth
"""

import torch
import json
import os
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel
from config import MODEL_CONFIG, LORA_CONFIG, TRAINING_CONFIG, DATA_CONFIG
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
    Setup the model and tokenizer with QLoRA configuration
    """
    print("Loading model and tokenizer...")
    
    # Check if we're on Apple Silicon and need to use MPS
    import torch
    if torch.backends.mps.is_available():
        print("Apple Silicon detected - using MPS backend")
        device = "mps"
    elif torch.cuda.is_available():
        print("CUDA detected - using CUDA backend")
        device = "cuda"
    else:
        print("Using CPU backend")
        device = "cpu"
    
    try:
        # This model doesn't require authentication
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/llama-3.1-8b-instruct-bnb-4bit",
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e
    
    # Add LoRA adapters
    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    return model, tokenizer

def prepare_dataset(train_data, test_data):
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
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    return train_dataset, test_dataset

def setup_training_arguments():
    """
    Setup training arguments optimized for Apple Silicon
    """
    # Detect device and set appropriate settings
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    use_fp16 = device == "mps"  # Use fp16 for MPS, bf16 for CUDA
    
    print(f"Using device: {device}")
    print(f"Using fp16: {use_fp16}")
    
    training_args = TrainingArguments(
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        warmup_steps=TRAINING_CONFIG["warmup_steps"],
        max_steps=TRAINING_CONFIG["max_steps"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        lr_scheduler_type=TRAINING_CONFIG["lr_scheduler_type"],
        seed=TRAINING_CONFIG["seed"],
        output_dir=TRAINING_CONFIG["output_dir"],
        save_strategy=TRAINING_CONFIG["save_strategy"],
        save_steps=TRAINING_CONFIG["save_steps"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        optim=TRAINING_CONFIG["optim"],
        fp16=use_fp16,
        bf16=not use_fp16,
        dataloader_pin_memory=False,  # Disable for Apple Silicon
        remove_unused_columns=False,
        report_to=None,  # Disable wandb/tensorboard
    )
    
    return training_args

def train_model(model, tokenizer, train_dataset, test_dataset, training_args):
    """
    Train the model using SFTTrainer
    """
    print("Setting up trainer...")
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        dataset_text_field="text",
        max_seq_length=MODEL_CONFIG["max_seq_length"],
        dataset_num_proc=2,
        args=training_args,
    )
    
    print("Starting training...")
    print(f"Total training steps: {training_args.max_steps}")
    print(f"Batch size: {training_args.per_device_train_batch_size}")
    print(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    
    # Start training
    trainer.train()
    
    return trainer

def save_model(model, tokenizer, output_dir="financial_sql_model"):
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
        "model_config": MODEL_CONFIG,
        "lora_config": LORA_CONFIG,
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
    print("Financial Text-to-SQL Fine-tuning")
    print("=" * 60)
    
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
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print(f"Model saved to: financial_sql_model/")
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
            outputs = model.generate(**inputs, max_length=512, temperature=0.1, do_sample=True)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        sql = response.split("### Response:")[1].strip() if "### Response:" in response else response
        
        print(f"\nQuery: {query}")
        print(f"Generated SQL: {sql}")

if __name__ == "__main__":
    main() 