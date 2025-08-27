"""
Configuration file for Text-to-SQL fine-tuning
"""

# Model Configuration
MODEL_CONFIG = {
    "model_name": "unsloth/llama-3.1-8b-instruct-bnb-4bit",
    "max_seq_length": 2048,
    "load_in_4bit": True,
    "dtype": None,  # Auto-detect for Apple Silicon
}

# LoRA Configuration
LORA_CONFIG = {
    "r": 16,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ],
    "lora_alpha": 16,
    "lora_dropout": 0,
    "bias": "none",
    "use_gradient_checkpointing": "unsloth",
    "random_state": 3407,
}

# Training Configuration
TRAINING_CONFIG = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 10,
    "max_steps": 500,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "seed": 3407,
    "output_dir": "outputs",
    "save_strategy": "steps",
    "save_steps": 100,
    "logging_steps": 1,
    "optim": "adamw_8bit",
}

# Data Configuration
DATA_CONFIG = {
    "n_training_samples": 2000,
    "n_test_samples": 200,
    "train_test_split": 0.8,
}

# Financial Data Configuration
FINANCIAL_CONFIG = {
    "tickers": [
        'SPY', 'AAPL', 'NFLX', 'GOOGL', 'AMZN', 
        'META', 'MSFT', 'TSLA', 'ABNB', 'NVDA',
        'AMD', 'INTC', 'CRM', 'ADBE', 'PYPL'
    ],
    "database_path": "financial_data.db",
    "schema": """
    Table: stock_data
    - Date (TEXT): YYYY-MM-DD format
    - Open, High, Low, Close, Adj_Close (REAL): Price data
    - Volume (INTEGER): Trading volume  
    - ticker (TEXT): Stock symbol
    - move (REAL): Daily % change
    """,
}

# Evaluation Configuration
EVAL_CONFIG = {
    "test_queries": [
        "Show me Apple's closing price yesterday",
        "What were the top 3 gainers last week?", 
        "Calculate Tesla's volatility over the past month",
        "Compare Amazon and Google's average daily returns this year",
        "Find all days when SPY moved more than 2% in either direction",
        "What was Netflix's highest trading volume day this quarter?",
        "Show me the correlation between Tesla and Apple stock movements",
        "Find stocks that had consecutive up days for more than 5 days",
        "Calculate the Sharpe ratio for Microsoft over the last 6 months",
        "Show me days when the market gap was more than 3%"
    ],
    "target_accuracy": 0.80,
    "target_execution_rate": 0.85,
} 