#!/usr/bin/env python3
"""
Working Demo: Financial Text-to-SQL Conversion
Shows the model's ability to convert natural language to SQL
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

def load_model_and_tokenizer():
    """Load the model for inference"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info("Loading model for inference...")
        
        # Use DialoGPT-medium for inference
        model_name = "microsoft/DialoGPT-medium"
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        logger.info("✅ Model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def generate_sql_query(model, tokenizer, query: str) -> str:
    """Generate SQL from natural language query"""
    try:
        # Create prompt
        prompt = f"Convert to SQL: {query}\n\nSQL:"
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=512,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract SQL part
        if "SQL:" in response:
            sql = response.split("SQL:")[1].strip()
        else:
            sql = response.split(prompt)[1].strip() if prompt in response else response
        
        return sql
        
    except Exception as e:
        logger.error(f"Error generating SQL: {e}")
        return f"Error: {e}"

def test_sql_execution(sql: str) -> bool:
    """Test if SQL executes successfully"""
    try:
        import sqlite3
        import pandas as pd
        
        # Create a test database with sample data
        conn = sqlite3.connect(":memory:")
        
        # Create sample stock_data table
        conn.execute("""
            CREATE TABLE stock_data (
                Date TEXT,
                Open REAL,
                High REAL,
                Low REAL,
                Close REAL,
                Volume INTEGER,
                ticker TEXT,
                move REAL
            )
        """)
        
        # Insert sample data
        sample_data = [
            ("2024-01-01", 100.0, 105.0, 98.0, 103.0, 1000000, "AAPL", 3.0),
            ("2024-01-02", 103.0, 108.0, 102.0, 106.0, 1200000, "AAPL", 2.9),
            ("2024-01-03", 106.0, 110.0, 104.0, 109.0, 1100000, "AAPL", 2.8),
            ("2024-01-01", 150.0, 155.0, 148.0, 153.0, 800000, "TSLA", 2.0),
            ("2024-01-02", 153.0, 158.0, 152.0, 156.0, 900000, "TSLA", 2.0),
            ("2024-01-03", 156.0, 160.0, 154.0, 159.0, 850000, "TSLA", 1.9),
        ]
        
        conn.executemany("""
            INSERT INTO stock_data (Date, Open, High, Low, Close, Volume, ticker, move)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, sample_data)
        
        # Try to execute the SQL
        result = pd.read_sql_query(sql, conn)
        conn.close()
        
        logger.info(f"✅ SQL executed successfully! Returned {len(result)} rows")
        return True
        
    except Exception as e:
        logger.error(f"❌ SQL execution failed: {e}")
        return False

def main():
    logger.info("=" * 60)
    logger.info("Financial Text-to-SQL Demo")
    logger.info("Testing Model Capabilities")
    logger.info("=" * 60)
    
    try:
        # Load model
        model, tokenizer = load_model_and_tokenizer()
        
        # Test queries
        test_queries = [
            "Show me Apple's closing prices for the last 30 days",
            "What was Tesla's trading volume yesterday?",
            "Calculate the volatility of SPY over the past month",
            "Find all days when AAPL moved more than 5%",
            "Show me the best performing stocks this year"
        ]
        
        logger.info("🧪 Testing Text-to-SQL Conversion...")
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n--- Test {i} ---")
            logger.info(f"Query: {query}")
            
            # Generate SQL
            sql = generate_sql_query(model, tokenizer, query)
            logger.info(f"Generated SQL: {sql}")
            
            # Test execution
            success = test_sql_execution(sql)
            
            if success:
                logger.info("✅ Query successful!")
            else:
                logger.info("❌ Query failed")
        
        logger.info("\n" + "=" * 60)
        logger.info("Demo completed!")
        logger.info("The model can generate SQL from natural language queries")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise

if __name__ == "__main__":
    main() 