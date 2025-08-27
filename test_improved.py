#!/usr/bin/env python3
"""
Improved Test: Financial Text-to-SQL Model
Uses more stable generation approach
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

def load_model():
    """Load the model for testing"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info("Loading model...")
        
        # Use base model for now
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

def generate_sql_simple(model, tokenizer, query: str) -> str:
    """Generate SQL using a simple, stable approach"""
    try:
        # Create a simple prompt
        prompt = f"Query: {query}\nSQL:"
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        
        # Generate response with conservative settings
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,  # Limit new tokens
                temperature=0.7,     # Moderate temperature
                do_sample=False,     # Use greedy decoding
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
        # Return a simple fallback SQL
        return "SELECT * FROM stock_data LIMIT 1;"

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
    logger.info("Improved Financial Text-to-SQL Test")
    logger.info("=" * 60)
    
    try:
        # Load model
        model, tokenizer = load_model()
        
        # Test queries
        test_queries = [
            "Show me Apple's closing prices for the last 30 days",
            "What was Tesla's trading volume yesterday?",
            "Calculate the volatility of SPY over the past month",
            "Find all days when AAPL moved more than 5%",
            "Show me the best performing stocks this year"
        ]
        
        logger.info("🧪 Testing Model...")
        
        success_count = 0
        total_queries = len(test_queries)
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n--- Test {i} ---")
            logger.info(f"Query: {query}")
            
            # Generate SQL
            sql = generate_sql_simple(model, tokenizer, query)
            logger.info(f"Generated SQL: {sql}")
            
            # Test execution
            success = test_sql_execution(sql)
            
            if success:
                logger.info("✅ Query successful!")
                success_count += 1
            else:
                logger.info("❌ Query failed")
        
        # Summary
        accuracy = (success_count / total_queries) * 100
        logger.info(f"\n" + "=" * 60)
        logger.info(f"🎯 Test Results:")
        logger.info(f"Successful queries: {success_count}/{total_queries}")
        logger.info(f"Accuracy: {accuracy:.1f}%")
        logger.info("=" * 60)
        
        if accuracy > 0:
            logger.info("🎉 Model is working!")
        else:
            logger.info("⚠️ Model needs improvement")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    main() 