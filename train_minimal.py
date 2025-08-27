#!/usr/bin/env python3
"""
Minimal training demonstration for text-to-SQL
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import FINANCIAL_CONFIG

def load_training_data():
    """Load the generated training data"""
    with open("train_data.json", "r") as f:
        return json.load(f)

def test_model_capabilities():
    """Test the model's ability to generate SQL"""
    print("=" * 60)
    print("Testing Model SQL Generation Capabilities")
    print("=" * 60)
    
    # Load a pre-trained model
    model_name = "microsoft/DialoGPT-medium"
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Test queries
    test_queries = [
        "Show me Apple's closing prices for the last 30 days",
        "Calculate Tesla's volatility over the past month",
        "What was the best performing stock this week?",
        "Find stocks that moved more than 5% today",
        "Compare Apple and Microsoft performance this year"
    ]
    
    print("\nTesting SQL generation for financial queries:")
    print("-" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: {query}")
        
        # Create a simple prompt
        prompt = f"Convert to SQL: {query}"
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt", max_length=100, truncation=True)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {response}")
    
    print("\n" + "=" * 60)
    print("Model testing complete!")
    print("=" * 60)

def analyze_training_data():
    """Analyze the generated training data"""
    print("=" * 60)
    print("Training Data Analysis")
    print("=" * 60)
    
    data = load_training_data()
    
    print(f"Total training examples: {len(data)}")
    
    # Analyze query types
    query_types = {}
    for item in data[:100]:  # Sample first 100
        query = item['input'].split('Query: ')[1]
        
        # Categorize queries
        if 'volatility' in query.lower():
            query_types['volatility'] = query_types.get('volatility', 0) + 1
        elif 'compare' in query.lower():
            query_types['comparison'] = query_types.get('comparison', 0) + 1
        elif 'best' in query.lower() or 'top' in query.lower():
            query_types['ranking'] = query_types.get('ranking', 0) + 1
        elif 'average' in query.lower() or 'avg' in query.lower():
            query_types['aggregation'] = query_types.get('aggregation', 0) + 1
        else:
            query_types['basic'] = query_types.get('basic', 0) + 1
    
    print("\nQuery type distribution (sample of 100):")
    for query_type, count in query_types.items():
        print(f"  {query_type}: {count}")
    
    # Show sample examples
    print("\nSample training examples:")
    for i, item in enumerate(data[:3]):
        query = item['input'].split('Query: ')[1]
        sql = item['output']
        print(f"\nExample {i+1}:")
        print(f"  Query: {query}")
        print(f"  SQL: {sql}")

def demonstrate_sql_execution():
    """Demonstrate SQL execution on the database"""
    print("=" * 60)
    print("SQL Execution Demonstration")
    print("=" * 60)
    
    import sqlite3
    import pandas as pd
    
    # Sample SQL queries to test
    test_sqls = [
        "SELECT COUNT(*) FROM stock_data WHERE ticker = 'AAPL'",
        "SELECT AVG(move) FROM stock_data WHERE ticker = 'TSLA'",
        "SELECT ticker, AVG(move) FROM stock_data GROUP BY ticker LIMIT 5",
        "SELECT Date, Close FROM stock_data WHERE ticker = 'GOOGL' ORDER BY Date DESC LIMIT 5"
    ]
    
    try:
        conn = sqlite3.connect("financial_data.db")
        
        for i, sql in enumerate(test_sqls, 1):
            print(f"\nTest {i}: {sql}")
            try:
                result = pd.read_sql_query(sql, conn)
                print(f"  Result: {len(result)} rows")
                if not result.empty:
                    print(f"  Sample: {result.head(2).to_string()}")
            except Exception as e:
                print(f"  Error: {e}")
        
        conn.close()
        
    except Exception as e:
        print(f"Database error: {e}")

def main():
    """Main demonstration function"""
    print("🚀 Financial Text-to-SQL Framework Demo")
    print("=" * 60)
    
    # Analyze training data
    analyze_training_data()
    
    # Test model capabilities
    test_model_capabilities()
    
    # Demonstrate SQL execution
    demonstrate_sql_execution()
    
    print("\n" + "=" * 60)
    print("✅ Demo Complete!")
    print("=" * 60)
    print("\nNext steps for full training:")
    print("1. Use a larger model (e.g., Llama 3.1 8B)")
    print("2. Implement proper fine-tuning with LoRA")
    print("3. Use Unsloth for Apple Silicon optimization")
    print("4. Train for more steps (500-1000)")
    print("5. Evaluate on test set")
    
    print("\nCurrent setup includes:")
    print("✓ 2000 training examples")
    print("✓ 200 test examples") 
    print("✓ Financial database with real data")
    print("✓ Comprehensive evaluation framework")
    print("✓ Visualization capabilities")

if __name__ == "__main__":
    main() 