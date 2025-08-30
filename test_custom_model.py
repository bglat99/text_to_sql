#!/usr/bin/env python3
"""
Test Custom Trained Financial Model
Evaluate performance of your custom CSV-trained model
"""

import argparse
import torch
import json
import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def load_custom_model(model_path):
    """Load the custom trained model"""
    logger.info(f"Loading custom model from {model_path}")
    
    try:
        # Check if training info exists
        info_path = os.path.join(model_path, "training_info.json")
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                training_info = json.load(f)
            logger.info(f"Model training info:")
            for key, value in training_info.items():
                logger.info(f"  {key}: {value}")
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            token="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",
            torch_dtype=torch.float16,
            device_map="cpu"
        )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        logger.info("✅ Custom model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"❌ Error loading custom model: {e}")
        return None, None

def generate_sql(model, tokenizer, query, max_length=200):
    """Generate SQL for a financial query"""
    try:
        prompt = f"Convert to SQL: {query}"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        logger.info(f"🤖 Processing: {query}")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.1,
                repetition_penalty=1.1
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract SQL part
        if prompt in response:
            sql_part = response[len(prompt):].strip()
        else:
            sql_part = response.strip()
            
        return sql_part
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return f"Error: {e}"

def test_financial_queries(model, tokenizer):
    """Test the model with various financial queries"""
    
    test_queries = [
        # Basic price queries
        "What was Apple's closing price yesterday?",
        "Show me Tesla's opening price today",
        
        # Volume queries
        "What's Google's trading volume for last week?",
        "Display Microsoft's average daily volume",
        
        # Performance queries
        "Which stock performed best last month?",
        "Calculate Tesla's 30-day return",
        
        # Volatility queries
        "What is Apple's realized volatility?",
        "Show me the risk metrics for Google",
        
        # Comparison queries
        "Compare Apple and Tesla performance",
        "Which is more volatile: Google or Microsoft?",
        
        # Time-based queries
        "Show me Amazon's price movement over the last quarter",
        "Get Netflix's returns for the past year"
    ]
    
    logger.info("🧪 **TESTING CUSTOM FINANCIAL MODEL**")
    logger.info("=" * 50)
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        logger.info(f"\n{i}. Query: {query}")
        
        sql_result = generate_sql(model, tokenizer, query)
        
        # Basic validation
        is_sql = any(keyword in sql_result.upper() for keyword in ['SELECT', 'FROM', 'WHERE', 'SQL'])
        
        if is_sql:
            logger.info(f"   ✅ Generated SQL: {sql_result}")
            results.append({'query': query, 'sql': sql_result, 'valid': True})
        else:
            logger.info(f"   ❌ Invalid response: {sql_result}")
            results.append({'query': query, 'sql': sql_result, 'valid': False})
    
    # Summary
    valid_count = sum(1 for r in results if r['valid'])
    total_count = len(results)
    success_rate = (valid_count / total_count) * 100
    
    logger.info("\n" + "=" * 50)
    logger.info("📊 **TEST RESULTS SUMMARY**")
    logger.info(f"Total queries tested: {total_count}")
    logger.info(f"Valid SQL generated: {valid_count}")
    logger.info(f"Success rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        logger.info("🎉 **EXCELLENT** - Model performing very well!")
    elif success_rate >= 60:
        logger.info("👍 **GOOD** - Model showing solid performance!")
    elif success_rate >= 40:
        logger.info("⚠️  **FAIR** - Consider more training data")
    else:
        logger.info("❌ **NEEDS IMPROVEMENT** - More training recommended")
    
    return results

def interactive_test(model, tokenizer):
    """Interactive testing mode"""
    logger.info("\n🎮 **INTERACTIVE TESTING MODE**")
    logger.info("=" * 35)
    logger.info("Enter financial queries to test your custom model")
    logger.info("Type 'quit' to exit\n")
    
    while True:
        try:
            query = input("🔍 Your query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
                
            if not query:
                continue
            
            sql_result = generate_sql(model, tokenizer, query)
            
            print(f"📝 Generated SQL:")
            print(f"   {sql_result}")
            
            # Quick validation
            if any(keyword in sql_result.upper() for keyword in ['SELECT', 'FROM', 'WHERE']):
                print("✅ Looks like valid SQL!")
            else:
                print("⚠️  May need refinement")
                
            print()
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("👋 Thanks for testing your custom model!")

def compare_models(custom_model_path, baseline_model_path=None):
    """Compare custom model vs baseline"""
    logger.info("🔄 **MODEL COMPARISON**")
    
    # Load custom model
    custom_model, custom_tokenizer = load_custom_model(custom_model_path)
    if not custom_model:
        return
    
    # Test queries
    test_query = "What was Tesla's closing price yesterday?"
    
    logger.info(f"Test query: {test_query}")
    
    # Custom model result
    custom_sql = generate_sql(custom_model, custom_tokenizer, test_query)
    logger.info(f"Custom model: {custom_sql}")
    
    logger.info("📊 Custom model loaded and ready for testing!")

def main():
    parser = argparse.ArgumentParser(description="Test custom trained financial model")
    parser.add_argument("--model_path", required=True, help="Path to custom trained model")
    parser.add_argument("--interactive", action="store_true", help="Run interactive testing")
    parser.add_argument("--compare", action="store_true", help="Compare with baseline model")
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model_path):
        logger.error(f"Model path not found: {args.model_path}")
        return
    
    # Load model
    model, tokenizer = load_custom_model(args.model_path)
    if not model:
        logger.error("Failed to load custom model")
        return
    
    try:
        if args.interactive:
            interactive_test(model, tokenizer)
        elif args.compare:
            compare_models(args.model_path)
        else:
            # Default: run test suite
            test_financial_queries(model, tokenizer)
            
    except Exception as e:
        logger.error(f"Testing failed: {e}")

if __name__ == "__main__":
    main()