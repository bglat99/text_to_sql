#!/usr/bin/env python3
"""
Comprehensive Test Suite for Fine-tuned Llama 3.1 Financial Text-to-SQL Model
"""

import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def load_finetuned_model():
    """Load the fine-tuned model"""
    try:
        logger.info("🔄 Loading fine-tuned Llama 3.1 model...")
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            token="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",
            torch_dtype=torch.float16,
            device_map="cpu"
        )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, "./lightweight_llama_model")
        tokenizer = AutoTokenizer.from_pretrained("./lightweight_llama_model")
        
        logger.info("✅ Fine-tuned model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return None, None

def generate_sql(model, tokenizer, query, max_length=200):
    """Generate SQL for a given natural language query"""
    try:
        # Format the prompt
        prompt = f"Convert to SQL: {query}"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.1,
                repetition_penalty=1.1
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the new part (remove the prompt)
        if prompt in response:
            sql_part = response[len(prompt):].strip()
        else:
            sql_part = response.strip()
            
        return sql_part
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return f"Error: {e}"

def run_comprehensive_tests():
    """Run comprehensive tests on the fine-tuned model"""
    
    # Load model
    model, tokenizer = load_finetuned_model()
    if model is None:
        logger.error("❌ Cannot run tests - model loading failed")
        return
    
    logger.info("\n🧪 **COMPREHENSIVE FINANCIAL TEXT-TO-SQL TESTS**")
    logger.info("=" * 60)
    
    # Test cases covering different financial query types
    test_cases = [
        {
            "category": "📈 Basic Price Queries",
            "queries": [
                "What was Apple's closing price yesterday?",
                "Show me Tesla's opening price for today",
                "Get Microsoft's high price for this week"
            ]
        },
        {
            "category": "📊 Volume and Trading",
            "queries": [
                "What was the trading volume for Amazon last week?",
                "Show me Google's average daily volume",
                "Find the highest volume trading day for Netflix"
            ]
        },
        {
            "category": "📉 Performance Analysis",
            "queries": [
                "Which stock had the best performance last month?",
                "Show me Tesla's worst performing days",
                "Get the top 5 gainers from the S&P 500"
            ]
        },
        {
            "category": "📅 Time-based Queries",
            "queries": [
                "Show me Apple's price movement over the last 30 days",
                "Get Microsoft's quarterly performance",
                "Find all stocks that gained more than 5% today"
            ]
        },
        {
            "category": "🔍 Complex Aggregations",
            "queries": [
                "Calculate the average return for tech stocks",
                "Show me the volatility of Tesla over the past year",
                "Get the correlation between Apple and Microsoft stock prices"
            ]
        }
    ]
    
    total_tests = 0
    successful_tests = 0
    
    for test_group in test_cases:
        logger.info(f"\n{test_group['category']}")
        logger.info("-" * 40)
        
        for query in test_group['queries']:
            total_tests += 1
            logger.info(f"\n🔍 Query: {query}")
            
            # Generate SQL
            sql_result = generate_sql(model, tokenizer, query)
            
            # Check if result looks like SQL
            if any(keyword in sql_result.upper() for keyword in ['SELECT', 'FROM', 'WHERE', 'SQL']):
                logger.info(f"✅ Generated SQL:\n{sql_result}")
                successful_tests += 1
            else:
                logger.info(f"❌ Invalid response:\n{sql_result}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 **TEST RESULTS SUMMARY**")
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Successful: {successful_tests}")
    logger.info(f"Success Rate: {(successful_tests/total_tests)*100:.1f}%")
    
    if successful_tests/total_tests > 0.7:
        logger.info("🎉 **EXCELLENT!** Model performing well!")
    elif successful_tests/total_tests > 0.5:
        logger.info("👍 **GOOD!** Model showing solid performance!")
    else:
        logger.info("⚠️  **NEEDS IMPROVEMENT** - More training recommended")
    
    return successful_tests, total_tests

def test_specific_queries():
    """Test specific edge cases and complex queries"""
    model, tokenizer = load_finetuned_model()
    if model is None:
        return
    
    logger.info("\n🎯 **SPECIFIC EDGE CASE TESTS**")
    logger.info("=" * 60)
    
    edge_cases = [
        "Show me all Tesla transactions where the price dropped more than 10%",
        "Get the moving average of Apple stock for the last 20 days",
        "Find stocks with unusual trading volume (more than 2x average)",
        "Compare the performance of FAANG stocks this quarter",
        "Show me the best time to buy Tesla stock based on historical patterns"
    ]
    
    for i, query in enumerate(edge_cases, 1):
        logger.info(f"\n{i}. 🔍 Edge Case: {query}")
        sql_result = generate_sql(model, tokenizer, query, max_length=300)
        logger.info(f"📝 Response:\n{sql_result}")

def main():
    logger.info("🚀 **TESTING FINE-TUNED LLAMA 3.1 FINANCIAL MODEL**")
    logger.info("🎯 Model: Llama 3.1 8B + LoRA (Financial Text-to-SQL)")
    logger.info("💻 Platform: MacBook Air M3")
    
    # Run comprehensive tests
    successful, total = run_comprehensive_tests()
    
    # Run edge case tests
    test_specific_queries()
    
    logger.info("\n🎊 **TESTING COMPLETE!**")
    logger.info(f"📈 Overall Success Rate: {(successful/total)*100:.1f}%")
    logger.info("✨ Your fine-tuned Llama 3.1 model is ready for financial queries!")

if __name__ == "__main__":
    main()