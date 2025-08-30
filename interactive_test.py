#!/usr/bin/env python3
"""
Interactive Test for Fine-tuned Llama 3.1 Model
Memory-efficient testing on MacBook Air M3
"""

import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Set up logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise

def load_model_safely():
    """Load model with minimal memory usage"""
    try:
        print("🔄 Loading your fine-tuned Llama 3.1 model...")
        print("⏳ This may take 1-2 minutes...")
        
        # Load base model with memory optimization
        base_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            token="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",
            torch_dtype=torch.float16,
            device_map="cpu",  # Force CPU to avoid memory issues
            low_cpu_mem_usage=True
        )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, "./lightweight_llama_model")
        tokenizer = AutoTokenizer.from_pretrained("./lightweight_llama_model")
        
        print("✅ Model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Try running 'python3 quick_test.py' instead for a memory-free test")
        return None, None

def generate_sql(model, tokenizer, query):
    """Generate SQL for a query"""
    try:
        prompt = f"Convert to SQL: {query}"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        print("🤖 Generating SQL...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.1
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract SQL part
        if prompt in response:
            sql_part = response[len(prompt):].strip()
        else:
            sql_part = response.strip()
            
        return sql_part
        
    except Exception as e:
        return f"Error: {e}"

def interactive_session():
    """Run interactive testing session"""
    print("🧪 **INTERACTIVE LLAMA 3.1 FINANCIAL SQL TESTING**")
    print("=" * 60)
    
    # Load model
    model, tokenizer = load_model_safely()
    if model is None:
        return
    
    print("\n🎯 **Ready for testing!**")
    print("💡 Try queries like:")
    print("  - What was Apple's closing price yesterday?")
    print("  - Show me Tesla's volume for last week")
    print("  - Find the best performing tech stocks")
    print("  - Get Microsoft's average return")
    print("\n📝 Type 'quit' to exit\n")
    
    while True:
        try:
            # Get user input
            query = input("🔍 Enter your financial query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
                
            if not query:
                continue
            
            print(f"\n📋 Query: {query}")
            
            # Generate SQL
            sql_result = generate_sql(model, tokenizer, query)
            
            print("🔧 Generated SQL:")
            print("─" * 40)
            print(sql_result)
            print("─" * 40)
            
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
    
    print("\n👋 Thanks for testing your fine-tuned model!")

def quick_demo():
    """Show a few example generations without loading the full model"""
    print("🚀 **QUICK DEMO - Example Model Outputs**")
    print("=" * 60)
    
    examples = [
        {
            "query": "What was Tesla's closing price yesterday?",
            "sql": """SELECT Date, Close 
FROM stock_data 
WHERE ticker = 'TSLA' 
ORDER BY Date DESC 
LIMIT 1;"""
        },
        {
            "query": "Show me Apple's best performing days",
            "sql": """SELECT Date, Close, move 
FROM stock_data 
WHERE ticker = 'AAPL' AND move > 0
ORDER BY move DESC 
LIMIT 10;"""
        },
        {
            "query": "Get average volume for tech stocks",
            "sql": """SELECT ticker, AVG(Volume) as avg_volume
FROM stock_data 
WHERE ticker IN ('AAPL', 'MSFT', 'GOOGL', 'TSLA')
GROUP BY ticker;"""
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. 🔍 Query: {example['query']}")
        print("   🔧 Generated SQL:")
        print(f"   {example['sql']}")
    
    print(f"\n✨ These are examples of what your model can generate!")
    print("🧪 For live testing, choose option 1 above (if you have enough memory)")

def main():
    print("🎯 **LLAMA 3.1 MODEL TESTING OPTIONS**")
    print("💻 MacBook Air M3 Optimized")
    print("=" * 60)
    
    print("\nChoose testing option:")
    print("1. 🧪 Interactive Testing (loads full model - may use lots of memory)")
    print("2. 🚀 Quick Demo (shows examples without loading model)")
    print("3. 📊 Full Test Results (from quick_test.py)")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            interactive_session()
        elif choice == "2":
            quick_demo()
        elif choice == "3":
            import subprocess
            subprocess.run(["python3", "quick_test.py"])
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()