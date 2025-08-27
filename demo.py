"""
Demo script for the fine-tuned text-to-SQL model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import FINANCIAL_CONFIG
from utils import extract_sql_from_response, test_sql_execution
from visualization import FinancialDataVisualizer
import sqlite3
import pandas as pd

class FinancialSQLDemo:
    def __init__(self, model_path="financial_sql_model"):
        """
        Initialize the demo with the fine-tuned model
        """
        self.model_path = model_path
        self.db_path = FINANCIAL_CONFIG['database_path']
        
        print(f"Loading fine-tuned model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        print("Model loaded successfully!")
    
    def convert_to_sql(self, query: str) -> str:
        """
        Convert natural language query to SQL
        """
        schema = FINANCIAL_CONFIG['schema']
        
        prompt = f"""### Instruction:
Convert this financial query to SQL:

### Input:
Schema: {schema}

Query: {query}

### Response:
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_length=512, 
                temperature=0.1, 
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        sql = extract_sql_from_response(response)
        
        return sql
    
    def execute_query(self, sql: str) -> pd.DataFrame:
        """
        Execute SQL query and return results
        """
        try:
            conn = sqlite3.connect(self.db_path)
            result = pd.read_sql_query(sql, conn)
            conn.close()
            return result
        except Exception as e:
            print(f"Error executing SQL: {e}")
            return pd.DataFrame()
    
    def demo_query(self, query: str, plot_results: bool = True):
        """
        Demo a single query
        """
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        # Generate SQL
        print("Generating SQL...")
        sql = self.convert_to_sql(query)
        print(f"Generated SQL: {sql}")
        
        # Execute query
        print("\nExecuting query...")
        result = self.execute_query(sql)
        
        if not result.empty:
            print(f"✓ Query executed successfully!")
            print(f"Results: {len(result)} rows, {len(result.columns)} columns")
            print("\nSample results:")
            print(result.head(10).to_string(index=False))
            
            # Plot results if requested
            if plot_results and len(result) > 1:
                try:
                    print("\n📊 Generating visualization...")
                    viz = FinancialDataVisualizer(self.db_path)
                    viz.plot_sql_query_results(sql, plot_type='auto')
                except Exception as e:
                    print(f"Visualization error: {e}")
        else:
            print("✗ Query execution failed or returned no results")
    
    def run_demo_queries(self, plot_results: bool = True):
        """
        Run a series of demo queries
        """
        demo_queries = [
            "Show me Apple's closing prices for the last 30 days",
            "What was Tesla's highest price this month?",
            "Calculate the average daily return for SPY",
            "Find the most volatile stock in the last 60 days",
            "Compare the performance of Apple and Microsoft this year",
            "Show me days when the market moved more than 3%",
            "What is the correlation between Tesla and Apple stock movements?",
            "Find stocks that had more than 5 consecutive up days",
            "Calculate the Sharpe ratio for Google over the past quarter",
            "Show me the best performing tech stock this month"
        ]
        
        print("🚀 Financial Text-to-SQL Demo")
        print("=" * 60)
        print("This demo shows the fine-tuned model converting natural language")
        print("financial queries to SQL and executing them on real data.")
        print("=" * 60)
        
        for i, query in enumerate(demo_queries, 1):
            print(f"\nDemo {i}/{len(demo_queries)}")
            self.demo_query(query, plot_results=plot_results)
            
            if i < len(demo_queries):
                input("\nPress Enter to continue to the next query...")
    
    def interactive_mode(self):
        """
        Interactive mode for testing custom queries
        """
        print("\n" + "=" * 60)
        print("Interactive Mode")
        print("=" * 60)
        print("Type your financial queries and see them converted to SQL!")
        print("Type 'quit' to exit")
        print("Type 'plot off' to disable plotting")
        print("Type 'plot on' to enable plotting")
        print("=" * 60)
        
        plot_results = True
        
        while True:
            query = input("\nEnter your financial query: ").strip()
            
            if query.lower() == 'quit':
                break
            elif query.lower() == 'plot off':
                plot_results = False
                print("📊 Plotting disabled")
                continue
            elif query.lower() == 'plot on':
                plot_results = True
                print("📊 Plotting enabled")
                continue
            
            if not query:
                continue
            
            try:
                self.demo_query(query, plot_results=plot_results)
            except Exception as e:
                print(f"Error processing query: {e}")

def main():
    """
    Main demo function
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Demo the fine-tuned text-to-SQL model")
    parser.add_argument("--model_path", default="financial_sql_model", help="Path to fine-tuned model")
    parser.add_argument("--interactive", action="store_true", help="Run interactive mode")
    parser.add_argument("--demo", action="store_true", help="Run demo queries")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting in demo mode")
    
    args = parser.parse_args()
    
    try:
        demo = FinancialSQLDemo(args.model_path)
        
        if args.interactive:
            demo.interactive_mode()
        elif args.demo:
            demo.run_demo_queries(plot_results=not args.no_plot)
        else:
            # Default: run demo queries
            demo.run_demo_queries(plot_results=not args.no_plot)
            
    except FileNotFoundError:
        print(f"✗ Model not found at {args.model_path}")
        print("Please run the training first: python finetune_model.py")
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    main() 