"""
Comprehensive evaluation script for the fine-tuned text-to-SQL model
"""

import torch
import sqlite3
import pandas as pd
import json
import time
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import EVAL_CONFIG, FINANCIAL_CONFIG
from utils import validate_sql_syntax, extract_sql_from_response, calculate_accuracy_metrics

class SQLEvaluator:
    def __init__(self, model_path: str, db_path: str = None):
        """
        Initialize the SQL evaluator
        """
        self.model_path = model_path
        self.db_path = db_path or FINANCIAL_CONFIG['database_path']
        
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Test database connection
        try:
            conn = sqlite3.connect(self.db_path)
            conn.close()
            print(f"Database connection successful: {self.db_path}")
        except Exception as e:
            print(f"Warning: Database connection failed: {e}")
    
    def generate_sql(self, natural_query: str, schema: str = None) -> str:
        """
        Generate SQL from natural language query
        """
        if schema is None:
            schema = FINANCIAL_CONFIG['schema']
        
        prompt = f"""### Instruction:
Convert this financial query to SQL:

### Input:
Schema: {schema}

Query: {natural_query}

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
    
    def test_sql_execution(self, sql: str) -> Tuple[bool, any]:
        """
        Test if SQL executes without errors and return results
        """
        try:
            conn = sqlite3.connect(self.db_path)
            result = pd.read_sql_query(sql, conn)
            conn.close()
            return True, result
        except Exception as e:
            return False, str(e)
    
    def evaluate_single_query(self, query: str, expected_sql: str = None) -> Dict:
        """
        Evaluate a single query
        """
        start_time = time.time()
        
        # Generate SQL
        generated_sql = self.generate_sql(query)
        generation_time = time.time() - start_time
        
        # Validate syntax
        syntax_valid, syntax_error = validate_sql_syntax(generated_sql)
        
        # Test execution
        execution_success, execution_result = self.test_sql_execution(generated_sql)
        
        # Calculate result metrics
        result_metrics = {}
        if execution_success and isinstance(execution_result, pd.DataFrame):
            result_metrics = {
                'row_count': len(execution_result),
                'column_count': len(execution_result.columns),
                'has_data': not execution_result.empty
            }
        
        return {
            'query': query,
            'generated_sql': generated_sql,
            'expected_sql': expected_sql,
            'syntax_valid': syntax_valid,
            'syntax_error': syntax_error,
            'execution_success': execution_success,
            'execution_error': execution_result if not execution_success else None,
            'result_metrics': result_metrics,
            'generation_time': generation_time
        }
    
    def evaluate_test_set(self, test_queries: List[str] = None) -> Dict:
        """
        Evaluate model on a set of test queries
        """
        if test_queries is None:
            test_queries = EVAL_CONFIG['test_queries']
        
        print(f"Evaluating {len(test_queries)} test queries...")
        
        results = []
        total_time = 0
        
        for i, query in enumerate(test_queries):
            print(f"Evaluating query {i+1}/{len(test_queries)}: {query[:50]}...")
            result = self.evaluate_single_query(query)
            results.append(result)
            total_time += result['generation_time']
        
        # Calculate aggregate metrics
        metrics = self._calculate_aggregate_metrics(results)
        metrics['total_evaluation_time'] = total_time
        metrics['avg_generation_time'] = total_time / len(results)
        
        return {
            'results': results,
            'metrics': metrics
        }
    
    def _calculate_aggregate_metrics(self, results: List[Dict]) -> Dict:
        """
        Calculate aggregate metrics from evaluation results
        """
        total = len(results)
        syntax_correct = sum(1 for r in results if r['syntax_valid'])
        execution_success = sum(1 for r in results if r['execution_success'])
        
        # Calculate average result metrics
        avg_row_count = 0
        avg_column_count = 0
        successful_executions = [r for r in results if r['execution_success']]
        
        if successful_executions:
            avg_row_count = sum(r['result_metrics']['row_count'] for r in successful_executions) / len(successful_executions)
            avg_column_count = sum(r['result_metrics']['column_count'] for r in successful_executions) / len(successful_executions)
        
        return {
            'total_queries': total,
            'syntax_accuracy': syntax_correct / total if total > 0 else 0,
            'execution_rate': execution_success / total if total > 0 else 0,
            'avg_row_count': avg_row_count,
            'avg_column_count': avg_column_count,
            'successful_executions': execution_success,
            'failed_executions': total - execution_success
        }
    
    def interactive_evaluation(self):
        """
        Interactive evaluation mode
        """
        print("\n" + "=" * 60)
        print("Interactive SQL Generation Mode")
        print("=" * 60)
        print("Type 'quit' to exit")
        
        while True:
            query = input("\nEnter your financial query: ").strip()
            
            if query.lower() == 'quit':
                break
            
            if not query:
                continue
            
            print(f"\nGenerating SQL for: {query}")
            result = self.evaluate_single_query(query)
            
            print(f"\nGenerated SQL:")
            print(f"  {result['generated_sql']}")
            
            if result['syntax_valid']:
                print(f"✓ Syntax: Valid")
            else:
                print(f"✗ Syntax: {result['syntax_error']}")
            
            if result['execution_success']:
                print(f"✓ Execution: Successful")
                if isinstance(result['execution_result'], pd.DataFrame):
                    print(f"  Results: {len(result['execution_result'])} rows, {len(result['execution_result'].columns)} columns")
                    if not result['execution_result'].empty:
                        print(f"  Sample data:")
                        print(result['execution_result'].head(3).to_string())
            else:
                print(f"✗ Execution: {result['execution_error']}")
            
            print(f"Generation time: {result['generation_time']:.2f}s")
    
    def save_evaluation_results(self, results: Dict, output_file: str = "evaluation_results.json"):
        """
        Save evaluation results to file
        """
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Evaluation results saved to {output_file}")

def run_comprehensive_evaluation(model_path: str = "financial_sql_model"):
    """
    Run comprehensive evaluation with multiple test scenarios
    """
    print("=" * 60)
    print("Comprehensive Model Evaluation")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = SQLEvaluator(model_path)
    
    # Test scenarios
    test_scenarios = {
        "Basic Queries": [
            "Show me Apple's closing prices for the last 30 days",
            "What was Tesla's highest price this month?",
            "Calculate the average daily return for SPY"
        ],
        "Financial Metrics": [
            "Calculate Apple's volatility over the past month",
            "What is Tesla's Sharpe ratio for the last quarter?",
            "Show me the maximum drawdown for Google"
        ],
        "Multi-Stock Analysis": [
            "Compare the performance of Apple and Microsoft this year",
            "Find the best performing tech stock this quarter",
            "Show me the correlation between Tesla and Apple"
        ],
        "Complex Queries": [
            "Find stocks that had consecutive up days for more than 5 days",
            "Show me days when the market gap was more than 3%",
            "Calculate the rolling beta of Tesla against SPY"
        ],
        "Edge Cases": [
            "What was the stock price on a specific date that doesn't exist?",
            "Show me data for a ticker that doesn't exist",
            "Calculate metrics for a very short time period"
        ]
    }
    
    all_results = {}
    overall_metrics = {
        'total_queries': 0,
        'syntax_correct': 0,
        'execution_success': 0
    }
    
    for scenario_name, queries in test_scenarios.items():
        print(f"\nEvaluating {scenario_name}...")
        results = evaluator.evaluate_test_set(queries)
        
        all_results[scenario_name] = results
        overall_metrics['total_queries'] += results['metrics']['total_queries']
        overall_metrics['syntax_correct'] += int(results['metrics']['syntax_accuracy'] * results['metrics']['total_queries'])
        overall_metrics['execution_success'] += results['metrics']['successful_executions']
        
        # Print scenario results
        print(f"  Syntax Accuracy: {results['metrics']['syntax_accuracy']:.2%}")
        print(f"  Execution Rate: {results['metrics']['execution_rate']:.2%}")
        print(f"  Avg Generation Time: {results['metrics']['avg_generation_time']:.2f}s")
    
    # Calculate overall metrics
    overall_accuracy = overall_metrics['syntax_correct'] / overall_metrics['total_queries']
    overall_execution_rate = overall_metrics['execution_success'] / overall_metrics['total_queries']
    
    print("\n" + "=" * 60)
    print("Overall Evaluation Results")
    print("=" * 60)
    print(f"Total Queries: {overall_metrics['total_queries']}")
    print(f"Overall Syntax Accuracy: {overall_accuracy:.2%}")
    print(f"Overall Execution Rate: {overall_execution_rate:.2%}")
    
    # Check against targets
    target_accuracy = EVAL_CONFIG['target_accuracy']
    target_execution_rate = EVAL_CONFIG['target_execution_rate']
    
    print(f"\nTarget Accuracy: {target_accuracy:.2%}")
    print(f"Target Execution Rate: {target_execution_rate:.2%}")
    
    if overall_accuracy >= target_accuracy:
        print("✓ Syntax accuracy target achieved!")
    else:
        print("✗ Syntax accuracy below target")
    
    if overall_execution_rate >= target_execution_rate:
        print("✓ Execution rate target achieved!")
    else:
        print("✗ Execution rate below target")
    
    # Save results
    final_results = {
        'overall_metrics': overall_metrics,
        'scenario_results': all_results,
        'targets': {
            'accuracy': target_accuracy,
            'execution_rate': target_execution_rate
        }
    }
    
    evaluator.save_evaluation_results(final_results)
    
    return final_results

def main():
    """
    Main evaluation function
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned text-to-SQL model")
    parser.add_argument("--model_path", default="financial_sql_model", help="Path to fine-tuned model")
    parser.add_argument("--interactive", action="store_true", help="Run interactive evaluation mode")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive evaluation")
    
    args = parser.parse_args()
    
    if args.interactive:
        evaluator = SQLEvaluator(args.model_path)
        evaluator.interactive_evaluation()
    elif args.comprehensive:
        run_comprehensive_evaluation(args.model_path)
    else:
        # Default: run comprehensive evaluation
        run_comprehensive_evaluation(args.model_path)

if __name__ == "__main__":
    main() 