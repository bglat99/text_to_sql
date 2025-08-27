"""
Utility functions for text-to-SQL fine-tuning
"""

import re
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import random

def validate_sql_syntax(sql: str) -> Tuple[bool, str]:
    """
    Basic SQL syntax validation
    """
    try:
        # Remove comments and normalize whitespace
        sql_clean = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
        sql_clean = ' '.join(sql_clean.split())
        
        # Basic checks
        if not sql_clean.upper().startswith('SELECT'):
            return False, "Query must start with SELECT"
        
        if 'FROM' not in sql_clean.upper():
            return False, "Missing FROM clause"
        
        # Check for balanced parentheses
        if sql_clean.count('(') != sql_clean.count(')'):
            return False, "Unbalanced parentheses"
        
        return True, "Valid SQL syntax"
    except Exception as e:
        return False, f"Syntax validation error: {str(e)}"

def extract_sql_from_response(response: str) -> str:
    """
    Extract SQL query from model response
    """
    # Look for SQL after "### Response:" or similar markers
    patterns = [
        r'### Response:\s*(.*?)(?:\n\n|$)',
        r'SQL:\s*(.*?)(?:\n\n|$)',
        r'```sql\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # If no pattern matches, return the last line that looks like SQL
    lines = response.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and ('SELECT' in line.upper() or 'FROM' in line.upper()):
            return line
    
    return response.strip()

def calculate_financial_metrics(data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate common financial metrics
    """
    if data.empty:
        return {}
    
    metrics = {}
    
    if 'move' in data.columns:
        metrics['volatility'] = data['move'].std()
        metrics['avg_return'] = data['move'].mean()
        metrics['sharpe_ratio'] = data['move'].mean() / data['move'].std() if data['move'].std() > 0 else 0
        metrics['max_gain'] = data['move'].max()
        metrics['max_loss'] = data['move'].min()
    
    if 'Volume' in data.columns:
        metrics['avg_volume'] = data['Volume'].mean()
        metrics['max_volume'] = data['Volume'].max()
    
    if 'Close' in data.columns:
        metrics['price_range'] = data['Close'].max() - data['Close'].min()
        metrics['current_price'] = data['Close'].iloc[-1] if len(data) > 0 else 0
    
    return metrics

def generate_date_range(days: int = 30) -> Tuple[str, str]:
    """
    Generate a date range for SQL queries
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

def create_sample_database(db_path: str = "financial_data.db"):
    """
    Create a sample database with synthetic financial data
    """
    conn = sqlite3.connect(db_path)
    
    # Create table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stock_data (
            Date TEXT,
            Open REAL,
            High REAL,
            Low REAL,
            Close REAL,
            Adj_Close REAL,
            Volume INTEGER,
            ticker TEXT,
            move REAL
        )
    """)
    
    # Generate synthetic data
    tickers = ['SPY', 'AAPL', 'NFLX', 'GOOGL', 'AMZN', 'META', 'MSFT', 'TSLA', 'ABNB']
    start_date = datetime(2023, 1, 1)
    
    for ticker in tickers:
        base_price = random.uniform(50, 500)
        current_price = base_price
        
        for i in range(365):  # One year of data
            date = start_date + timedelta(days=i)
            
            # Generate realistic price movements
            daily_move = random.normalvariate(0, 0.02)  # 2% daily volatility
            current_price *= (1 + daily_move)
            
            open_price = current_price
            high_price = current_price * random.uniform(1.0, 1.05)
            low_price = current_price * random.uniform(0.95, 1.0)
            close_price = current_price * random.uniform(0.98, 1.02)
            adj_close = close_price
            
            volume = random.randint(1000000, 10000000)
            
            conn.execute("""
                INSERT INTO stock_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                date.strftime('%Y-%m-%d'),
                open_price,
                high_price,
                low_price,
                close_price,
                adj_close,
                volume,
                ticker,
                daily_move * 100  # Convert to percentage
            ))
    
    conn.commit()
    conn.close()
    print(f"Sample database created at {db_path}")

def format_training_example(instruction: str, input_text: str, output: str) -> str:
    """
    Format a training example for the model
    """
    return f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""

def test_sql_execution(sql: str, db_path: str = "financial_data.db") -> Tuple[bool, any]:
    """
    Test if SQL executes without errors
    """
    try:
        conn = sqlite3.connect(db_path)
        result = pd.read_sql_query(sql, conn)
        conn.close()
        return True, len(result)
    except Exception as e:
        return False, str(e)

def calculate_accuracy_metrics(predictions: List[str], ground_truth: List[str], 
                             db_path: str = "financial_data.db") -> Dict[str, float]:
    """
    Calculate comprehensive accuracy metrics
    """
    syntax_correct = 0
    execution_success = 0
    semantic_correct = 0
    
    for pred, truth in zip(predictions, ground_truth):
        # Syntax validation
        is_valid, _ = validate_sql_syntax(pred)
        if is_valid:
            syntax_correct += 1
            
            # Execution test
            try:
                conn = sqlite3.connect(db_path)
                result = pd.read_sql_query(pred, conn)
                conn.close()
                execution_success += 1
                
                # Basic semantic check (compare result shapes)
                try:
                    truth_result = pd.read_sql_query(truth, conn)
                    if len(result) == len(truth_result):
                        semantic_correct += 1
                except:
                    pass
                    
            except Exception:
                pass
    
    total = len(predictions)
    return {
        'syntax_accuracy': syntax_correct / total if total > 0 else 0,
        'execution_rate': execution_success / total if total > 0 else 0,
        'semantic_accuracy': semantic_correct / total if total > 0 else 0,
        'total_queries': total
    } 