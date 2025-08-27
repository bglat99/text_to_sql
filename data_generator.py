"""
Comprehensive data generator for text-to-SQL fine-tuning
"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from config import FINANCIAL_CONFIG, DATA_CONFIG
from utils import format_training_example

class FinancialSQLGenerator:
    def __init__(self):
        self.tickers = FINANCIAL_CONFIG['tickers']
        self.schema = FINANCIAL_CONFIG['schema']
        self.n_samples = DATA_CONFIG['n_training_samples']
        
        # Define comprehensive query templates
        self.templates = self._create_query_templates()
    
    def _create_query_templates(self) -> List[Tuple[str, str]]:
        """
        Create diverse query templates covering various financial scenarios
        """
        templates = [
            # Basic price queries
            ("Show me {ticker}'s closing prices for the last {days} days", 
             "SELECT Date, Close FROM stock_data WHERE ticker = '{ticker}' AND Date >= date('now', '-{days} days') ORDER BY Date;"),
            
            ("What was {ticker}'s highest price in the last {days} days?", 
             "SELECT Date, High FROM stock_data WHERE ticker = '{ticker}' AND Date >= date('now', '-{days} days') ORDER BY High DESC LIMIT 1;"),
            
            ("Show me {ticker}'s daily returns for the past {days} days", 
             "SELECT Date, move FROM stock_data WHERE ticker = '{ticker}' AND Date >= date('now', '-{days} days') ORDER BY Date;"),
            
            # Volume analysis
            ("What was {ticker}'s highest trading volume day this month?", 
             "SELECT Date, Volume FROM stock_data WHERE ticker = '{ticker}' AND Date >= date('now', '-30 days') ORDER BY Volume DESC LIMIT 1;"),
            
            ("Show me {ticker}'s average daily volume over the last quarter", 
             "SELECT AVG(Volume) as avg_volume FROM stock_data WHERE ticker = '{ticker}' AND Date >= date('now', '-90 days');"),
            
            # Financial metrics
            ("Calculate {ticker}'s volatility over the past {days} days", 
             "SELECT SQRT(AVG(move * move)) as volatility FROM stock_data WHERE ticker = '{ticker}' AND Date >= date('now', '-{days} days');"),
            
            ("What is {ticker}'s Sharpe ratio for the last {days} days?", 
             "SELECT AVG(move) / SQRT(AVG(move * move)) as sharpe_ratio FROM stock_data WHERE ticker = '{ticker}' AND Date >= date('now', '-{days} days');"),
            
            ("Show me {ticker}'s maximum drawdown in the last {days} days", 
             "SELECT MIN(move) as max_drawdown FROM stock_data WHERE ticker = '{ticker}' AND Date >= date('now', '-{days} days');"),
            
            # Time-based analysis
            ("How many up days did {ticker} have this month?", 
             "SELECT COUNT(*) as up_days FROM stock_data WHERE ticker = '{ticker}' AND move > 0 AND Date >= date('now', '-30 days');"),
            
            ("Show me {ticker}'s best performing day this quarter", 
             "SELECT Date, move FROM stock_data WHERE ticker = '{ticker}' AND Date >= date('now', '-90 days') ORDER BY move DESC LIMIT 1;"),
            
            ("What was {ticker}'s worst day in the last {days} days?", 
             "SELECT Date, move FROM stock_data WHERE ticker = '{ticker}' AND Date >= date('now', '-{days} days') ORDER BY move ASC LIMIT 1;"),
            
            # Gap analysis
            ("Show me days when {ticker} gapped up more than {percent}%", 
             "SELECT Date, Open, Close, ((Open - LAG(Close) OVER (ORDER BY Date)) / LAG(Close) OVER (ORDER BY Date) * 100) as gap FROM stock_data WHERE ticker = '{ticker}' HAVING gap > {percent};"),
            
            ("Find {ticker} days with gaps larger than {percent}%", 
             "SELECT Date, Open, Close, ABS((Open - LAG(Close) OVER (ORDER BY Date)) / LAG(Close) OVER (ORDER BY Date) * 100) as gap FROM stock_data WHERE ticker = '{ticker}' HAVING gap > {percent};"),
            
            # Multi-stock comparisons
            ("Compare the average returns of {ticker1} and {ticker2} this year", 
             "SELECT ticker, AVG(move) as avg_return FROM stock_data WHERE ticker IN ('{ticker1}', '{ticker2}') AND Date >= '2024-01-01' GROUP BY ticker;"),
            
            ("Which stock performed better between {ticker1} and {ticker2} last month?", 
             "SELECT ticker, SUM(move) as total_return FROM stock_data WHERE ticker IN ('{ticker1}', '{ticker2}') AND Date >= date('now', '-30 days') GROUP BY ticker ORDER BY total_return DESC;"),
            
            ("Show me the correlation between {ticker1} and {ticker2} movements", 
             "SELECT CORR(a.move, b.move) as correlation FROM stock_data a JOIN stock_data b ON a.Date = b.Date WHERE a.ticker = '{ticker1}' AND b.ticker = '{ticker2}' AND a.Date >= date('now', '-90 days');"),
            
            # Market analysis
            ("Find the top {n} performing stocks this month", 
             "SELECT ticker, AVG(move) as avg_return FROM stock_data WHERE Date >= date('now', '-30 days') GROUP BY ticker ORDER BY avg_return DESC LIMIT {n};"),
            
            ("Show me stocks that had more than {n} consecutive up days", 
             "SELECT ticker, COUNT(*) as consecutive_up_days FROM (SELECT ticker, Date, move, ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY Date) - ROW_NUMBER() OVER (PARTITION BY ticker, CASE WHEN move > 0 THEN 1 ELSE 0 END ORDER BY Date) as grp FROM stock_data WHERE move > 0) t GROUP BY ticker, grp HAVING COUNT(*) >= {n};"),
            
            # Complex aggregations
            ("What is the average daily market movement across all stocks?", 
             "SELECT AVG(ABS(move)) as avg_market_movement FROM stock_data WHERE Date >= date('now', '-30 days');"),
            
            ("Show me the most volatile stock in the last {days} days", 
             "SELECT ticker, SQRT(AVG(move * move)) as volatility FROM stock_data WHERE Date >= date('now', '-{days} days') GROUP BY ticker ORDER BY volatility DESC LIMIT 1;"),
            
            # Date-specific queries
            ("What was {ticker}'s price on {date}?", 
             "SELECT Date, Close FROM stock_data WHERE ticker = '{ticker}' AND Date = '{date}';"),
            
            ("Show me {ticker}'s performance between {start_date} and {end_date}", 
             "SELECT Date, Open, High, Low, Close, move FROM stock_data WHERE ticker = '{ticker}' AND Date BETWEEN '{start_date}' AND '{end_date}' ORDER BY Date;"),
            
            # Technical analysis
            ("Calculate {ticker}'s {period}-day moving average", 
             "SELECT Date, Close, AVG(Close) OVER (ORDER BY Date ROWS BETWEEN {period}-1 PRECEDING AND CURRENT ROW) as moving_average FROM stock_data WHERE ticker = '{ticker}' ORDER BY Date;"),
            
            ("Show me when {ticker} crossed above its {period}-day moving average", 
             "SELECT a.Date, a.Close, a.moving_average FROM (SELECT Date, Close, AVG(Close) OVER (ORDER BY Date ROWS BETWEEN {period}-1 PRECEDING AND CURRENT ROW) as moving_average FROM stock_data WHERE ticker = '{ticker}') a WHERE a.Close > a.moving_average AND LAG(a.Close) OVER (ORDER BY a.Date) <= LAG(a.moving_average) OVER (ORDER BY a.Date);"),
            
            # Risk analysis
            ("What is {ticker}'s Value at Risk (95% confidence) for the last {days} days?", 
             "SELECT PERCENTILE(move, 5) as var_95 FROM stock_data WHERE ticker = '{ticker}' AND Date >= date('now', '-{days} days');"),
            
            ("Show me {ticker}'s worst {n} days", 
             "SELECT Date, move FROM stock_data WHERE ticker = '{ticker}' ORDER BY move ASC LIMIT {n};"),
            
            # Sector/group analysis
            ("Compare FAANG stocks performance this year", 
             "SELECT ticker, AVG(move) as avg_return FROM stock_data WHERE ticker IN ('META', 'AAPL', 'AMZN', 'NFLX', 'GOOGL') AND Date >= '2024-01-01' GROUP BY ticker ORDER BY avg_return DESC;"),
            
            ("Show me the best performing tech stock this quarter", 
             "SELECT ticker, AVG(move) as avg_return FROM stock_data WHERE ticker IN ('AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA') AND Date >= date('now', '-90 days') GROUP BY ticker ORDER BY avg_return DESC LIMIT 1;"),
        ]
        
        return templates
    
    def _fill_template(self, template: Tuple[str, str]) -> Tuple[str, str]:
        """
        Fill template with random values
        """
        text_template, sql_template = template
        
        # Random parameters
        ticker = random.choice(self.tickers)
        ticker1, ticker2 = random.sample(self.tickers, 2)
        days = random.choice([7, 14, 30, 60, 90, 180, 365])
        percent = random.choice([1, 2, 3, 5, 10])
        n = random.choice([3, 5, 10])
        period = random.choice([5, 10, 20, 50])
        
        # Date ranges
        end_date = datetime.now()
        start_date = end_date - timedelta(days=random.randint(30, 365))
        specific_date = end_date - timedelta(days=random.randint(1, 30))
        
        # Replace placeholders
        text = text_template.format(
            ticker=ticker, ticker1=ticker1, ticker2=ticker2,
            days=days, percent=percent, n=n, period=period,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            date=specific_date.strftime('%Y-%m-%d')
        )
        
        sql = sql_template.format(
            ticker=ticker, ticker1=ticker1, ticker2=ticker2,
            days=days, percent=percent, n=n, period=period,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            date=specific_date.strftime('%Y-%m-%d')
        )
        
        return text, sql
    
    def generate_training_data(self, n_samples=None) -> List[Dict]:
        """
        Generate diverse text-to-SQL pairs
        """
        if n_samples is None:
            n_samples = self.n_samples
        
        pairs = []
        
        for _ in range(n_samples):
            template = random.choice(self.templates)
            text, sql = self._fill_template(template)
            
            pairs.append({
                "instruction": "Convert this financial query to SQL:",
                "input": f"Schema: {self.schema}\n\nQuery: {text}",
                "output": sql
            })
        
        return pairs
    
    def generate_test_data(self, n_samples=None) -> List[Dict]:
        """
        Generate test data with more complex queries
        """
        if n_samples is None:
            n_samples = DATA_CONFIG['n_test_samples']
        
        # More complex test templates
        test_templates = [
            ("Find stocks that had both high volume and high price movement on the same day", 
             "SELECT ticker, Date, Volume, move FROM stock_data WHERE Volume > (SELECT AVG(Volume) * 2 FROM stock_data) AND ABS(move) > 5 ORDER BY Date DESC;"),
            
            ("Show me the correlation matrix between all major tech stocks", 
             "SELECT a.ticker as stock1, b.ticker as stock2, CORR(a.move, b.move) as correlation FROM stock_data a JOIN stock_data b ON a.Date = b.Date WHERE a.ticker IN ('AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA') AND b.ticker IN ('AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA') AND a.ticker < b.ticker GROUP BY a.ticker, b.ticker;"),
            
            ("Calculate the rolling beta of {ticker} against the market (SPY)", 
             "SELECT a.Date, COVAR(a.move, b.move) / VAR(b.move) as beta FROM stock_data a JOIN stock_data b ON a.Date = b.Date WHERE a.ticker = '{ticker}' AND b.ticker = 'SPY' AND a.Date >= date('now', '-90 days');"),
            
            ("Find days when multiple stocks had significant moves in the same direction", 
             "SELECT Date, COUNT(*) as stocks_moved FROM stock_data WHERE ABS(move) > 3 GROUP BY Date HAVING COUNT(*) >= 5 ORDER BY Date DESC;"),
        ]
        
        pairs = []
        for _ in range(n_samples):
            if random.random() < 0.7:  # 70% complex templates, 30% regular
                template = random.choice(test_templates)
            else:
                template = random.choice(self.templates)
            
            text, sql = self._fill_template(template)
            
            pairs.append({
                "instruction": "Convert this financial query to SQL:",
                "input": f"Schema: {self.schema}\n\nQuery: {text}",
                "output": sql
            })
        
        return pairs
    
    def save_data(self, train_data: List[Dict], test_data: List[Dict] = None, 
                  train_file: str = "train_data.json", test_file: str = "test_data.json"):
        """
        Save generated data to JSON files
        """
        # Save training data
        with open(train_file, 'w') as f:
            json.dump(train_data, f, indent=2)
        print(f"Saved {len(train_data)} training examples to {train_file}")
        
        # Save test data if provided
        if test_data:
            with open(test_file, 'w') as f:
                json.dump(test_data, f, indent=2)
            print(f"Saved {len(test_data)} test examples to {test_file}")
    
    def load_data(self, train_file: str = "train_data.json", test_file: str = "test_data.json") -> Tuple[List[Dict], List[Dict]]:
        """
        Load data from JSON files
        """
        with open(train_file, 'r') as f:
            train_data = json.load(f)
        
        test_data = []
        try:
            with open(test_file, 'r') as f:
                test_data = json.load(f)
        except FileNotFoundError:
            print(f"Test file {test_file} not found")
        
        return train_data, test_data

def main():
    """
    Main function to generate training data
    """
    print("Generating financial text-to-SQL training data...")
    
    generator = FinancialSQLGenerator()
    
    # Generate training data
    print(f"Generating {DATA_CONFIG['n_training_samples']} training examples...")
    train_data = generator.generate_training_data()
    
    # Generate test data
    print(f"Generating {DATA_CONFIG['n_test_samples']} test examples...")
    test_data = generator.generate_test_data()
    
    # Save data
    generator.save_data(train_data, test_data)
    
    # Show sample examples
    print("\nSample training examples:")
    for i, example in enumerate(train_data[:3]):
        print(f"\nExample {i+1}:")
        print(f"Query: {example['input'].split('Query: ')[1]}")
        print(f"SQL: {example['output']}")
    
    print(f"\nData generation complete!")
    print(f"Training examples: {len(train_data)}")
    print(f"Test examples: {len(test_data)}")

if __name__ == "__main__":
    main() 