"""
Database setup script for financial data
"""

import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import random
from config import FINANCIAL_CONFIG
from utils import create_sample_database

def download_real_data(tickers=None, start_date=None, end_date=None):
    """
    Download real financial data using yfinance
    """
    if tickers is None:
        tickers = FINANCIAL_CONFIG['tickers']
    
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365)
    
    if end_date is None:
        end_date = datetime.now()
    
    all_data = []
    
    for ticker in tickers:
        try:
            print(f"Downloading data for {ticker}...")
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            
            if not data.empty:
                # Calculate daily move percentage
                data['move'] = data['Close'].pct_change() * 100
                data['ticker'] = ticker
                data['Date'] = data.index.strftime('%Y-%m-%d')
                
                # Reset index to make Date a column
                data = data.reset_index(drop=True)
                
                # Select and rename columns
                data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Close', 'Volume', 'ticker', 'move']]
                data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume', 'ticker', 'move']
                
                all_data.append(data)
                print(f"Downloaded {len(data)} records for {ticker}")
            else:
                print(f"No data available for {ticker}")
                
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
            continue
    
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def create_database_with_real_data(db_path="financial_data.db", use_real_data=True):
    """
    Create database with either real or synthetic data
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
    
    if use_real_data:
        try:
            print("Downloading real financial data...")
            data = download_real_data()
            
            if not data.empty:
                # Insert data into database
                data.to_sql('stock_data', conn, if_exists='replace', index=False)
                print(f"Inserted {len(data)} records into database")
                
                # Show sample data
                sample = conn.execute("SELECT * FROM stock_data LIMIT 5").fetchall()
                print("Sample data:")
                for row in sample:
                    print(row)
            else:
                print("No real data available, creating synthetic data...")
                create_sample_database(db_path)
                
        except Exception as e:
            print(f"Error downloading real data: {e}")
            print("Creating synthetic data instead...")
            create_sample_database(db_path)
    else:
        print("Creating synthetic data...")
        create_sample_database(db_path)
    
    # Create indexes for better performance
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON stock_data(ticker)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON stock_data(Date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ticker_date ON stock_data(ticker, Date)")
    
    # Show database statistics
    total_records = conn.execute("SELECT COUNT(*) FROM stock_data").fetchone()[0]
    unique_tickers = conn.execute("SELECT COUNT(DISTINCT ticker) FROM stock_data").fetchone()[0]
    date_range = conn.execute("SELECT MIN(Date), MAX(Date) FROM stock_data").fetchone()
    
    print(f"\nDatabase Statistics:")
    print(f"Total records: {total_records}")
    print(f"Unique tickers: {unique_tickers}")
    print(f"Date range: {date_range[0]} to {date_range[1]}")
    
    conn.close()
    print(f"Database created successfully at {db_path}")

def verify_database(db_path="financial_data.db"):
    """
    Verify the database structure and data
    """
    conn = sqlite3.connect(db_path)
    
    # Check table structure
    cursor = conn.execute("PRAGMA table_info(stock_data)")
    columns = cursor.fetchall()
    print("Table structure:")
    for col in columns:
        print(f"  {col[1]} ({col[2]})")
    
    # Check data quality
    print("\nData quality checks:")
    
    # Check for null values
    null_counts = conn.execute("""
        SELECT 
            COUNT(*) as total_rows,
            SUM(CASE WHEN Date IS NULL THEN 1 ELSE 0 END) as null_dates,
            SUM(CASE WHEN ticker IS NULL THEN 1 ELSE 0 END) as null_tickers,
            SUM(CASE WHEN Close IS NULL THEN 1 ELSE 0 END) as null_prices
        FROM stock_data
    """).fetchone()
    
    print(f"Total rows: {null_counts[0]}")
    print(f"Null dates: {null_counts[1]}")
    print(f"Null tickers: {null_counts[2]}")
    print(f"Null prices: {null_counts[3]}")
    
    # Check price ranges
    price_stats = conn.execute("""
        SELECT 
            MIN(Close) as min_price,
            MAX(Close) as max_price,
            AVG(Close) as avg_price,
            MIN(move) as min_move,
            MAX(move) as max_move,
            AVG(move) as avg_move
        FROM stock_data
    """).fetchone()
    
    print(f"\nPrice statistics:")
    print(f"Min price: ${price_stats[0]:.2f}")
    print(f"Max price: ${price_stats[1]:.2f}")
    print(f"Avg price: ${price_stats[2]:.2f}")
    print(f"Min move: {price_stats[3]:.2f}%")
    print(f"Max move: {price_stats[4]:.2f}%")
    print(f"Avg move: {price_stats[5]:.2f}%")
    
    # Sample queries to test
    test_queries = [
        "SELECT COUNT(*) FROM stock_data WHERE ticker = 'AAPL'",
        "SELECT AVG(move) FROM stock_data WHERE ticker = 'TSLA'",
        "SELECT Date, Close FROM stock_data WHERE ticker = 'GOOGL' ORDER BY Date DESC LIMIT 5"
    ]
    
    print(f"\nTesting sample queries:")
    for query in test_queries:
        try:
            result = conn.execute(query).fetchall()
            print(f"✓ {query}: {len(result)} results")
        except Exception as e:
            print(f"✗ {query}: {e}")
    
    conn.close()

if __name__ == "__main__":
    print("Setting up financial database...")
    
    # Create database with real data (fallback to synthetic if needed)
    create_database_with_real_data(use_real_data=True)
    
    # Verify the database
    print("\nVerifying database...")
    verify_database()
    
    print("\nDatabase setup complete!") 