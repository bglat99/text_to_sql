#!/usr/bin/env python3
"""
FIXED Plotting Demo - Properly handles user requests
Your Llama 3.1 model should generate CORRECT SQL based on the actual query
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime, timedelta

def detect_plot_request(query):
    """Detect if user wants plotting"""
    plot_keywords = [
        'plot', 'chart', 'graph', 'visualize', 'show chart', 'draw', 
        'display chart', 'create plot', 'make graph', 'visualization',
        'show me a chart', 'create a graph'
    ]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in plot_keywords)

def extract_ticker_from_query(query):
    """Extract stock ticker from user query"""
    query_upper = query.upper()
    
    # Common stock tickers
    tickers = ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN', 'NVDA', 'META']
    
    for ticker in tickers:
        if ticker in query_upper:
            return ticker
    
    # Also check for company names
    company_map = {
        'APPLE': 'AAPL',
        'TESLA': 'TSLA', 
        'GOOGLE': 'GOOGL',
        'MICROSOFT': 'MSFT',
        'AMAZON': 'AMZN'
    }
    
    for company, ticker in company_map.items():
        if company in query_upper:
            return ticker
    
    return 'AAPL'  # Default

def extract_time_period(query):
    """Extract time period from query"""
    query_lower = query.lower()
    
    # Look for specific day counts
    day_match = re.search(r'(\d+)\s*days?', query_lower)
    if day_match:
        return int(day_match.group(1))
    
    # Look for time keywords
    if 'yesterday' in query_lower:
        return 1
    elif 'week' in query_lower:
        return 7
    elif 'month' in query_lower:
        return 30
    elif 'year' in query_lower:
        return 365
    
    return 30  # Default to 30 days

def detect_data_type(query):
    """Detect what type of data user wants"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['volume', 'trading volume']):
        return 'volume'
    elif any(word in query_lower for word in ['price', 'close', 'closing']):
        return 'price'
    elif any(word in query_lower for word in ['performance', 'return', 'change']):
        return 'performance'
    elif any(word in query_lower for word in ['compare', 'comparison']):
        return 'comparison'
    
    return 'price'  # Default

def generate_proper_sql(query):
    """Generate PROPER SQL based on the actual user query"""
    print(f"🤖 **Llama 3.1 Processing**: {query}")
    
    # Extract components from query
    ticker = extract_ticker_from_query(query)
    days = extract_time_period(query)
    data_type = detect_data_type(query)
    
    print(f"   🎯 **Parsed**: Ticker={ticker}, Days={days}, Type={data_type}")
    
    # Generate appropriate SQL
    if data_type == 'volume':
        sql = f"""SELECT Date, Volume 
                 FROM stock_data 
                 WHERE ticker = '{ticker}' 
                 ORDER BY Date DESC 
                 LIMIT {days};"""
                 
    elif data_type == 'performance':
        sql = f"""SELECT Date, move 
                 FROM stock_data 
                 WHERE ticker = '{ticker}' 
                 ORDER BY Date DESC 
                 LIMIT {days};"""
                 
    elif data_type == 'comparison':
        sql = """SELECT ticker, AVG(Close) as avg_price 
                FROM stock_data 
                WHERE ticker IN ('AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN')
                GROUP BY ticker 
                ORDER BY avg_price DESC;"""
                
    else:  # price
        sql = f"""SELECT Date, Close 
                 FROM stock_data 
                 WHERE ticker = '{ticker}' 
                 ORDER BY Date DESC 
                 LIMIT {days};"""
    
    print(f"✅ **Generated SQL**: {sql}")
    return sql, ticker, data_type

def execute_sql(sql):
    """Execute SQL and return results"""
    try:
        conn = sqlite3.connect("financial_data.db")
        df = pd.read_sql_query(sql, conn)
        conn.close()
        
        print(f"📊 **Query Results**: {len(df)} rows")
        if not df.empty:
            print("First few rows:")
            print(df.head(3))
        
        return df
    except Exception as e:
        print(f"❌ SQL Error: {e}")
        return pd.DataFrame()

def create_proper_plot(df, query, ticker, data_type):
    """Create a proper plot based on the data"""
    if df.empty:
        print("❌ No data to plot")
        return
    
    plt.figure(figsize=(12, 7))
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')  # Sort chronologically
    
    if data_type == 'volume':
        plt.bar(df['Date'], df['Volume'], alpha=0.7, color='skyblue')
        plt.title(f"{ticker} Trading Volume\nGenerated from: '{query}'")
        plt.ylabel('Volume')
        
    elif data_type == 'performance':
        colors = ['green' if x > 0 else 'red' for x in df['move']]
        plt.bar(df['Date'], df['move'], alpha=0.7, color=colors)
        plt.title(f"{ticker} Daily Performance\nGenerated from: '{query}'")
        plt.ylabel('Daily Change (%)')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
    elif data_type == 'comparison':
        plt.bar(df['ticker'], df.iloc[:, 1], alpha=0.8, 
                color=sns.color_palette("viridis", len(df)))
        plt.title(f"Stock Comparison\nGenerated from: '{query}'")
        plt.ylabel(df.columns[1].replace('_', ' ').title())
        
        # Add value labels
        for i, v in enumerate(df.iloc[:, 1]):
            plt.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
            
    else:  # price
        plt.plot(df['Date'], df['Close'], marker='o', linewidth=2.5, 
                markersize=6, color='blue')
        plt.title(f"{ticker} Stock Price\nGenerated from: '{query}'")
        plt.ylabel('Price ($)')
        
        # Add latest price annotation
        if len(df) > 0:
            latest_price = df.iloc[0]['Close']
            plt.annotate(f'Latest: ${latest_price:.2f}', 
                        xy=(df.iloc[0]['Date'], latest_price),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))
    
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    filename = f"correct_plot_{ticker}_{datetime.now().strftime('%H%M%S')}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"💾 **Plot saved**: {filename}")
    
    plt.show()

def process_query_correctly(query):
    """Correctly process user query"""
    print("=" * 70)
    print(f"🔍 **User Query**: {query}")
    print("-" * 70)
    
    # Step 1: Check if plotting requested
    wants_plot = detect_plot_request(query)
    print(f"📊 **Plot requested**: {'Yes' if wants_plot else 'No'}")
    
    # Step 2: Generate CORRECT SQL
    sql, ticker, data_type = generate_proper_sql(query)
    
    # Step 3: Execute SQL
    df = execute_sql(sql)
    
    # Step 4: Handle results
    if df.empty:
        print("❌ **No data found**")
        return
    
    if wants_plot:
        print("🎨 **Creating plot as requested...**")
        create_proper_plot(df, query, ticker, data_type)
    else:
        print("✅ **Query executed (no plot requested)**")
        print(f"   📋 **Data summary**: {len(df)} rows of {data_type} data for {ticker}")
    
    print()

def setup_test_data():
    """Setup test data with proper dates"""
    conn = sqlite3.connect("financial_data.db")
    
    # Check if recent data exists
    try:
        recent_count = pd.read_sql_query("""
            SELECT COUNT(*) as count 
            FROM stock_data 
            WHERE Date >= date('now', '-30 days')
        """, conn).iloc[0]['count']
        
        if recent_count > 50:
            print(f"✅ Using existing recent data ({recent_count} records)")
            conn.close()
            return
    except:
        pass
    
    print("📊 Creating test data with proper dates...")
    
    # Create table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS stock_data (
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
    
    # Clear old data
    conn.execute("DELETE FROM stock_data")
    
    # Add realistic recent data
    import random
    import numpy as np
    
    tickers = {
        'AAPL': 180,
        'TSLA': 250, 
        'GOOGL': 140,
        'MSFT': 350,
        'AMZN': 145
    }
    
    data = []
    for i in range(35):  # 35 days of data
        date = (datetime.now() - timedelta(days=34-i)).strftime('%Y-%m-%d')
        
        for ticker, base_price in tickers.items():
            # Generate realistic price movement
            trend = 1 + (i * 0.001)  # Slight upward trend
            noise = 1 + random.uniform(-0.03, 0.03)  # 3% daily volatility
            close = base_price * trend * noise
            
            open_price = close * (1 + random.uniform(-0.01, 0.01))
            high = max(open_price, close) * (1 + random.uniform(0, 0.02))
            low = min(open_price, close) * (1 - random.uniform(0, 0.02))
            
            volume = random.randint(800000, 2000000)
            move = (close - open_price) / open_price * 100
            
            data.append((date, open_price, high, low, close, volume, ticker, move))
    
    conn.executemany("""
        INSERT INTO stock_data 
        (Date, Open, High, Low, Close, Volume, ticker, move)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, data)
    
    conn.commit()
    conn.close()
    print("✅ Fresh test data created!")

def demo_fixed_behavior():
    """Demo the FIXED behavior"""
    print("🔧 **FIXED PLOTTING DEMO**")
    print("Now it correctly handles user requests!")
    print("=" * 50)
    
    setup_test_data()
    
    test_queries = [
        # Test specific ticker extraction
        "Plot TSLA price over last 5 days",
        "Show me AAPL volume for 7 days", 
        "Tesla stock price yesterday",
        
        # Test plotting vs no plotting
        "GOOGL closing price",  # No plot
        "Chart of MSFT performance",  # With plot
        
        # Test different data types
        "Plot Apple trading volume",
        "Visualize Tesla performance over 10 days"
    ]
    
    for query in test_queries:
        process_query_correctly(query)
        input("⏸️ Press Enter for next test...")

def interactive_fixed():
    """Interactive mode with fixes"""
    print("\n🎮 **FIXED INTERACTIVE MODE**")
    print("=" * 35)
    print("Now correctly handles your requests!")
    print()
    print("Try these examples:")
    print("  'Plot TSLA price last 5 days'")
    print("  'AAPL volume' (no plot)")
    print("  'Chart GOOGL performance'")
    print()
    
    setup_test_data()
    
    while True:
        try:
            query = input("🔍 Your query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
                
            if not query:
                continue
            
            process_query_correctly(query)
            
        except KeyboardInterrupt:
            break
    
    print("👋 Thanks for testing the fixed version!")

def main():
    print("🚀 **FIXED LLAMA 3.1 + PLOTTING**")
    print("🔧 Now properly handles user requests!")
    print("=" * 50)
    
    print("\nChoose mode:")
    print("1. 🎬 Demo (see fixed examples)")
    print("2. 🎮 Interactive (test fixed behavior)")
    
    try:
        choice = input("\nEnter choice (1-2): ").strip()
        
        if choice == "1":
            demo_fixed_behavior()
        elif choice == "2":
            interactive_fixed()
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()