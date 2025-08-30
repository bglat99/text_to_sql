#!/usr/bin/env python3
"""
Demo: Llama 3.1 Model + SQL Generation + Data Plotting
Shows how your fine-tuned model can generate SQL AND visualize results
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_sample_data():
    """Create sample financial data for plotting demo"""
    print("📊 Creating sample financial data...")
    
    conn = sqlite3.connect("financial_data.db")
    
    # Create table if it doesn't exist
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
    
    # Generate sample data for multiple stocks
    tickers = ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN']
    base_prices = {'AAPL': 150, 'TSLA': 250, 'GOOGL': 2800, 'MSFT': 300, 'AMZN': 3200}
    
    data = []
    for i in range(30):  # 30 days of data
        date = (datetime.now() - timedelta(days=29-i)).strftime('%Y-%m-%d')
        
        for ticker in tickers:
            base_price = base_prices[ticker]
            
            # Generate realistic price movements
            daily_change = np.random.normal(0, 0.02)  # 2% volatility
            price = base_price * (1 + daily_change * (i/10))  # Slight trend
            
            open_price = price * (1 + np.random.normal(0, 0.005))
            high_price = max(open_price, price) * (1 + abs(np.random.normal(0, 0.01)))
            low_price = min(open_price, price) * (1 - abs(np.random.normal(0, 0.01)))
            close_price = price
            volume = int(np.random.normal(1000000, 200000))
            move = (close_price - open_price) / open_price * 100
            
            data.append((date, open_price, high_price, low_price, close_price, volume, ticker, move))
    
    # Insert data
    conn.executemany("""
    INSERT OR REPLACE INTO stock_data 
    (Date, Open, High, Low, Close, Volume, ticker, move)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, data)
    
    conn.commit()
    conn.close()
    print("✅ Sample data created!")

def simulate_model_sql_generation(query):
    """Simulate what your fine-tuned Llama 3.1 model would generate"""
    
    print(f"🤖 **Llama 3.1 Model Processing**: {query}")
    
    # These are the types of SQL your model generates
    query_lower = query.lower()
    
    if "tesla" in query_lower and "price" in query_lower:
        sql = "SELECT Date, Close FROM stock_data WHERE ticker = 'TSLA' ORDER BY Date;"
        title = "Tesla Stock Price Over Time"
        
    elif "apple" in query_lower and "volume" in query_lower:
        sql = "SELECT Date, Volume FROM stock_data WHERE ticker = 'AAPL' ORDER BY Date;"
        title = "Apple Trading Volume Over Time"
        
    elif "compare" in query_lower and "performance" in query_lower:
        sql = """SELECT ticker, AVG(move) as avg_return 
                FROM stock_data 
                WHERE ticker IN ('AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN')
                GROUP BY ticker 
                ORDER BY avg_return DESC;"""
        title = "Average Stock Performance Comparison"
        
    elif "volatility" in query_lower:
        sql = """SELECT ticker, SQRT(AVG(move * move)) as volatility
                FROM stock_data 
                WHERE ticker IN ('AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN')
                GROUP BY ticker 
                ORDER BY volatility DESC;"""
        title = "Stock Volatility Comparison"
        
    elif "best" in query_lower and "day" in query_lower:
        sql = """SELECT Date, ticker, move 
                FROM stock_data 
                WHERE move = (SELECT MAX(move) FROM stock_data)
                LIMIT 5;"""
        title = "Best Performing Days"
        
    else:
        sql = "SELECT Date, Close, ticker FROM stock_data WHERE ticker = 'AAPL' ORDER BY Date;"
        title = "Default: Apple Stock Price"
    
    print(f"✅ **Generated SQL**:")
    print(f"   {sql}")
    
    return sql, title

def execute_and_plot(sql, title, query):
    """Execute SQL and create visualization"""
    try:
        # Execute SQL
        conn = sqlite3.connect("financial_data.db")
        df = pd.read_sql_query(sql, conn)
        conn.close()
        
        if df.empty:
            print("❌ No data returned from query")
            return
        
        print(f"📊 **Data Retrieved**: {len(df)} rows")
        print(f"   Columns: {list(df.columns)}")
        
        # Create plot based on data structure
        plt.figure(figsize=(12, 8))
        
        if 'Date' in df.columns and 'Close' in df.columns:
            # Time series plot
            df['Date'] = pd.to_datetime(df['Date'])
            plt.plot(df['Date'], df['Close'], marker='o', linewidth=2, markersize=6)
            plt.xlabel('Date')
            plt.ylabel('Price ($)')
            plt.title(f'{title}\n🤖 Generated by Llama 3.1 + Executed')
            plt.xticks(rotation=45)
            
        elif 'Date' in df.columns and 'Volume' in df.columns:
            # Volume bar chart
            df['Date'] = pd.to_datetime(df['Date'])
            plt.bar(df['Date'], df['Volume'], alpha=0.7)
            plt.xlabel('Date')
            plt.ylabel('Volume')
            plt.title(f'{title}\n🤖 Generated by Llama 3.1 + Executed')
            plt.xticks(rotation=45)
            
        elif 'ticker' in df.columns and len(df.columns) == 2:
            # Bar chart for comparisons
            y_col = [col for col in df.columns if col != 'ticker'][0]
            plt.bar(df['ticker'], df[y_col], alpha=0.8, color=sns.color_palette("viridis", len(df)))
            plt.xlabel('Stock Ticker')
            plt.ylabel(y_col.replace('_', ' ').title())
            plt.title(f'{title}\n🤖 Generated by Llama 3.1 + Executed')
            
            # Add value labels on bars
            for i, v in enumerate(df[y_col]):
                plt.text(i, v + max(df[y_col]) * 0.01, f'{v:.2f}', 
                        ha='center', va='bottom', fontweight='bold')
        
        else:
            # Generic scatter plot
            if len(df.columns) >= 2:
                x_col, y_col = df.columns[0], df.columns[1]
                plt.scatter(df[x_col], df[y_col], alpha=0.7, s=100)
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.title(f'{title}\n🤖 Generated by Llama 3.1 + Executed')
        
        # Add query as subtitle
        plt.figtext(0.5, 0.02, f'Original Query: "{query}"', 
                   ha='center', fontsize=10, style='italic', color='gray')
        
        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        filename = f"plot_{datetime.now().strftime('%H%M%S')}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"💾 **Plot saved**: {filename}")
        
        plt.show()
        
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_model_with_plots():
    """Demo showing model + plotting workflow"""
    print("🎯 **LLAMA 3.1 MODEL + PLOTTING DEMO**")
    print("=" * 50)
    print("This shows how your fine-tuned model generates SQL + creates plots!")
    print()
    
    # Create sample data
    create_sample_data()
    print()
    
    # Test queries that your model can handle
    test_queries = [
        "Show me Tesla's stock price over time",
        "What's Apple's trading volume pattern?", 
        "Compare performance of all tech stocks",
        "Show me stock volatility comparison",
        "Find the best performing days"
    ]
    
    print("🧪 **Testing Model + Plotting Pipeline**")
    print("-" * 40)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. 🔍 **Query**: {query}")
        print("   " + "─" * 40)
        
        # Step 1: Model generates SQL
        sql, title = simulate_model_sql_generation(query)
        
        # Step 2: Execute and plot
        print("   🎨 **Creating visualization...**")
        execute_and_plot(sql, title, query)
        
        if i < len(test_queries):
            input("   ⏸️  Press Enter for next demo...")

def interactive_mode():
    """Interactive mode for testing"""
    print("\n🎮 **INTERACTIVE MODE**")
    print("=" * 30)
    print("Type your financial queries and see SQL + plots!")
    print("Examples:")
    print("  - Tesla stock price")
    print("  - Apple volume")  
    print("  - Compare tech stocks")
    print("Type 'quit' to exit\n")
    
    create_sample_data()
    
    while True:
        try:
            query = input("🔍 Your query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
                
            if not query:
                continue
            
            print()
            sql, title = simulate_model_sql_generation(query)
            execute_and_plot(sql, title, query)
            print()
            
        except KeyboardInterrupt:
            break
    
    print("👋 Thanks for testing!")

def main():
    print("🚀 **LLAMA 3.1 + PLOTTING CAPABILITIES**")
    print("🤖 Your fine-tuned model generates SQL")
    print("📊 Built-in plotting executes and visualizes results")
    print("=" * 60)
    
    print("\nChoose mode:")
    print("1. 🎬 Demo Mode (see all examples)")
    print("2. 🎮 Interactive Mode (try your own queries)")
    print("3. 📈 Quick Test (single example)")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            demo_model_with_plots()
        elif choice == "2":
            interactive_mode()
        elif choice == "3":
            create_sample_data()
            sql, title = simulate_model_sql_generation("Show me Tesla's stock price")
            execute_and_plot(sql, title, "Tesla stock price")
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()