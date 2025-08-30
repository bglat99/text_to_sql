#!/usr/bin/env python3
"""
Smart Plotting Demo - Plots ONLY when user asks for it
Your Llama 3.1 model generates SQL, and we plot only if requested
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

def detect_plot_request(query):
    """Detect if user wants plotting based on their natural language"""
    plot_keywords = [
        'plot', 'chart', 'graph', 'visualize', 'show chart', 'draw', 
        'display chart', 'create plot', 'make graph', 'visualization'
    ]
    
    query_lower = query.lower()
    wants_plot = any(keyword in query_lower for keyword in plot_keywords)
    
    return wants_plot

def simulate_llama_sql_generation(query):
    """Simulate your fine-tuned Llama 3.1 generating SQL"""
    print(f"🤖 **Llama 3.1 Processing**: {query}")
    
    query_lower = query.lower()
    
    # Your model generates appropriate SQL based on the query
    if "tesla" in query_lower and ("price" in query_lower or "close" in query_lower):
        sql = "SELECT Date, Close FROM stock_data WHERE ticker = 'TSLA' ORDER BY Date DESC LIMIT 30;"
        
    elif "apple" in query_lower and "volume" in query_lower:
        sql = "SELECT Date, Volume FROM stock_data WHERE ticker = 'AAPL' ORDER BY Date DESC LIMIT 30;"
        
    elif "compare" in query_lower or "comparison" in query_lower:
        sql = """SELECT ticker, AVG(Close) as avg_price 
                FROM stock_data 
                WHERE ticker IN ('AAPL', 'TSLA', 'GOOGL') 
                GROUP BY ticker;"""
        
    elif "performance" in query_lower:
        sql = """SELECT ticker, AVG(move) as avg_return 
                FROM stock_data 
                WHERE ticker IN ('AAPL', 'TSLA', 'GOOGL')
                GROUP BY ticker 
                ORDER BY avg_return DESC;"""
        
    else:
        # Default query
        sql = "SELECT Date, Close FROM stock_data WHERE ticker = 'AAPL' ORDER BY Date DESC LIMIT 10;"
    
    print(f"✅ **Generated SQL**: {sql}")
    return sql

def execute_sql(sql):
    """Execute the SQL and return results"""
    try:
        conn = sqlite3.connect("financial_data.db")
        df = pd.read_sql_query(sql, conn)
        conn.close()
        
        print(f"📊 **Query Results**: {len(df)} rows")
        if not df.empty:
            print("First few rows:")
            print(df.head())
        
        return df
    except Exception as e:
        print(f"❌ SQL Error: {e}")
        return pd.DataFrame()

def create_plot(df, query):
    """Create plot only if data exists"""
    if df.empty:
        print("❌ No data to plot")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Smart plotting based on data structure
    if 'Date' in df.columns and 'Close' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        plt.plot(df['Date'], df['Close'], marker='o', linewidth=2)
        plt.title(f"Stock Price Chart\n(Generated from: '{query}')")
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.xticks(rotation=45)
        
    elif 'Date' in df.columns and 'Volume' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        plt.bar(df['Date'], df['Volume'], alpha=0.7)
        plt.title(f"Volume Chart\n(Generated from: '{query}')")
        plt.xlabel('Date')
        plt.ylabel('Volume')
        plt.xticks(rotation=45)
        
    elif 'ticker' in df.columns and len(df.columns) == 2:
        value_col = [col for col in df.columns if col != 'ticker'][0]
        plt.bar(df['ticker'], df[value_col], alpha=0.8)
        plt.title(f"Stock Comparison\n(Generated from: '{query}')")
        plt.xlabel('Stock')
        plt.ylabel(value_col.replace('_', ' ').title())
        
        # Add value labels
        for i, v in enumerate(df[value_col]):
            plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    import datetime
    filename = f"user_requested_plot_{datetime.datetime.now().strftime('%H%M%S')}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"💾 **Plot saved**: {filename}")
    
    plt.show()

def process_user_query(query):
    """Complete workflow: detect intent, generate SQL, execute, optionally plot"""
    print("=" * 60)
    print(f"🔍 **User Query**: {query}")
    print("-" * 60)
    
    # Step 1: Check if user wants plotting
    wants_plot = detect_plot_request(query)
    print(f"📊 **Plot requested**: {'Yes' if wants_plot else 'No'}")
    
    # Step 2: Your Llama model generates SQL
    sql = simulate_llama_sql_generation(query)
    
    # Step 3: Execute SQL
    df = execute_sql(sql)
    
    # Step 4: Plot ONLY if requested
    if wants_plot and not df.empty:
        print("🎨 **Creating plot as requested...**")
        create_plot(df, query)
    elif wants_plot and df.empty:
        print("⚠️ **User wanted plot but no data available**")
    else:
        print("✅ **SQL executed successfully (no plot requested)**")
    
    print()

def demo_smart_plotting():
    """Demo showing smart plotting behavior"""
    print("🎯 **SMART PLOTTING DEMO**")
    print("🤖 Llama 3.1 generates SQL")
    print("📊 Plots ONLY when user explicitly asks")
    print("=" * 60)
    
    # Ensure we have data
    setup_sample_data()
    
    # Test queries - some request plots, some don't
    test_queries = [
        # No plotting requested
        "What was Tesla's closing price yesterday?",
        "Show me Apple's trading volume",
        "Get Tesla stock data",
        
        # Plotting explicitly requested  
        "Plot Tesla's stock price over time",
        "Create a chart of Apple's volume",
        "Visualize stock performance comparison",
        "Show me a graph of Tesla prices"
    ]
    
    for query in test_queries:
        process_user_query(query)
        input("⏸️ Press Enter for next example...")

def setup_sample_data():
    """Quick setup of sample data"""
    conn = sqlite3.connect("financial_data.db")
    
    # Check if data exists
    try:
        count = pd.read_sql_query("SELECT COUNT(*) as count FROM stock_data", conn).iloc[0]['count']
        if count > 0:
            print(f"✅ Using existing data ({count} records)")
            conn.close()
            return
    except:
        pass
    
    print("📊 Setting up sample data...")
    
    # Create table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS stock_data (
        Date TEXT,
        Close REAL,
        Volume INTEGER,
        ticker TEXT,
        move REAL
    )
    """)
    
    # Add sample data
    import datetime
    import random
    
    data = []
    for i in range(30):
        date = (datetime.datetime.now() - datetime.timedelta(days=29-i)).strftime('%Y-%m-%d')
        for ticker, base_price in [('AAPL', 150), ('TSLA', 250), ('GOOGL', 2800)]:
            price = base_price * (1 + random.uniform(-0.05, 0.05))
            volume = random.randint(800000, 1200000)
            move = random.uniform(-3, 3)
            data.append((date, price, volume, ticker, move))
    
    conn.executemany("INSERT INTO stock_data VALUES (?, ?, ?, ?, ?)", data)
    conn.commit()
    conn.close()
    print("✅ Sample data created!")

def interactive_mode():
    """Interactive mode to test your own queries"""
    print("\n🎮 **INTERACTIVE MODE**")
    print("=" * 30)
    print("Type queries to see the smart plotting behavior!")
    print()
    print("📝 **Examples**:")
    print("  Without plotting: 'Tesla stock price'")
    print("  With plotting: 'Plot Tesla stock price'")
    print("  With plotting: 'Show me a chart of Apple volume'")
    print()
    print("Type 'quit' to exit")
    print()
    
    setup_sample_data()
    
    while True:
        try:
            query = input("🔍 Your query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
                
            if not query:
                continue
            
            process_user_query(query)
            
        except KeyboardInterrupt:
            break
    
    print("👋 Thanks for testing!")

def main():
    print("🚀 **LLAMA 3.1 + SMART PLOTTING**")
    print("🧠 Plots ONLY when you ask for it!")
    print("=" * 50)
    
    print("\nChoose mode:")
    print("1. 🎬 Demo (see examples of smart plotting)")
    print("2. 🎮 Interactive (try your own queries)")
    
    try:
        choice = input("\nEnter choice (1-2): ").strip()
        
        if choice == "1":
            demo_smart_plotting()
        elif choice == "2":
            interactive_mode()
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()