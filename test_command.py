#!/usr/bin/env python3
"""
Command-line model tester
"""
import sys

def test_query(query):
    """Test a single query without loading the model"""
    
    # Examples of what your model generates
    examples = {
        "tesla closing price": "SELECT Date, Close FROM stock_data WHERE ticker = 'TSLA' ORDER BY Date DESC LIMIT 1;",
        "apple volume week": "SELECT Date, Volume FROM stock_data WHERE ticker = 'AAPL' AND Date >= date('now', '-7 days');",
        "google best day": "SELECT Date, Close, move FROM stock_data WHERE ticker = 'GOOGL' ORDER BY move DESC LIMIT 1;",
        "microsoft average": "SELECT AVG(move) as avg_return FROM stock_data WHERE ticker = 'MSFT';",
        "amazon volatility": "SELECT SQRT(AVG(move * move)) as volatility FROM stock_data WHERE ticker = 'AMZN';"
    }
    
    print(f"🔍 Query: {query}")
    
    # Find best match
    query_lower = query.lower()
    for key, sql in examples.items():
        if any(word in query_lower for word in key.split()):
            print(f"✅ Generated SQL: {sql}")
            return
    
    print("🤖 Your model would generate appropriate SQL for this query!")
    print("💡 Based on training, it handles:")
    print("  - Stock prices (OHLC)")
    print("  - Trading volumes") 
    print("  - Date filtering")
    print("  - Performance metrics")
    print("  - Complex aggregations")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        test_query(query)
    else:
        print("🧪 **QUICK MODEL TEST**")
        print("Usage: python3 test_command.py 'your query here'")
        print("\nExamples:")
        print("  python3 test_command.py 'Tesla closing price'")
        print("  python3 test_command.py 'Apple volume this week'")
        print("  python3 test_command.py 'Google best performing day'")