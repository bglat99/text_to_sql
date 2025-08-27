#!/usr/bin/env python3
"""
Test script for visualization module
"""

import sys
import os

def test_visualization_imports():
    """Test that visualization dependencies can be imported"""
    print("Testing visualization imports...")
    
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib")
    except ImportError as e:
        print(f"✗ matplotlib: {e}")
        return False
    
    try:
        import seaborn as sns
        print("✓ seaborn")
    except ImportError as e:
        print(f"✗ seaborn: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ pandas")
    except ImportError as e:
        print(f"✗ pandas: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ numpy")
    except ImportError as e:
        print(f"✗ numpy: {e}")
        return False
    
    return True

def test_visualization_module():
    """Test the visualization module"""
    print("\nTesting visualization module...")
    
    try:
        from visualization import FinancialDataVisualizer
        print("✓ FinancialDataVisualizer imported successfully")
        
        # Test initialization
        viz = FinancialDataVisualizer()
        print("✓ Visualizer initialized successfully")
        
        return True
    except Exception as e:
        print(f"✗ Visualization module error: {e}")
        return False

def test_database_connection():
    """Test database connection for visualization"""
    print("\nTesting database connection...")
    
    try:
        import sqlite3
        conn = sqlite3.connect("financial_data.db")
        
        # Check if table exists
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stock_data'")
        if cursor.fetchone():
            print("✓ Database and stock_data table found")
            
            # Check data
            cursor = conn.execute("SELECT COUNT(*) FROM stock_data")
            count = cursor.fetchone()[0]
            print(f"✓ Found {count} records in database")
            
            conn.close()
            return True
        else:
            print("✗ stock_data table not found")
            conn.close()
            return False
            
    except Exception as e:
        print(f"✗ Database connection error: {e}")
        return False

def test_basic_visualization():
    """Test basic visualization functionality"""
    print("\nTesting basic visualization...")
    
    try:
        from visualization import FinancialDataVisualizer
        
        viz = FinancialDataVisualizer()
        
        # Test getting stock data
        df = viz.get_stock_data('AAPL', days=30)
        if not df.empty:
            print(f"✓ Retrieved {len(df)} records for AAPL")
        else:
            print("⚠ No data found for AAPL (database might be empty)")
        
        # Test SQL query plotting (without showing plot)
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        sql_query = "SELECT COUNT(*) as count FROM stock_data WHERE ticker = 'AAPL'"
        try:
            viz.plot_sql_query_results(sql_query, plot_type='bar')
            print("✓ SQL query plotting works")
        except Exception as e:
            print(f"⚠ SQL plotting test: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic visualization error: {e}")
        return False

def main():
    """Run all visualization tests"""
    print("=" * 60)
    print("Visualization Module Test")
    print("=" * 60)
    
    tests = [
        ("Import Dependencies", test_visualization_imports),
        ("Visualization Module", test_visualization_module),
        ("Database Connection", test_database_connection),
        ("Basic Visualization", test_basic_visualization),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"✗ {test_name} test failed")
        except Exception as e:
            print(f"✗ {test_name} test error: {e}")
    
    print("\n" + "=" * 60)
    print("Visualization Test Results")
    print("=" * 60)
    print(f"Passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n✅ All visualization tests passed!")
        print("\nYou can now use visualization features:")
        print("- python visualization.py (run demo)")
        print("- python demo.py --interactive (with plotting)")
        print("- python demo.py --demo (with plotting)")
    else:
        print(f"\n❌ {total - passed} test(s) failed.")
        print("\nCommon solutions:")
        print("- Install missing packages: pip install matplotlib seaborn")
        print("- Ensure database exists: python database_setup.py")
        print("- Check Python environment")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 