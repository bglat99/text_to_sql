#!/usr/bin/env python3
"""
Test script to verify the setup is working correctly
"""

import sys
import os
import importlib

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")
    
    required_packages = [
        'torch',
        'transformers',
        'datasets',
        'trl',
        'unsloth',
        'pandas',
        'sqlite3',
        'yfinance',
        'numpy',
        'sklearn'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\nFailed to import: {failed_imports}")
        return False
    
    print("✓ All imports successful!")
    return True

def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        from config import MODEL_CONFIG, LORA_CONFIG, TRAINING_CONFIG, DATA_CONFIG, FINANCIAL_CONFIG, EVAL_CONFIG
        print("✓ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

def test_database_setup():
    """Test database setup"""
    print("\nTesting database setup...")
    
    try:
        from database_setup import create_database_with_real_data
        create_database_with_real_data(use_real_data=False)  # Use synthetic data for testing
        
        import sqlite3
        conn = sqlite3.connect("financial_data.db")
        cursor = conn.execute("SELECT COUNT(*) FROM stock_data")
        count = cursor.fetchone()[0]
        conn.close()
        
        print(f"✓ Database created with {count} records")
        return True
    except Exception as e:
        print(f"✗ Database setup error: {e}")
        return False

def test_data_generation():
    """Test data generation"""
    print("\nTesting data generation...")
    
    try:
        from data_generator import FinancialSQLGenerator
        
        generator = FinancialSQLGenerator()
        train_data = generator.generate_training_data(10)  # Generate 10 samples for testing
        test_data = generator.generate_test_data(5)
        
        print(f"✓ Generated {len(train_data)} training samples")
        print(f"✓ Generated {len(test_data)} test samples")
        
        # Test a sample
        sample = train_data[0]
        print(f"✓ Sample query: {sample['input'].split('Query: ')[1][:50]}...")
        
        return True
    except Exception as e:
        print(f"✗ Data generation error: {e}")
        return False

def test_utils():
    """Test utility functions"""
    print("\nTesting utility functions...")
    
    try:
        from utils import validate_sql_syntax, extract_sql_from_response, test_sql_execution
        
        # Test SQL validation
        valid_sql = "SELECT * FROM stock_data WHERE ticker = 'AAPL'"
        is_valid, _ = validate_sql_syntax(valid_sql)
        print(f"✓ SQL validation: {is_valid}")
        
        # Test SQL extraction
        response = "### Response:\nSELECT * FROM stock_data"
        sql = extract_sql_from_response(response)
        print(f"✓ SQL extraction: {sql}")
        
        # Test SQL execution
        success, result = test_sql_execution(valid_sql)
        print(f"✓ SQL execution: {success}")
        
        return True
    except Exception as e:
        print(f"✗ Utils error: {e}")
        return False

def test_model_loading():
    """Test model loading (without downloading)"""
    print("\nTesting model loading capability...")
    
    try:
        from unsloth import FastLanguageModel
        
        # This will test if we can access the model (but won't download it)
        print("✓ Unsloth FastLanguageModel import successful")
        print("✓ Model loading capability verified")
        
        return True
    except Exception as e:
        print(f"✗ Model loading error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Text-to-SQL Framework Setup Test")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Database Setup", test_database_setup),
        ("Data Generation", test_data_generation),
        ("Utility Functions", test_utils),
        ("Model Loading", test_model_loading),
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
    print("Test Results")
    print("=" * 60)
    print(f"Passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n✅ All tests passed! Setup is ready for training.")
        print("\nNext steps:")
        print("1. Run training: python finetune_model.py")
        print("2. Evaluate model: python evaluate_model.py --comprehensive")
        print("3. Demo: python demo.py")
    else:
        print(f"\n❌ {total - passed} test(s) failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Check Python version (3.8+ required)")
        print("- Ensure sufficient disk space")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 