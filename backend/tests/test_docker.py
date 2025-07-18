#!/usr/bin/env python3
"""
Test script to verify Docker setup and dependencies
"""

import os
import sys

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import flask
        print("✓ Flask imported successfully")
    except ImportError as e:
        print(f"✗ Flask import failed: {e}")
        return False
    
    try:
        import pandas
        print("✓ Pandas imported successfully")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        import numpy
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import sklearn
        print("✓ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"✗ Scikit-learn import failed: {e}")
        return False
    
    try:
        import matplotlib
        print("✓ Matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    try:
        import seaborn
        print("✓ Seaborn imported successfully")
    except ImportError as e:
        print(f"✗ Seaborn import failed: {e}")
        return False
    
    try:
        import openpyxl
        print("✓ OpenPyXL imported successfully")
    except ImportError as e:
        print(f"✗ OpenPyXL import failed: {e}")
        return False
    
    try:
        import reportlab
        print("✓ ReportLab imported successfully")
    except ImportError as e:
        print(f"✗ ReportLab import failed: {e}")
        return False
    
    try:
        import mysqlclient
        print("✓ MySQL client imported successfully")
    except ImportError as e:
        print(f"✗ MySQL client import failed: {e}")
        return False
    
    try:
        import pymysql
        print("✓ PyMySQL imported successfully")
    except ImportError as e:
        print(f"✗ PyMySQL import failed: {e}")
        return False
    
    try:
        import pyodbc
        print("✓ PyODBC imported successfully")
    except ImportError as e:
        print(f"✗ PyODBC import failed: {e}")
        return False
    
    return True

def test_model_file():
    """Test if the model file exists"""
    print("\nTesting model file...")
    
    model_path = "heart_disease_rf_model.pkl"
    if os.path.exists(model_path):
        print(f"✓ Model file found: {model_path}")
        print(f"  Size: {os.path.getsize(model_path)} bytes")
        return True
    else:
        print(f"✗ Model file not found: {model_path}")
        return False

def test_model_loading():
    """Test if the model can be loaded"""
    print("\nTesting model loading...")
    
    try:
        import pickle
        model_path = "heart_disease_rf_model.pkl"
        
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        
        print("✓ Model loaded successfully")
        print(f"  Model type: {type(model)}")
        return True
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        from config import config
        print("✓ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Docker Setup Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_model_file,
        test_model_loading,
        test_config
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! Docker setup is ready.")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 