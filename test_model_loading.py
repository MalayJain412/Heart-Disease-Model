#!/usr/bin/env python3
"""
Model Loading Test Script
This script helps debug model loading issues on Render deployment.
"""

import os
import sys
import pickle
from pathlib import Path

def test_model_loading():
    """Test model loading with detailed error reporting"""
    print("🔍 Model Loading Test")
    print("=" * 50)
    
    # Check current working directory
    print(f"Current working directory: {os.getcwd()}")
    
    # List all files in current directory
    print("\n📁 Files in current directory:")
    for file in os.listdir('.'):
        if os.path.isfile(file):
            size = os.path.getsize(file)
            print(f"  - {file} ({size} bytes)")
    
    # Check for model file
    model_paths = [
        'heart_disease_rf_model.pkl',
        './heart_disease_rf_model.pkl',
        '/app/heart_disease_rf_model.pkl',
        'mount/src/heart_disease_rf_model.pkl'
    ]
    
    print("\n🔍 Checking model file locations:")
    for path in model_paths:
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        print(f"  - {path}: {'✅ EXISTS' if exists else '❌ NOT FOUND'} ({size} bytes)")
    
    # Try to load model from each location
    print("\n🧠 Testing model loading:")
    for path in model_paths:
        if os.path.exists(path):
            try:
                print(f"  Trying to load from: {path}")
                with open(path, 'rb') as file:
                    model = pickle.load(file)
                print(f"  ✅ SUCCESS: Model loaded from {path}")
                print(f"     Model type: {type(model)}")
                print(f"     Model attributes: {dir(model)}")
                return True
            except Exception as e:
                print(f"  ❌ FAILED: {e}")
    
    # Check environment variables
    print("\n🌍 Environment variables:")
    model_path_env = os.environ.get('MODEL_PATH')
    print(f"  MODEL_PATH: {model_path_env}")
    
    if model_path_env and os.path.exists(model_path_env):
        try:
            print(f"  Trying to load from MODEL_PATH: {model_path_env}")
            with open(model_path_env, 'rb') as file:
                model = pickle.load(file)
            print(f"  ✅ SUCCESS: Model loaded from MODEL_PATH")
            return True
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
    
    print("\n❌ No model could be loaded from any location")
    return False

def test_scikit_learn():
    """Test scikit-learn installation"""
    print("\n🔬 Testing scikit-learn installation:")
    try:
        import sklearn
        print(f"  ✅ scikit-learn version: {sklearn.__version__}")
        
        from sklearn.ensemble import RandomForestClassifier
        print("  ✅ RandomForestClassifier imported successfully")
        
        # Test creating a simple model
        model = RandomForestClassifier()
        print("  ✅ RandomForestClassifier created successfully")
        
        return True
    except Exception as e:
        print(f"  ❌ scikit-learn test failed: {e}")
        return False

def test_pickle():
    """Test pickle functionality"""
    print("\n🥒 Testing pickle functionality:")
    try:
        import pickle
        print("  ✅ pickle imported successfully")
        
        # Test basic pickle operations
        test_data = {'test': 'data'}
        pickled = pickle.dumps(test_data)
        unpickled = pickle.loads(pickled)
        
        if test_data == unpickled:
            print("  ✅ pickle operations work correctly")
            return True
        else:
            print("  ❌ pickle operations failed")
            return False
    except Exception as e:
        print(f"  ❌ pickle test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Heart Disease Model Loading Test")
    print("=" * 60)
    
    # Test basic dependencies
    sklearn_ok = test_scikit_learn()
    pickle_ok = test_pickle()
    
    # Test model loading
    model_ok = test_model_loading()
    
    # Summary
    print("\n📊 Test Summary:")
    print("=" * 30)
    print(f"  scikit-learn: {'✅ OK' if sklearn_ok else '❌ FAILED'}")
    print(f"  pickle: {'✅ OK' if pickle_ok else '❌ FAILED'}")
    print(f"  model loading: {'✅ OK' if model_ok else '❌ FAILED'}")
    
    if model_ok:
        print("\n🎉 All tests passed! Model should work correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 