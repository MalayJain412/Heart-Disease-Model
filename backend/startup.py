#!/usr/bin/env python3
"""
Startup Script for Heart Disease Prediction System
This script ensures proper initialization before starting the Flask app.
"""

import os
import sys
import time
import pickle
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_model():
    """Verify that the model file exists and can be loaded"""
    print("🔍 Verifying model file...")
    
    # Check environment variable
    model_path = os.environ.get('MODEL_PATH', 'data/heart_disease_rf_model.pkl')
    print(f"Model path from environment: {model_path}")
    
    # Check if file exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at: {model_path}")
        
        # Try alternative paths
        alternative_paths = [
            'heart_disease_rf_model.pkl',
            './heart_disease_rf_model.pkl',
            '/app/heart_disease_rf_model.pkl',
            'mount/src/heart_disease_rf_model.pkl'
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                print(f"✅ Found model at alternative path: {alt_path}")
                # Update environment variable
                os.environ['MODEL_PATH'] = alt_path
                model_path = alt_path
                break
        else:
            print("❌ Model file not found in any location")
            return False
    
    # Try to load the model
    try:
        print(f"🧠 Loading model from: {model_path}")
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        print(f"✅ Model loaded successfully! Type: {type(model)}")
        return True
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

def verify_dependencies():
    """Verify that all required dependencies are available"""
    print("🔬 Verifying dependencies...")
    
    required_packages = [
        'flask',
        'flask_sqlalchemy', 
        'flask_login',
        'werkzeug',
        'pandas',
        'numpy',
        'sklearn',
        'openpyxl',
        'reportlab',
        'matplotlib',
        'seaborn',
        'dotenv',
        'pyodbc',
        'gunicorn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {missing_packages}")
        return False
    
    print("✅ All dependencies verified")
    return True

def verify_environment():
    """Verify environment variables"""
    print("🌍 Verifying environment variables...")
    
    required_vars = [
        'SECRET_KEY',
        'DATABASE_URL',
        'FLASK_ENV'
    ]
    
    missing_vars = []
    
    for var in required_vars:
        value = os.environ.get(var)
        if value:
            print(f"  ✅ {var}: {'*' * len(value)} (hidden)")
        else:
            print(f"  ❌ {var}: NOT SET")
            missing_vars.append(var)
    
    # Check optional variables
    optional_vars = ['MODEL_PATH', 'SESSION_COOKIE_SECURE', 'LOG_LEVEL']
    for var in optional_vars:
        value = os.environ.get(var)
        if value:
            print(f"  ✅ {var}: {value}")
        else:
            print(f"  ⚠️  {var}: NOT SET (optional)")
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        return False
    
    print("✅ Environment variables verified")
    return True

def check_database_connection():
    """Check database connection"""
    try:
        database_url = os.environ.get('DATABASE_URL')
        if not database_url:
            logger.error("DATABASE_URL environment variable not set")
            return False
        
        logger.info("Testing database connection...")
        engine = create_engine(database_url)
        
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            logger.info("✅ Database connection successful")
            return True
            
    except SQLAlchemyError as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error during database connection: {e}")
        return False

def check_model_file():
    """Check if model file exists and can be loaded"""
    try:
        # Updated model path to data directory
        model_path = os.environ.get('MODEL_PATH', 'data/heart_disease_rf_model.pkl')
        
        if not os.path.exists(model_path):
            logger.error(f"❌ Model file not found: {model_path}")
            return False
        
        logger.info(f"Testing model loading from {model_path}...")
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        
        logger.info("✅ Model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return False

def main():
    """Main startup verification"""
    print("🚀 Heart Disease Prediction System - Startup Verification")
    print("=" * 60)
    
    # Run all verifications
    deps_ok = verify_dependencies()
    env_ok = verify_environment()
    model_ok = verify_model()
    
    print("\n📊 Verification Summary:")
    print("=" * 30)
    print(f"  Dependencies: {'✅ OK' if deps_ok else '⚠️  WARNING'}")
    print(f"  Environment: {'✅ OK' if env_ok else '❌ FAILED'}")
    print(f"  Model: {'✅ OK' if model_ok else '❌ FAILED'}")
    
    # Only fail if critical components are missing
    if env_ok and model_ok:
        print("\n🎉 Critical verifications passed! Starting application...")
        return True
    else:
        print("\n❌ Critical verifications failed. Cannot start application.")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 