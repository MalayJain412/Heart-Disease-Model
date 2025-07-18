# Model Loading Troubleshooting Guide

## Issue: Model Not Loading Successfully on Render

### 🔍 Problem Identification

The model loading issue on Render can be caused by several factors:

1. **Model file not included in deployment**
2. **Incorrect file path configuration**
3. **Model loading timing issues**
4. **File permissions problems**
5. **Dependency issues**

### 🛠️ Solutions Implemented

#### 1. Fixed Model Loading Timing
**Problem**: Model was loaded only in `if __name__ == '__main__':` block, which doesn't execute with Gunicorn.

**Solution**: Moved model loading to module level in `app.py`:
```python
# Load model immediately after app configuration
print("Loading machine learning model...")
model = load_model()
if model is None:
    print("WARNING: Model could not be loaded. Predictions will not work.")
else:
    print("Model loaded successfully!")
```

#### 2. Enhanced Dockerfile (Updated for New Structure)
**Problem**: Insufficient model file verification during build.

**Solution**: Added comprehensive model verification with updated paths:
```dockerfile
# Verify model file exists and show details
RUN echo "=== Model File Verification ===" && \
    ls -la data/heart_disease_rf_model.pkl && \
    echo "=== File size ===" && \
    du -h data/heart_disease_rf_model.pkl && \
    echo "=== File type ===" && \
    file data/heart_disease_rf_model.pkl

# Test model loading during build (if test file exists)
RUN if [ -f "tests/test_model_loading.py" ]; then \
        echo "=== Testing Model Loading ===" && \
        python tests/test_model_loading.py; \
    else \
        echo "=== Model Loading Test Skipped (test file not found) ===" && \
        echo "Testing basic model loading..." && \
        python -c "import pickle; model = pickle.load(open('data/heart_disease_rf_model.pkl', 'rb')); print('✅ Model loaded successfully!')"; \
    fi
```

#### 3. Created Diagnostic Scripts (Updated Locations)
**Problem**: No way to debug model loading issues in production.

**Solution**: Created `tests/test_model_loading.py` and `startup.py` for diagnostics.

### 🔧 Manual Troubleshooting Steps

#### Step 1: Check Model File in Repository (Updated Path)
```bash
# Verify model file exists in data directory
ls -la data/heart_disease_rf_model.pkl

# Check file size (should be ~2.9MB)
du -h data/heart_disease_rf_model.pkl

# Verify it's a valid pickle file
file data/heart_disease_rf_model.pkl
```

#### Step 2: Test Model Loading Locally (Updated Command)
```bash
# Run the test script locally
python tests/test_model_loading.py

# Expected output:
# ✅ Model loaded successfully! Type: <class 'sklearn.ensemble._forest.RandomForestClassifier'>
```

#### Step 3: Check Render Build Logs
1. Go to your Render dashboard
2. Click on your web service
3. Go to "Logs" tab
4. Look for these messages:
   - `=== Model File Verification ===`
   - `=== Testing Model Loading ===`
   - `✅ Model loaded successfully!`

#### Step 4: Verify Environment Variables (Updated Path)
In Render dashboard → Environment tab, ensure:
```env
MODEL_PATH=data/heart_disease_rf_model.pkl
FLASK_ENV=production
SECRET_KEY=your-secret-key
DATABASE_URL=your-azure-connection-string
```

### 🚨 Common Issues and Fixes

#### Issue 1: "Model file not found" (Updated Path)
**Symptoms**: `FileNotFoundError: Model file not found: data/heart_disease_rf_model.pkl`

**Causes**:
- Model file not committed to Git
- Model file in `.gitignore`
- Wrong file path

**Solutions**:
```bash
# 1. Check if file is tracked by Git
git ls-files | grep heart_disease_rf_model.pkl

# 2. If not tracked, add it
git add data/heart_disease_rf_model.pkl
git commit -m "Add model file to data directory"
git push

# 3. Check .gitignore
cat .dockerignore | grep -i data
# Ensure data/ directory is not excluded
```

#### Issue 2: "Pickle protocol error"
**Symptoms**: `ValueError: unsupported pickle protocol`

**Causes**:
- Model saved with newer Python version
- Incompatible pickle protocol

**Solutions**:
```python
# Re-save model with compatible protocol
import pickle
import joblib

# Load and re-save with protocol 4
with open('data/heart_disease_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('data/heart_disease_rf_model.pkl', 'wb') as f:
    pickle.dump(model, f, protocol=4)
```

#### Issue 3: "scikit-learn version mismatch"
**Symptoms**: `AttributeError` or import errors

**Causes**:
- Different scikit-learn versions between training and deployment

**Solutions**:
```bash
# Check scikit-learn version in requirements.txt
grep scikit-learn requirements.txt

# Update to specific version
pip install scikit-learn==1.4.2
```

#### Issue 4: "Memory issues"
**Symptoms**: Container crashes or timeout

**Causes**:
- Model file too large for free tier
- Insufficient memory allocation

**Solutions**:
1. **Upgrade Render plan** to paid tier
2. **Optimize model size**:
   ```python
   # Use joblib for smaller file size
   import joblib
   joblib.dump(model, 'data/heart_disease_rf_model.joblib')
   ```

### 🔍 Debugging Commands (Updated)

#### Check Render Logs
```bash
# View recent logs
render logs --service your-service-name

# Follow logs in real-time
render logs --service your-service-name --follow
```

#### Test Model Loading in Render Shell (Updated Path)
```bash
# Access Render shell (if available)
render shell --service your-service-name

# Then run:
python tests/test_model_loading.py
```

#### Verify File Structure in Container (Updated Paths)
```bash
# Check what files are in the container
ls -la /
ls -la /app/
ls -la /app/data/
find /app -name "*.pkl"
```

### 📋 Pre-Deployment Checklist (Updated)

#### ✅ Repository Setup
- [ ] Model file exists in `data/heart_disease_rf_model.pkl`
- [ ] Model file is tracked by Git (`git ls-files | grep heart_disease_rf_model.pkl`)
- [ ] Model file is not in `.gitignore` or `.dockerignore`
- [ ] Test script exists in `tests/test_model_loading.py`

#### ✅ Local Testing
- [ ] Model loads locally: `python tests/test_model_loading.py`
- [ ] Docker build succeeds: `docker build -t heart-disease-app .`
- [ ] Docker run works: `docker run -p 5000:5000 heart-disease-app`
- [ ] Startup verification passes: `python startup.py`

#### ✅ Environment Configuration
- [ ] `MODEL_PATH=data/heart_disease_rf_model.pkl` in environment variables
- [ ] `FLASK_ENV=production` set
- [ ] `SECRET_KEY` configured
- [ ] `DATABASE_URL` configured

#### ✅ Render Configuration
- [ ] Environment variables set in Render dashboard
- [ ] Build command: `docker build -t heart-disease-app .`
- [ ] Start command: `gunicorn --bind 0.0.0.0:$PORT app:app`
- [ ] Health check configured

### 🛠️ Advanced Troubleshooting

#### Debug Environment Variables
```bash
# Run environment debugging script
python scripts/debug_env.py

# Check specific environment variable
echo $MODEL_PATH
```

#### Test Configuration Loading
```bash
# Test config loading
python -c "from config import config; print(config['development'].MODEL_PATH)"
```

#### Verify Model File Integrity
```bash
# Check file integrity
python -c "
import pickle
import os
model_path = 'data/heart_disease_rf_model.pkl'
print(f'File exists: {os.path.exists(model_path)}')
print(f'File size: {os.path.getsize(model_path)} bytes')
with open(model_path, 'rb') as f:
    model = pickle.load(f)
print(f'Model type: {type(model)}')
print('✅ Model loaded successfully!')
"
```

### 📊 Model Loading Test Script (Updated)

The `tests/test_model_loading.py` script now includes the new directory structure:

```python
import os
import sys
import pickle
import logging

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config

def test_model_loading():
    """Test if the model can be loaded successfully"""
    try:
        # Get model path from config
        model_path = config['development'].MODEL_PATH
        
        logger.info(f"Testing model loading from: {model_path}")
        
        # Check if file exists
        if not os.path.exists(model_path):
            logger.error(f"❌ Model file not found: {model_path}")
            return False
        
        # Try to load the model
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        
        logger.info("✅ Model loaded successfully!")
        logger.info(f"Model type: {type(model)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return False
```

### 🎯 Success Indicators

When model loading is working correctly, you should see:

1. **Local Testing**:
   ```
   ✅ Model loaded successfully!
   Model type: <class 'sklearn.ensemble._forest.RandomForestClassifier'>
   ```

2. **Docker Build**:
   ```
   === Model File Verification ===
   -rw-r--r-- 1 root root 3048576 Jan 1 12:00 data/heart_disease_rf_model.pkl
   === File size ===
   2.9M    data/heart_disease_rf_model.pkl
   === Testing Model Loading ===
   ✅ Model loaded successfully!
   ```

3. **Application Startup**:
   ```
   Loading machine learning model...
   Model loaded successfully from data/heart_disease_rf_model.pkl
   Model loaded successfully!
   ```

4. **Render Deployment**:
   - Build completes successfully
   - Application starts without errors
   - Health checks pass
   - Predictions work in the web interface

This troubleshooting guide has been updated to reflect the new organized directory structure with the model file located in the `data/` directory and test scripts in the `tests/` directory. 