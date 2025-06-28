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

#### 2. Enhanced Dockerfile
**Problem**: Insufficient model file verification during build.

**Solution**: Added comprehensive model verification:
```dockerfile
# Verify model file exists and show details
RUN echo "=== Model File Verification ===" && \
    ls -la heart_disease_rf_model.pkl && \
    echo "=== File size ===" && \
    du -h heart_disease_rf_model.pkl && \
    echo "=== File type ===" && \
    file heart_disease_rf_model.pkl

# Test model loading during build
RUN echo "=== Testing Model Loading ===" && \
    python test_model_loading.py
```

#### 3. Created Diagnostic Scripts
**Problem**: No way to debug model loading issues in production.

**Solution**: Created `test_model_loading.py` and `startup.py` for diagnostics.

### 🔧 Manual Troubleshooting Steps

#### Step 1: Check Model File in Repository
```bash
# Verify model file exists in your repository
ls -la heart_disease_rf_model.pkl

# Check file size (should be ~2.9MB)
du -h heart_disease_rf_model.pkl

# Verify it's a valid pickle file
file heart_disease_rf_model.pkl
```

#### Step 2: Test Model Loading Locally
```bash
# Run the test script locally
python test_model_loading.py

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

#### Step 4: Verify Environment Variables
In Render dashboard → Environment tab, ensure:
```env
MODEL_PATH=heart_disease_rf_model.pkl
FLASK_ENV=production
SECRET_KEY=your-secret-key
DATABASE_URL=your-azure-connection-string
```

### 🚨 Common Issues and Fixes

#### Issue 1: "Model file not found"
**Symptoms**: `FileNotFoundError: Model file not found: heart_disease_rf_model.pkl`

**Causes**:
- Model file not committed to Git
- Model file in `.gitignore`
- Wrong file path

**Solutions**:
```bash
# 1. Check if file is tracked by Git
git ls-files | grep heart_disease_rf_model.pkl

# 2. If not tracked, add it
git add heart_disease_rf_model.pkl
git commit -m "Add model file"
git push

# 3. Check .gitignore
cat .gitignore | grep -i model
# Remove any lines that exclude the model file
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
with open('heart_disease_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('heart_disease_rf_model.pkl', 'wb') as f:
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
   joblib.dump(model, 'heart_disease_rf_model.joblib')
   ```

### 🔍 Debugging Commands

#### Check Render Logs
```bash
# View recent logs
render logs --service your-service-name

# Follow logs in real-time
render logs --service your-service-name --follow
```

#### Test Model Loading in Render Shell
```bash
# Access Render shell (if available)
render shell --service your-service-name

# Then run:
python test_model_loading.py
```

#### Verify File Structure in Container
```bash
# Check what files are in the container
ls -la /
ls -la /app/
find /app -name "*.pkl"
```

### 📋 Pre-Deployment Checklist

Before deploying to Render, ensure:

- [ ] Model file is committed to Git repository
- [ ] Model file is not in `.gitignore`
- [ ] `requirements.txt` includes correct scikit-learn version
- [ ] `MODEL_PATH` environment variable is set in Render
- [ ] Model file size is reasonable (< 10MB for free tier)
- [ ] Model can be loaded locally with `python test_model_loading.py`

### 🚀 Deployment Verification

After deployment, verify:

1. **Build Success**: Check Render build logs for model verification messages
2. **Startup Success**: Look for "Model loaded successfully!" in logs
3. **Functionality**: Test prediction feature in the web app
4. **Performance**: Monitor memory usage and response times

### 📞 Getting Help

If issues persist:

1. **Check Render Status**: [status.render.com](https://status.render.com)
2. **Review Logs**: Look for specific error messages
3. **Test Locally**: Ensure everything works locally first
4. **Contact Support**: Use Render support if needed

### 🔄 Quick Fix Commands

```bash
# 1. Force rebuild on Render
# Go to Render dashboard → Manual Deploy → Clear build cache & deploy

# 2. Update environment variables
# Set MODEL_PATH=heart_disease_rf_model.pkl in Render dashboard

# 3. Check file in repository
git add heart_disease_rf_model.pkl
git commit -m "Ensure model file is included"
git push

# 4. Test locally
python test_model_loading.py
python startup.py
```

---

**Note**: This troubleshooting guide should help resolve most model loading issues on Render. If problems persist, check the specific error messages in your Render logs for more targeted solutions. 