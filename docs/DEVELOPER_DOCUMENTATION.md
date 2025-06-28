# Heart Disease Prediction System - Developer Documentation

## 🏗️ Architecture Overview

The Heart Disease Prediction System is built using a modern Flask-based architecture with role-based authentication, Azure SQL Database integration, and machine learning capabilities.

**Current Status**: ✅ **Successfully Deployed and Working**

## 🛠️ Technology Stack

### Backend Framework
- **Flask 2.3.3**: Web framework
- **Flask-SQLAlchemy 3.0.5**: Database ORM
- **Flask-Login 0.6.3**: Authentication management
- **Werkzeug 2.3.7**: Security utilities

### Database
- **Azure SQL Database**: Production database
- **SQLite**: Development database
- **pyodbc**: ODBC driver for Azure SQL
- **SQLAlchemy**: Database abstraction layer

### Machine Learning
- **scikit-learn 1.4.2**: ML framework
- **pandas 2.2.1**: Data manipulation
- **numpy 1.26.4**: Numerical computing
- **Random Forest**: Classification algorithm

### Frontend
- **Bootstrap 5**: CSS framework
- **Chart.js**: Data visualization
- **JavaScript**: Client-side functionality
- **HTML5/CSS3**: Markup and styling

### Production
- **Docker**: Containerization
- **Gunicorn**: WSGI server
- **Render**: Cloud deployment platform
- **Azure App Service**: Alternative deployment

## 📁 Project Structure

```
Heart-Disease-Model/
├── 📁 Production Files (Root Directory)
│   ├── app.py                          # Main Flask application
│   ├── config.py                       # Configuration management
│   ├── requirements.txt                # Python dependencies
│   ├── startup.py                      # Application startup script
│   ├── env.example                     # Environment variables template
│   ├── env.production                  # Production environment config
│   ├── .dockerignore                   # Docker ignore patterns
│   ├── Dockerfile                      # Production Docker configuration
│   ├── docker-compose.yml              # Docker Compose configuration
│   ├── render.yaml                     # Render deployment config
│   └── README.md                       # Main project documentation
├── 📁 docs/                            # Documentation Directory
│   ├── DEVELOPER_DOCUMENTATION.md      # This file
│   ├── DEPLOYMENT.md                   # Deployment guide
│   ├── USER_DOCUMENTATION.md           # End-user guide
│   ├── PROJECT_STRUCTURE.md            # Detailed project structure
│   ├── MODEL_LOADING_TROUBLESHOOTING.md # Model troubleshooting guide
│   ├── RENDER_DEPLOYMENT.md            # Render-specific deployment
│   └── PROJECT_DOCUMENTATION.md        # Comprehensive project docs
├── 📁 scripts/                         # Setup and Utility Scripts
│   ├── add_admin_user.py               # Add default admin user
│   ├── setup_azure_database.py         # Azure SQL setup script
│   ├── debug_env.py                    # Environment debugging tool
│   ├── update_env.py                   # Environment file updater
│   ├── fix_env.py                      # Environment file fixer
│   ├── streamlit_app.py                # Alternative Streamlit interface
│   ├── coopy_utils.py                  # Utility functions
│   └── all_code_snippets.txt           # Code documentation
├── 📁 tests/                           # Testing Files
│   ├── test_model_loading.py           # Model loading tests
│   ├── test_azure_connection.py        # Azure connection testing
│   ├── test_admin_credentials.py       # Admin user testing
│   ├── test_database_comprehensive.py  # Comprehensive DB tests
│   ├── test_azure_db.py                # Azure database testing
│   └── test_docker.py                  # Docker testing
├── 📁 data/                            # Data Files
│   ├── heart_disease_rf_model.pkl      # Pre-trained ML model
│   └── Heart D Dataset.csv             # Training dataset
├── 📁 app/                             # Flask Application Structure
│   ├── 📁 templates/                   # HTML templates
│   │   ├── base.html                   # Base template
│   │   ├── 📁 auth/                    # Authentication templates
│   │   ├── 📁 doctor/                  # Doctor role templates
│   │   ├── 📁 dean/                    # Dean role templates
│   │   └── 📁 chairman/                # Chairman role templates
│   └── 📁 static/                      # Static assets
│       ├── 📁 css/                     # Stylesheets
│       └── 📁 js/                      # JavaScript files
├── 📁 instance/                        # Database Instance
│   └── heart_disease_app.db            # Local SQLite database
└── 📁 mount/                           # Docker Mount Directory
    └── src/
        └── heart_disease_rf_model.pkl  # Model for Docker mounting
```

## 🔧 Core Components

### 1. Flask Application (`app.py`)

#### Application Initialization
```python
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import os
from config import config

app = Flask(__name__)
app.config.from_object(config['development'])
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
```

#### Model Loading (Updated Implementation)
```python
# Load the pre-trained model
def load_model():
    try:
        model_path = app.config['MODEL_PATH']  # Now points to data/heart_disease_rf_model.pkl
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Initialize model (will be loaded when app starts)
model = None

# Load model immediately after app configuration
print("Loading machine learning model...")
model = load_model()
if model is None:
    print("WARNING: Model could not be loaded. Predictions will not work.")
else:
    print("Model loaded successfully!")
```

#### Database Models
```python
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # doctor, dean, chairman
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    patients = db.relationship('Patient', backref='doctor', lazy=True)

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    # ... 30+ clinical parameters
    prediction_result = db.Column(db.String(20), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
```

### 2. Configuration Management (`config.py`)

#### Environment-Based Configuration (Updated)
```python
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Database configuration - use DATABASE_URL from environment
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    
    # Model path - updated to data directory
    MODEL_PATH = os.environ.get('MODEL_PATH') or 'data/heart_disease_rf_model.pkl'
    
    # Flask configuration
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Security settings
    SESSION_COOKIE_SECURE = os.environ.get('SESSION_COOKIE_SECURE', 'False').lower() == 'true'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # File upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = 'uploads'
    
    # Logging configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

class DevelopmentConfig(Config):
    DEBUG = True
    
    # Azure SQL specific settings for development
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
        'pool_size': 5,
        'max_overflow': 10,
        'connect_args': {
            'timeout': 30,
            'autocommit': True
        }
    }

class ProductionConfig(Config):
    DEBUG = False
    SESSION_COOKIE_SECURE = True
    
    # Azure SQL specific settings for production
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
        'pool_size': 10,
        'max_overflow': 20,
        'connect_args': {
            'timeout': 30,
            'autocommit': True
        }
    }

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
```

### 3. Startup Verification (`startup.py`)

#### Application Startup Script
```python
import os
import sys
import time
import pickle
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    logger.info("🚀 Starting Heart Disease Prediction System...")
    
    # Check environment variables
    if not check_environment_variables():
        logger.error("❌ Environment check failed")
        sys.exit(1)
    
    # Check database connection
    if not check_database_connection():
        logger.error("❌ Database check failed")
        sys.exit(1)
    
    # Check model file
    if not check_model_file():
        logger.error("❌ Model check failed")
        sys.exit(1)
    
    logger.info("✅ All startup checks passed!")
    logger.info("🎉 Application is ready to start!")

if __name__ == "__main__":
    main()
```

### 4. Machine Learning Integration

#### Prediction Function
```python
def predict_heart_disease(patient_data):
    if model is None:
        return "Model not loaded", 0.0
    
    try:
        # Prepare features for prediction
        features = prepare_features(patient_data)
        
        # Make prediction
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0]
        
        result = "Heart Disease Risk" if prediction == 1 else "Normal"
        confidence = max(probability) * 100
        
        return result, confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error", 0.0
```

#### Feature Preparation
```python
def prepare_features(patient_data):
    features = []
    feature_order = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]
    
    for feature in feature_order:
        value = patient_data.get(feature, 0)
        features.append(float(value))
    
    return features
```

## 🔐 Authentication & Authorization

### User Authentication
```python
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('auth/login.html')
```

## 🧪 Testing Framework

### Test Organization
The project includes a comprehensive testing suite organized in the `tests/` directory:

#### Model Testing (`tests/test_model_loading.py`)
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
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return False
```

#### Database Testing (`tests/test_database_comprehensive.py`)
```python
# Comprehensive database functionality testing
# Tests CRUD operations, user management, patient data
# Usage: python tests/test_database_comprehensive.py
```

#### Azure Connection Testing (`tests/test_azure_connection.py`)
```python
# Azure SQL Database connection testing
# Tests connection strings, authentication, queries
# Usage: python tests/test_azure_connection.py
```

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test
python tests/test_model_loading.py
python tests/test_azure_connection.py
python tests/test_database_comprehensive.py
python tests/test_docker.py
```

## 🔧 Development Setup

### Local Development
```bash
# Clone repository
git clone <repository-url>
cd Heart-Disease-Model

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp env.example .env
# Edit .env with your configuration

# Set up database
python scripts/setup_azure_database.py

# Add admin user
python scripts/add_admin_user.py

# Run tests
python tests/test_model_loading.py
python tests/test_azure_connection.py

# Start development server
python app.py
```

### Docker Development
```bash
# Build development image
docker build -f Dockerfile.dev -t heart-disease-dev .

# Run with Docker Compose
docker-compose up --build

# Test Docker build
python tests/test_docker.py
```

## 🚀 Production Deployment

### Docker Production Build
```bash
# Build production image
docker build -t heart-disease-app .

# Run production container
docker run -p 5000:5000 -e DATABASE_URL="your-connection-string" heart-disease-app
```

### Environment Variables
```bash
# Required for production
FLASK_ENV=production
SECRET_KEY=your-super-secret-production-key
DATABASE_URL=mssql+pyodbc://username:password@server.database.windows.net:1433/heart_disease_db?driver=ODBC+Driver+18+for+SQL+Server
MODEL_PATH=data/heart_disease_rf_model.pkl
SESSION_COOKIE_SECURE=true
LOG_LEVEL=INFO
```

## 📊 Database Schema

### User Table
```sql
CREATE TABLE user (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(80) UNIQUE NOT NULL,
    email VARCHAR(120) UNIQUE NOT NULL,
    password_hash VARCHAR(120) NOT NULL,
    role VARCHAR(20) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### Patient Table
```sql
CREATE TABLE patient (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(100) NOT NULL,
    gender VARCHAR(10) NOT NULL,
    age INTEGER NOT NULL,
    chest_pain BOOLEAN DEFAULT FALSE,
    shortness_of_breath BOOLEAN DEFAULT FALSE,
    fatigue BOOLEAN DEFAULT FALSE,
    systolic INTEGER NOT NULL,
    diastolic INTEGER NOT NULL,
    heart_rate INTEGER NOT NULL,
    lung_sounds BOOLEAN DEFAULT FALSE,
    cholesterol INTEGER NOT NULL,
    ldl INTEGER NOT NULL,
    hdl INTEGER NOT NULL,
    diabetes BOOLEAN DEFAULT FALSE,
    atrial_fibrillation BOOLEAN DEFAULT FALSE,
    rheumatic_fever BOOLEAN DEFAULT FALSE,
    mitral_stenosis BOOLEAN DEFAULT FALSE,
    aortic_stenosis BOOLEAN DEFAULT FALSE,
    tricuspid_stenosis BOOLEAN DEFAULT FALSE,
    pulmonary_stenosis BOOLEAN DEFAULT FALSE,
    dilated_cardiomyopathy BOOLEAN DEFAULT FALSE,
    hypertrophic_cardiomyopathy BOOLEAN DEFAULT FALSE,
    drug_use BOOLEAN DEFAULT FALSE,
    fever BOOLEAN DEFAULT FALSE,
    chills BOOLEAN DEFAULT FALSE,
    alcoholism BOOLEAN DEFAULT FALSE,
    hypertension BOOLEAN DEFAULT FALSE,
    fainting BOOLEAN DEFAULT FALSE,
    dizziness BOOLEAN DEFAULT FALSE,
    smoking BOOLEAN DEFAULT FALSE,
    obesity BOOLEAN DEFAULT FALSE,
    murmur BOOLEAN DEFAULT FALSE,
    prediction_result VARCHAR(100) NOT NULL,
    doctor_id INTEGER NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doctor_id) REFERENCES user (id)
);
```

## 🔍 Debugging and Troubleshooting

### Environment Debugging
```bash
# Debug environment variables
python scripts/debug_env.py

# Fix environment file issues
python scripts/fix_env.py

# Update environment file
python scripts/update_env.py
```

### Model Loading Issues
```bash
# Test model loading
python tests/test_model_loading.py

# Check model file
ls -la data/heart_disease_rf_model.pkl
```

### Database Issues
```bash
# Test database connection
python tests/test_azure_connection.py

# Test comprehensive database functionality
python tests/test_database_comprehensive.py
```

### Docker Issues
```bash
# Test Docker build
python tests/test_docker.py

# Check Docker logs
docker logs <container-id>
```

## 📈 Performance Optimization

### Database Optimization
- Connection pooling configured
- Query optimization with SQLAlchemy
- Indexed foreign keys
- Efficient pagination

### Application Optimization
- Model loaded once at startup
- Cached predictions
- Optimized template rendering
- Static file compression

### Production Optimization
- Gunicorn with multiple workers
- Docker container optimization
- Environment-specific configurations
- Health checks and monitoring

## 🔒 Security Considerations

### Authentication Security
- Password hashing with Werkzeug
- Session management with Flask-Login
- CSRF protection
- Secure cookie settings

### Data Security
- Input validation and sanitization
- SQL injection prevention with ORM
- XSS protection
- HTTPS enforcement in production

### Access Control
- Role-based permissions
- Route-level authorization
- User session management
- Audit trail logging

## 📝 Code Style and Standards

### Python Style Guide
- Follow PEP 8 standards
- Use type hints where appropriate
- Document functions and classes
- Use meaningful variable names

### Flask Best Practices
- Blueprint organization for large apps
- Error handling and logging
- Configuration management
- Testing with pytest

### Database Best Practices
- Use migrations for schema changes
- Optimize queries
- Use transactions appropriately
- Regular backups

## 🤝 Contributing

### Development Workflow
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Update documentation
6. Submit a pull request

### Testing Requirements
- All new features must have tests
- Run existing test suite before submitting
- Maintain test coverage
- Test both development and production environments

### Documentation Updates
- Update relevant documentation files
- Include code examples
- Update project structure if needed
- Maintain consistency across docs

This developer documentation provides comprehensive guidance for working with the Heart Disease Prediction System, including the new organized directory structure and updated file paths. 