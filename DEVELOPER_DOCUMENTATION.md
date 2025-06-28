# Heart Disease Prediction System - Developer Documentation

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Development Setup](#development-setup)
3. [Code Structure](#code-structure)
4. [Database Design](#database-design)
5. [API Reference](#api-reference)
6. [Deployment Guide](#deployment-guide)
7. [Testing](#testing)
8. [Security](#security)
9. [Performance Optimization](#performance-optimization)
10. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

### System Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   Database      │
│   (Templates)   │◄──►│   (Flask App)   │◄──►│   (Azure SQL)   │
│   + Static      │    │   + Models      │    │   + ML Model    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Technology Stack
- **Backend Framework**: Flask 2.3.3
- **Database ORM**: SQLAlchemy 3.0.5
- **Authentication**: Flask-Login 0.6.3
- **Database**: Azure SQL Database (Production), SQLite (Development)
- **ML Framework**: scikit-learn 1.4.2
- **Frontend**: Bootstrap 5, Chart.js, HTML5/CSS3
- **Deployment**: Docker, Render, Azure App Service

### Design Patterns
- **MVC Pattern**: Model-View-Controller separation
- **Factory Pattern**: Configuration management
- **Repository Pattern**: Database abstraction
- **Decorator Pattern**: Route protection

---

## Development Setup

### Prerequisites
```bash
Python 3.8+
pip
Git
Azure SQL Database (for production testing)
Docker (optional)
```

### Local Development Environment

#### 1. Clone Repository
```bash
git clone <repository-url>
cd Heart-Disease-Model
```

#### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Environment Configuration
```bash
# Copy environment template
cp env.example .env

# Edit .env file
# SECRET_KEY=your-secret-key
# DATABASE_URL=sqlite:///heart_disease_app.db (for local dev)
```

#### 5. Initialize Database
```bash
python app.py
# This will create the database and default admin user
```

#### 6. Run Application
```bash
python app.py
# Access at http://localhost:5000
```

### Docker Development
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t heart-disease-app .
docker run -p 5000:5000 heart-disease-app
```

---

## Code Structure

### Main Application (`app.py`)

#### Application Initialization
```python
# Flask app creation
app = Flask(__name__, template_folder='app/templates')

# Configuration loading
config_name = os.environ.get('FLASK_ENV', 'development')
app.config.from_object(config[config_name])

# Database and login manager setup
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
```

#### Database Models
```python
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'doctor', 'dean', 'chairman'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    patients = db.relationship('Patient', backref='doctor', lazy=True)

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    # 30+ clinical parameters
    prediction_result = db.Column(db.String(100), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
```

#### ML Model Integration
```python
def load_model():
    try:
        model_path = app.config['MODEL_PATH']
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Model prediction function
def predict_heart_disease(patient_data):
    # Preprocess input data
    # Make prediction
    # Return result
```

### Configuration Management (`config.py`)

#### Configuration Classes
```python
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MODEL_PATH = os.environ.get('MODEL_PATH') or 'heart_disease_rf_model.pkl'

class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
        'pool_size': 5,
        'max_overflow': 10
    }

class ProductionConfig(Config):
    DEBUG = False
    SESSION_COOKIE_SECURE = True
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
        'pool_size': 10,
        'max_overflow': 20
    }
```

### Template Structure

#### Base Template (`app/templates/base.html`)
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <!-- Bootstrap 5, Google Fonts, Font Awesome -->
    <!-- Custom CSS with medical theme -->
</head>
<body>
    <!-- Navigation bar -->
    <!-- Flash messages -->
    <!-- Content block -->
    <!-- Bootstrap JS -->
</body>
</html>
```

#### Role-specific Templates
- `auth/`: Login and password change
- `doctor/`: Patient management
- `dean/`: Analytics and doctor registration
- `chairman/`: User management and system overview

---

## Database Design

### Schema Overview

#### User Table
```sql
CREATE TABLE [user] (
    id INT PRIMARY KEY IDENTITY(1,1),
    username VARCHAR(80) UNIQUE NOT NULL,
    email VARCHAR(120) UNIQUE NOT NULL,
    password_hash VARCHAR(120) NOT NULL,
    role VARCHAR(20) NOT NULL,  -- 'doctor', 'dean', 'chairman'
    created_at DATETIME DEFAULT GETDATE()
);
```

#### Patient Table
```sql
CREATE TABLE patient (
    id INT PRIMARY KEY IDENTITY(1,1),
    name VARCHAR(100) NOT NULL,
    gender VARCHAR(10) NOT NULL,
    age INT NOT NULL,
    
    -- Vital Signs
    systolic INT NOT NULL,
    diastolic INT NOT NULL,
    heart_rate INT NOT NULL,
    
    -- Symptoms
    chest_pain BIT DEFAULT 0,
    shortness_of_breath BIT DEFAULT 0,
    fatigue BIT DEFAULT 0,
    lung_sounds BIT DEFAULT 0,
    
    -- Cholesterol
    cholesterol INT NOT NULL,
    ldl INT NOT NULL,
    hdl INT NOT NULL,
    
    -- Medical Conditions
    diabetes BIT DEFAULT 0,
    atrial_fibrillation BIT DEFAULT 0,
    rheumatic_fever BIT DEFAULT 0,
    mitral_stenosis BIT DEFAULT 0,
    aortic_stenosis BIT DEFAULT 0,
    tricuspid_stenosis BIT DEFAULT 0,
    pulmonary_stenosis BIT DEFAULT 0,
    dilated_cardiomyopathy BIT DEFAULT 0,
    hypertrophic_cardiomyopathy BIT DEFAULT 0,
    
    -- Lifestyle Factors
    drug_use BIT DEFAULT 0,
    fever BIT DEFAULT 0,
    chills BIT DEFAULT 0,
    alcoholism BIT DEFAULT 0,
    hypertension BIT DEFAULT 0,
    fainting BIT DEFAULT 0,
    dizziness BIT DEFAULT 0,
    smoking BIT DEFAULT 0,
    obesity BIT DEFAULT 0,
    murmur BIT DEFAULT 0,
    
    -- Results
    prediction_result VARCHAR(100) NOT NULL,
    doctor_id INT FOREIGN KEY REFERENCES [user](id),
    timestamp DATETIME DEFAULT GETDATE()
);
```

### Database Relationships
- **One-to-Many**: User → Patients (one doctor can have many patients)
- **Foreign Key**: Patient.doctor_id → User.id

### Indexes
```sql
-- Performance indexes
CREATE INDEX idx_patient_doctor_id ON patient(doctor_id);
CREATE INDEX idx_patient_timestamp ON patient(timestamp);
CREATE INDEX idx_user_username ON [user](username);
CREATE INDEX idx_user_email ON [user](email);
```

---

## API Reference

### Authentication Endpoints

#### POST /login
**Purpose**: User authentication
**Request Body**:
```json
{
    "username": "string",
    "password": "string"
}
```
**Response**: Redirect to dashboard or error message

#### GET /logout
**Purpose**: User logout
**Response**: Redirect to login page

#### POST /change_password
**Purpose**: Change user password
**Request Body**:
```json
{
    "current_password": "string",
    "new_password": "string",
    "confirm_password": "string"
}
```

### Doctor Endpoints

#### GET /doctor/dashboard
**Purpose**: Doctor dashboard view
**Response**: HTML template with patient list

#### POST /doctor/add_patient
**Purpose**: Add new patient and get prediction
**Request Body**: Form data with patient parameters
**Response**: Prediction result and patient saved

### Dean Endpoints

#### GET /dean/dashboard
**Purpose**: Dean dashboard view
**Response**: HTML template with system overview

#### POST /dean/register_doctor
**Purpose**: Register new doctor
**Request Body**: User registration form data

#### GET /dean/analytics
**Purpose**: Analytics dashboard
**Response**: HTML template with charts

### Chairman Endpoints

#### GET /chairman/dashboard
**Purpose**: Chairman dashboard view
**Response**: HTML template with full system overview

#### POST /chairman/register_user
**Purpose**: Register new users (any role)
**Request Body**: User registration form data

### Export Endpoints

#### GET /export/excel
**Purpose**: Export patient data to Excel
**Query Parameters**:
- `start_date`: Filter start date
- `end_date`: Filter end date
- `doctor_id`: Filter by doctor
- `disease_type`: Filter by prediction result

#### GET /export/pdf
**Purpose**: Export patient data to PDF
**Query Parameters**: Same as Excel export

### API Endpoints

#### GET /api/analytics
**Purpose**: Get analytics data as JSON
**Response**:
```json
{
    "total_patients": 150,
    "normal_cases": 120,
    "disease_cases": 30,
    "doctor_stats": [...],
    "prediction_distribution": {...}
}
```

---

## Deployment Guide

### Docker Deployment

#### Production Dockerfile
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ pkg-config curl gnupg2 \
    apt-transport-https ca-certificates \
    wget software-properties-common

# Install ODBC Driver 18 for Azure SQL
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18 unixodbc-dev

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE $PORT

# Run with Gunicorn
CMD gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 app:app
```

#### Docker Compose (Development)
```yaml
version: '3.8'
services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=development
      - DATABASE_URL=sqlite:///heart_disease_app.db
    volumes:
      - .:/app
```

### Render Deployment

#### render.yaml Configuration
```yaml
services:
  - type: web
    name: heart-disease-predictor
    env: docker
    region: oregon
    plan: free
    buildCommand: ""
    startCommand: ""
    envVars:
      - key: FLASK_ENV
        value: production
      - key: SECRET_KEY
        generateValue: true
      - key: DATABASE_URL
        sync: false
```

#### Environment Variables
```env
FLASK_ENV=production
SECRET_KEY=your-production-secret-key
DATABASE_URL=mssql+pyodbc://user:pass@server/db?driver=ODBC+Driver+18
MODEL_PATH=heart_disease_rf_model.pkl
SESSION_COOKIE_SECURE=True
```

### Azure App Service Deployment

#### Azure CLI Deployment
```bash
# Login to Azure
az login

# Create resource group
az group create --name myResourceGroup --location eastus

# Create App Service plan
az appservice plan create --name myAppServicePlan --resource-group myResourceGroup --sku B1

# Create web app
az webapp create --name myWebApp --resource-group myResourceGroup --plan myAppServicePlan --runtime "PYTHON|3.8"

# Deploy application
az webapp up --name myWebApp --resource-group myResourceGroup
```

#### Environment Variables in Azure
```bash
# Set environment variables
az webapp config appsettings set --name myWebApp --resource-group myResourceGroup --settings \
    SECRET_KEY="your-secret-key" \
    DATABASE_URL="mssql+pyodbc://user:pass@server/db?driver=ODBC+Driver+17" \
    FLASK_ENV="production"
```

---

## Testing

### Test Structure
```
tests/
├── test_auth.py              # Authentication tests
├── test_models.py            # Database model tests
├── test_routes.py            # Route functionality tests
├── test_ml.py                # ML model tests
├── test_export.py            # Export functionality tests
└── conftest.py               # Test configuration
```

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=app --cov-report=html

# Run specific test file
python -m pytest tests/test_auth.py

# Run with verbose output
python -m pytest -v
```

### Test Examples

#### Authentication Tests
```python
def test_login_success(client):
    response = client.post('/login', data={
        'username': 'admin',
        'password': 'admin123'
    })
    assert response.status_code == 302  # Redirect after login

def test_login_failure(client):
    response = client.post('/login', data={
        'username': 'admin',
        'password': 'wrongpassword'
    })
    assert b'Invalid username or password' in response.data
```

#### Model Tests
```python
def test_user_creation():
    user = User(
        username='testuser',
        email='test@example.com',
        password_hash='hashed_password',
        role='doctor'
    )
    db.session.add(user)
    db.session.commit()
    
    assert user.id is not None
    assert user.username == 'testuser'
```

#### Route Tests
```python
def test_doctor_dashboard_access(client, auth):
    auth.login('doctor')
    response = client.get('/doctor/dashboard')
    assert response.status_code == 200
    assert b'Add Patient' in response.data

def test_unauthorized_access(client):
    response = client.get('/doctor/dashboard')
    assert response.status_code == 302  # Redirect to login
```

### Integration Tests
```python
def test_patient_prediction_workflow(client, auth):
    # Login as doctor
    auth.login('doctor')
    
    # Add patient
    response = client.post('/doctor/add_patient', data={
        'name': 'John Doe',
        'age': 45,
        'gender': 'Male',
        # ... other patient data
    })
    
    assert response.status_code == 200
    assert b'Prediction Result' in response.data
```

---

## Security

### Authentication Security
```python
# Password hashing
from werkzeug.security import generate_password_hash, check_password_hash

# Password validation
def validate_password(password):
    if len(password) < 8:
        return False
    # Add more validation rules
    return True

# Session security
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
```

### Route Protection
```python
from functools import wraps
from flask_login import login_required, current_user

def role_required(role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated or current_user.role != role:
                flash('Access denied', 'error')
                return redirect(url_for('index'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Usage
@app.route('/doctor/dashboard')
@login_required
@role_required('doctor')
def doctor_dashboard():
    # Route implementation
```

### Input Validation
```python
from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, BooleanField
from wtforms.validators import DataRequired, Email, Length, NumberRange

class PatientForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired(), Length(min=2, max=100)])
    age = IntegerField('Age', validators=[DataRequired(), NumberRange(min=1, max=120)])
    # ... other fields
```

### SQL Injection Prevention
```python
# Use SQLAlchemy ORM (prevents SQL injection)
patients = Patient.query.filter_by(doctor_id=current_user.id).all()

# Parameterized queries for raw SQL
result = db.session.execute(
    text("SELECT * FROM patient WHERE doctor_id = :doctor_id"),
    {"doctor_id": current_user.id}
)
```

### CSRF Protection
```python
# Enable CSRF protection
app.config['WTF_CSRF_ENABLED'] = True
app.config['WTF_CSRF_SECRET_KEY'] = 'your-csrf-secret-key'

# In templates
<form method="POST">
    {{ form.csrf_token }}
    <!-- form fields -->
</form>
```

---

## Performance Optimization

### Database Optimization

#### Connection Pooling
```python
# Configure connection pooling
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
```

#### Query Optimization
```python
# Use eager loading for relationships
patients = Patient.query.options(
    db.joinedload(Patient.doctor)
).filter_by(doctor_id=current_user.id).all()

# Use pagination for large datasets
patients = Patient.query.paginate(
    page=page, per_page=20, error_out=False
)
```

#### Database Indexes
```sql
-- Create indexes for frequently queried columns
CREATE INDEX idx_patient_doctor_timestamp ON patient(doctor_id, timestamp);
CREATE INDEX idx_patient_prediction ON patient(prediction_result);
CREATE INDEX idx_user_role ON [user](role);
```

### Application Optimization

#### Caching
```python
from flask_caching import Cache

cache = Cache(config={'CACHE_TYPE': 'simple'})
cache.init_app(app)

@cache.memoize(timeout=300)
def get_analytics_data():
    # Expensive analytics calculation
    return analytics_data
```

#### Static File Optimization
```python
# Configure static file serving
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 31536000  # 1 year

# Use CDN for external libraries
# Bootstrap, Chart.js, Font Awesome
```

#### Background Tasks
```python
from celery import Celery

celery = Celery('tasks', broker='redis://localhost:6379/0')

@celery.task
def generate_export_file(data, format_type):
    # Generate export file in background
    pass
```

### Monitoring and Logging
```python
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
if not app.debug:
    file_handler = RotatingFileHandler('logs/heart_disease.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Heart Disease Prediction startup')
```

---

## Troubleshooting

### Common Development Issues

#### Database Connection Errors
```bash
# Check connection string
echo $DATABASE_URL

# Test connection
python test_azure_connection.py

# Check firewall settings
# Add your IP to Azure SQL Database firewall
```

#### Model Loading Issues
```python
# Check model file exists
import os
model_path = 'heart_disease_rf_model.pkl'
print(f"Model exists: {os.path.exists(model_path)}")

# Check file permissions
print(f"Model readable: {os.access(model_path, os.R_OK)}")
```

#### Import Errors
```bash
# Check virtual environment
which python
pip list

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Production Issues

#### Deployment Failures
```bash
# Check build logs
docker build -t test-app . 2>&1 | tee build.log

# Check runtime logs
docker logs <container_id>

# Test locally
docker run -p 5000:5000 test-app
```

#### Performance Issues
```python
# Monitor database queries
from flask_sqlalchemy import get_debug_queries

@app.after_request
def after_request(response):
    for query in get_debug_queries():
        if query.duration >= 0.5:
            app.logger.warning(f"Slow query: {query.statement}")
    return response
```

#### Memory Issues
```python
# Monitor memory usage
import psutil
import os

def log_memory_usage():
    process = psutil.Process(os.getpid())
    app.logger.info(f"Memory usage: {process.memory_info().rss / 1024 / 1024} MB")
```

### Debug Tools

#### Environment Debugging
```bash
# Run environment debug script
python debug_env.py

# Check environment variables
python -c "import os; print(os.environ.get('DATABASE_URL'))"
```

#### Database Debugging
```bash
# Test database connection
python test_azure_connection.py

# Comprehensive database test
python test_database_comprehensive.py

# Check database schema
python -c "
from app import db
from app import User, Patient
print('Tables:', db.engine.table_names())
"
```

#### Application Debugging
```python
# Enable debug mode
app.config['DEBUG'] = True

# Add debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use Flask debugger
if app.debug:
    app.logger.debug('Debug mode enabled')
```

### Performance Monitoring

#### Application Metrics
```python
import time
from functools import wraps

def monitor_performance(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        app.logger.info(f"{f.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return decorated_function
```

#### Database Performance
```python
# Monitor slow queries
from flask_sqlalchemy import get_debug_queries

@app.after_request
def after_request(response):
    queries = get_debug_queries()
    slow_queries = [q for q in queries if q.duration >= 0.5]
    if slow_queries:
        app.logger.warning(f"Found {len(slow_queries)} slow queries")
    return response
```

---

## Contributing

### Development Workflow
1. **Fork** the repository
2. **Create** feature branch: `git checkout -b feature/new-feature`
3. **Make** changes and test thoroughly
4. **Commit** changes: `git commit -m 'Add new feature'`
5. **Push** to branch: `git push origin feature/new-feature`
6. **Create** Pull Request

### Code Standards
- **PEP 8**: Python code style
- **Docstrings**: Document all functions and classes
- **Type Hints**: Use type annotations
- **Error Handling**: Proper exception handling
- **Testing**: Write tests for new features

### Testing Requirements
- **Unit Tests**: Test individual functions
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Coverage**: Maintain >80% code coverage

---

**Note**: This documentation should be updated as the codebase evolves. For the latest information, always refer to the current code and deployment guides. 