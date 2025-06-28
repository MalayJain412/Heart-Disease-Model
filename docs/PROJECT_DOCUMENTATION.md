# Heart Disease Prediction System - Complete Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [User Documentation](#user-documentation)
3. [Developer Documentation](#developer-documentation)
4. [Project Structure](#project-structure)
5. [File Descriptions](#file-descriptions)
6. [Deployment Guide](#deployment-guide)
7. [Troubleshooting](#troubleshooting)

---

## Project Overview

The Heart Disease Prediction System is a comprehensive Flask-based web application designed for medical professionals to predict cardiovascular diseases using machine learning. The system features role-based authentication, patient management, analytics, and export capabilities.

### Key Features
- **Role-based Access Control**: Three user roles (Doctor, Dean, Chairman)
- **ML-powered Predictions**: Pre-trained Random Forest model for heart disease prediction
- **Patient Management**: Comprehensive patient data collection and storage
- **Analytics Dashboard**: Real-time statistics and visualizations
- **Export Capabilities**: Excel and PDF export functionality
- **Modern UI/UX**: Responsive design with Bootstrap 5

### Technology Stack
- **Backend**: Flask, SQLAlchemy, Flask-Login
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5, Chart.js
- **Database**: Azure SQL Database (Production), SQLite (Development)
- **ML**: scikit-learn, pandas, numpy
- **Deployment**: Docker, Render, Azure App Service

---

## User Documentation

### Getting Started

#### 1. Accessing the Application
- **Local Development**: `http://localhost:5000`
- **Production**: Your deployed URL (e.g., `https://your-app.onrender.com`)

#### 2. Default Login Credentials
- **Username**: `admin`
- **Password**: `admin123`
- **Role**: Chairman (Full system access)

#### 3. User Roles and Permissions

##### 👨‍⚕️ Doctor
**Capabilities:**
- Add new patients and make heart disease predictions
- View only their own patients
- Change personal password
- Access comprehensive patient input form

**Workflow:**
1. Login with doctor credentials
2. Navigate to "Add Patient" from dashboard
3. Fill in patient details (30+ clinical parameters)
4. Submit for ML prediction
5. View prediction results and patient history

##### 👨‍🏫 Dean
**Capabilities:**
- View all patients across all doctors
- Register new doctors
- Access analytics dashboard with charts
- Export patient data (Excel/PDF)
- Filter patients by date, doctor, or disease type

**Workflow:**
1. Login with dean credentials
2. View analytics dashboard for insights
3. Register new doctors as needed
4. Export patient data for reporting
5. Monitor doctor performance

##### 👨‍💼 Chairman
**Capabilities:**
- Full system access
- Register doctors and deans
- View all users and patients
- Access analytics and exports
- User management capabilities

**Workflow:**
1. Login with chairman credentials
2. Manage user accounts (add/remove users)
3. Monitor system-wide analytics
4. Export comprehensive reports
5. Oversee system administration

### Using the Application

#### Adding a Patient (Doctors)
1. **Login** as a doctor
2. **Click** "Add Patient" from dashboard
3. **Fill** the comprehensive form with:
   - Patient demographics (name, age, gender)
   - Vital signs (blood pressure, heart rate)
   - Symptoms (chest pain, shortness of breath, fatigue)
   - Medical conditions (diabetes, hypertension, etc.)
   - Lifestyle factors (smoking, obesity, alcoholism)
4. **Submit** the form
5. **View** the ML prediction result
6. **Patient** is automatically saved to database

#### Viewing Analytics (Deans/Chairmen)
1. **Login** with appropriate credentials
2. **Navigate** to Analytics dashboard
3. **View** real-time statistics:
   - Total patients
   - Normal vs disease cases
   - Doctor performance metrics
4. **Interact** with charts for detailed insights

#### Exporting Data
1. **Login** as Dean or Chairman
2. **Navigate** to Export section
3. **Choose** export format (Excel or PDF)
4. **Apply** filters if needed (date range, doctor, disease type)
5. **Download** the generated file

#### Changing Password
1. **Click** username in top-right corner
2. **Select** "Change Password"
3. **Enter** current password
4. **Enter** new password twice
5. **Submit** to update

---

## Developer Documentation

### Architecture Overview

The application follows a traditional Flask MVC pattern with the following components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   Database      │
│   (Templates)   │◄──►│   (Flask App)   │◄──►│   (Azure SQL)   │
│   + Static      │    │   + Models      │    │   + ML Model    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Components

#### 1. Flask Application (`app.py`)
- **Main entry point** for the web application
- **Route definitions** for all endpoints
- **Database models** (User, Patient)
- **Authentication logic** with Flask-Login
- **ML model integration** for predictions
- **Export functionality** (Excel/PDF)
- **Analytics and chart generation**
- **Authentication and authorization**

#### 2. Configuration Management (`config.py`)
- **Environment-based configuration**
- **Database connection settings**
- **Security configurations**
- **Development/Production profiles**
- **Updated model path**: `data/heart_disease_rf_model.pkl`

#### 3. Database Models
```python
class User(UserMixin, db.Model):
    # User authentication and role management
    id, username, email, password_hash, role, created_at

class Patient(db.Model):
    # Patient data and prediction results
    id, name, gender, age, [30+ clinical parameters], 
    prediction_result, doctor_id, timestamp
```

#### 4. Machine Learning Integration
- **Pre-trained model**: `data/heart_disease_rf_model.pkl` (updated path)
- **Input processing**: 30+ clinical parameters
- **Prediction pipeline**: scikit-learn Random Forest
- **Result storage**: Database integration

### Development Setup

#### Prerequisites
```bash
Python 3.8+
pip
Git
Azure SQL Database (for production)
```

#### Local Development (Updated Commands)
```bash
# 1. Clone repository
git clone <repository-url>
cd Heart-Disease-Model

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment
cp env.example .env
# Edit .env with your configuration

# 4. Set up database
python scripts/setup_azure_database.py

# 5. Add admin user
python scripts/add_admin_user.py

# 6. Run tests
python tests/test_model_loading.py
python tests/test_azure_connection.py

# 7. Start development server
python app.py
```

---

## Project Structure

### Complete Directory Organization

The project is organized with a **production-first** approach, keeping only deployment-essential files in the root directory while organizing supporting files into logical directories:

```
Heart-Disease-Model/
├── 📁 Production Files (Root Directory)
│   ├── app.py                          # Main Flask application (23KB, 588 lines)
│   ├── config.py                       # Configuration management (2.2KB, 80 lines)
│   ├── requirements.txt                # Python dependencies (479B, 20 lines)
│   ├── startup.py                      # Application startup script (6.3KB, 207 lines)
│   ├── env.example                     # Environment variables template (707B, 25 lines)
│   ├── env.production                  # Production environment config (784B, 28 lines)
│   ├── .dockerignore                   # Docker ignore patterns (798B, 79 lines)
│   ├── Dockerfile                      # Production Docker configuration (2.8KB, 85 lines)
│   ├── Dockerfile.alternative          # Alternative Docker setup (2.0KB, 70 lines)
│   ├── Dockerfile.dev                  # Development Docker setup (783B, 37 lines)
│   ├── docker-compose.yml              # Docker Compose configuration (673B, 32 lines)
│   ├── render.yaml                     # Render deployment config (590B, 22 lines)
│   └── README.md                       # Main project documentation (8.7KB, 345 lines)
├── 📁 docs/                            # Documentation Directory
│   ├── DEVELOPER_DOCUMENTATION.md      # Developer technical guide (27KB, 972 lines)
│   ├── DEPLOYMENT.md                   # Deployment guide (4.8KB, 160 lines)
│   ├── USER_DOCUMENTATION.md           # End-user guide (13KB, 501 lines)
│   ├── PROJECT_STRUCTURE.md            # Detailed project structure (18KB, 538 lines)
│   ├── MODEL_LOADING_TROUBLESHOOTING.md # Model troubleshooting guide (6.5KB, 250 lines)
│   ├── RENDER_DEPLOYMENT.md            # Render-specific deployment (4.2KB, 153 lines)
│   └── PROJECT_DOCUMENTATION.md        # This comprehensive guide (22KB, 747 lines)
├── 📁 scripts/                         # Setup and Utility Scripts
│   ├── add_admin_user.py               # Add default admin user (3.7KB, 104 lines)
│   ├── setup_azure_database.py         # Azure SQL setup script (6.0KB, 140 lines)
│   ├── debug_env.py                    # Environment debugging tool (2.5KB, 67 lines)
│   ├── update_env.py                   # Environment file updater (1.7KB, 59 lines)
│   ├── fix_env.py                      # Environment file fixer (2.1KB, 62 lines)
│   ├── streamlit_app.py                # Alternative Streamlit interface (4.4KB, 121 lines)
│   ├── coopy_utils.py                  # Utility functions (1.3KB, 33 lines)
│   └── all_code_snippets.txt           # Code documentation (58KB, 1313 lines)
├── 📁 tests/                           # Testing Files
│   ├── test_model_loading.py           # Model loading tests (4.6KB, 142 lines)
│   ├── test_azure_connection.py        # Azure connection testing (7.1KB, 151 lines)
│   ├── test_admin_credentials.py       # Admin user testing (5.0KB, 105 lines)
│   ├── test_database_comprehensive.py  # Comprehensive DB tests (18KB, 430 lines)
│   ├── test_azure_db.py                # Azure database testing (2.3KB, 58 lines)
│   └── test_docker.py                  # Docker testing (4.3KB, 166 lines)
├── 📁 data/                            # Data Files
│   ├── heart_disease_rf_model.pkl      # Pre-trained ML model (2.9MB)
│   └── Heart D Dataset.csv             # Training dataset (129KB, 1002 lines)
├── 📁 app/                             # Flask Application Structure
│   ├── 📁 templates/                   # HTML templates
│   │   ├── base.html                   # Base template (7.6KB, 224 lines)
│   │   ├── 📁 auth/                    # Authentication templates
│   │   │   ├── login.html              # Login page (2.1KB, 48 lines)
│   │   │   └── change_password.html    # Password change (2.7KB, 56 lines)
│   │   ├── 📁 doctor/                  # Doctor role templates
│   │   │   ├── dashboard.html          # Doctor dashboard (5.8KB, 127 lines)
│   │   │   └── add_patient.html        # Patient input form (19KB, 344 lines)
│   │   ├── 📁 dean/                    # Dean role templates
│   │   │   ├── dashboard.html          # Dean dashboard (6.6KB, 136 lines)
│   │   │   ├── analytics.html          # Analytics dashboard (7.4KB, 198 lines)
│   │   │   └── register_doctor.html    # Doctor registration (4.7KB, 97 lines)
│   │   └── 📁 chairman/                # Chairman role templates
│   │       ├── dashboard.html          # Chairman dashboard (8.1KB, 166 lines)
│   │       └── register_user.html      # User registration (5.2KB, 105 lines)
│   └── 📁 static/                      # Static assets
│       ├── 📁 css/                     # Stylesheets
│       └── 📁 js/                      # JavaScript files
├── 📁 instance/                        # Database Instance
│   └── heart_disease_app.db            # Local SQLite database
└── 📁 mount/                           # Docker Mount Directory
    └── src/
        └── heart_disease_rf_model.pkl  # Model for Docker mounting
```

### Directory Organization Philosophy

#### 🎯 **Production-First Organization**
The project is organized with a **production-first** approach, keeping only deployment-essential files in the root directory while organizing supporting files into logical directories.

#### 📁 **Directory Purposes**

**Root Directory** - Production Files Only
- Contains only files needed for deployment and production
- Core application files (`app.py`, `config.py`)
- Deployment configurations (`Dockerfile`, `docker-compose.yml`)
- Environment templates (`env.example`, `env.production`)
- Main documentation (`README.md`)

**docs/** - Complete Documentation
- All technical documentation and guides
- User manuals and developer guides
- Deployment instructions and troubleshooting
- Project structure and architecture docs

**scripts/** - Setup and Utilities
- Database setup and configuration scripts
- Environment management tools
- Alternative interfaces (Streamlit)
- Utility functions and code snippets

**tests/** - Comprehensive Testing
- Model loading and validation tests
- Database connection and functionality tests
- Azure integration tests
- Docker and deployment tests

**data/** - ML Assets
- Pre-trained machine learning model
- Training datasets and reference data
- Model artifacts and configurations

### Benefits of New Organization

#### ✅ **Clean Production Directory**
- Only deployment-essential files in root
- Reduced clutter and confusion
- Clear separation of concerns

#### ✅ **Logical File Grouping**
- Documentation in dedicated `docs/` folder
- Scripts organized in `scripts/` folder
- Tests centralized in `tests/` folder
- Data assets in `data/` folder

#### ✅ **Improved Maintainability**
- Easier to find specific files
- Better organization for new contributors
- Clearer project structure

#### ✅ **Enhanced Development Workflow**
- Dedicated testing directory
- Organized utility scripts
- Centralized documentation

#### ✅ **Production Optimization**
- Docker ignores non-essential directories
- Faster builds with focused file selection
- Cleaner deployment packages

---

## File Descriptions

### Production Files (Root Directory)

#### `app.py` (Main Application)
- **Purpose**: Main Flask application entry point
- **Key Functions**:
  - Flask app initialization and configuration
  - Database model definitions (User, Patient)
  - Route definitions for all user roles
  - ML model integration and prediction logic
  - Export functionality (Excel/PDF)
  - Analytics and chart generation
  - Authentication and authorization
- **Size**: 23KB, 588 lines
- **Critical**: Yes - Main application logic

#### `config.py` (Configuration Management)
- **Purpose**: Centralized configuration management
- **Key Functions**:
  - Environment-specific configurations (dev/prod/test)
  - Database connection settings
  - Security configurations
  - Azure SQL specific settings
  - **Updated**: Model path now points to `data/heart_disease_rf_model.pkl`
- **Size**: 2.2KB, 80 lines
- **Critical**: Yes - Application configuration

#### `startup.py` (Application Startup)
- **Purpose**: Application startup verification and initialization
- **Key Functions**:
  - Model file verification
  - Database connection testing
  - Environment variable validation
  - Dependency verification
  - **Updated**: Model path references updated to data directory
- **Size**: 6.3KB, 207 lines
- **Critical**: Yes - Production startup verification

#### `requirements.txt` (Dependencies)
- **Purpose**: Python package dependencies
- **Key Contents**:
  - Flask and extensions
  - Database drivers (pyodbc for Azure SQL)
  - ML libraries (scikit-learn, pandas, numpy)
  - Export libraries (openpyxl, reportlab)
  - Production server (gunicorn)
- **Size**: 479B, 20 lines
- **Critical**: Yes - Dependency management

### Documentation Files (docs/)

#### `DEVELOPER_DOCUMENTATION.md`
- **Purpose**: Comprehensive developer guide
- **Key Contents**:
  - Technical architecture overview
  - API documentation
  - Database schema
  - Development setup instructions
  - Code style guidelines
- **Size**: 27KB, 972 lines
- **Audience**: Developers and contributors

#### `USER_DOCUMENTATION.md`
- **Purpose**: End-user guide and manual
- **Key Contents**:
  - User role descriptions
  - Feature walkthroughs
  - Screenshots and examples
  - Troubleshooting guide
- **Size**: 13KB, 501 lines
- **Audience**: End users and administrators

#### `DEPLOYMENT.md`
- **Purpose**: Deployment instructions
- **Key Contents**:
  - Azure App Service deployment
  - Docker deployment
  - Environment configuration
  - Production setup
- **Size**: 4.8KB, 160 lines
- **Audience**: DevOps and system administrators

#### `PROJECT_STRUCTURE.md`
- **Purpose**: Detailed project structure documentation
- **Key Contents**:
  - Complete file organization
  - File purposes and functions
  - Directory explanations
  - Architecture overview
- **Size**: 18KB, 538 lines
- **Audience**: Developers and project maintainers

### Scripts Directory (scripts/)

#### `add_admin_user.py` (Admin User Setup)
- **Purpose**: Create default admin user in database
- **Key Functions**:
  - Database connection setup
  - User creation with hashed password
  - Role assignment (chairman)
  - **Updated**: Import paths updated for new structure
- **Size**: 3.7KB, 104 lines
- **Usage**: `python scripts/add_admin_user.py`

#### `setup_azure_database.py` (Azure DB Setup)
- **Purpose**: Azure SQL Database initialization
- **Key Functions**:
  - Database creation
  - User setup and permissions
  - Connection testing
  - Schema initialization
- **Size**: 6.0KB, 140 lines
- **Usage**: `python scripts/setup_azure_database.py`

#### `debug_env.py` (Environment Debugging)
- **Purpose**: Environment variable debugging
- **Key Functions**:
  - Environment variable validation
  - Configuration testing
  - Connection string verification
- **Size**: 2.5KB, 67 lines
- **Usage**: `python scripts/debug_env.py`

### Tests Directory (tests/)

#### `test_model_loading.py` (Model Tests)
- **Purpose**: ML model loading and validation
- **Key Functions**:
  - Model file existence verification
  - Model loading testing
  - Prediction functionality testing
  - **Updated**: Paths updated to reference data directory
- **Size**: 4.6KB, 142 lines
- **Usage**: `python tests/test_model_loading.py`

#### `test_azure_connection.py` (Azure Tests)
- **Purpose**: Azure SQL Database connection testing
- **Key Functions**:
  - Connection string validation
  - Database connectivity testing
  - Query execution testing
- **Size**: 7.1KB, 151 lines
- **Usage**: `python tests/test_azure_connection.py`

#### `test_database_comprehensive.py` (DB Tests)
- **Purpose**: Comprehensive database functionality testing
- **Key Functions**:
  - CRUD operations testing
  - User management testing
  - Patient data testing
  - Performance testing
- **Size**: 18KB, 430 lines
- **Usage**: `python tests/test_database_comprehensive.py`

### Data Directory (data/)

#### `heart_disease_rf_model.pkl` (ML Model)
- **Purpose**: Pre-trained Random Forest model
- **Key Features**:
  - Trained on heart disease dataset
  - 30+ clinical parameters
  - Binary classification (Normal/Disease)
  - **Location**: Now in dedicated data directory
- **Size**: 2.9MB
- **Critical**: Yes - Core ML functionality

#### `Heart D Dataset.csv` (Training Data)
- **Purpose**: Original training dataset
- **Key Features**:
  - 1002 patient records
  - 30+ clinical parameters
  - Used for model training
  - **Location**: Now in dedicated data directory
- **Size**: 129KB, 1002 lines
- **Critical**: No - Reference data only

### Docker & Deployment Files

#### `Dockerfile` (Production Docker)
- **Purpose**: Production Docker container configuration
- **Key Features**:
  - Python 3.11 slim base
  - ODBC Driver 18 installation
  - Non-root user for security
  - Gunicorn production server
  - **Updated**: Model path updated to `data/heart_disease_rf_model.pkl`
  - **Updated**: Test path updated to `tests/test_model_loading.py`
- **Size**: 2.8KB, 85 lines
- **Critical**: Yes - Production deployment

#### `docker-compose.yml` (Docker Compose)
- **Purpose**: Multi-container Docker setup
- **Key Features**:
  - Application container
  - Database container (optional)
  - Volume mounting
  - Environment configuration
- **Size**: 673B, 32 lines
- **Critical**: Yes - Local development

#### `render.yaml` (Render Deployment)
- **Purpose**: Render cloud deployment configuration
- **Key Features**:
  - Build configuration
  - Environment variables
  - Health checks
  - Auto-deployment
- **Size**: 590B, 22 lines
- **Critical**: Yes - Cloud deployment

---

## Deployment Guide

### Updated Commands and Usage

#### 🚀 **Setup Commands**
```bash
# Database setup
python scripts/setup_azure_database.py

# Admin user creation
python scripts/add_admin_user.py

# Environment debugging
python scripts/debug_env.py
```

#### 🧪 **Testing Commands**
```bash
# Model loading tests
python tests/test_model_loading.py

# Azure connection tests
python tests/test_azure_connection.py

# Database functionality tests
python tests/test_database_comprehensive.py

# Docker tests
python tests/test_docker.py
```

#### 📚 **Documentation Access**
```bash
# View user documentation
cat docs/USER_DOCUMENTATION.md

# View developer documentation
cat docs/DEVELOPER_DOCUMENTATION.md

# View deployment guide
cat docs/DEPLOYMENT.md
```

#### 🐳 **Docker Commands**
```bash
# Build production image
docker build -t heart-disease-app .

# Run with Docker Compose
docker-compose up -d

# Test Docker build
python tests/test_docker.py
```

### Migration Notes

#### 🔄 **Path Updates**
- Model path: `heart_disease_rf_model.pkl` → `data/heart_disease_rf_model.pkl`
- Test paths: `test_*.py` → `tests/test_*.py`
- Script paths: `*_script.py` → `scripts/*_script.py`
- Documentation: `*.md` → `docs/*.md`

#### 🔧 **Configuration Changes**
- `config.py`: Updated MODEL_PATH to data directory
- `Dockerfile`: Updated model and test file paths
- `.dockerignore`: Updated to exclude new directories
- `startup.py`: Updated model path references

#### 📝 **Documentation Updates**
- All documentation files updated to reflect new structure
- Commands updated to use new paths
- Examples updated with new directory references

---

## Troubleshooting

### Common Issues and Solutions

#### Model Loading Issues
```bash
# Test model loading
python tests/test_model_loading.py

# Check model file
ls -la data/heart_disease_rf_model.pkl

# Debug environment variables
python scripts/debug_env.py
```

#### Database Connection Issues
```bash
# Test database connection
python tests/test_azure_connection.py

# Test comprehensive database functionality
python tests/test_database_comprehensive.py

# Debug environment
python scripts/debug_env.py
```

#### Docker Issues
```bash
# Test Docker build
python tests/test_docker.py

# Check Docker logs
docker logs <container-id>

# Build and test locally
docker build -t heart-disease-app .
docker run -p 5000:5000 heart-disease-app
```

#### Environment Issues
```bash
# Debug environment variables
python scripts/debug_env.py

# Fix environment file issues
python scripts/fix_env.py

# Update environment file
python scripts/update_env.py
```

### Debugging and Troubleshooting

#### Environment Debugging
```bash
# Run environment debugging script
python scripts/debug_env.py

# Check specific environment variable
echo $MODEL_PATH
```

#### Model Loading Issues
```bash
# Test model loading
python tests/test_model_loading.py

# Check model file
ls -la data/heart_disease_rf_model.pkl
```

#### Database Issues
```bash
# Test database connection
python tests/test_azure_connection.py

# Test comprehensive database functionality
python tests/test_database_comprehensive.py
```

#### Docker Issues
```bash
# Test Docker build
python tests/test_docker.py

# Check Docker logs
docker logs <container-id>
```

This comprehensive documentation has been updated to reflect the new organized directory structure with the model file located in the `data/` directory and updated paths throughout the documentation. 