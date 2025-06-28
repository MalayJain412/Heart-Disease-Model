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

#### 2. Configuration Management (`config.py`)
- **Environment-based configuration**
- **Database connection settings**
- **Security configurations**
- **Development/Production profiles**

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
- **Pre-trained model**: `heart_disease_rf_model.pkl`
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

#### Local Development
```bash
# 1. Clone repository
git clone <repository-url>
cd Heart-Disease-Model

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp env.example .env
# Edit .env with your settings

# 5. Run application
python app.py
```

#### Environment Variables
```env
# Required
SECRET_KEY=your-secret-key
DATABASE_URL=mssql+pyodbc://user:pass@server/db?driver=ODBC+Driver+18

# Optional
FLASK_ENV=development
MODEL_PATH=heart_disease_rf_model.pkl
SESSION_COOKIE_SECURE=False
```

### Database Schema

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
    -- 30+ clinical parameters (Boolean/Integer)
    prediction_result VARCHAR(100) NOT NULL,
    doctor_id INT FOREIGN KEY REFERENCES [user](id),
    timestamp DATETIME DEFAULT GETDATE()
);
```

### API Endpoints

#### Authentication
- `GET/POST /login` - User login
- `GET /logout` - User logout
- `GET/POST /change_password` - Password change

#### Doctor Routes
- `GET /doctor/dashboard` - Doctor dashboard
- `GET/POST /doctor/add_patient` - Add new patient

#### Dean Routes
- `GET /dean/dashboard` - Dean dashboard
- `GET/POST /dean/register_doctor` - Register new doctor
- `GET /dean/analytics` - Analytics dashboard

#### Chairman Routes
- `GET /chairman/dashboard` - Chairman dashboard
- `GET/POST /chairman/register_user` - Register new users

#### Export Routes
- `GET /export/excel` - Export to Excel
- `GET /export/pdf` - Export to PDF

#### API Routes
- `GET /api/analytics` - Analytics data (JSON)

### Security Features

#### Authentication & Authorization
- **Flask-Login**: Session management
- **Password Hashing**: Werkzeug security
- **Role-based Access**: Route protection
- **CSRF Protection**: Built-in Flask protection

#### Data Protection
- **Input Validation**: Form validation
- **SQL Injection Prevention**: SQLAlchemy ORM
- **XSS Protection**: Template escaping
- **Secure Headers**: Production security headers

### Testing

#### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=app tests/

# Run specific test file
python -m pytest tests/test_auth.py
```

#### Test Files
- `test_azure_connection.py` - Database connection tests
- `test_admin_credentials.py` - Authentication tests
- `test_database_comprehensive.py` - Database operation tests
- `test_docker.py` - Docker deployment tests

---

## Project Structure

```
Heart-Disease-Model/
├── 📁 Root Files
│   ├── app.py                          # Main Flask application
│   ├── config.py                       # Configuration management
│   ├── requirements.txt                # Python dependencies
│   ├── Dockerfile                      # Production Docker configuration
│   ├── docker-compose.yml              # Docker Compose for development
│   ├── render.yaml                     # Render deployment configuration
│   ├── README.md                       # Project overview
│   ├── RENDER_DEPLOYMENT.md            # Render deployment guide
│   ├── DEPLOYMENT.md                   # General deployment guide
│   └── PROJECT_DOCUMENTATION.md        # This documentation file
│
├── 📁 Environment & Configuration
│   ├── env.example                     # Environment variables template
│   ├── env.production                  # Production environment settings
│   ├── .dockerignore                   # Docker build exclusions
│   └── .gitignore                      # Git exclusions
│
├── 📁 Application Core
│   ├── heart_disease_rf_model.pkl      # Pre-trained ML model
│   ├── Heart D Dataset.csv             # Training dataset
│   └── streamlit_app.py                # Alternative Streamlit interface
│
├── 📁 Templates (app/templates/)
│   ├── base.html                       # Base template with styling
│   ├── 📁 auth/                        # Authentication templates
│   │   ├── login.html                  # Login page
│   │   └── change_password.html        # Password change form
│   ├── 📁 doctor/                      # Doctor-specific templates
│   │   ├── dashboard.html              # Doctor dashboard
│   │   └── add_patient.html            # Patient input form
│   ├── 📁 dean/                        # Dean-specific templates
│   │   ├── dashboard.html              # Dean dashboard
│   │   ├── analytics.html              # Analytics dashboard
│   │   └── register_doctor.html        # Doctor registration form
│   └── 📁 chairman/                    # Chairman-specific templates
│       ├── dashboard.html              # Chairman dashboard
│       └── register_user.html          # User registration form
│
├── 📁 Static Files (app/static/)
│   ├── 📁 css/                         # Custom CSS stylesheets
│   └── 📁 js/                          # JavaScript files
│
├── 📁 Helper Scripts
│   ├── add_admin_user.py               # Add default admin user
│   ├── setup_azure_database.py         # Azure SQL setup script
│   ├── test_azure_connection.py        # Database connection testing
│   ├── test_admin_credentials.py       # Authentication testing
│   ├── test_database_comprehensive.py  # Comprehensive DB tests
│   ├── test_docker.py                  # Docker testing
│   ├── test_azure_db.py                # Azure DB specific tests
│   ├── debug_env.py                    # Environment debugging
│   ├── fix_env.py                      # Environment fixing
│   ├── update_env.py                   # Environment updating
│   └── coopy_utils.py                  # Utility functions
│
├── 📁 Docker Files
│   ├── Dockerfile                      # Production Dockerfile
│   ├── Dockerfile.alternative          # Alternative Docker configuration
│   └── Dockerfile.dev                  # Development Dockerfile
│
├── 📁 Data & Models
│   ├── instance/                       # Local database files
│   │   └── heart_disease_app.db        # SQLite database (dev)
│   └── mount/                          # Mounted model files
│       └── src/
│           └── heart_disease_rf_model.pkl
│
└── 📁 Documentation
    └── all_code_snippets.txt           # Code documentation
```

---

## File Descriptions

### Core Application Files

#### `app.py` (23KB, 588 lines)
**Purpose**: Main Flask application entry point
**Key Components**:
- Flask app initialization and configuration
- Database models (User, Patient)
- Authentication routes and logic
- Role-based route protection
- ML model integration for predictions
- Export functionality (Excel/PDF)
- Analytics API endpoints
- Database initialization

**Why Used**: Central application file that orchestrates all functionality

#### `config.py` (2.2KB, 80 lines)
**Purpose**: Configuration management for different environments
**Key Components**:
- Base configuration class
- Development, Production, and Testing configs
- Azure SQL Database specific settings
- Security configurations
- Environment variable management

**Why Used**: Separates configuration from code, enables environment-specific settings

#### `requirements.txt` (479B, 20 lines)
**Purpose**: Python dependencies specification
**Key Dependencies**:
- Flask ecosystem (Flask, SQLAlchemy, Login)
- ML libraries (scikit-learn, pandas, numpy)
- Export libraries (openpyxl, reportlab)
- Azure SQL support (pyodbc)
- Production server (gunicorn)

**Why Used**: Ensures consistent dependency versions across environments

### Template Files

#### `app/templates/base.html` (7.6KB, 224 lines)
**Purpose**: Base template with common styling and layout
**Key Features**:
- Bootstrap 5 integration
- Custom CSS with medical theme
- Responsive navigation
- Flash message handling
- User authentication display

**Why Used**: Provides consistent UI across all pages

#### `app/templates/doctor/add_patient.html` (19KB, 344 lines)
**Purpose**: Comprehensive patient input form
**Key Features**:
- 30+ clinical parameter inputs
- Form validation
- Real-time input processing
- ML prediction integration
- Responsive design

**Why Used**: Main interface for doctors to input patient data

#### `app/templates/dean/analytics.html` (7.4KB, 198 lines)
**Purpose**: Analytics dashboard with charts
**Key Features**:
- Chart.js integration
- Real-time statistics
- Interactive visualizations
- Doctor performance metrics
- Prediction distribution charts

**Why Used**: Provides insights for deans and chairmen

### Helper Scripts

#### `add_admin_user.py` (3.7KB, 104 lines)
**Purpose**: Add default admin user to database
**Key Features**:
- Database connection management
- Password hashing
- User verification
- Error handling

**Why Used**: Simplifies initial setup and user creation

#### `setup_azure_database.py` (6.0KB, 140 lines)
**Purpose**: Azure SQL Database setup and user creation
**Key Features**:
- Database creation
- User authentication setup
- Permission management
- Connection testing

**Why Used**: Automates Azure SQL Database configuration

#### `test_azure_connection.py` (7.1KB, 151 lines)
**Purpose**: Comprehensive database connection testing
**Key Features**:
- Connection string validation
- Driver compatibility testing
- Query execution testing
- Error diagnosis

**Why Used**: Ensures database connectivity before deployment

### Deployment Files

#### `Dockerfile` (1.8KB, 65 lines)
**Purpose**: Production Docker container configuration
**Key Features**:
- Python 3.11 slim base
- Azure SQL ODBC driver installation
- Security hardening (non-root user)
- Health checks
- Gunicorn production server

**Why Used**: Enables containerized deployment on Render/Azure

#### `render.yaml` (590B, 22 lines)
**Purpose**: Render deployment configuration
**Key Features**:
- Service definition
- Environment variables
- Build and start commands
- Health check configuration

**Why Used**: Automates Render deployment process

#### `docker-compose.yml` (673B, 32 lines)
**Purpose**: Local development with Docker
**Key Features**:
- Multi-container setup
- Volume mounting
- Port mapping
- Environment variable management

**Why Used**: Simplifies local development setup

### Data Files

#### `heart_disease_rf_model.pkl` (2.9MB)
**Purpose**: Pre-trained Random Forest model
**Key Features**:
- Serialized scikit-learn model
- Trained on heart disease dataset
- 30+ feature input
- Binary classification output

**Why Used**: Provides ML predictions for heart disease

#### `Heart D Dataset.csv` (129KB, 1002 lines)
**Purpose**: Training dataset for the ML model
**Key Features**:
- Clinical parameters
- Patient outcomes
- Feature engineering data
- Model training source

**Why Used**: Source data for model training and validation

### Alternative Interface

#### `streamlit_app.py` (4.4KB, 121 lines)
**Purpose**: Alternative Streamlit-based interface
**Key Features**:
- Simplified UI for quick predictions
- Same ML model integration
- Interactive input widgets
- Real-time predictions

**Why Used**: Provides alternative interface for different use cases

---

## Deployment Guide

### Local Development
```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment
cp env.example .env
# Edit .env with your settings

# 3. Run application
python app.py
```

### Docker Development
```bash
# 1. Build and run with Docker Compose
docker-compose up --build

# 2. Access application
# http://localhost:5000
```

### Render Deployment
```bash
# 1. Push code to GitHub
git add .
git commit -m "Deploy to Render"
git push origin main

# 2. Configure Render service
# - Connect GitHub repository
# - Set environment variables
# - Deploy automatically
```

### Azure App Service Deployment
```bash
# 1. Install Azure CLI
# 2. Login to Azure
az login

# 3. Deploy application
az webapp up --name <app-name> --resource-group <resource-group>
```

### Environment Variables Setup

#### Development
```env
SECRET_KEY=your-dev-secret-key
DATABASE_URL=sqlite:///heart_disease_app.db
FLASK_ENV=development
FLASK_DEBUG=True
```

#### Production
```env
SECRET_KEY=your-production-secret-key
DATABASE_URL=mssql+pyodbc://user:pass@server/db?driver=ODBC+Driver+18
FLASK_ENV=production
FLASK_DEBUG=False
SESSION_COOKIE_SECURE=True
```

---

## Troubleshooting

### Common Issues

#### 1. Database Connection Errors
**Symptoms**: "Connection failed" or "Driver not found"
**Solutions**:
- Verify DATABASE_URL format
- Install ODBC drivers
- Check firewall settings
- Validate credentials

#### 2. Model Loading Errors
**Symptoms**: "Model file not found"
**Solutions**:
- Verify model file exists
- Check MODEL_PATH environment variable
- Ensure file permissions

#### 3. Authentication Issues
**Symptoms**: "Invalid credentials" or login failures
**Solutions**:
- Reset admin password
- Check user table in database
- Verify password hashing

#### 4. Deployment Failures
**Symptoms**: Build errors or runtime crashes
**Solutions**:
- Check Dockerfile syntax
- Verify requirements.txt
- Review deployment logs
- Test locally first

### Debugging Tools

#### Environment Debugging
```bash
python debug_env.py
```

#### Database Testing
```bash
python test_azure_connection.py
python test_database_comprehensive.py
```

#### Docker Testing
```bash
python test_docker.py
```

### Performance Optimization

#### Database Optimization
- Use connection pooling
- Implement query optimization
- Add database indexes
- Monitor query performance

#### Application Optimization
- Enable caching
- Optimize static files
- Use CDN for assets
- Implement lazy loading

### Security Best Practices

#### Production Security
- Use strong SECRET_KEY
- Enable HTTPS
- Implement rate limiting
- Regular security updates
- Database backup strategy

#### Data Protection
- Encrypt sensitive data
- Implement access logging
- Regular security audits
- Compliance with healthcare regulations

---

## Support and Maintenance

### Regular Maintenance Tasks
- Update dependencies monthly
- Monitor application performance
- Review security patches
- Backup database regularly
- Check error logs

### Monitoring
- Application health checks
- Database performance monitoring
- User activity tracking
- Error rate monitoring

### Updates and Upgrades
- Test thoroughly before deployment
- Use staging environment
- Implement rollback strategy
- Update documentation

---

**Note**: This documentation should be updated regularly as the project evolves. For the latest information, always refer to the current codebase and deployment guides. 