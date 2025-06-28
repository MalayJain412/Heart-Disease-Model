# Heart Disease Prediction System - Project Structure

## Complete File Structure

```
Heart-Disease-Model/
├── 📁 Core Application Files
│   ├── app.py                          # Main Flask application (23KB, 588 lines)
│   ├── config.py                       # Configuration management (2.2KB, 80 lines)
│   ├── requirements.txt                # Python dependencies (479B, 20 lines)
│   └── streamlit_app.py                # Alternative Streamlit interface (4.4KB, 121 lines)
│
├── 📁 Database & ML Files
│   ├── heart_disease_rf_model.pkl      # Pre-trained Random Forest model (2.9MB)
│   ├── Heart D Dataset.csv             # Training dataset (129KB, 1002 lines)
│   └── instance/
│       └── heart_disease_app.db        # Local SQLite database
│
├── 📁 Environment & Configuration
│   ├── env.example                     # Environment variables template (707B, 25 lines)
│   ├── env.production                  # Production environment config (784B, 28 lines)
│   ├── .dockerignore                   # Docker ignore patterns (788B, 78 lines)
│   └── .devcontainer/                  # VS Code dev container config
│
├── 📁 Docker & Deployment
│   ├── Dockerfile                      # Production Docker configuration (1.8KB, 65 lines)
│   ├── Dockerfile.alternative          # Alternative Docker setup (2.0KB, 70 lines)
│   ├── Dockerfile.dev                  # Development Docker setup (783B, 37 lines)
│   ├── docker-compose.yml              # Docker Compose configuration (673B, 32 lines)
│   ├── render.yaml                     # Render deployment config (590B, 22 lines)
│   └── mount/
│       └── src/
│           └── heart_disease_rf_model.pkl  # Model for Docker mounting
│
├── 📁 Helper Scripts
│   ├── add_admin_user.py               # Add default admin user (3.7KB, 104 lines)
│   ├── setup_azure_database.py         # Azure SQL setup script (6.0KB, 140 lines)
│   ├── debug_env.py                    # Environment debugging tool (2.5KB, 67 lines)
│   ├── update_env.py                   # Environment file updater (1.7KB, 59 lines)
│   ├── fix_env.py                      # Environment file fixer (2.1KB, 62 lines)
│   ├── coopy_utils.py                  # Utility functions (1.3KB, 33 lines)
│   └── all_code_snippets.txt           # Code documentation (58KB, 1313 lines)
│
├── 📁 Testing Scripts
│   ├── test_azure_connection.py        # Azure connection testing (7.1KB, 151 lines)
│   ├── test_admin_credentials.py       # Admin user testing (5.0KB, 105 lines)
│   ├── test_database_comprehensive.py  # Comprehensive DB tests (18KB, 430 lines)
│   ├── test_azure_db.py                # Azure database testing (2.3KB, 58 lines)
│   └── test_docker.py                  # Docker testing (4.3KB, 166 lines)
│
├── 📁 Documentation
│   ├── README.md                       # Main project documentation (9.0KB, 345 lines)
│   ├── DEPLOYMENT.md                   # Deployment guide (4.9KB, 160 lines)
│   ├── RENDER_DEPLOYMENT.md            # Render-specific deployment (4.3KB, 153 lines)
│   ├── USER_DOCUMENTATION.md           # End-user guide
│   ├── DEVELOPER_DOCUMENTATION.md      # Developer technical guide
│   └── PROJECT_STRUCTURE.md            # This file
│
├── 📁 Application Structure (app/)
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
│
└── 📁 Miscellaneous
    ├── -p/                             # Temporary directory
    └── __pycache__/                    # Python cache files
```

## File Purposes and Functions

### 🚀 Core Application Files

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
- **Size**: 2.2KB, 80 lines
- **Critical**: Yes - Application configuration

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

#### `streamlit_app.py` (Alternative Interface)
- **Purpose**: Streamlit-based alternative interface
- **Key Functions**:
  - Simplified patient input form
  - Real-time prediction display
  - Interactive data visualization
- **Size**: 4.4KB, 121 lines
- **Critical**: No - Alternative interface

### 🗄️ Database & ML Files

#### `heart_disease_rf_model.pkl` (ML Model)
- **Purpose**: Pre-trained Random Forest model
- **Key Features**:
  - Trained on heart disease dataset
  - 30+ clinical parameters
  - Binary classification (Normal/Disease)
- **Size**: 2.9MB
- **Critical**: Yes - Core ML functionality

#### `Heart D Dataset.csv` (Training Data)
- **Purpose**: Original training dataset
- **Key Features**:
  - 1002 patient records
  - 30+ clinical parameters
  - Used for model training
- **Size**: 129KB, 1002 lines
- **Critical**: No - Reference data only

#### `instance/heart_disease_app.db` (Local Database)
- **Purpose**: Local SQLite database for development
- **Key Features**:
  - User and patient tables
  - Local development data
- **Size**: Variable
- **Critical**: No - Development only

### ⚙️ Environment & Configuration

#### `env.example` (Environment Template)
- **Purpose**: Template for environment variables
- **Key Contents**:
  - Database connection strings
  - Security keys
  - Configuration flags
- **Size**: 707B, 25 lines
- **Critical**: Yes - Setup reference

#### `env.production` (Production Config)
- **Purpose**: Production environment configuration
- **Key Contents**:
  - Production database settings
  - Security configurations
  - Performance settings
- **Size**: 784B, 28 lines
- **Critical**: Yes - Production deployment

#### `.dockerignore` (Docker Ignore)
- **Purpose**: Files to exclude from Docker builds
- **Key Contents**:
  - Development files
  - Cache directories
  - Documentation files
- **Size**: 788B, 78 lines
- **Critical**: Yes - Docker optimization

### 🐳 Docker & Deployment

#### `Dockerfile` (Production Docker)
- **Purpose**: Production Docker container configuration
- **Key Features**:
  - Python 3.11 slim base
  - ODBC Driver 18 installation
  - Non-root user for security
  - Gunicorn production server
- **Size**: 1.8KB, 65 lines
- **Critical**: Yes - Production deployment

#### `Dockerfile.alternative` (Alternative Docker)
- **Purpose**: Alternative Docker configuration
- **Key Features**:
  - Different base image
  - Alternative package installation
  - Different security setup
- **Size**: 2.0KB, 70 lines
- **Critical**: No - Alternative option

#### `Dockerfile.dev` (Development Docker)
- **Purpose**: Development Docker configuration
- **Key Features**:
  - Development-friendly setup
  - Volume mounting for live code changes
  - Debug mode enabled
- **Size**: 783B, 37 lines
- **Critical**: No - Development only

#### `docker-compose.yml` (Docker Compose)
- **Purpose**: Multi-container Docker setup
- **Key Features**:
  - Web service configuration
  - Environment variable management
  - Volume mounting
- **Size**: 673B, 32 lines
- **Critical**: Yes - Local development

#### `render.yaml` (Render Deployment)
- **Purpose**: Render cloud deployment configuration
- **Key Features**:
  - Service definition
  - Environment variables
  - Build configuration
- **Size**: 590B, 22 lines
- **Critical**: Yes - Render deployment

### 🛠️ Helper Scripts

#### `add_admin_user.py` (Admin User Setup)
- **Purpose**: Add default admin user to database
- **Key Functions**:
  - Create admin user with hashed password
  - Verify user creation
  - Database connection testing
- **Size**: 3.7KB, 104 lines
- **Critical**: Yes - Initial setup

#### `setup_azure_database.py` (Azure DB Setup)
- **Purpose**: Azure SQL Database initialization
- **Key Functions**:
  - Database creation
  - User setup
  - Permission configuration
  - Connection testing
- **Size**: 6.0KB, 140 lines
- **Critical**: Yes - Azure setup

#### `debug_env.py` (Environment Debugging)
- **Purpose**: Debug environment configuration
- **Key Functions**:
  - Environment variable checking
  - Database connection testing
  - Configuration validation
- **Size**: 2.5KB, 67 lines
- **Critical**: No - Debugging tool

#### `update_env.py` (Environment Updater)
- **Purpose**: Update environment file
- **Key Functions**:
  - Environment variable updates
  - Configuration file management
- **Size**: 1.7KB, 59 lines
- **Critical**: No - Utility script

#### `fix_env.py` (Environment Fixer)
- **Purpose**: Fix common environment issues
- **Key Functions**:
  - Connection string formatting
  - Configuration validation
  - Error correction
- **Size**: 2.1KB, 62 lines
- **Critical**: No - Utility script

#### `coopy_utils.py` (Utility Functions)
- **Purpose**: General utility functions
- **Key Functions**:
  - Helper functions
  - Common operations
- **Size**: 1.3KB, 33 lines
- **Critical**: No - Utility functions

### 🧪 Testing Scripts

#### `test_azure_connection.py` (Azure Connection Test)
- **Purpose**: Test Azure SQL Database connection
- **Key Functions**:
  - Connection string validation
  - Database connectivity testing
  - Error reporting
- **Size**: 7.1KB, 151 lines
- **Critical**: Yes - Connection validation

#### `test_admin_credentials.py` (Admin Credentials Test)
- **Purpose**: Test admin user authentication
- **Key Functions**:
  - Password verification
  - Login testing
  - User validation
- **Size**: 5.0KB, 105 lines
- **Critical**: Yes - Authentication testing

#### `test_database_comprehensive.py` (Comprehensive DB Test)
- **Purpose**: Comprehensive database testing
- **Key Functions**:
  - Schema validation
  - Data integrity testing
  - Performance testing
  - User and patient operations
- **Size**: 18KB, 430 lines
- **Critical**: Yes - Database validation

#### `test_azure_db.py` (Azure DB Test)
- **Purpose**: Azure database specific testing
- **Key Functions**:
  - Azure-specific features testing
  - Connection pooling validation
  - Performance testing
- **Size**: 2.3KB, 58 lines
- **Critical**: No - Azure-specific testing

#### `test_docker.py` (Docker Test)
- **Purpose**: Docker container testing
- **Key Functions**:
  - Container build testing
  - Runtime testing
  - Environment validation
- **Size**: 4.3KB, 166 lines
- **Critical**: No - Docker testing

### 📚 Documentation

#### `README.md` (Main Documentation)
- **Purpose**: Comprehensive project overview
- **Key Contents**:
  - Feature descriptions
  - Installation guide
  - User roles and permissions
  - Deployment instructions
- **Size**: 9.0KB, 345 lines
- **Critical**: Yes - Project overview

#### `DEPLOYMENT.md` (Deployment Guide)
- **Purpose**: Detailed deployment instructions
- **Key Contents**:
  - Azure App Service deployment
  - Environment configuration
  - Post-deployment steps
- **Size**: 4.9KB, 160 lines
- **Critical**: Yes - Deployment guide

#### `RENDER_DEPLOYMENT.md` (Render Deployment)
- **Purpose**: Render-specific deployment guide
- **Key Contents**:
  - Render setup instructions
  - Environment variables
  - Troubleshooting
- **Size**: 4.3KB, 153 lines
- **Critical**: Yes - Render deployment

### 🎨 Application Structure (app/)

#### `app/templates/base.html` (Base Template)
- **Purpose**: Base HTML template for all pages
- **Key Features**:
  - Bootstrap 5 integration
  - Medical theme styling
  - Navigation structure
  - Common layout elements
- **Size**: 7.6KB, 224 lines
- **Critical**: Yes - Template foundation

#### Authentication Templates (`app/templates/auth/`)
- **`login.html`**: User login interface
- **`change_password.html`**: Password change form
- **Purpose**: User authentication interfaces
- **Critical**: Yes - Authentication

#### Doctor Templates (`app/templates/doctor/`)
- **`dashboard.html`**: Doctor's main dashboard
- **`add_patient.html`**: Comprehensive patient input form
- **Purpose**: Doctor role interfaces
- **Critical**: Yes - Core functionality

#### Dean Templates (`app/templates/dean/`)
- **`dashboard.html`**: Dean's overview dashboard
- **`analytics.html`**: Analytics and charts
- **`register_doctor.html`**: Doctor registration form
- **Purpose**: Dean role interfaces
- **Critical**: Yes - Management features

#### Chairman Templates (`app/templates/chairman/`)
- **`dashboard.html`**: Chairman's system overview
- **`register_user.html`**: User registration form
- **Purpose**: Chairman role interfaces
- **Critical**: Yes - Administration

#### Static Assets (`app/static/`)
- **`css/`**: Custom stylesheets
- **`js/`**: JavaScript files
- **Purpose**: Frontend assets
- **Critical**: Yes - UI/UX

## File Dependencies and Relationships

### Core Dependencies
```
app.py
├── config.py (Configuration)
├── requirements.txt (Dependencies)
├── heart_disease_rf_model.pkl (ML Model)
└── app/templates/ (Templates)
    ├── base.html (Base template)
    └── role-specific templates
```

### Database Dependencies
```
Database Operations
├── setup_azure_database.py (Initial setup)
├── add_admin_user.py (User creation)
├── test_azure_connection.py (Connection testing)
└── test_database_comprehensive.py (Validation)
```

### Deployment Dependencies
```
Production Deployment
├── Dockerfile (Container)
├── requirements.txt (Dependencies)
├── env.production (Configuration)
└── render.yaml (Platform config)
```

### Development Dependencies
```
Development Environment
├── docker-compose.yml (Local setup)
├── Dockerfile.dev (Dev container)
├── env.example (Config template)
└── debug_env.py (Debugging)
```

## Critical vs Non-Critical Files

### 🔴 Critical Files (Required for Production)
- `app.py` - Main application
- `config.py` - Configuration
- `requirements.txt` - Dependencies
- `heart_disease_rf_model.pkl` - ML model
- `Dockerfile` - Production container
- `env.production` - Production config
- `render.yaml` - Deployment config
- `app/templates/` - All templates
- `setup_azure_database.py` - Database setup
- `add_admin_user.py` - User setup

### 🟡 Important Files (Development/Testing)
- `test_*.py` - Testing scripts
- `debug_env.py` - Debugging tools
- `docker-compose.yml` - Local development
- `Dockerfile.dev` - Development container
- `env.example` - Configuration template

### 🟢 Optional Files (Utilities/Alternatives)
- `streamlit_app.py` - Alternative interface
- `Dockerfile.alternative` - Alternative Docker
- `coopy_utils.py` - Utility functions
- `Heart D Dataset.csv` - Reference data
- `all_code_snippets.txt` - Code documentation

## File Size Analysis

### Large Files (>1MB)
- `heart_disease_rf_model.pkl` (2.9MB) - ML model
- `all_code_snippets.txt` (58KB) - Documentation

### Medium Files (1KB-10KB)
- `app.py` (23KB) - Main application
- `test_database_comprehensive.py` (18KB) - Testing
- `app/templates/doctor/add_patient.html` (19KB) - Patient form
- `Heart D Dataset.csv` (129KB) - Training data

### Small Files (<1KB)
- Most configuration files
- Helper scripts
- Template files
- Documentation files

## Security Considerations

### Sensitive Files
- `env.production` - Contains production secrets
- `instance/heart_disease_app.db` - Local database
- `.env` files - Environment variables

### Public Files
- `env.example` - Template only
- `requirements.txt` - Dependencies
- `README.md` - Documentation
- Templates - HTML files

### Access Control
- Database files should be secured
- Environment files should be private
- ML model should be protected
- Logs should be secured

## Maintenance and Updates

### Regular Updates
- `requirements.txt` - Dependency updates
- `Dockerfile` - Base image updates
- Documentation files - Keep current

### Version Control
- Track all source code changes
- Ignore sensitive files (.env, database files)
- Document major changes
- Tag releases

### Backup Strategy
- Database backups (Azure SQL)
- Configuration backups
- ML model backups
- Documentation backups

---

**Note**: This structure documentation should be updated whenever files are added, removed, or modified. The criticality ratings help prioritize maintenance and deployment efforts. 