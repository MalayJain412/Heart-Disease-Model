# Heart Disease Prediction System

A comprehensive Flask web application for predicting cardiovascular diseases using machine learning. The system features role-based authentication, patient management, analytics, and export capabilities.

## 🚀 Features

### 🔐 Authentication & User Management
- **Role-based Access Control**: Three user roles (Doctor, Dean, Chairman)
- **Secure Login/Logout**: Flask-Login integration
- **Password Management**: Change password functionality
- **User Registration**: Chairmen can add Doctors/Deans, Deans can add Doctors

### 🧠 Heart Disease Prediction
- **ML Model Integration**: Pre-trained Random Forest model (`heart_disease_rf_model.pkl`)
- **Comprehensive Input Form**: 30+ clinical parameters including:
  - Vital signs (BP, Heart Rate, Age)
  - Cholesterol levels (Total, LDL, HDL)
  - Symptoms (Chest Pain, Shortness of Breath, Fatigue)
  - Medical conditions (Diabetes, Hypertension, etc.)
  - Lifestyle factors (Smoking, Obesity, Alcoholism)

### 📊 Analytics Dashboard
- **Prediction Distribution**: Pie charts showing disease vs normal cases
- **Doctor Performance**: Bar charts showing patients per doctor
- **Real-time Statistics**: Total patients, normal cases, disease cases
- **Interactive Charts**: Chart.js integration for dynamic visualizations

### 📤 Export Capabilities
- **Excel Export**: Complete patient records in .xlsx format
- **PDF Export**: Formatted patient reports in PDF
- **Filtered Exports**: Filter by date, doctor, or disease type

### 🎨 Modern UI/UX
- **Bootstrap 5**: Responsive, mobile-friendly design
- **Medical Theme**: Professional blue/white color scheme
- **Google Fonts**: Poppins and Open Sans typography
- **Font Awesome**: Rich iconography throughout
- **Accessibility**: ARIA labels and semantic HTML

## 🛠️ Technology Stack

- **Backend**: Flask, SQLAlchemy, Flask-Login
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5, Chart.js
- **Database**: SQLite (dev), Azure SQL (prod)
- **ML**: scikit-learn, pandas, numpy
- **Export**: openpyxl, reportlab
- **Deployment**: Azure App Service

## 📋 Prerequisites

- Python 3.8+
- pip (Python package installer)
- Git

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Heart-Disease-Model
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
```bash
# Copy environment example
cp env.example .env

# Edit .env file with your configuration
# SECRET_KEY=your-secret-key
# DATABASE_URL=sqlite:///heart_disease_app.db
```

### 5. Initialize Database
```bash
python app.py
```
The application will automatically create the database and a default chairman account:
- **Username**: `admin`
- **Password**: `admin123`

### 6. Run the Application
```bash
python app.py
```

Access the application at: `http://localhost:5000`

## 🏥 User Roles & Permissions

### 👨‍⚕️ Doctor
- Add new patients and make predictions
- View only their own patients
- Change password
- Access to comprehensive patient form

### 👨‍🏫 Dean
- View all patients across all doctors
- Register new doctors
- Access analytics dashboard
- Export patient data (Excel/PDF)
- Filter patients by date, doctor, disease

### 👨‍💼 Chairman
- Full system access
- Register doctors and deans
- View all users and patients
- Access analytics and exports
- User management capabilities

## 📊 Database Schema

### User Table
- `id` (Primary Key)
- `username` (Unique)
- `email` (Unique)
- `password_hash`
- `role` (doctor/dean/chairman)
- `created_at`

### Patient Table
- `id` (Primary Key)
- `name`, `gender`, `age`
- All clinical parameters (30+ fields)
- `prediction_result`
- `doctor_id` (Foreign Key)
- `timestamp`

## 🚀 Deployment to Azure App Service

### 1. Prepare for Deployment

#### Update requirements.txt for Azure
```txt
Flask==2.3.3
Flask-SQLAlchemy==3.0.5
Flask-Login==0.6.3
Werkzeug==2.3.7
pandas==2.2.1
numpy==1.26.4
scikit-learn==1.4.2
openpyxl==3.1.2
reportlab==4.0.4
matplotlib==3.8.2
seaborn==0.13.0
python-dotenv==1.0.0
gunicorn==21.2.0
```

#### Create startup file
Create `startup.txt`:
```txt
gunicorn --bind=0.0.0.0 --timeout 600 app:app
```

### 2. Azure App Service Setup

#### Create App Service
1. Go to Azure Portal
2. Create new App Service
3. Choose Python runtime (3.8+)
4. Select appropriate region

#### Configure Environment Variables
In Azure App Service Configuration:
```
SECRET_KEY=your-production-secret-key
FLASK_ENV=production
DATABASE_URL=mssql+pyodbc://username:password@server.database.windows.net:1433/database?driver=ODBC+Driver+17+for+SQL+Server
MODEL_PATH=heart_disease_rf_model.pkl
SESSION_COOKIE_SECURE=True
```

#### Azure SQL Database Setup
1. Create Azure SQL Database
2. Configure firewall rules
3. Create database user
4. Update connection string in environment variables

### 3. Deploy Application

#### Using Azure CLI
```bash
# Login to Azure
az login

# Set subscription
az account set --subscription <subscription-id>

# Deploy to App Service
az webapp up --name <app-name> --resource-group <resource-group> --runtime "PYTHON|3.8"
```

#### Using GitHub Actions
Create `.github/workflows/deploy.yml`:
```yaml
name: Deploy to Azure
on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Deploy to Azure Web App
      uses: azure/webapps-deploy@v2
      with:
        app-name: 'your-app-name'
        publish-profile: ${{ secrets.AZURE_WEBAPP_PUBLISH_PROFILE }}
```

### 4. Post-Deployment

1. **Update Default Credentials**: Change admin password after first login
2. **Configure Custom Domain**: Set up SSL certificate
3. **Monitor Application**: Use Azure Application Insights
4. **Backup Database**: Set up automated backups

## 🔧 Configuration

### Development
```python
# config.py
class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///heart_disease_app.db'
```

### Production
```python
# config.py
class ProductionConfig(Config):
    DEBUG = False
    SESSION_COOKIE_SECURE = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
```

## 📁 Project Structure

```
Heart-Disease-Model/
├── app.py                 # Main Flask application
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── env.example          # Environment variables template
├── heart_disease_rf_model.pkl  # Pre-trained ML model
├── app/
│   ├── templates/       # HTML templates
│   │   ├── auth/       # Authentication templates
│   │   ├── doctor/     # Doctor dashboard templates
│   │   ├── dean/       # Dean dashboard templates
│   │   └── chairman/   # Chairman dashboard templates
│   └── static/         # Static files (CSS, JS)
├── instance/           # Database files
└── README.md          # This file
```

## 🔒 Security Features

- **Password Hashing**: Werkzeug security for password protection
- **Session Management**: Secure session handling with Flask-Login
- **CSRF Protection**: Built-in CSRF protection
- **Input Validation**: Form validation and sanitization
- **Role-based Access**: Strict permission controls
- **Secure Headers**: Security headers for production

## 🧪 Testing

```bash
# Run tests (if implemented)
python -m pytest tests/

# Run with coverage
python -m pytest --cov=app tests/
```

## 📈 Monitoring & Logging

### Application Logs
```python
import logging
logging.basicConfig(level=logging.INFO)
```

### Azure Application Insights
Add to `app.py`:
```python
from opencensus.ext.azure.log_exporter import AzureLogHandler
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the documentation

## 🔄 Updates & Maintenance

### Regular Maintenance Tasks
- Update dependencies monthly
- Monitor Azure costs
- Review security patches
- Backup database regularly
- Monitor application performance

### Version Updates
- Test thoroughly before deployment
- Update requirements.txt
- Migrate database if needed
- Update documentation

---

**Note**: This application is designed for medical professionals and should be used in accordance with healthcare regulations and data protection laws.