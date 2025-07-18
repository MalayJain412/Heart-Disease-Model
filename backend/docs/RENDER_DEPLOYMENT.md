# Render Deployment Guide

## Prerequisites

1. **Azure SQL Database** - Already configured ✅
2. **GitHub Repository** - Your code should be in a GitHub repo
3. **Render Account** - Sign up at [render.com](https://render.com)

## Deployment Steps

### 1. Prepare Your Repository

Make sure your repository contains the following organized structure:

```
Heart-Disease-Model/
├── 📁 Production Files (Root Directory)
│   ├── app.py                          # Main Flask application
│   ├── config.py                       # Configuration management
│   ├── requirements.txt                # Python dependencies
│   ├── startup.py                      # Application startup script
│   ├── Dockerfile                      # Production-ready Docker configuration
│   ├── .dockerignore                   # Optimizes Docker build
│   └── render.yaml                     # Render deployment configuration
├── 📁 data/                            # Data Files
│   └── heart_disease_rf_model.pkl      # ML model file
├── 📁 scripts/                         # Setup scripts
├── 📁 tests/                           # Testing files
└── 📁 docs/                            # Documentation
```

**Required Files:**
- ✅ `app.py` - Main Flask application
- ✅ `requirements.txt` - Python dependencies
- ✅ `Dockerfile` - Production-ready Docker configuration
- ✅ `.dockerignore` - Optimizes Docker build
- ✅ `data/heart_disease_rf_model.pkl` - ML model file (updated path)
- ✅ `config.py` - Configuration management
- ✅ `startup.py` - Application startup verification

### 2. Create Render Web Service

1. **Go to Render Dashboard**
   - Visit [dashboard.render.com](https://dashboard.render.com)
   - Click "New +" → "Web Service"

2. **Connect GitHub Repository**
   - Connect your GitHub account
   - Select your repository: `Heart-Disease-Model`

3. **Configure Web Service**

   **Basic Settings:**
   - **Name**: `heart-disease-predictor` (or your preferred name)
   - **Environment**: `Docker`
   - **Region**: Choose closest to your users
   - **Branch**: `main` (or your default branch)

   **Advanced Settings:**
   - **Build Command**: Leave empty (Docker handles this)
   - **Start Command**: Leave empty (Docker handles this)

### 3. Set Environment Variables

In Render dashboard, go to your web service → "Environment" tab and add these variables:

```env
# Flask Configuration
FLASK_ENV=production
FLASK_DEBUG=False
SECRET_KEY=your-super-secret-production-key-change-this

# Azure SQL Database Configuration
DATABASE_URL=mssql+pyodbc://Malay:Greenscan%4012@greenscanserver.database.windows.net:1433/heart_disease_db?driver={ODBC Driver 18 for SQL Server}&Encrypt=yes&TrustServerCertificate=no&Connection Timeout=30

# Model Configuration (Updated Path)
MODEL_PATH=data/heart_disease_rf_model.pkl

# Security Settings
SESSION_COOKIE_SECURE=True
SESSION_COOKIE_HTTPONLY=True
SESSION_COOKIE_SAMESITE=Lax

# File Upload Settings
MAX_CONTENT_LENGTH=16777216
UPLOAD_FOLDER=uploads

# Logging
LOG_LEVEL=INFO

# Render Configuration
PORT=5000
```

### 4. Deploy

1. **Click "Create Web Service"**
2. **Wait for build** (5-10 minutes)
3. **Monitor logs** for any issues

### 5. Configure Azure SQL Database Firewall

Add Render's IP addresses to your Azure SQL Database firewall:

1. **Get Render IPs** from the deployment logs
2. **Go to Azure Portal** → SQL Database → Networking
3. **Add firewall rules** for Render's IP ranges

### 6. Test Deployment

1. **Visit your Render URL**: `https://your-app-name.onrender.com`
2. **Login with admin credentials**:
   - Username: `admin`
   - Password: `admin123`

## Pre-Deployment Testing

### Local Testing (Updated Commands)
```bash
# Test model loading with new path
python tests/test_model_loading.py

# Test database connection
python tests/test_azure_connection.py

# Test Docker build
python tests/test_docker.py

# Test startup verification
python startup.py

# Build and test Docker locally
docker build -t heart-disease-app .
docker run -p 5000:5000 heart-disease-app
```

### Environment Verification
```bash
# Debug environment variables
python scripts/debug_env.py

# Test configuration loading
python -c "from config import config; print(config['development'].MODEL_PATH)"
```

## Troubleshooting

### Common Issues

1. **Build Fails**
   - Check Dockerfile syntax
   - Verify all files are in repository
   - Check requirements.txt
   - Ensure model file is in `data/` directory

2. **Database Connection Fails**
   - Verify DATABASE_URL in environment variables
   - Check Azure SQL Database firewall rules
   - Ensure database is running
   - Run: `python tests/test_azure_connection.py`

3. **Model File Not Found** (Updated Path)
   - Verify `data/heart_disease_rf_model.pkl` is in repository
   - Check file path in MODEL_PATH: `data/heart_disease_rf_model.pkl`
   - Run: `python tests/test_model_loading.py`

4. **Startup Verification Fails**
   - Check startup logs for specific errors
   - Verify environment variables
   - Run: `python startup.py`

### Logs

- **Build Logs**: Check during deployment
- **Runtime Logs**: Available in Render dashboard
- **Application Logs**: Check Flask app logs
- **Startup Logs**: Look for startup verification messages

### Debugging Commands (Updated)
```bash
# Test model loading
python tests/test_model_loading.py

# Test database connection
python tests/test_azure_connection.py

# Debug environment
python scripts/debug_env.py

# Fix environment issues
python scripts/fix_env.py

# Update environment
python scripts/update_env.py
```

## Security Considerations

1. **Change SECRET_KEY** in production
2. **Use HTTPS** (Render provides automatically)
3. **Secure DATABASE_URL** (keep private)
4. **Regular backups** of Azure SQL Database
5. **Update admin password** after first login

## Monitoring

1. **Health Checks**: Dockerfile includes health check
2. **Logs**: Monitor in Render dashboard
3. **Performance**: Check Render metrics
4. **Startup Verification**: Monitor startup.py logs

## Scaling

- **Free Tier**: Limited resources
- **Paid Plans**: Better performance and uptime
- **Auto-scaling**: Available on paid plans

## Support

- **Render Documentation**: [docs.render.com](https://docs.render.com)
- **Azure SQL Documentation**: [docs.microsoft.com](https://docs.microsoft.com/en-us/azure/azure-sql/)
- **Flask Documentation**: [flask.palletsprojects.com](https://flask.palletsprojects.com/)

## Project Structure for Deployment

The project is organized with a production-first approach:

```
Heart-Disease-Model/
├── 📁 Production Files (Root)
│   ├── app.py                          # Main Flask application
│   ├── config.py                       # Configuration
│   ├── requirements.txt                # Dependencies
│   ├── Dockerfile                      # Production container
│   ├── docker-compose.yml              # Local development
│   ├── render.yaml                     # Render deployment
│   ├── startup.py                      # Startup verification
│   ├── env.example                     # Development template
│   ├── env.production                  # Production template
│   └── README.md                       # Main documentation
├── 📁 docs/                            # Documentation
├── 📁 scripts/                         # Setup scripts
├── 📁 tests/                           # Testing files
├── 📁 data/                            # ML model and data
└── 📁 app/                             # Flask templates and static files
```

### Key Deployment Files:
- **Root Directory**: Contains only production-essential files
- **Dockerfile**: Updated to reference `data/heart_disease_rf_model.pkl`
- **config.py**: Updated MODEL_PATH to data directory
- **startup.py**: Production startup verification
- **env.example**: Development environment template
- **env.production**: Production environment template
- **scripts/**: Database setup and utility scripts
- **tests/**: Comprehensive testing suite

## Next Steps

After successful deployment:

1. **Test all features** (login, add patients, predictions)
2. **Set up monitoring** and alerts
3. **Configure custom domain** (optional)
4. **Set up CI/CD** for automatic deployments
5. **Backup strategy** for database
6. **Update admin password** for security
7. **Configure user roles** as needed

This deployment guide has been updated to reflect the new organized directory structure with the model file located in the `data/` directory and updated paths throughout the documentation. 