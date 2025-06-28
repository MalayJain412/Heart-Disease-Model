# Deployment Guide

## Prerequisites

1. **Azure Account** - For Azure SQL Database
2. **Render Account** - For hosting the application
3. **Docker** - For containerization
4. **Git** - For version control

## Step 1: Set up Azure SQL Database

### 1.1 Create Azure SQL Database
1. Go to Azure Portal (https://portal.azure.com)
2. Create a new SQL Database
3. Choose your subscription and resource group
4. Set database name: `heart_disease_db`
5. Choose server or create new one
6. Set authentication method to "SQL authentication"
7. Set admin username and password
8. Choose pricing tier (Basic is fine for testing)

### 1.2 Configure Firewall
1. In your SQL Database, go to "Networking"
2. Add your IP address to firewall rules
3. Enable "Allow Azure services and resources to access this server"

### 1.3 Get Connection String
1. Go to "Connection strings" in your database
2. Copy the ADO.NET connection string
3. Replace placeholders with your actual values:
   ```
   Server=tcp:your-server.database.windows.net,1433;Initial Catalog=heart_disease_db;Persist Security Info=False;User ID=your-username;Password=your-password;MultipleActiveResultSets=False;Encrypt=True;TrustServerCertificate=False;Connection Timeout=30;
   ```

## Step 2: Prepare for Render Deployment

### 2.1 Update Environment Variables
Create a `.env` file for production:
```env
FLASK_ENV=production
SECRET_KEY=your-super-secret-production-key
DATABASE_URL=mssql+pyodbc://username:password@server.database.windows.net:1433/heart_disease_db?driver=ODBC+Driver+18+for+SQL+Server
MODEL_PATH=data/heart_disease_rf_model.pkl
SESSION_COOKIE_SECURE=true
LOG_LEVEL=INFO
```

### 2.2 Test Docker Build Locally
```bash
# Build the Docker image
docker build -t heart-disease-app .

# Test locally with docker-compose
docker-compose up --build
```

## Step 3: Deploy to Render

### 3.1 Connect Repository
1. Go to Render Dashboard
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Choose the repository with your Flask app

### 3.2 Configure Service
1. **Name**: `heart-disease-prediction`
2. **Environment**: `Docker`
3. **Region**: Choose closest to your users
4. **Branch**: `main` (or your default branch)
5. **Build Command**: `docker build -t heart-disease-app .`
6. **Start Command**: `gunicorn --bind 0.0.0.0:$PORT app:app`

### 3.3 Set Environment Variables
In Render dashboard, add these environment variables:
- `FLASK_ENV`: `production`
- `SECRET_KEY`: Generate a secure random key
- `DATABASE_URL`: Your Azure SQL connection string
- `MODEL_PATH`: `data/heart_disease_rf_model.pkl`
- `SESSION_COOKIE_SECURE`: `true`
- `LOG_LEVEL`: `INFO`

### 3.4 Deploy
1. Click "Create Web Service"
2. Render will automatically build and deploy your app
3. Monitor the build logs for any issues

## Step 4: Post-Deployment

### 4.1 Test the Application
1. Visit your Render URL
2. Test login with default admin credentials:
   - Username: `admin`
   - Password: `admin123`
3. Test all functionality (add patients, view analytics, etc.)

### 4.2 Set up Custom Domain (Optional)
1. In Render dashboard, go to "Settings"
2. Add your custom domain
3. Configure DNS records as instructed

### 4.3 Monitor and Scale
1. Monitor application logs in Render dashboard
2. Set up alerts for errors
3. Scale up if needed (upgrade plan)

## Pre-Deployment Testing

### 4.1 Test Database Setup
```bash
# Test Azure database connection
python scripts/setup_azure_database.py

# Test database functionality
python tests/test_database_comprehensive.py
```

### 4.2 Test Model Loading
```bash
# Test model loading
python tests/test_model_loading.py

# Test Azure connection
python tests/test_azure_connection.py
```

### 4.3 Test Docker Build
```bash
# Test Docker build process
python tests/test_docker.py

# Build and test locally
docker build -t heart-disease-app .
docker run -p 5000:5000 heart-disease-app
```

### 4.4 Test Admin User Creation
```bash
# Create admin user
python scripts/add_admin_user.py

# Test admin credentials
python tests/test_admin_credentials.py
```

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Check Azure SQL firewall settings
   - Verify connection string format
   - Ensure ODBC driver is installed in Docker
   - Run: `python scripts/debug_env.py`

2. **Build Failures**
   - Check Dockerfile syntax
   - Verify all dependencies in requirements.txt
   - Check build logs in Render
   - Test locally: `docker build -t heart-disease-app .`

3. **Runtime Errors**
   - Check application logs in Render
   - Verify environment variables
   - Test locally with Docker
   - Run startup verification: `python startup.py`

4. **Model Loading Issues**
   - Verify model file exists in `data/` directory
   - Check MODEL_PATH environment variable
   - Run model test: `python tests/test_model_loading.py`

### Useful Commands

```bash
# Test Docker build locally
docker build -t heart-disease-app .

# Run locally with Docker
docker run -p 5000:5000 -e DATABASE_URL="your-connection-string" heart-disease-app

# Check Docker logs
docker logs <container-id>

# Test database connection
python -c "from app import db; db.create_all()"

# Debug environment variables
python scripts/debug_env.py

# Fix environment file issues
python scripts/fix_env.py

# Update environment file
python scripts/update_env.py
```

## Security Considerations

1. **Environment Variables**: Never commit sensitive data to Git
2. **Database Security**: Use Azure SQL firewall rules
3. **HTTPS**: Render provides SSL certificates automatically
4. **Session Security**: Use secure cookies in production
5. **Input Validation**: Validate all user inputs
6. **SQL Injection**: Use SQLAlchemy ORM (already implemented)

## Cost Optimization

1. **Azure SQL**: Start with Basic tier, scale as needed
2. **Render**: Use free tier for testing, upgrade for production
3. **Monitoring**: Set up alerts to avoid unexpected costs

## Backup Strategy

1. **Database**: Azure SQL provides automatic backups
2. **Application**: Use Git for version control
3. **Data**: Export patient data regularly via the app's export features

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
- **scripts/**: Database setup and utility scripts
- **tests/**: Comprehensive testing suite 