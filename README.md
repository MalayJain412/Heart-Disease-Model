# Heart Disease Prediction System

A comprehensive Flask-based web application for heart disease prediction using machine learning, with role-based access control and Azure SQL Database integration.

## 📁 Project Structure

```
Heart-Disease-Model/
├── docs/                           # Documentation files
│   ├── DEVELOPER_DOCUMENTATION.md
│   ├── DEPLOYMENT.md
│   ├── USER_DOCUMENTATION.md
│   ├── PROJECT_STRUCTURE.md
│   ├── MODEL_LOADING_TROUBLESHOOTING.md
│   ├── RENDER_DEPLOYMENT.md
│   └── PROJECT_DOCUMENTATION.md
├── scripts/                        # Setup and utility scripts
│   ├── add_admin_user.py
│   ├── setup_azure_database.py
│   ├── debug_env.py
│   ├── update_env.py
│   ├── fix_env.py
│   ├── streamlit_app.py
│   ├── coopy_utils.py
│   └── all_code_snippets.txt
├── tests/                          # Testing files
│   ├── test_model_loading.py
│   ├── test_azure_connection.py
│   ├── test_admin_credentials.py
│   ├── test_database_comprehensive.py
│   ├── test_docker.py
│   └── test_azure_db.py
├── data/                           # Data files
│   ├── heart_disease_rf_model.pkl
│   └── Heart D Dataset.csv
├── app/                            # Flask application
│   ├── templates/
│   └── static/
├── instance/                       # Database instance
├── mount/                          # Mount directory
├── app.py                          # Main Flask application
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Production Docker configuration
├── docker-compose.yml              # Docker Compose configuration
├── render.yaml                     # Render deployment configuration
├── startup.py                      # Application startup script
├── env.example                     # Environment variables template
├── env.production                  # Production environment variables
├── .dockerignore                   # Docker ignore file
└── README.md                       # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Azure SQL Database
- ODBC Driver 18 for SQL Server

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Heart-Disease-Model
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your Azure SQL Database credentials
   ```

4. **Set up the database**
   ```bash
   python scripts/setup_azure_database.py
   ```

5. **Add admin user**
   ```bash
   python scripts/add_admin_user.py
   ```

6. **Run the application**
   ```bash
   python app.py
   ```

## 🧪 Testing

Run tests from the project root:
```bash
# Test model loading
python tests/test_model_loading.py

# Test Azure connection
python tests/test_azure_connection.py

# Test database functionality
python tests/test_database_comprehensive.py

# Test Docker build
python tests/test_docker.py
```

## 🐳 Docker Deployment

### Build and run with Docker
```bash
docker build -t heart-disease-app .
docker run -p 5000:5000 heart-disease-app
```

### Using Docker Compose
```bash
docker-compose up -d
```

## 📚 Documentation

- **[User Documentation](docs/USER_DOCUMENTATION.md)** - Complete user guide
- **[Developer Documentation](docs/DEVELOPER_DOCUMENTATION.md)** - Developer guide
- **[Deployment Guide](docs/DEPLOYMENT.md)** - Deployment instructions
- **[Project Structure](docs/PROJECT_STRUCTURE.md)** - Detailed project structure

## 🔧 Development

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test
python tests/test_model_loading.py
```

### Debugging
```bash
# Debug environment variables
python scripts/debug_env.py

# Fix environment file
python scripts/fix_env.py

# Update environment file
python scripts/update_env.py
```

## ✨ Features

- **Role-based Access Control**: Doctor, Dean, and Chairman roles
- **Patient Management**: Add, view, and manage patient records
- **Heart Disease Prediction**: ML-powered risk assessment
- **Analytics Dashboard**: Comprehensive data visualization
- **Export Functionality**: Excel and PDF report generation
- **Azure SQL Integration**: Scalable cloud database
- **Docker Support**: Containerized deployment
- **Security**: Password hashing, session management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For support and questions:
- Check the [documentation](docs/)
- Review [troubleshooting guide](docs/MODEL_LOADING_TROUBLESHOOTING.md)
- Open an issue on GitHub