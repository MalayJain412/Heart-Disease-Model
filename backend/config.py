import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base configuration class"""
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

    @staticmethod
    def get_engine_options():
        db_url = os.environ.get('DATABASE_URL', '')
        if db_url.startswith('mysql'):
            return {
                'pool_pre_ping': True,
                'pool_recycle': 300,
                'pool_size': 5,
                'max_overflow': 10,
                'connect_args': {
                    'connect_timeout': 30
                }
            }
        else:  # Assume Azure SQL/pyodbc
            return {
                'pool_pre_ping': True,
                'pool_recycle': 300,
                'pool_size': 5,
                'max_overflow': 10,
                'connect_args': {
                    'timeout': 30,
                    'autocommit': True
                }
            }

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    
    # Azure SQL specific settings for development
    SQLALCHEMY_ENGINE_OPTIONS = Config.get_engine_options()

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SESSION_COOKIE_SECURE = True
    
    # Azure SQL specific settings for production
    SQLALCHEMY_ENGINE_OPTIONS = Config.get_engine_options()
    
    # Azure App Service specific settings
    WEBSITES_PORT = os.environ.get('WEBSITES_PORT', '5000')

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
} 