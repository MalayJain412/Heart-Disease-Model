import os
from dotenv import load_dotenv

print("🔍 Debugging Environment Variables")
print("=" * 40)

# Check if .env file exists
env_file_path = '.env'
if os.path.exists(env_file_path):
    print(f"✅ .env file found at: {os.path.abspath(env_file_path)}")
else:
    print(f"❌ .env file not found at: {os.path.abspath(env_file_path)}")

# Load environment variables
load_dotenv()

# Check DATABASE_URL
database_url = os.environ.get('DATABASE_URL')
print(f"\nDATABASE_URL: {database_url}")

if database_url:
    if 'mssql' in database_url:
        print("✅ DATABASE_URL contains 'mssql' - should use Azure SQL")
    elif 'mysql' in database_url:
        print("❌ DATABASE_URL contains 'mysql' - this is the problem!")
    else:
        print(f"⚠️ DATABASE_URL contains neither 'mssql' nor 'mysql': {database_url}")
else:
    print("❌ DATABASE_URL is not set")

# Check other environment variables
print(f"\nFLASK_ENV: {os.environ.get('FLASK_ENV')}")
print(f"FLASK_DEBUG: {os.environ.get('FLASK_DEBUG')}")
print(f"SECRET_KEY: {os.environ.get('SECRET_KEY', 'Not set')}")

# Test configuration loading
print("\n🔍 Testing Configuration Loading")
print("=" * 40)

try:
    from config import config
    
    config_name = os.environ.get('FLASK_ENV', 'development')
    print(f"Config name: {config_name}")
    
    config_class = config[config_name]
    config_instance = config_class()
    
    print(f"SQLALCHEMY_DATABASE_URI: {config_instance.SQLALCHEMY_DATABASE_URI}")
    
    if config_instance.SQLALCHEMY_DATABASE_URI:
        if 'mssql' in config_instance.SQLALCHEMY_DATABASE_URI:
            print("✅ Configuration has mssql connection string")
        else:
            print(f"❌ Configuration has wrong connection string: {config_instance.SQLALCHEMY_DATABASE_URI}")
    else:
        print("❌ SQLALCHEMY_DATABASE_URI is None")
        
except Exception as e:
    print(f"❌ Error loading configuration: {e}")

print("\n📋 Instructions:")
print("1. Make sure you have created a .env file in the project root")
print("2. The .env file should contain:")
print("   DATABASE_URL=mssql+pyodbc://Malay:Greenscan%4012@greenscanserver.database.windows.net:1433/heart_disease_db?driver={ODBC Driver 18 for SQL Server}&Encrypt=yes&TrustServerCertificate=no&Connection Timeout=30")
print("3. The file should be named exactly '.env' (with the dot)")
print("4. Make sure there are no spaces around the = sign in the .env file") 