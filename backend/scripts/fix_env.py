import os

# Correct .env content with proper driver syntax
correct_env_content = """# Flask Configuration
SECRET_KEY=your-super-secret-key-change-this-in-production
FLASK_DEBUG=True
FLASK_ENV=development

# Azure SQL Database Configuration
DATABASE_URL=mssql+pyodbc://Malay:Greenscan%4012@greenscanserver.database.windows.net:1433/heart_disease_db?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&TrustServerCertificate=no&Connection+Timeout=30

# Model Configuration
MODEL_PATH=heart_disease_rf_model.pkl

# Security Settings
SESSION_COOKIE_SECURE=False
SESSION_COOKIE_HTTPONLY=True
SESSION_COOKIE_SAMESITE=Lax

# File Upload Settings
MAX_CONTENT_LENGTH=16777216
UPLOAD_FOLDER=uploads

# Logging
LOG_LEVEL=INFO

# Azure App Service Configuration
WEBSITES_PORT=5000
"""

def fix_env_file():
    """Fix the .env file by removing curly braces from driver name"""
    env_file_path = '.env'
    
    # Backup the current .env file
    if os.path.exists(env_file_path):
        backup_path = '.env.backup2'
        with open(env_file_path, 'r') as f:
            current_content = f.read()
        
        with open(backup_path, 'w') as f:
            f.write(current_content)
        print(f"✅ Backed up current .env to {backup_path}")
    
    # Write the corrected .env content
    with open(env_file_path, 'w') as f:
        f.write(correct_env_content)
    
    print("✅ Fixed .env file - removed curly braces from driver name")
    print("\nKey changes:")
    print("- Removed {} from around 'ODBC Driver 18 for SQL Server'")
    print("- Changed spaces to + in driver name")
    print("- Changed 'Connection Timeout' to 'Connection+Timeout'")
    
    print("\nNew DATABASE_URL:")
    print("mssql+pyodbc://Malay:Greenscan%4012@greenscanserver.database.windows.net:1433/heart_disease_db?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&TrustServerCertificate=no&Connection+Timeout=30")

if __name__ == "__main__":
    print("🔧 Fixing .env file...")
    fix_env_file()
    print("\n🎉 .env file fixed successfully!")
    print("You can now run: python app.py") 