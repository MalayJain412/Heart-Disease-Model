import os

# New .env content with Azure SQL Database configuration
new_env_content = """# Flask Configuration
SECRET_KEY=your-super-secret-key-change-this-in-production
FLASK_DEBUG=True
FLASK_ENV=development

# Azure SQL Database Configuration
DATABASE_URL=mssql+pyodbc://Malay:Greenscan%4012@greenscanserver.database.windows.net:1433/heart_disease_db?driver={ODBC Driver 18 for SQL Server}&Encrypt=yes&TrustServerCertificate=no&Connection Timeout=30

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

def update_env_file():
    """Update the .env file with Azure SQL Database configuration"""
    env_file_path = '.env'
    
    # Backup the current .env file
    if os.path.exists(env_file_path):
        backup_path = '.env.backup'
        with open(env_file_path, 'r') as f:
            current_content = f.read()
        
        with open(backup_path, 'w') as f:
            f.write(current_content)
        print(f"✅ Backed up current .env to {backup_path}")
    
    # Write the new .env content
    with open(env_file_path, 'w') as f:
        f.write(new_env_content)
    
    print("✅ Updated .env file with Azure SQL Database configuration")
    print("\nNew .env content:")
    print("=" * 50)
    print(new_env_content)
    print("=" * 50)

if __name__ == "__main__":
    print("🔄 Updating .env file...")
    update_env_file()
    print("\n🎉 .env file updated successfully!")
    print("You can now run: python app.py") 