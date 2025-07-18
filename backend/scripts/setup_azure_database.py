import os
import sys
from urllib.parse import quote_plus

# Azure SQL Database Configuration
AZURE_DB_SERVER = "greenscanserver.database.windows.net"
AZURE_DB_NAME = "heart_disease_db"
AZURE_DB_USERNAME = "admin"
AZURE_DB_PASSWORD = "Greenscan@12"

def test_master_connection():
    """Test connection to master database to create user"""
    print("🔍 Testing connection to master database...")
    
    try:
        from sqlalchemy import create_engine, text
        import pyodbc
        
        # Try to connect to master database with server admin credentials
        # You'll need to provide the server admin credentials here
        print("To create a user in your Azure SQL Database, you need the server admin credentials.")
        print("These are the credentials you used when creating the Azure SQL Server.")
        print("\nPlease provide:")
        
        server_admin = input("Server Admin Username (usually like 'admin@greenscanserver'): ").strip()
        server_password = input("Server Admin Password: ").strip()
        
        if not server_admin or not server_password:
            print("❌ Server admin credentials are required")
            return None
        
        # URL-encode the password
        password_encoded = quote_plus(server_password)
        
        # Connect to master database
        conn_str = f"mssql+pyodbc://{server_admin}:{password_encoded}@{AZURE_DB_SERVER}:1433/master?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&TrustServerCertificate=no&Connection+Timeout=30"
        
        engine = create_engine(conn_str)
        with engine.connect() as connection:
            print("✅ Connected to master database successfully!")
            
            # Check if heart_disease_db exists
            result = connection.execute(text("SELECT name FROM sys.databases WHERE name = 'heart_disease_db'"))
            db_exists = result.fetchone()
            
            if not db_exists:
                print("❌ Database 'heart_disease_db' does not exist")
                print("Creating database...")
                connection.execute(text("CREATE DATABASE heart_disease_db"))
                connection.commit()
                print("✅ Database 'heart_disease_db' created successfully!")
            else:
                print("✅ Database 'heart_disease_db' exists")
            
            # Create user in the database
            print(f"Creating user '{AZURE_DB_USERNAME}' in database...")
            
            # Switch to heart_disease_db
            connection.execute(text(f"USE heart_disease_db"))
            
            # Create login if it doesn't exist
            connection.execute(text(f"IF NOT EXISTS (SELECT * FROM sys.server_principals WHERE name = '{AZURE_DB_USERNAME}') CREATE LOGIN [{AZURE_DB_USERNAME}] WITH PASSWORD = '{AZURE_DB_PASSWORD}'"))
            
            # Create user in the database
            connection.execute(text(f"IF NOT EXISTS (SELECT * FROM sys.database_principals WHERE name = '{AZURE_DB_USERNAME}') CREATE USER [{AZURE_DB_USERNAME}] FOR LOGIN [{AZURE_DB_USERNAME}]"))
            
            # Grant permissions
            connection.execute(text(f"ALTER ROLE db_owner ADD MEMBER [{AZURE_DB_USERNAME}]"))
            
            connection.commit()
            print(f"✅ User '{AZURE_DB_USERNAME}' created successfully with db_owner permissions!")
            
            return engine
            
    except Exception as e:
        print(f"❌ Failed to set up database: {e}")
        return None

def test_new_user_connection():
    """Test connection with the newly created user"""
    print("\n🔍 Testing connection with new user...")
    
    try:
        from sqlalchemy import create_engine, text
        
        password_encoded = quote_plus(AZURE_DB_PASSWORD)
        conn_str = f"mssql+pyodbc://{AZURE_DB_USERNAME}:{password_encoded}@{AZURE_DB_SERVER}:1433/{AZURE_DB_NAME}?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&TrustServerCertificate=no&Connection+Timeout=30"
        
        engine = create_engine(conn_str)
        with engine.connect() as connection:
            print("✅ Connection successful with new user!")
            
            # Test basic query
            result = connection.execute(text("SELECT @@VERSION"))
            version = result.fetchone()[0]
            print(f"SQL Server Version: {version}")
            
            # Test if tables exist
            result = connection.execute(text("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'"))
            tables = [row[0] for row in result.fetchall()]
            print(f"✅ Found tables: {tables}")
            
            return True
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Azure SQL Database Setup")
    print("=" * 40)
    
    print("This script will help you set up your Azure SQL Database with proper user authentication.")
    print("You'll need your Azure SQL Server admin credentials to create a new user.")
    
    choice = input("\nDo you want to proceed with database setup? (y/n): ").strip().lower()
    
    if choice != 'y':
        print("Setup cancelled.")
        return
    
    # Test master connection and create user
    engine = test_master_connection()
    
    if engine:
        # Test the new user connection
        if test_new_user_connection():
            print("\n🎉 Database setup completed successfully!")
            print(f"You can now use these credentials in your Flask app:")
            print(f"  - Username: {AZURE_DB_USERNAME}")
            print(f"  - Password: {AZURE_DB_PASSWORD}")
            print(f"  - Database: {AZURE_DB_NAME}")
        else:
            print("\n❌ New user connection failed. Please check the credentials.")
    else:
        print("\n❌ Database setup failed.")

if __name__ == "__main__":
    main() 