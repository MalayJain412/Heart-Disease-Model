import os
import sys
from urllib.parse import quote_plus

# Azure SQL Database Configuration
AZURE_DB_SERVER = "greenscanserver.database.windows.net"
AZURE_DB_NAME = "heart_disease_db"
AZURE_DB_USERNAME = "Malay"
AZURE_DB_PASSWORD = "Greenscan@12"

def test_azure_connection():
    """Test Azure SQL Database connection"""
    print("🔍 Testing Azure SQL Database connection...")
    
    try:
        # Try to import required modules
        try:
            from sqlalchemy import create_engine, text
        except ImportError as e:
            print(f"❌ SQLAlchemy not available: {e}")
            print("Please install: pip install sqlalchemy")
            return False
        
        try:
            import pyodbc
        except ImportError as e:
            print(f"❌ pyodbc not available: {e}")
            print("Please install: pip install pyodbc")
            return False
        
        # Check available ODBC drivers
        print("Available ODBC drivers:")
        for driver in pyodbc.drivers():
            print(f"  - {driver}")
        
        # URL-encode the password
        password_encoded = quote_plus(AZURE_DB_PASSWORD)
        
        # Try different connection string formats with better error handling
        connection_strings = [
            # Format 1: Standard format with ODBC Driver 18
            f"mssql+pyodbc://{AZURE_DB_USERNAME}:{password_encoded}@{AZURE_DB_SERVER}:1433/{AZURE_DB_NAME}?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&TrustServerCertificate=no&Connection+Timeout=30",
            
            # Format 2: Try without database name first (connect to master)
            f"mssql+pyodbc://{AZURE_DB_USERNAME}:{password_encoded}@{AZURE_DB_SERVER}:1433/master?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&TrustServerCertificate=no&Connection+Timeout=30",
            
            # Format 3: Try with different authentication
            f"mssql+pyodbc://{AZURE_DB_USERNAME}:{password_encoded}@{AZURE_DB_SERVER}:1433/{AZURE_DB_NAME}?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&TrustServerCertificate=no&Connection+Timeout=30&Authentication=ActiveDirectoryPassword",
            
            # Format 4: Direct connection string
            f"mssql+pyodbc:///?odbc_connect=DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={AZURE_DB_SERVER};DATABASE={AZURE_DB_NAME};UID={AZURE_DB_USERNAME};PWD={AZURE_DB_PASSWORD};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30"
        ]
        
        for i, conn_str in enumerate(connection_strings, 1):
            print(f"\n--- Trying connection format {i} ---")
            print(f"Connection string: {conn_str[:100]}...")
            
            try:
                engine = create_engine(conn_str)
                with engine.connect() as connection:
                    print("✅ Connection successful!")
                    
                    # Test basic query
                    result = connection.execute(text("SELECT @@VERSION"))
                    version = result.fetchone()[0]
                    print(f"SQL Server Version: {version}")
                    
                    # Test if we can access the database
                    if i == 2:  # Format 2 connects to master database
                        result = connection.execute(text("SELECT name FROM sys.databases WHERE name = 'heart_disease_db'"))
                        db_exists = result.fetchone()
                        if db_exists:
                            print("✅ heart_disease_db database exists")
                        else:
                            print("❌ heart_disease_db database not found")
                    else:
                        # Test if tables exist
                        result = connection.execute(text("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'"))
                        tables = [row[0] for row in result.fetchall()]
                        print(f"✅ Found tables: {tables}")
                    
                    return True
                    
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Format {i} failed: {error_msg}")
                
                # Provide specific guidance based on error
                if "Login failed" in error_msg:
                    print("   → This suggests incorrect username/password or user doesn't exist")
                elif "Cannot open server" in error_msg:
                    print("   → This suggests firewall/IP access issue")
                elif "Data source name not found" in error_msg:
                    print("   → This suggests ODBC driver issue")
                
                continue
        
        print("\n❌ All connection formats failed")
        print("\n--- Troubleshooting Steps ---")
        print("1. Verify your Azure SQL Database credentials:")
        print(f"   - Server: {AZURE_DB_SERVER}")
        print(f"   - Database: {AZURE_DB_NAME}")
        print(f"   - Username: {AZURE_DB_USERNAME}")
        print(f"   - Password: {AZURE_DB_PASSWORD}")
        print("2. Check if the user 'Malay' exists in your Azure SQL Database")
        print("3. Verify the user has proper permissions")
        print("4. Try connecting via Azure Data Studio to verify credentials")
        print("5. Check if the database is paused in Azure portal")
        return False
            
    except Exception as e:
        print(f"❌ Azure SQL connection failed: {e}")
        return False

def test_with_different_credentials():
    """Test with common Azure SQL Database credential patterns"""
    print("\n🔍 Testing common credential patterns...")
    
    # Common Azure SQL Database username patterns
    test_credentials = [
        {"username": "Malay", "password": "Greenscan@12"},
        {"username": "Malay@greenscanserver", "password": "Greenscan@12"},
        {"username": "Malay@greenscanserver.database.windows.net", "password": "Greenscan@12"},
        {"username": "admin", "password": "Greenscan@12"},
        {"username": "admin@greenscanserver", "password": "Greenscan@12"},
    ]
    
    for cred in test_credentials:
        print(f"\n--- Testing username: {cred['username']} ---")
        try:
            from sqlalchemy import create_engine, text
            password_encoded = quote_plus(cred['password'])
            
            conn_str = f"mssql+pyodbc://{cred['username']}:{password_encoded}@{AZURE_DB_SERVER}:1433/{AZURE_DB_NAME}?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&TrustServerCertificate=no&Connection+Timeout=30"
            
            engine = create_engine(conn_str)
            with engine.connect() as connection:
                print("✅ Connection successful with these credentials!")
                return True
                
        except Exception as e:
            print(f"❌ Failed: {str(e)[:100]}...")
            continue
    
    return False

if __name__ == "__main__":
    # First try the original credentials
    if not test_azure_connection():
        # If that fails, try different credential patterns
        test_with_different_credentials() 