import os
import sys
from urllib.parse import quote_plus

# Azure SQL Database Configuration
AZURE_DB_SERVER = "greenscanserver.database.windows.net"
AZURE_DB_NAME = "master"  # Connect to master database

def test_admin_credentials():
    """Test common Azure SQL Server admin credential formats"""
    print("🔍 Testing common Azure SQL Server admin credential formats...")
    
    try:
        from sqlalchemy import create_engine, text
        import pyodbc
        
        # Common admin username patterns
        test_credentials = [
            {"username": "admin", "password": "admin123"},
            {"username": "admin@greenscanserver", "password": "admin123"},
            {"username": "admin@greenscanserver.database.windows.net", "password": "admin123"},
            {"username": "admin", "password": "Greenscan@12"},
            {"username": "admin@greenscanserver", "password": "Greenscan@12"},
            {"username": "admin@greenscanserver.database.windows.net", "password": "Greenscan@12"},
            {"username": "sa", "password": "admin123"},
            {"username": "sa", "password": "Greenscan@12"},
            {"username": "sqladmin", "password": "admin123"},
            {"username": "sqladmin", "password": "Greenscan@12"},
            {"username": "greenscanserver", "password": "admin123"},
            {"username": "greenscanserver", "password": "Greenscan@12"},
        ]
        
        for i, cred in enumerate(test_credentials, 1):
            print(f"\n--- Test {i}: {cred['username']} ---")
            
            try:
                password_encoded = quote_plus(cred['password'])
                conn_str = f"mssql+pyodbc://{cred['username']}:{password_encoded}@{AZURE_DB_SERVER}:1433/{AZURE_DB_NAME}?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&TrustServerCertificate=no&Connection+Timeout=30"
                
                engine = create_engine(conn_str)
                with engine.connect() as connection:
                    print("✅ CONNECTION SUCCESSFUL!")
                    print(f"   Username: {cred['username']}")
                    print(f"   Password: {cred['password']}")
                    
                    # Test basic query
                    result = connection.execute(text("SELECT @@VERSION"))
                    version = result.fetchone()[0]
                    print(f"   SQL Server Version: {version}")
                    
                    # Check databases
                    result = connection.execute(text("SELECT name FROM sys.databases"))
                    databases = [row[0] for row in result.fetchall()]
                    print(f"   Available databases: {databases}")
                    
                    return cred
                    
            except Exception as e:
                error_msg = str(e)
                if "Login failed" in error_msg:
                    print(f"   ❌ Login failed")
                elif "Cannot open server" in error_msg:
                    print(f"   ❌ Server access denied (firewall)")
                else:
                    print(f"   ❌ {error_msg[:50]}...")
                continue
        
        print("\n❌ All credential combinations failed")
        print("\n--- Next Steps ---")
        print("1. Check your Azure portal for the correct server admin credentials")
        print("2. Go to SQL Server 'greenscanserver' → Active Directory admin")
        print("3. Or reset the admin password in Azure portal")
        print("4. The username is usually in format: username@servername")
        return None
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None

def get_connection_string_from_portal():
    """Show how to get connection string from Azure portal"""
    print("\n📋 How to get connection string from Azure Portal:")
    print("1. Go to https://portal.azure.com")
    print("2. Navigate to your SQL Database: heart_disease_db")
    print("3. Click 'Connection strings' in the left menu")
    print("4. Copy the ADO.NET connection string")
    print("5. Look for 'User ID=' and 'Password=' in the connection string")
    print("\nExample connection string format:")
    print("Server=tcp:greenscanserver.database.windows.net,1433;Initial Catalog=heart_disease_db;Persist Security Info=False;User ID=your_username;Password=your_password;MultipleActiveResultSets=False;Encrypt=True;TrustServerCertificate=False;Connection Timeout=30;")

if __name__ == "__main__":
    print("🚀 Azure SQL Server Admin Credential Test")
    print("=" * 50)
    
    # Test common credential combinations
    working_creds = test_admin_credentials()
    
    if working_creds:
        print(f"\n🎉 Found working credentials!")
        print(f"Username: {working_creds['username']}")
        print(f"Password: {working_creds['password']}")
        print("\nYou can now use these credentials to set up your database.")
    else:
        # Show how to get connection string from portal
        get_connection_string_from_portal() 