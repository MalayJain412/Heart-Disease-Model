import os
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

# --- USER CONFIGURATION ---
# Please fill in your Azure SQL admin username below
AZURE_DB_USERNAME = "admin"
# --------------------------

# Details from your Azure portal screenshot and message
AZURE_DB_SERVER = "greenscanserver.database.windows.net"
AZURE_DB_NAME = "HeartModel"
AZURE_DB_PASSWORD = "Greenscan@12" # The script will handle encoding

def test_azure_sql_connection():
    """Tests the connection to the Azure SQL Database."""
    print("Testing Azure SQL Database connection...")

    if AZURE_DB_USERNAME == "admin":
        print("\nERROR: Please open 'test_azure_db.py' and replace 'your_admin_username_here' with your actual Azure SQL admin username.")
        return

    try:
        # URL-encode the password
        password_encoded = quote_plus(AZURE_DB_PASSWORD)

        # Construct the connection string
        driver = "{ODBC Driver 18 for SQL Server}"
        conn_str = (
            f"mssql+pyodbc://{AZURE_DB_USERNAME}:{password_encoded}@"
            f"{AZURE_DB_SERVER}:1433/{AZURE_DB_NAME}?driver={driver}"
            "&Encrypt=yes&TrustServerCertificate=no&Connection Timeout=30"
        )

        print(f"Connecting with server: {AZURE_DB_SERVER} and database: {AZURE_DB_NAME}...")

        # Create an engine and try to connect
        engine = create_engine(conn_str)
        with engine.connect() as connection:
            print("Connection successful!")
            # Test query
            result = connection.execute(text("SELECT @@VERSION"))
            for row in result:
                print(f"SQL Server Version: {row[0]}")
            print("\n✅ Azure SQL Database connection test PASSED!")

    except Exception as e:
        print("\n❌ Azure SQL Database connection test FAILED.")
        print("\n--- Error Details ---")
        print(e)
        print("\n--- Troubleshooting ---")
        print("1. Is the database 'Resumed' (not 'Paused') in the Azure portal?")
        print("2. Is your admin username correct?")
        print("3. Have you added your current IP address to the server's firewall rules in Azure?")
        print("4. Is the password correct?")

if __name__ == "__main__":
    test_azure_sql_connection() 