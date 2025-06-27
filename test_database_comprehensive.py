import os
import sys
import time
from sqlalchemy import create_engine, text, inspect
from urllib.parse import quote_plus
from werkzeug.security import generate_password_hash
import pandas as pd

# --- CONFIGURATION ---
# Azure SQL Database Configuration
AZURE_DB_SERVER = "greenscanserver.database.windows.net"
AZURE_DB_NAME = "HeartModel"
AZURE_DB_USERNAME = "admin"
AZURE_DB_PASSWORD = "Greenscan@12"

# Local MySQL Configuration (alternative)
LOCAL_MYSQL_HOST = "localhost"
LOCAL_MYSQL_USER = "root"
LOCAL_MYSQL_PASSWORD = ""
LOCAL_MYSQL_DB = "heart_disease_db"

def test_azure_sql_connection():
    """Test Azure SQL Database connection"""
    print("🔍 Testing Azure SQL Database connection...")
    
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
        
        print(f"Connecting to: {AZURE_DB_SERVER}/{AZURE_DB_NAME}")
        
        # Create engine and test connection
        engine = create_engine(conn_str)
        with engine.connect() as connection:
            print("✅ Connection successful!")
            
            # Test basic query
            result = connection.execute(text("SELECT @@VERSION"))
            version = result.fetchone()[0]
            print(f"SQL Server Version: {version}")
            
            return engine
            
    except Exception as e:
        print(f"❌ Azure SQL connection failed: {e}")
        return None

def test_local_mysql_connection():
    """Test local MySQL connection"""
    print("🔍 Testing Local MySQL connection...")
    
    try:
        conn_str = f"mysql+pymysql://{LOCAL_MYSQL_USER}:{LOCAL_MYSQL_PASSWORD}@{LOCAL_MYSQL_HOST}/{LOCAL_MYSQL_DB}"
        
        print(f"Connecting to: {LOCAL_MYSQL_HOST}/{LOCAL_MYSQL_DB}")
        
        engine = create_engine(conn_str)
        with engine.connect() as connection:
            print("✅ Local MySQL connection successful!")
            
            # Test basic query
            result = connection.execute(text("SELECT VERSION()"))
            version = result.fetchone()[0]
            print(f"MySQL Version: {version}")
            
            return engine
            
    except Exception as e:
        print(f"❌ Local MySQL connection failed: {e}")
        return None

def create_database_schema(engine, db_type="azure"):
    """Create the database schema"""
    print(f"\n🔧 Creating database schema for {db_type}...")
    
    try:
        with engine.connect() as connection:
            # Create user table
            if db_type == "azure":
                user_table_sql = """
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='user' AND xtype='U')
                CREATE TABLE [user] (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    username NVARCHAR(80) UNIQUE NOT NULL,
                    email NVARCHAR(120) UNIQUE NOT NULL,
                    password_hash NVARCHAR(120) NOT NULL,
                    role NVARCHAR(20) NOT NULL,
                    created_at DATETIME2 DEFAULT GETDATE()
                )
                """
            else:
                user_table_sql = """
                CREATE TABLE IF NOT EXISTS `user` (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(80) UNIQUE NOT NULL,
                    email VARCHAR(120) UNIQUE NOT NULL,
                    password_hash VARCHAR(120) NOT NULL,
                    role VARCHAR(20) NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            
            connection.execute(text(user_table_sql))
            print("✅ User table created/verified")
            
            # Create patient table
            if db_type == "azure":
                patient_table_sql = """
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='patient' AND xtype='U')
                CREATE TABLE patient (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    name NVARCHAR(100) NOT NULL,
                    gender NVARCHAR(10) NOT NULL,
                    age INT NOT NULL,
                    chest_pain BIT DEFAULT 0,
                    shortness_of_breath BIT DEFAULT 0,
                    fatigue BIT DEFAULT 0,
                    systolic INT NOT NULL,
                    diastolic INT NOT NULL,
                    heart_rate INT NOT NULL,
                    lung_sounds BIT DEFAULT 0,
                    cholesterol INT NOT NULL,
                    ldl INT NOT NULL,
                    hdl INT NOT NULL,
                    diabetes BIT DEFAULT 0,
                    atrial_fibrillation BIT DEFAULT 0,
                    rheumatic_fever BIT DEFAULT 0,
                    mitral_stenosis BIT DEFAULT 0,
                    aortic_stenosis BIT DEFAULT 0,
                    tricuspid_stenosis BIT DEFAULT 0,
                    pulmonary_stenosis BIT DEFAULT 0,
                    dilated_cardiomyopathy BIT DEFAULT 0,
                    hypertrophic_cardiomyopathy BIT DEFAULT 0,
                    drug_use BIT DEFAULT 0,
                    fever BIT DEFAULT 0,
                    chills BIT DEFAULT 0,
                    alcoholism BIT DEFAULT 0,
                    hypertension BIT DEFAULT 0,
                    fainting BIT DEFAULT 0,
                    dizziness BIT DEFAULT 0,
                    smoking BIT DEFAULT 0,
                    obesity BIT DEFAULT 0,
                    murmur BIT DEFAULT 0,
                    prediction_result NVARCHAR(100) NOT NULL,
                    doctor_id INT NOT NULL,
                    timestamp DATETIME2 DEFAULT GETDATE(),
                    FOREIGN KEY (doctor_id) REFERENCES [user](id)
                )
                """
            else:
                patient_table_sql = """
                CREATE TABLE IF NOT EXISTS patient (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    gender VARCHAR(10) NOT NULL,
                    age INT NOT NULL,
                    chest_pain BOOLEAN DEFAULT FALSE,
                    shortness_of_breath BOOLEAN DEFAULT FALSE,
                    fatigue BOOLEAN DEFAULT FALSE,
                    systolic INT NOT NULL,
                    diastolic INT NOT NULL,
                    heart_rate INT NOT NULL,
                    lung_sounds BOOLEAN DEFAULT FALSE,
                    cholesterol INT NOT NULL,
                    ldl INT NOT NULL,
                    hdl INT NOT NULL,
                    diabetes BOOLEAN DEFAULT FALSE,
                    atrial_fibrillation BOOLEAN DEFAULT FALSE,
                    rheumatic_fever BOOLEAN DEFAULT FALSE,
                    mitral_stenosis BOOLEAN DEFAULT FALSE,
                    aortic_stenosis BOOLEAN DEFAULT FALSE,
                    tricuspid_stenosis BOOLEAN DEFAULT FALSE,
                    pulmonary_stenosis BOOLEAN DEFAULT FALSE,
                    dilated_cardiomyopathy BOOLEAN DEFAULT FALSE,
                    hypertrophic_cardiomyopathy BOOLEAN DEFAULT FALSE,
                    drug_use BOOLEAN DEFAULT FALSE,
                    fever BOOLEAN DEFAULT FALSE,
                    chills BOOLEAN DEFAULT FALSE,
                    alcoholism BOOLEAN DEFAULT FALSE,
                    hypertension BOOLEAN DEFAULT FALSE,
                    fainting BOOLEAN DEFAULT FALSE,
                    dizziness BOOLEAN DEFAULT FALSE,
                    smoking BOOLEAN DEFAULT FALSE,
                    obesity BOOLEAN DEFAULT FALSE,
                    murmur BOOLEAN DEFAULT FALSE,
                    prediction_result VARCHAR(100) NOT NULL,
                    doctor_id INT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (doctor_id) REFERENCES `user`(id)
                )
                """
            
            connection.execute(text(patient_table_sql))
            print("✅ Patient table created/verified")
            
            # Create indexes
            if db_type == "azure":
                indexes_sql = [
                    "IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name='IX_patient_doctor_id') CREATE INDEX IX_patient_doctor_id ON patient(doctor_id)",
                    "IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name='IX_patient_timestamp') CREATE INDEX IX_patient_timestamp ON patient(timestamp)",
                    "IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name='IX_user_role') CREATE INDEX IX_user_role ON [user](role)"
                ]
            else:
                indexes_sql = [
                    "CREATE INDEX IF NOT EXISTS idx_patient_doctor_id ON patient(doctor_id)",
                    "CREATE INDEX IF NOT EXISTS idx_patient_timestamp ON patient(timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_user_role ON `user`(role)"
                ]
            
            for index_sql in indexes_sql:
                try:
                    connection.execute(text(index_sql))
                except:
                    pass  # Index might already exist
            
            print("✅ Indexes created/verified")
            connection.commit()
            
    except Exception as e:
        print(f"❌ Schema creation failed: {e}")
        return False
    
    return True

def test_crud_operations(engine, db_type="azure"):
    """Test CRUD operations"""
    print(f"\n🧪 Testing CRUD operations for {db_type}...")
    
    try:
        with engine.connect() as connection:
            # Test 1: Insert a test user
            password_hash = generate_password_hash("test123")
            
            if db_type == "azure":
                insert_user_sql = """
                IF NOT EXISTS (SELECT 1 FROM [user] WHERE username = 'test_doctor')
                INSERT INTO [user] (username, email, password_hash, role) 
                VALUES ('test_doctor', 'test@example.com', :password_hash, 'doctor')
                """
            else:
                insert_user_sql = """
                INSERT IGNORE INTO `user` (username, email, password_hash, role) 
                VALUES ('test_doctor', 'test@example.com', :password_hash, 'doctor')
                """
            
            connection.execute(text(insert_user_sql), {"password_hash": password_hash})
            print("✅ Test user created")
            
            # Test 2: Get user ID
            if db_type == "azure":
                get_user_sql = "SELECT id FROM [user] WHERE username = 'test_doctor'"
            else:
                get_user_sql = "SELECT id FROM `user` WHERE username = 'test_doctor'"
            
            result = connection.execute(text(get_user_sql))
            user_id = result.fetchone()[0]
            print(f"✅ Retrieved user ID: {user_id}")
            
            # Test 3: Insert a test patient
            if db_type == "azure":
                insert_patient_sql = """
                INSERT INTO patient (
                    name, gender, age, chest_pain, shortness_of_breath, fatigue,
                    systolic, diastolic, heart_rate, lung_sounds, cholesterol, ldl, hdl,
                    diabetes, atrial_fibrillation, rheumatic_fever, mitral_stenosis,
                    aortic_stenosis, tricuspid_stenosis, pulmonary_stenosis,
                    dilated_cardiomyopathy, hypertrophic_cardiomyopathy, drug_use,
                    fever, chills, alcoholism, hypertension, fainting, dizziness,
                    smoking, obesity, murmur, prediction_result, doctor_id
                ) VALUES (
                    'John Doe', 'Male', 45, 1, 0, 1, 140, 90, 75, 0, 200, 120, 50,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'Normal', :doctor_id
                )
                """
            else:
                insert_patient_sql = """
                INSERT INTO patient (
                    name, gender, age, chest_pain, shortness_of_breath, fatigue,
                    systolic, diastolic, heart_rate, lung_sounds, cholesterol, ldl, hdl,
                    diabetes, atrial_fibrillation, rheumatic_fever, mitral_stenosis,
                    aortic_stenosis, tricuspid_stenosis, pulmonary_stenosis,
                    dilated_cardiomyopathy, hypertrophic_cardiomyopathy, drug_use,
                    fever, chills, alcoholism, hypertension, fainting, dizziness,
                    smoking, obesity, murmur, prediction_result, doctor_id
                ) VALUES (
                    'John Doe', 'Male', 45, TRUE, FALSE, TRUE, 140, 90, 75, FALSE, 200, 120, 50,
                    FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, 'Normal', :doctor_id
                )
                """
            
            connection.execute(text(insert_patient_sql), {"doctor_id": user_id})
            print("✅ Test patient created")
            
            # Test 4: Query patients
            if db_type == "azure":
                query_sql = """
                SELECT p.name, p.age, p.prediction_result, u.username as doctor_name
                FROM patient p
                JOIN [user] u ON p.doctor_id = u.id
                WHERE u.username = 'test_doctor'
                """
            else:
                query_sql = """
                SELECT p.name, p.age, p.prediction_result, u.username as doctor_name
                FROM patient p
                JOIN `user` u ON p.doctor_id = u.id
                WHERE u.username = 'test_doctor'
                """
            
            result = connection.execute(text(query_sql))
            patients = result.fetchall()
            print(f"✅ Retrieved {len(patients)} patients")
            
            for patient in patients:
                print(f"   - {patient[0]} (Age: {patient[1]}, Prediction: {patient[2]}, Doctor: {patient[3]})")
            
            # Test 5: Update patient
            if db_type == "azure":
                update_sql = "UPDATE patient SET prediction_result = 'High Risk' WHERE name = 'John Doe'"
            else:
                update_sql = "UPDATE patient SET prediction_result = 'High Risk' WHERE name = 'John Doe'"
            
            connection.execute(text(update_sql))
            print("✅ Patient updated")
            
            # Test 6: Delete test data
            if db_type == "azure":
                delete_sql = "DELETE FROM patient WHERE name = 'John Doe'"
            else:
                delete_sql = "DELETE FROM patient WHERE name = 'John Doe'"
            
            connection.execute(text(delete_sql))
            print("✅ Test patient deleted")
            
            connection.commit()
            
    except Exception as e:
        print(f"❌ CRUD operations failed: {e}")
        return False
    
    return True

def test_table_structure(engine, db_type="azure"):
    """Test table structure and constraints"""
    print(f"\n📋 Testing table structure for {db_type}...")
    
    try:
        inspector = inspect(engine)
        
        # Check tables exist
        tables = inspector.get_table_names()
        print(f"✅ Found tables: {tables}")
        
        # Check user table columns
        user_columns = inspector.get_columns('user')
        print(f"✅ User table has {len(user_columns)} columns")
        
        # Check patient table columns
        patient_columns = inspector.get_columns('patient')
        print(f"✅ Patient table has {len(patient_columns)} columns")
        
        # Check foreign key constraints
        if db_type == "azure":
            with engine.connect() as connection:
                result = connection.execute(text("""
                    SELECT 
                        fk.name as constraint_name,
                        OBJECT_NAME(fk.parent_object_id) as table_name,
                        COL_NAME(fkc.parent_object_id, fkc.parent_column_id) as column_name,
                        OBJECT_NAME(fk.referenced_object_id) as referenced_table_name,
                        COL_NAME(fkc.referenced_object_id, fkc.referenced_column_id) as referenced_column_name
                    FROM sys.foreign_keys fk
                    INNER JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id
                    WHERE OBJECT_NAME(fk.parent_object_id) = 'patient'
                """))
                fks = result.fetchall()
                print(f"✅ Found {len(fks)} foreign key constraints")
                
    except Exception as e:
        print(f"❌ Table structure test failed: {e}")
        return False
    
    return True

def main():
    """Main test function"""
    print("🚀 Starting Comprehensive Database Test")
    print("=" * 50)
    
    # Test Azure SQL Database
    print("\n1. TESTING AZURE SQL DATABASE")
    print("-" * 30)
    
    azure_engine = test_azure_sql_connection()
    if azure_engine:
        if create_database_schema(azure_engine, "azure"):
            test_crud_operations(azure_engine, "azure")
            test_table_structure(azure_engine, "azure")
        else:
            print("❌ Azure schema creation failed")
    else:
        print("❌ Azure connection failed")
    
    # Test Local MySQL Database
    print("\n\n2. TESTING LOCAL MYSQL DATABASE")
    print("-" * 30)
    
    mysql_engine = test_local_mysql_connection()
    if mysql_engine:
        if create_database_schema(mysql_engine, "mysql"):
            test_crud_operations(mysql_engine, "mysql")
            test_table_structure(mysql_engine, "mysql")
        else:
            print("❌ MySQL schema creation failed")
    else:
        print("❌ MySQL connection failed")
    
    print("\n" + "=" * 50)
    print("🏁 Database testing completed!")

if __name__ == "__main__":
    main() 