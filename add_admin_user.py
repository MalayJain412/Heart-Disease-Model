import os
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash
from sqlalchemy import create_engine, text

# Load environment variables
load_dotenv()

def add_admin_user():
    """Add the default admin user to the database"""
    print("🔧 Adding admin user to database...")
    
    try:
        # Get database URL from environment
        database_url = os.environ.get('DATABASE_URL')
        if not database_url:
            print("❌ DATABASE_URL not found in environment variables")
            return False
        
        print(f"Connecting to database...")
        
        # Create engine
        engine = create_engine(database_url)
        
        with engine.connect() as connection:
            # Check if admin user already exists
            result = connection.execute(text("SELECT id FROM [user] WHERE username = 'admin'"))
            existing_user = result.fetchone()
            
            if existing_user:
                print("✅ Admin user already exists in database")
                return True
            
            # Create admin user
            admin_password_hash = generate_password_hash('admin123')
            
            insert_query = """
            INSERT INTO [user] (username, email, password_hash, role, created_at)
            VALUES ('admin', 'admin@hospital.com', :password_hash, 'chairman', GETDATE())
            """
            
            connection.execute(text(insert_query), {"password_hash": admin_password_hash})
            connection.commit()
            
            print("✅ Admin user created successfully!")
            print("   Username: admin")
            print("   Password: admin123")
            print("   Role: chairman")
            print("   Email: admin@hospital.com")
            
            return True
            
    except Exception as e:
        print(f"❌ Error adding admin user: {e}")
        return False

def verify_admin_user():
    """Verify that the admin user was created successfully"""
    print("\n🔍 Verifying admin user...")
    
    try:
        database_url = os.environ.get('DATABASE_URL')
        engine = create_engine(database_url)
        
        with engine.connect() as connection:
            # Check admin user
            result = connection.execute(text("SELECT username, email, role, created_at FROM [user] WHERE username = 'admin'"))
            admin_user = result.fetchone()
            
            if admin_user:
                print("✅ Admin user verified:")
                print(f"   Username: {admin_user[0]}")
                print(f"   Email: {admin_user[1]}")
                print(f"   Role: {admin_user[2]}")
                print(f"   Created: {admin_user[3]}")
                
                # Count total users
                result = connection.execute(text("SELECT COUNT(*) FROM [user]"))
                total_users = result.fetchone()[0]
                print(f"\n📊 Total users in database: {total_users}")
                
                return True
            else:
                print("❌ Admin user not found")
                return False
                
    except Exception as e:
        print(f"❌ Error verifying admin user: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Adding Default Admin User")
    print("=" * 40)
    
    # Add admin user
    if add_admin_user():
        # Verify the user was created
        verify_admin_user()
        print("\n🎉 Admin user setup completed!")
        print("You can now login to your Flask app with:")
        print("   Username: admin")
        print("   Password: admin123")
    else:
        print("\n❌ Failed to add admin user") 