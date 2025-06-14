# setup_database.py
import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

async def setup_database():
    """Setup database with step-by-step error handling"""
    try:
        print("🔧 Step 1: Importing core modules...")
        from app.core.config import settings
        from app.core.database import engine, AsyncSessionLocal
        print("✅ Core modules imported")
        
        print("🔧 Step 2: Testing database connection...")
        async with AsyncSessionLocal() as session:
            from sqlalchemy import text
            result = await session.execute(text("SELECT version()"))
            version = result.scalar()
            print(f"✅ Connected to PostgreSQL: {version}")
        
        print("🔧 Step 3: Importing base model...")
        from app.models.base import Base
        print("✅ Base model imported")
        
        print("🔧 Step 4: Importing user model...")
        from app.models.user import User
        print("✅ User model imported")
        
        print("🔧 Step 5: Importing leads model...")
        from app.models.leads import Lead
        print("✅ Leads model imported")
        
        print("🔧 Step 6: Importing contacts model...")
        from app.models.contacts import Contact
        print("✅ Contacts model imported")
        
        print("🔧 Step 7: Importing sentiment model...")
        from app.models.sentiment import SentimentAnalysis
        print("✅ Sentiment model imported")
        
        print("🔧 Step 8: Creating tables...")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("✅ All tables created successfully!")
        
        print("🔧 Step 9: Creating admin user...")
        from app.core.security import get_password_hash
        from sqlalchemy import select
        
        async with AsyncSessionLocal() as session:
            # Check if admin exists
            result = await session.execute(
                select(User).where(User.email == "admin@leadgen.com")
            )
            existing_user = result.scalar_one_or_none()
            
            if existing_user:
                print("⚠️  Admin user already exists")
            else:
                admin_user = User(
                    email="admin@leadgen.com",
                    hashed_password=get_password_hash("admin123"),
                    full_name="Lead Gen Admin",
                    is_active=True,
                    is_superuser=True,
                    company="Lead Gen Inc",
                    job_title="System Administrator"
                )
                
                session.add(admin_user)
                await session.commit()
                
                print("✅ Admin user created!")
                print("📧 Email: admin@leadgen.com")
                print("🔑 Password: admin123")
        
        print("\n🎉 Database setup completed successfully!")
        
        # List created tables
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
            )
            tables = result.fetchall()
            print(f"📋 Tables created: {[table[0] for table in tables]}")
        
    except ImportError as e:
        print(f"❌ Import error at step: {e}")
        print("💡 Check that all files exist and have correct syntax")
    except Exception as e:
        print(f"❌ Setup error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(setup_database())