# check_tables.py
import asyncio
from app.core.database import AsyncSessionLocal
from sqlalchemy import text

async def check_all_tables():
    """Check what tables exist in database"""
    async with AsyncSessionLocal() as session:
        # List all tables
        result = await session.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename")
        )
        tables = result.fetchall()
        
        print("📋 Tables in database:")
        for table in tables:
            print(f"   • {table[0]}")
        
        print(f"\nTotal tables: {len(tables)}")
        
        # Check if sentiment_analysis should exist
        print("\n🔍 Expected tables:")
        expected = ['users', 'leads', 'contacts', 'sentiment_analysis']
        for table in expected:
            exists = any(t[0] == table for t in tables)
            status = "✅" if exists else "❌"
            print(f"   {status} {table}")

if __name__ == "__main__":
    asyncio.run(check_all_tables())