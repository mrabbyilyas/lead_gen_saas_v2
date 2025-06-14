# create_sentiment_table.py
import asyncio
from app.core.database import AsyncSessionLocal
from sqlalchemy import text

async def create_sentiment_table_manual():
    """Create sentiment_analysis table manually via SQL"""
    
    async with AsyncSessionLocal() as session:
        try:
            print("🔧 Creating sentiment_analysis table manually...")
            
            # Create table with SQL (avoiding SQLAlchemy relationship issues)
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS sentiment_analysis (
                    id SERIAL PRIMARY KEY,
                    lead_id INTEGER REFERENCES leads(id) ON DELETE CASCADE,
                    company_name VARCHAR(255) NOT NULL,
                    overall_sentiment_score REAL,
                    confidence_score REAL,
                    overall_sentiment VARCHAR(20),
                    sentiment_trend VARCHAR(20),
                    detailed_analysis TEXT,
                    key_topics JSONB,
                    news_sources JSONB,
                    news_count INTEGER,
                    expires_at TIMESTAMP WITH TIME ZONE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE
                )
            """))
            
            # Create index
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_sentiment_company_name 
                ON sentiment_analysis(company_name)
            """))
            
            await session.commit()
            print("✅ sentiment_analysis table created successfully!")
            
            # Verify all tables
            result = await session.execute(
                text("SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename")
            )
            tables = result.fetchall()
            
            print("📋 All tables in database:")
            for table in tables:
                print(f"   • {table[0]}")
            
            print(f"\nTotal tables: {len(tables)}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            await session.rollback()

if __name__ == "__main__":
    asyncio.run(create_sentiment_table_manual())