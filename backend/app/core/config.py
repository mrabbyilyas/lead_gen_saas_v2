from pydantic_settings import BaseSettings
from typing import List, Optional
import secrets

class Settings(BaseSettings):
    # Application
    PROJECT_NAME: str = "Lead Generation API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Security
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # Database (Azure PostgreSQL)
    DATABASE_URL: str = "postgresql+asyncpg://lead_gen_admin:VFBZ$dPcrI)QyAag@leadgen-mvp-db.postgres.database.azure.com:5432/postgres"
    
    # Redis (can use Azure Redis or local for development)
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_CACHE_TTL: int = 3600  # 1 hour
    
    # External APIs (Free alternatives focus)
    HUNTER_IO_API_KEY: Optional[str] = None  # For email finding (free tier available)
    CLEARBIT_API_KEY: Optional[str] = None   # For company enrichment (free tier)
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        # "https://yourdomain.com"
    ]
    
    # Scraping Configuration (Conservative for free approach)
    MAX_CONCURRENT_REQUESTS: int = 3  # Reduced for free proxies
    REQUEST_DELAY_MIN: float = 2.0    # Longer delays to be respectful
    REQUEST_DELAY_MAX: float = 5.0
    MAX_RETRIES: int = 2
    TIMEOUT_SECONDS: int = 20
    
    # Rate Limiting (Conservative)
    RATE_LIMIT_PER_MINUTE: int = 30
    RATE_LIMIT_PER_HOUR: int = 200
    
    # File Upload
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    UPLOAD_DIR: str = "./uploads"
    
    # Monitoring
    SENTRY_DSN: Optional[str] = None
    LOG_LEVEL: str = "INFO"
    
    # Background Tasks
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()