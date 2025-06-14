# ============================================================================
# COMPLETE LEAD GENERATION BACKEND - FastAPI
# ============================================================================

# requirements.txt
"""
fastapi==0.104.1
uvicorn[standard]==0.24.0
sqlalchemy==2.0.23
alembic==1.12.1
asyncpg==0.29.0
redis==5.0.1
celery==5.3.4
pydantic==2.5.0
pydantic-settings==2.1.0
playwright==1.40.0
beautifulsoup4==4.12.2
aiohttp==3.9.1
fake-useragent==1.4.0
python-multipart==0.0.6
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
pandas==2.1.4
numpy==1.24.3
scikit-learn==1.3.2
textblob==0.17.1
python-dotenv==1.0.0
httpx==0.25.2
loguru==0.7.2
prometheus-client==0.19.0
sentry-sdk==1.38.0
feedparser==6.0.10
dnspython==2.4.2
email-validator==2.1.0
phonenumbers==8.13.26
yfinance==0.2.28

# Optional: Transformers for advanced sentiment analysis
# Only install if you have sufficient compute resources
# transformers==4.36.2
# torch==2.1.1
"""

# ============================================================================
# PROJECT STRUCTURE
# ============================================================================
"""
backend/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── security.py
│   │   └── logging.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── deps.py
│   │   └── endpoints/
│   │       ├── __init__.py
│   │       ├── auth.py
│   │       ├── leads.py
│   │       ├── scraping.py
│   │       ├── sentiment.py
│   │       └── export.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── user.py
│   │   ├── leads.py
│   │   ├── contacts.py
│   │   └── sentiment.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── leads.py
│   │   ├── contacts.py
│   │   └── sentiment.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── scraping_service.py
│   │   ├── sentiment_service.py
│   │   ├── data_enrichment.py
│   │   ├── export_service.py
│   │   └── background_tasks.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── anti_bot.py
│   │   ├── data_validator.py
│   │   ├── email_finder.py
│   │   ├── proxy_manager.py
│   │   └── rate_limiter.py
│   └── tasks/
│       ├── __init__.py
│       ├── celery_app.py
│       └── scraping_tasks.py
├── alembic/
├── tests/
├── docker-compose.yml
├── Dockerfile
├── .env.example
└── requirements.txt
"""

# ============================================================================
# CORE CONFIGURATION
# ============================================================================

# app/core/config.py
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
        "https://yourdomain.com"
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

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

# app/core/database.py
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from typing import AsyncGenerator
import redis.asyncio as aioredis

from app.core.config import settings

# Database engine
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=True if settings.LOG_LEVEL == "DEBUG" else False,
    pool_pre_ping=True,
    pool_size=20,
    max_overflow=30,
)

AsyncSessionLocal = async_sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    class_=AsyncSession,
)

Base = declarative_base()

# Redis connection
redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)

# Dependency
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

async def get_redis() -> aioredis.Redis:
    return redis_client

# ============================================================================
# SECURITY & AUTHENTICATION
# ============================================================================

# app/core/security.py
from datetime import datetime, timedelta
from typing import Any, Union, Optional
from jose import jwt, JWTError
from passlib.context import CryptContext
from fastapi import HTTPException, status
from pydantic import ValidationError

from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_access_token(
    subject: Union[str, Any], expires_delta: timedelta = None
) -> str:
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    to_encode = {"exp": expire, "sub": str(subject)}
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# app/core/logging.py
import logging
import sys
from loguru import logger
from app.core.config import settings

class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

def setup_logging():
    logging.root.handlers = [InterceptHandler()]
    logging.root.setLevel(settings.LOG_LEVEL)
    
    for name in logging.root.manager.loggerDict.keys():
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True
    
    logger.configure(
        handlers=[
            {
                "sink": sys.stderr,
                "level": settings.LOG_LEVEL,
                "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
            },
            {
                "sink": "logs/app.log",
                "level": settings.LOG_LEVEL,
                "rotation": "10 MB",
                "retention": "1 week",
                "format": "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
            }
        ]
    )

# ============================================================================
# DATABASE MODELS
# ============================================================================

# app/models/base.py
from sqlalchemy import Column, Integer, DateTime, Boolean
from sqlalchemy.sql import func
from app.core.database import Base

class TimestampMixin:
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class SoftDeleteMixin:
    is_deleted = Column(Boolean, default=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True)

# app/models/user.py
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.orm import relationship
from app.models.base import TimestampMixin, SoftDeleteMixin
from app.core.database import Base

class User(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    
    # Profile
    company = Column(String(255))
    job_title = Column(String(255))
    phone = Column(String(50))
    
    # Usage tracking
    api_calls_today = Column(Integer, default=0)
    api_calls_month = Column(Integer, default=0)
    last_activity = Column(DateTime(timezone=True))

# app/models/leads.py
from sqlalchemy import Column, Integer, String, Text, Float, JSON, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from app.models.base import TimestampMixin, SoftDeleteMixin
from app.core.database import Base

class Lead(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "leads"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Company Information
    company_name = Column(String(255), nullable=False, index=True)
    domain = Column(String(255), index=True)
    industry = Column(String(100), index=True)
    employee_count = Column(Integer)
    location = Column(String(255))
    description = Column(Text)
    founded_year = Column(Integer)
    revenue_range = Column(String(50))
    
    # Contact Details
    company_email = Column(String(255))
    company_phone = Column(String(50))
    headquarters_address = Column(Text)
    
    # Social Media
    linkedin_url = Column(String(500))
    twitter_handle = Column(String(100))
    facebook_url = Column(String(500))
    
    # Financial Information
    funding_stage = Column(String(50))
    total_funding = Column(String(50))
    valuation = Column(String(50))
    
    # Technology Stack
    technologies = Column(JSON)  # List of technologies used
    
    # Quality Metrics
    data_quality_score = Column(Float, default=0.0)
    verification_status = Column(String(20), default="unverified")
    confidence_score = Column(Float, default=0.0)
    
    # Scraping Metadata
    source_urls = Column(JSON)  # List of source URLs
    scraping_difficulty = Column(Integer, default=1)  # 1-5 scale
    last_scraped = Column(DateTime(timezone=True))
    
    # Lead Scoring
    lead_score = Column(Float, default=0.0)
    priority = Column(String(20), default="medium")  # low, medium, high
    status = Column(String(20), default="new")  # new, contacted, qualified, converted
    
    # Relationships
    contacts = relationship("Contact", back_populates="lead", cascade="all, delete-orphan")
    sentiment_analyses = relationship("SentimentAnalysis", back_populates="lead")
    user = relationship("User")

# app/models/contacts.py
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Float, JSON
from sqlalchemy.orm import relationship
from app.models.base import TimestampMixin, SoftDeleteMixin
from app.core.database import Base

class Contact(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "contacts"
    
    id = Column(Integer, primary_key=True, index=True)
    lead_id = Column(Integer, ForeignKey("leads.id"), nullable=False)
    
    # Personal Information
    first_name = Column(String(100))
    last_name = Column(String(100))
    full_name = Column(String(255), index=True)
    title = Column(String(255))
    department = Column(String(100))
    seniority_level = Column(String(50))  # C-level, VP, Director, Manager, etc.
    
    # Contact Details
    email = Column(String(255), index=True)
    phone = Column(String(50))
    mobile_phone = Column(String(50))
    work_phone = Column(String(50))
    
    # Social Media
    linkedin_url = Column(String(500))
    twitter_handle = Column(String(100))
    github_username = Column(String(100))
    
    # Professional Details
    years_at_company = Column(Integer)
    previous_companies = Column(JSON)  # List of previous companies
    education = Column(JSON)  # Education background
    skills = Column(JSON)  # List of skills
    
    # Verification Status
    email_verified = Column(Boolean, default=False)
    phone_verified = Column(Boolean, default=False)
    linkedin_verified = Column(Boolean, default=False)
    
    # Quality Metrics
    contact_quality_score = Column(Float, default=0.0)
    deliverability_score = Column(Float)  # Email deliverability
    engagement_score = Column(Float)  # Likelihood to respond
    
    # Source Information
    data_source = Column(String(100))  # linkedin, company_website, etc.
    confidence_level = Column(Float, default=0.0)
    
    # Relationship
    lead = relationship("Lead", back_populates="contacts")

# app/models/sentiment.py
from sqlalchemy import Column, Integer, String, Text, Float, JSON, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from app.models.base import TimestampMixin
from app.core.database import Base

class SentimentAnalysis(Base, TimestampMixin):
    __tablename__ = "sentiment_analysis"
    
    id = Column(Integer, primary_key=True, index=True)
    lead_id = Column(Integer, ForeignKey("leads.id"), nullable=False)
    company_name = Column(String(255), nullable=False, index=True)
    
    # Sentiment Scores (-1 to 1)
    financial_sentiment_score = Column(Float)
    general_sentiment_score = Column(Float)
    media_sentiment_score = Column(Float)
    social_sentiment_score = Column(Float)
    overall_sentiment_score = Column(Float)
    
    # Confidence Metrics (0 to 1)
    confidence_score = Column(Float)
    data_quality_score = Column(Float)
    
    # Classifications
    overall_sentiment = Column(String(20))  # positive, negative, neutral
    sentiment_trend = Column(String(20))    # improving, declining, stable
    risk_level = Column(String(20))         # low, medium, high
    
    # Detailed Analysis
    detailed_analysis = Column(Text)  # AI-generated summary
    key_topics = Column(JSON)         # List of key topics found
    sentiment_breakdown = Column(JSON) # Detailed sentiment by category
    
    # News Sources
    news_sources = Column(JSON)       # List of news sources analyzed
    news_count = Column(Integer)      # Number of articles analyzed
    date_range_start = Column(DateTime(timezone=True))
    date_range_end = Column(DateTime(timezone=True))
    
    # Expiration
    expires_at = Column(DateTime(timezone=True))  # Cache expiration
    
    # Relationship
    lead = relationship("Lead", back_populates="sentiment_analyses")

# ============================================================================
# PYDANTIC SCHEMAS
# ============================================================================

# app/schemas/auth.py
from pydantic import BaseModel, EmailStr
from typing import Optional

class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None
    company: Optional[str] = None
    job_title: Optional[str] = None
    phone: Optional[str] = None

class UserCreate(UserBase):
    password: str

class UserUpdate(UserBase):
    password: Optional[str] = None

class UserInDB(UserBase):
    id: int
    is_active: bool
    is_superuser: bool
    api_calls_today: int
    api_calls_month: int
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# app/schemas/leads.py
from pydantic import BaseModel, HttpUrl, validator, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class PriorityEnum(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"

class StatusEnum(str, Enum):
    new = "new"
    contacted = "contacted"
    qualified = "qualified"
    converted = "converted"

class LeadSearchRequest(BaseModel):
    # Search Filters
    company_name: Optional[str] = None
    industry: Optional[List[str]] = None
    location: Optional[List[str]] = None
    employee_count_min: Optional[int] = Field(None, ge=1)
    employee_count_max: Optional[int] = Field(None, le=1000000)
    keywords: Optional[List[str]] = None
    domains: Optional[List[str]] = None
    technologies: Optional[List[str]] = None
    
    # Quality Filters
    min_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    verification_status: Optional[str] = None
    
    # Sentiment Filters
    sentiment_filter: Optional[str] = None  # positive, negative, neutral
    min_sentiment_score: Optional[float] = Field(None, ge=-1.0, le=1.0)
    
    # Pagination
    skip: int = Field(0, ge=0)
    limit: int = Field(100, ge=1, le=1000)
    
    # Sorting
    sort_by: str = "created_at"
    sort_order: str = "desc"
    
    @validator('sort_by')
    def validate_sort_by(cls, v):
        allowed_fields = [
            'created_at', 'updated_at', 'company_name', 
            'data_quality_score', 'lead_score', 'employee_count'
        ]
        if v not in allowed_fields:
            raise ValueError(f'sort_by must be one of {allowed_fields}')
        return v

class ContactBase(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    full_name: Optional[str] = None
    title: Optional[str] = None
    department: Optional[str] = None
    seniority_level: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin_url: Optional[str] = None
    contact_quality_score: Optional[float] = None

class LeadBase(BaseModel):
    company_name: str = Field(..., min_length=1, max_length=255)
    domain: Optional[str] = None
    industry: Optional[str] = None
    employee_count: Optional[int] = Field(None, ge=1)
    location: Optional[str] = None
    description: Optional[str] = None
    founded_year: Optional[int] = Field(None, ge=1800, le=2024)
    revenue_range: Optional[str] = None

class LeadCreate(LeadBase):
    contacts: Optional[List[ContactBase]] = []

class LeadUpdate(BaseModel):
    company_name: Optional[str] = None
    domain: Optional[str] = None
    industry: Optional[str] = None
    employee_count: Optional[int] = None
    location: Optional[str] = None
    description: Optional[str] = None
    priority: Optional[PriorityEnum] = None
    status: Optional[StatusEnum] = None

class LeadResponse(LeadBase):
    id: int
    user_id: int
    data_quality_score: float
    verification_status: str
    lead_score: float
    priority: str
    status: str
    contacts: List[ContactBase] = []
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class ScrapeRequest(BaseModel):
    companies: List[str] = Field(..., min_items=1, max_items=50)
    include_contacts: bool = True
    include_sentiment: bool = True
    include_technologies: bool = False
    priority: PriorityEnum = PriorityEnum.medium
    webhook_url: Optional[str] = None
    
    @validator('companies')
    def validate_companies(cls, v):
        if len(v) > 50:
            raise ValueError('Maximum 50 companies per request')
        return [company.strip() for company in v if company.strip()]

class ScrapeResponse(BaseModel):
    job_id: str
    status: str
    estimated_completion: datetime
    companies_count: int
    priority: str

# app/schemas/sentiment.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime

class SentimentRequest(BaseModel):
    company_names: List[str] = Field(..., min_items=1, max_items=20)
    days_back: int = Field(30, ge=1, le=365)
    include_social: bool = False
    force_refresh: bool = False

class SentimentResponse(BaseModel):
    company_name: str
    overall_sentiment_score: float
    overall_sentiment: str
    confidence_score: float
    detailed_analysis: str
    key_topics: List[str]
    news_count: int
    last_updated: datetime
    
class BulkSentimentResponse(BaseModel):
    results: List[SentimentResponse]
    total_companies: int
    successful_analyses: int
    failed_analyses: int

# ============================================================================
# ANTI-BOT UTILITIES
# ============================================================================

# app/utils/anti_bot.py
import random
import asyncio
from typing import List, Dict, Optional
from playwright.async_api import Browser, BrowserContext, Page, Playwright
from fake_useragent import UserAgent
import json

class StealthBrowser:
    def __init__(self):
        self.user_agent = UserAgent()
        self.proxy_manager = ProxyManager()
        
    async def create_stealth_page(self, playwright: Playwright) -> Page:
        """Create a stealth browser page with anti-detection measures"""
        
        # Random browser selection
        browsers = [playwright.chromium, playwright.firefox]
        browser_engine = random.choice(browsers)
        
        # Proxy configuration
        proxy = await self.proxy_manager.get_random_proxy()
        
        # Launch browser with stealth options
        browser = await browser_engine.launch(
            headless=True,
            proxy=proxy,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-web-security',
                '--disable-features=VizDisplayCompositor',
                '--disable-extensions',
                '--disable-plugins',
                '--disable-images',  # Faster loading
                '--disable-javascript',  # For some sites
                f'--user-agent={self.user_agent.random}'
            ]
        )
        
        # Create context with random properties
        context = await browser.new_context(
            user_agent=self.user_agent.random,
            viewport={
                'width': random.randint(1200, 1920),
                'height': random.randint(800, 1080)
            },
            locale=random.choice(['en-US', 'en-GB', 'en-CA']),
            timezone_id=random.choice([
                'America/New_York', 'America/Los_Angeles', 
                'America/Chicago', 'Europe/London'
            ]),
            extra_http_headers={
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
        )
        
        # Create page
        page = await context.new_page()
        
        # Remove automation indicators
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
            
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5],
            });
            
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en'],
            });
            
            window.chrome = {
                runtime: {},
            };
            
            Object.defineProperty(navigator, 'permissions', {
                get: () => ({
                    query: () => Promise.resolve({ state: 'granted' }),
                }),
            });
        """)
        
        # Random delays and human-like behavior
        await self._add_human_behavior(page)
        
        return page
    
    async def _add_human_behavior(self, page: Page):
        """Add human-like behavior to the page"""
        
        # Random mouse movements
        await page.add_init_script("""
            setInterval(() => {
                const x = Math.random() * window.innerWidth;
                const y = Math.random() * window.innerHeight;
                document.dispatchEvent(new MouseEvent('mousemove', {
                    clientX: x,
                    clientY: y
                }));
            }, Math.random() * 10000 + 5000);
        """)
        
        # Random scrolling
        await page.add_init_script("""
            setInterval(() => {
                window.scrollBy(0, Math.random() * 100 - 50);
            }, Math.random() * 15000 + 10000);
        """)

# app/utils/proxy_manager.py
import aiohttp
import random
from typing import List, Dict, Optional
from app.core.config import settings

class ProxyManager:
    def __init__(self):
        self.proxy_list = []
        self.working_proxies = []
        self.failed_proxies = set()
        
    async def load_proxies(self):
        """Load proxy list from various sources"""
        if settings.BRIGHT_DATA_USERNAME and settings.BRIGHT_DATA_PASSWORD:
            # Bright Data proxies (premium)
            self.proxy_list.extend(await self._get_bright_data_proxies())
        
        # Free proxy sources (backup)
        self.proxy_list.extend(await self._get_free_proxies())
    
    async def get_random_proxy(self) -> Optional[Dict]:
        """Get a random working proxy"""
        if not self.working_proxies and self.proxy_list:
            await self._test_proxies()
        
        if self.working_proxies:
            proxy_config = random.choice(self.working_proxies)
            return {
                'server': f"http://{proxy_config['ip']}:{proxy_config['port']}",
                'username': proxy_config.get('username'),
                'password': proxy_config.get('password')
            }
        return None
    
# app/utils/proxy_manager.py
import aiohttp
import random
from typing import List, Dict, Optional
from app.core.config import settings

class ProxyManager:
    def __init__(self):
        self.proxy_list = []
        self.working_proxies = []
        self.failed_proxies = set()
        
    async def load_proxies(self):
        """Load proxy list from free sources"""
        # Focus on free proxy sources
        self.proxy_list.extend(await self._get_free_proxies())
    
    async def get_random_proxy(self) -> Optional[Dict]:
        """Get a random working proxy"""
        if not self.working_proxies and self.proxy_list:
            await self._test_proxies()
        
        if self.working_proxies:
            proxy_config = random.choice(self.working_proxies)
            return {
                'server': f"http://{proxy_config['ip']}:{proxy_config['port']}",
                'username': proxy_config.get('username'),
                'password': proxy_config.get('password')
            }
        return None
    
    async def _get_free_proxies(self) -> List[Dict]:
        """Get free proxies from public sources"""
        proxies = []
        
        try:
            async with aiohttp.ClientSession() as session:
                # ProxyList API
                try:
                    async with session.get('https://www.proxy-list.download/api/v1/get?type=http', timeout=10) as response:
                        if response.status == 200:
                            proxy_text = await response.text()
                            for line in proxy_text.strip().split('\n')[:5]:  # Limit to 5 proxies
                                if ':' in line:
                                    ip, port = line.strip().split(':')
                                    proxies.append({'ip': ip, 'port': int(port)})
                except Exception:
                    pass
                
                # Backup: ProxyScrape API
                try:
                    async with session.get('https://api.proxyscrape.com/v2/?request=get&protocol=http&timeout=10000&country=all', timeout=10) as response:
                        if response.status == 200:
                            proxy_text = await response.text()
                            for line in proxy_text.strip().split('\n')[:5]:  # Limit to 5 proxies
                                if ':' in line:
                                    ip, port = line.strip().split(':')
                                    proxies.append({'ip': ip, 'port': int(port)})
                except Exception:
                    pass
                    
        except Exception as e:
            print(f"Error fetching free proxies: {e}")
        
        return proxies[:10]  # Limit to 10 free proxies total
    
    async def _test_proxies(self):
        """Test proxy connectivity and speed"""
        working = []
        
        for proxy in self.proxy_list:
            if f"{proxy['ip']}:{proxy['port']}" in self.failed_proxies:
                continue
                
            try:
                proxy_url = f"http://{proxy['ip']}:{proxy['port']}"
                timeout = aiohttp.ClientTimeout(total=5)  # Quick test
                
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(
                        'http://httpbin.org/ip',
                        proxy=proxy_url
                    ) as response:
                        if response.status == 200:
                            working.append(proxy)
            except Exception:
                self.failed_proxies.add(f"{proxy['ip']}:{proxy['port']}")
        
        self.working_proxies = working
    
    async def _get_free_proxies(self) -> List[Dict]:
        """Get free proxies from public sources"""
        proxies = []
        
        try:
            async with aiohttp.ClientSession() as session:
                # ProxyList API
                async with session.get('https://www.proxy-list.download/api/v1/get?type=http') as response:
                    if response.status == 200:
                        proxy_text = await response.text()
                        for line in proxy_text.strip().split('\n'):
                            if ':' in line:
                                ip, port = line.strip().split(':')
                                proxies.append({'ip': ip, 'port': int(port)})
        except Exception as e:
            print(f"Error fetching free proxies: {e}")
        
        return proxies[:10]  # Limit to 10 free proxies
    
    async def _test_proxies(self):
        """Test proxy connectivity and speed"""
        working = []
        
        for proxy in self.proxy_list:
            if f"{proxy['ip']}:{proxy['port']}" in self.failed_proxies:
                continue
                
            try:
                proxy_url = f"http://{proxy['ip']}:{proxy['port']}"
                timeout = aiohttp.ClientTimeout(total=10)
                
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(
                        'http://httpbin.org/ip',
                        proxy=proxy_url
                    ) as response:
                        if response.status == 200:
                            working.append(proxy)
            except Exception:
                self.failed_proxies.add(f"{proxy['ip']}:{proxy['port']}")
        
        self.working_proxies = working

# app/utils/data_validator.py
import re
import dns.resolver
import requests
from typing import Dict, List, Optional, Tuple
from email_validator import validate_email, EmailNotValidError
import phonenumbers
from urllib.parse import urlparse

class DataValidator:
    def __init__(self):
        self.email_regex = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        self.phone_regex = re.compile(r'(\+\d{1,3}[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}')
        self.linkedin_regex = re.compile(r'linkedin\.com/in/[\w-]+')
        
    def validate_email_format(self, email: str) -> bool:
        """Validate email format"""
        try:
            validated_email = validate_email(email)
            return True
        except EmailNotValidError:
            return False
    
    def validate_email_deliverability(self, email: str) -> Tuple[bool, float]:
        """Check email deliverability"""
        if not self.validate_email_format(email):
            return False, 0.0
        
        domain = email.split('@')[1]
        
        try:
            # Check MX record
            mx_records = dns.resolver.resolve(domain, 'MX')
            if not mx_records:
                return False, 0.1
            
            # Basic deliverability score
            score = 0.8
            
            # Check for common disposable email domains
            disposable_domains = [
                '10minutemail.com', 'tempmail.org', 'guerrillamail.com',
                'mailinator.com', 'throwaway.email'
            ]
            
            if domain.lower() in disposable_domains:
                score = 0.2
            
            # Check for role-based emails
            role_prefixes = ['admin', 'info', 'support', 'sales', 'marketing', 'noreply']
            local_part = email.split('@')[0].lower()
            
            if any(prefix in local_part for prefix in role_prefixes):
                score -= 0.2
            
            return True, max(score, 0.1)
            
        except Exception:
            return False, 0.0
    
    def validate_phone_number(self, phone: str, region: str = None) -> Tuple[bool, str]:
        """Validate and format phone number"""
        try:
            parsed_number = phonenumbers.parse(phone, region)
            if phonenumbers.is_valid_number(parsed_number):
                formatted = phonenumbers.format_number(
                    parsed_number, 
                    phonenumbers.PhoneNumberFormat.INTERNATIONAL
                )
                return True, formatted
            return False, phone
        except Exception:
            return False, phone
    
    def validate_linkedin_url(self, url: str) -> bool:
        """Validate LinkedIn URL"""
        if not url:
            return False
        
        try:
            parsed = urlparse(url)
            if 'linkedin.com' not in parsed.netloc:
                return False
            
            # Check if it's a profile URL
            if '/in/' in parsed.path or '/company/' in parsed.path:
                return True
            
            return False
        except Exception:
            return False
    
    def validate_domain(self, domain: str) -> bool:
        """Validate domain format and existence"""
        if not domain:
            return False
        
        # Remove protocol if present
        domain = domain.replace('http://', '').replace('https://', '')
        domain = domain.split('/')[0]  # Remove path
        
        # Basic format validation
        domain_pattern = re.compile(
            r'^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$'
        )
        
        if not domain_pattern.match(domain):
            return False
        
        try:
            # Check if domain resolves
            dns.resolver.resolve(domain, 'A')
            return True
        except Exception:
            return False
    
    def calculate_data_quality_score(self, lead_data: Dict) -> float:
        """Calculate overall data quality score for a lead"""
        score = 0.0
        total_weight = 0.0
        
        # Company name (required)
        if lead_data.get('company_name'):
            score += 0.2
        total_weight += 0.2
        
        # Domain validation
        if lead_data.get('domain'):
            if self.validate_domain(lead_data['domain']):
                score += 0.15
            else:
                score += 0.05
        total_weight += 0.15
        
        # Contact information
        contacts = lead_data.get('contacts', [])
        if contacts:
            contact_score = 0.0
            for contact in contacts:
                contact_quality = 0.0
                
                # Email validation
                if contact.get('email'):
                    is_valid, deliverability = self.validate_email_deliverability(contact['email'])
                    if is_valid:
                        contact_quality += deliverability * 0.4
                
                # Phone validation
                if contact.get('phone'):
                    is_valid, _ = self.validate_phone_number(contact['phone'])
                    if is_valid:
                        contact_quality += 0.2
                
                # LinkedIn validation
                if contact.get('linkedin_url'):
                    if self.validate_linkedin_url(contact['linkedin_url']):
                        contact_quality += 0.2
                
                # Name completeness
                if contact.get('full_name') or (contact.get('first_name') and contact.get('last_name')):
                    contact_quality += 0.1
                
                # Title/position
                if contact.get('title'):
                    contact_quality += 0.1
                
                contact_score = max(contact_score, contact_quality)
            
            score += contact_score * 0.35
        total_weight += 0.35
        
        # Company information completeness
        info_score = 0.0
        if lead_data.get('industry'):
            info_score += 0.05
        if lead_data.get('employee_count'):
            info_score += 0.05
        if lead_data.get('location'):
            info_score += 0.05
        if lead_data.get('description'):
            info_score += 0.05
        if lead_data.get('founded_year'):
            info_score += 0.05
        
        score += info_score
        total_weight += 0.25
        
        # Source diversity bonus
        sources = lead_data.get('sources', [])
        if len(sources) > 1:
            score += 0.05
        total_weight += 0.05
        
        return min(score / total_weight if total_weight > 0 else 0.0, 1.0)

# app/utils/email_finder.py
import aiohttp
import re
from typing import List, Dict, Optional, Set
from bs4 import BeautifulSoup
import itertools

class EmailFinder:
    def __init__(self):
        self.email_patterns = [
            # Common patterns
            "{first}.{last}@{domain}",
            "{first}{last}@{domain}",
            "{first}@{domain}",
            "{last}@{domain}",
            "{first}_{last}@{domain}",
            "{first}-{last}@{domain}",
            "{f}{last}@{domain}",
            "{first}{l}@{domain}",
            # With numbers
            "{first}.{last}1@{domain}",
            "{first}{last}1@{domain}",
            # Role-based
            "info@{domain}",
            "contact@{domain}",
            "sales@{domain}",
            "support@{domain}",
            "hello@{domain}",
        ]
        
    async def find_emails_from_name(self, first_name: str, last_name: str, domain: str) -> List[str]:
        """Generate possible email addresses from name and domain"""
        if not all([first_name, last_name, domain]):
            return []
        
        first = first_name.lower().strip()
        last = last_name.lower().strip()
        f = first[0] if first else ""
        l = last[0] if last else ""
        
        emails = []
        for pattern in self.email_patterns:
            try:
                email = pattern.format(
                    first=first,
                    last=last,
                    f=f,
                    l=l,
                    domain=domain
                )
                emails.append(email)
            except KeyError:
                continue
        
        return list(set(emails))  # Remove duplicates
    
    async def verify_email_existence(self, email: str) -> bool:
        """Verify if email exists (simplified version)"""
        # In production, you'd use services like:
        # - Hunter.io API
        # - ZeroBounce API
        # - NeverBounce API
        
        # Basic SMTP verification (simplified)
        try:
            domain = email.split('@')[1]
            
            # Check MX record exists
            import dns.resolver
            mx_records = dns.resolver.resolve(domain, 'MX')
            return len(mx_records) > 0
            
        except Exception:
            return False
    
    async def extract_emails_from_website(self, url: str) -> Set[str]:
        """Extract emails from website content"""
        emails = set()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        # Extract emails using regex
                        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                        found_emails = re.findall(email_pattern, content)
                        
                        # Filter out common false positives
                        filtered_emails = []
                        for email in found_emails:
                            if not any(ignore in email.lower() for ignore in [
                                'example.com', 'test.com', 'placeholder',
                                'your-email', 'email@email', 'name@company'
                            ]):
                                filtered_emails.append(email.lower())
                        
                        emails.update(filtered_emails)
        
        except Exception as e:
            print(f"Error extracting emails from {url}: {e}")
        
        return emails
    
    async def find_emails_hunter_io(self, domain: str, api_key: str) -> List[Dict]:
        """Find emails using Hunter.io API"""
        if not api_key:
            return []
        
        url = f"https://api.hunter.io/v2/domain-search"
        params = {
            'domain': domain,
            'api_key': api_key,
            'limit': 10
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        emails = data.get('data', {}).get('emails', [])
                        
                        return [{
                            'email': email.get('value'),
                            'first_name': email.get('first_name'),
                            'last_name': email.get('last_name'),
                            'position': email.get('position'),
                            'confidence': email.get('confidence'),
                            'source': 'hunter.io'
                        } for email in emails]
        
        except Exception as e:
            print(f"Error with Hunter.io API: {e}")
        
        return []
    
    async def enrich_contact_with_emails(self, contact: Dict, domain: str) -> Dict:
        """Enrich a contact with possible email addresses"""
        if contact.get('email'):
            return contact  # Already has email
        
        first_name = contact.get('first_name', '')
        last_name = contact.get('last_name', '')
        
        if not (first_name and last_name and domain):
            return contact
        
        # Generate possible emails
        possible_emails = await self.find_emails_from_name(first_name, last_name, domain)
        
        # Verify emails (simplified)
        verified_emails = []
        for email in possible_emails[:5]:  # Limit to top 5
            if await self.verify_email_existence(email):
                verified_emails.append(email)
        
        if verified_emails:
            contact['email'] = verified_emails[0]  # Use the first verified email
            contact['possible_emails'] = verified_emails
            contact['email_confidence'] = 0.8  # Estimated confidence
        
        return contact

# ============================================================================
# MAIN APPLICATION
# ============================================================================

# app/main.py
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time

from app.core.config import settings
from app.core.database import engine
from app.core.logging import setup_logging
from app.models import user, leads, contacts, sentiment
from app.api.endpoints import auth, leads as leads_router, scraping, sentiment as sentiment_router, export

# Setup logging
setup_logging()

# Sentry integration
if settings.SENTRY_DSN:
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        integrations=[FastApiIntegration(auto_enabling=True)],
        traces_sample_rate=0.1,
    )

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Advanced Lead Generation and Sentiment Analysis API",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Metrics middleware
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    # Record metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path
    ).inc()
    
    REQUEST_DURATION.observe(time.time() - start_time)
    
    return response

# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc) if settings.LOG_LEVEL == "DEBUG" else "Something went wrong"
        }
    )

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "timestamp": time.time()
    }

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Include routers
app.include_router(
    auth.router,
    prefix=f"{settings.API_V1_STR}/auth",
    tags=["authentication"]
)

app.include_router(
    leads_router.router,
    prefix=f"{settings.API_V1_STR}/leads",
    tags=["leads"]
)

app.include_router(
    scraping.router,
    prefix=f"{settings.API_V1_STR}/scraping",
    tags=["scraping"]
)

app.include_router(
    sentiment_router.router,
    prefix=f"{settings.API_V1_STR}/sentiment",
    tags=["sentiment"]
)

app.include_router(
    export.router,
    prefix=f"{settings.API_V1_STR}/export",
    tags=["export"]
)

@app.get("/")
async def root():
    return {
        "message": f"{settings.PROJECT_NAME} is running",
        "version": settings.VERSION,
        "docs": f"{settings.API_V1_STR}/docs"
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True if settings.LOG_LEVEL == "DEBUG" else False,
        log_level=settings.LOG_LEVEL.lower()
    )

# ============================================================================
# API ENDPOINTS
# ============================================================================

# app/api/deps.py
from typing import Generator, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db, get_redis
from app.core.security import verify_token
from app.models.user import User
from sqlalchemy import select
import redis.asyncio as aioredis

security = HTTPBearer()

async def get_current_user(
    db: AsyncSession = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """Get current authenticated user"""
    token = credentials.credentials
    username = verify_token(token)
    
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )
    
    result = await db.execute(
        select(User).where(User.email == username, User.is_active == True)
    )
    user = result.scalar_one_or_none()
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    return user

async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user

# app/api/endpoints/leads.py
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc, asc
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.api.deps import get_current_active_user
from app.models.user import User
from app.models.leads import Lead, Contact
from app.schemas.leads import LeadResponse, LeadSearchRequest, LeadCreate, LeadUpdate
from app.services.data_enrichment import DataEnrichmentService

router = APIRouter()

@router.get("/", response_model=List[LeadResponse])
async def get_leads(
    search: LeadSearchRequest = Depends(),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get leads with advanced filtering and search"""
    
    # Build query
    query = select(Lead).where(
        and_(
            Lead.user_id == current_user.id,
            Lead.is_deleted == False
        )
    ).options(selectinload(Lead.contacts))
    
    # Apply filters
    if search.company_name:
        query = query.where(Lead.company_name.ilike(f"%{search.company_name}%"))
    
    if search.industry:
        query = query.where(Lead.industry.in_(search.industry))
    
    if search.location:
        location_filters = [Lead.location.ilike(f"%{loc}%") for loc in search.location]
        query = query.where(or_(*location_filters))
    
    if search.employee_count_min:
        query = query.where(Lead.employee_count >= search.employee_count_min)
    
    if search.employee_count_max:
        query = query.where(Lead.employee_count <= search.employee_count_max)
    
    if search.keywords:
        keyword_filters = []
        for keyword in search.keywords:
            keyword_filters.extend([
                Lead.company_name.ilike(f"%{keyword}%"),
                Lead.description.ilike(f"%{keyword}%"),
                Lead.industry.ilike(f"%{keyword}%")
            ])
        query = query.where(or_(*keyword_filters))
    
    if search.domains:
        query = query.where(Lead.domain.in_(search.domains))
    
    if search.min_quality_score:
        query = query.where(Lead.data_quality_score >= search.min_quality_score)
    
    if search.verification_status:
        query = query.where(Lead.verification_status == search.verification_status)
    
    # Sorting
    if search.sort_order == "desc":
        query = query.order_by(desc(getattr(Lead, search.sort_by)))
    else:
        query = query.order_by(asc(getattr(Lead, search.sort_by)))
    
    # Pagination
    query = query.offset(search.skip).limit(search.limit)
    
    # Execute query
    result = await db.execute(query)
    leads = result.scalars().all()
    
    return leads

@router.get("/{lead_id}", response_model=LeadResponse)
async def get_lead(
    lead_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get a specific lead by ID"""
    
    result = await db.execute(
        select(Lead).where(
            and_(
                Lead.id == lead_id,
                Lead.user_id == current_user.id,
                Lead.is_deleted == False
            )
        ).options(selectinload(Lead.contacts))
    )
    
    lead = result.scalar_one_or_none()
    
    if not lead:
        raise HTTPException(
            status_code=404,
            detail="Lead not found"
        )
    
    return lead

@router.post("/", response_model=LeadResponse)
async def create_lead(
    lead_data: LeadCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create a new lead"""
    
    # Check if lead already exists for this user
    existing = await db.execute(
        select(Lead).where(
            and_(
                Lead.company_name == lead_data.company_name,
                Lead.user_id == current_user.id,
                Lead.is_deleted == False
            )
        )
    )
    
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=400,
            detail="Lead with this company name already exists"
        )
    
    # Create lead
    lead = Lead(
        user_id=current_user.id,
        **lead_data.dict(exclude={'contacts'})
    )
    
    db.add(lead)
    await db.flush()  # Get the lead ID
    
    # Add contacts
    for contact_data in lead_data.contacts:
        contact = Contact(
            lead_id=lead.id,
            **contact_data.dict()
        )
        db.add(contact)
    
    await db.commit()
    await db.refresh(lead)
    
    # Load contacts
    result = await db.execute(
        select(Lead).where(Lead.id == lead.id).options(selectinload(Lead.contacts))
    )
    lead_with_contacts = result.scalar_one()
    
    return lead_with_contacts

@router.put("/{lead_id}", response_model=LeadResponse)
async def update_lead(
    lead_id: int,
    lead_update: LeadUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update a lead"""
    
    result = await db.execute(
        select(Lead).where(
            and_(
                Lead.id == lead_id,
                Lead.user_id == current_user.id,
                Lead.is_deleted == False
            )
        )
    )
    
    lead = result.scalar_one_or_none()
    
    if not lead:
        raise HTTPException(
            status_code=404,
            detail="Lead not found"
        )
    
    # Update fields
    update_data = lead_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(lead, field, value)
    
    await db.commit()
    await db.refresh(lead)
    
    return lead

@router.delete("/{lead_id}")
async def delete_lead(
    lead_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Soft delete a lead"""
    
    result = await db.execute(
        select(Lead).where(
            and_(
                Lead.id == lead_id,
                Lead.user_id == current_user.id,
                Lead.is_deleted == False
            )
        )
    )
    
    lead = result.scalar_one_or_none()
    
    if not lead:
        raise HTTPException(
            status_code=404,
            detail="Lead not found"
        )
    
    # Soft delete
    lead.is_deleted = True
    lead.deleted_at = func.now()
    
    await db.commit()
    
    return {"message": "Lead deleted successfully"}

@router.post("/{lead_id}/enrich")
async def enrich_lead(
    lead_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Enrich lead data with additional information"""
    
    result = await db.execute(
        select(Lead).where(
            and_(
                Lead.id == lead_id,
                Lead.user_id == current_user.id,
                Lead.is_deleted == False
            )
        ).options(selectinload(Lead.contacts))
    )
    
    lead = result.scalar_one_or_none()
    
    if not lead:
        raise HTTPException(
            status_code=404,
            detail="Lead not found"
        )
    
    # Enrich data
    enrichment_service = DataEnrichmentService()
    enriched_data = await enrichment_service.enrich_lead(lead)
    
    # Update lead with enriched data
    for field, value in enriched_data.items():
        if hasattr(lead, field) and value:
            setattr(lead, field, value)
    
    await db.commit()
    await db.refresh(lead)
    
    return {"message": "Lead enriched successfully", "lead": lead}

# ============================================================================
# SCRAPING SERVICE
# ============================================================================

# app/services/scraping_service.py
import asyncio
import aiohttp
from playwright.async_api import async_playwright
from typing import List, Dict, Optional, Set
import random
import json
import re
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import httpx

from app.core.config import settings
from app.utils.anti_bot import StealthBrowser
from app.utils.data_validator import DataValidator
from app.utils.email_finder import EmailFinder
from app.models.leads import Lead, Contact

class ScrapingService:
    def __init__(self):
        self.stealth_browser = StealthBrowser()
        self.data_validator = DataValidator()
        self.email_finder = EmailFinder()
        self.session_pool = {}
        
    async def scrape_companies_batch(self, company_names: List[str], user_id: int) -> List[Dict]:
        """Main scraping orchestrator for batch processing (free approach)"""
        
        # Initialize free proxy manager
        await self.stealth_browser.proxy_manager.load_proxies()
        
        # Conservative semaphore for free approach
        semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_REQUESTS)
        
        # Create scraping tasks
        tasks = []
        for company_name in company_names:
            task = asyncio.create_task(
                self._scrape_single_company_with_semaphore(semaphore, company_name, user_id)
            )
            tasks.append(task)
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return valid results
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                print(f"Scraping error: {result}")
            elif result:
                valid_results.append(result)
        
        return valid_results
    
    async def _scrape_single_company_with_semaphore(
        self, 
        semaphore: asyncio.Semaphore, 
        company_name: str, 
        user_id: int
    ) -> Optional[Dict]:
        """Scrape data for a single company with conservative rate limiting"""
        
        async with semaphore:
            # Longer delay for free approach
            delay = random.uniform(settings.REQUEST_DELAY_MIN, settings.REQUEST_DELAY_MAX)
            await asyncio.sleep(delay)
            
            return await self._scrape_single_company(company_name, user_id)
    
    async def _scrape_single_company(self, company_name: str, user_id: int) -> Optional[Dict]:
        """Scrape comprehensive data for a single company (free sources focus)"""
        
        company_data = {
            'company_name': company_name,
            'user_id': user_id,
            'contacts': [],
            'company_info': {},
            'sources': [],
            'scraping_metadata': {
                'scraped_at': datetime.utcnow().isoformat(),
                'difficulty_score': 1,
                'success_rate': 0.0
            }
        }
        
        # Simplified scraping strategy for free approach
        scrapers = [
            ('google_search', self._scrape_google_search),
            ('company_website', self._scrape_company_website),
            ('duckduckgo_search', self._scrape_duckduckgo_search),  # Alternative search engine
        ]
        
        successful_scrapers = 0
        total_scrapers = len(scrapers)
        
        for scraper_name, scraper_func in scrapers:
            try:
                print(f"Running {scraper_name} for {company_name}")
                data = await scraper_func(company_name)
                
                if data:
                    company_data = self._merge_company_data(company_data, data)
                    company_data['sources'].append(scraper_name)
                    successful_scrapers += 1
                    
                    # Early exit if we have basic good data
                    if (company_data.get('domain') and 
                        len(company_data.get('contacts', [])) >= 1):
                        break
                        
            except Exception as e:
                print(f"Error in {scraper_name} for {company_name}: {e}")
                continue
        
        # Calculate success rate and quality score
        company_data['scraping_metadata']['success_rate'] = successful_scrapers / total_scrapers
        company_data['data_quality_score'] = self.data_validator.calculate_data_quality_score(company_data)
        
        # Enrich contacts with email finding
        if company_data.get('domain'):
            enriched_contacts = []
            for contact in company_data.get('contacts', []):
                enriched_contact = await self.email_finder.enrich_contact_with_emails(
                    contact, company_data['domain']
                )
                enriched_contacts.append(enriched_contact)
            company_data['contacts'] = enriched_contacts
        
        return company_data if company_data.get('domain') or company_data.get('contacts') else None
    
    async def _scrape_google_search(self, company_name: str) -> Optional[Dict]:
        """Scrape basic company information from Google search"""
        
        search_query = f'"{company_name}" official website'
        search_url = f"https://www.google.com/search?q={search_query}&num=10"
        
        async with async_playwright() as p:
            page = await self.stealth_browser.create_stealth_page(p)
            
            try:
                await page.goto(search_url, wait_until='domcontentloaded', timeout=30000)
                await page.wait_for_timeout(2000)
                
                # Extract search results
                results = await page.evaluate("""
                    () => {
                        const results = [];
                        const searchResults = document.querySelectorAll('div.g');
                        
                        searchResults.forEach((result, index) => {
                            if (index < 5) { // Top 5 results
                                const titleElement = result.querySelector('h3');
                                const linkElement = result.querySelector('a[href]');
                                const snippetElement = result.querySelector('.VwiC3b');
                                
                                if (titleElement && linkElement) {
                                    results.push({
                                        title: titleElement.innerText,
                                        url: linkElement.href,
                                        snippet: snippetElement ? snippetElement.innerText : ''
                                    });
                                }
                            }
                        });
                        
                        return results;
                    }
                """)
                
                # Find the most likely official website
                domain = None
                for result in results:
                    url = result.get('url', '')
                    if any(word in result.get('title', '').lower() for word in ['official', 'home', company_name.lower()]):
                        try:
                            from urllib.parse import urlparse
                            parsed = urlparse(url)
                            domain = parsed.netloc.replace('www.', '')
                            break
                        except:
                            continue
                
                if not domain and results:
                    # Fallback to first result
                    try:
                        from urllib.parse import urlparse
                        parsed = urlparse(results[0]['url'])
                        domain = parsed.netloc.replace('www.', '')
                    except:
                        pass
                
                return {
                    'domain': domain,
                    'search_results': results,
                    'source': 'google_search'
                } if domain else None
                
            except Exception as e:
                print(f"Error in Google search for {company_name}: {e}")
                return None
            finally:
                await page.close()
    
    async def _scrape_company_website(self, company_name: str) -> Optional[Dict]:
        """Scrape company's main website for comprehensive information"""
        
        # First, find the domain
        domain_data = await self._scrape_google_search(company_name)
        if not domain_data or not domain_data.get('domain'):
            return None
        
        domain = domain_data['domain']
        website_url = f"https://{domain}"
        
        async with async_playwright() as p:
            page = await self.stealth_browser.create_stealth_page(p)
            
            try:
                await page.goto(website_url, wait_until='domcontentloaded', timeout=30000)
                await page.wait_for_timeout(3000)
                
                # Extract comprehensive company information
                company_info = await page.evaluate("""
                    () => {
                        const data = {};
                        
                        // Meta information
                        const description = document.querySelector('meta[name="description"]');
                        if (description) {
                            data.description = description.content;
                        }
                        
                        const keywords = document.querySelector('meta[name="keywords"]');
                        if (keywords) {
                            data.keywords = keywords.content.split(',').map(k => k.trim());
                        }
                        
                        // Company information from text
                        const bodyText = document.body.innerText.toLowerCase();
                        
                        // Extract founded year
                        const yearMatches = bodyText.match(/(?:founded|established|since)\\s+(?:in\\s+)?(19|20)\\d{2}/);
                        if (yearMatches) {
                            data.founded_year = parseInt(yearMatches[0].match(/(19|20)\\d{2}/)[0]);
                        }
                        
                        // Extract employee count indicators
                        const employeePatterns = [
                            /([0-9,]+)\\s*(?:\\+)?\\s*employees/,
                            /team\\s+of\\s+([0-9,]+)/,
                            /([0-9,]+)\\s*people/
                        ];
                        
                        for (const pattern of employeePatterns) {
                            const match = bodyText.match(pattern);
                            if (match) {
                                const count = parseInt(match[1].replace(/,/g, ''));
                                if (count > 1 && count < 1000000) {
                                    data.employee_count = count;
                                    break;
                                }
                            }
                        }
                        
                        // Extract location information
                        const locationPatterns = [
                            /(?:headquarters|based|located)\\s+in\\s+([a-z\\s,]+)(?:,\\s*[a-z]{2,})?/,
                            /([a-z\\s]+),\\s*(?:ca|ny|tx|fl|il|wa|ma|co|ga|nc|va|oh|pa|az|nj|tn|in|mo|md|wi|mn|or|sc|al|la|ky|ar|ia|ut|nv|nm|wv|ne|id|nh|me|ri|mt|de|sd|nd|ak|dc|hi|vt|wy)/
                        ];
                        
                        for (const pattern of locationPatterns) {
                            const match = bodyText.match(pattern);
                            if (match) {
                                data.location = match[1].replace(/^\\s+|\\s+$/g, '');
                                break;
                            }
                        }
                        
                        // Extract contact information
                        const emailRegex = /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}/g;
                        const phoneRegex = /(?:\\+?1[-.\s]?)?\\(?[0-9]{3}\\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}/g;
                        
                        const emails = [...new Set(document.body.innerText.match(emailRegex) || [])];
                        const phones = [...new Set(document.body.innerText.match(phoneRegex) || [])];
                        
                        // Filter emails (remove common false positives)
                        data.emails = emails.filter(email => 
                            !email.includes('example.com') && 
                            !email.includes('test.com') &&
                            !email.includes('placeholder')
                        ).slice(0, 5);
                        
                        data.phones = phones.slice(0, 3);
                        
                        // Extract social media links
                        const socialLinks = {};
                        const links = Array.from(document.querySelectorAll('a[href]'));
                        
                        links.forEach(link => {
                            const href = link.href.toLowerCase();
                            if (href.includes('linkedin.com')) {
                                socialLinks.linkedin = link.href;
                            } else if (href.includes('twitter.com')) {
                                socialLinks.twitter = link.href;
                            } else if (href.includes('facebook.com')) {
                                socialLinks.facebook = link.href;
                            }
                        });
                        
                        data.social_links = socialLinks;
                        
                        return data;
                    }
                """)
                
                # Scrape team/about page for contacts
                contacts = await self._scrape_team_pages(page, domain)
                
                # Extract additional emails from website
                website_emails = await self.email_finder.extract_emails_from_website(website_url)
                
                return {
                    'domain': domain,
                    'company_info': company_info,
                    'contacts': contacts,
                    'website_emails': list(website_emails),
                    'source': 'company_website'
                }
                
            except Exception as e:
                print(f"Error scraping website {website_url}: {e}")
                return None
            finally:
                await page.close()
    
    async def _scrape_team_pages(self, page, domain: str) -> List[Dict]:
        """Scrape team/about pages for contact information"""
        contacts = []
        
        # Common team page URLs
        team_urls = [
            f"https://{domain}/team",
            f"https://{domain}/about",
            f"https://{domain}/about-us",
            f"https://{domain}/leadership",
            f"https://{domain}/management",
            f"https://{domain}/people",
            f"https://{domain}/our-team",
        ]
        
        for url in team_urls:
            try:
                await page.goto(url, wait_until='domcontentloaded', timeout=15000)
                await page.wait_for_timeout(2000)
                
                # Check if page exists (not 404)
                if page.url != url:
                    continue
                
                # Extract team member information
                team_data = await page.evaluate("""
                    () => {
                        const teamMembers = [];
                        
                        // Look for common team member selectors
                        const selectors = [
                            '.team-member', '.staff-member', '.person', '.employee',
                            '.bio', '.profile', '[data-team]', '.leadership',
                            '.founder', '.executive'
                        ];
                        
                        const processedNames = new Set();
                        
                        selectors.forEach(selector => {
                            const elements = document.querySelectorAll(selector);
                            
                            elements.forEach(element => {
                                const member = {};
                                
                                // Extract name
                                const nameSelectors = ['h1', 'h2', 'h3', 'h4', '.name', '.title', 'strong'];
                                for (const nameSelector of nameSelectors) {
                                    const nameElement = element.querySelector(nameSelector);
                                    if (nameElement && nameElement.innerText.trim()) {
                                        const nameText = nameElement.innerText.trim();
                                        if (nameText.length > 2 && nameText.length < 50 && 
                                            /^[a-zA-Z\\s\\.\\-']+$/.test(nameText)) {
                                            member.full_name = nameText;
                                            break;
                                        }
                                    }
                                }
                                
                                if (!member.full_name) return;
                                
                                // Skip if already processed
                                if (processedNames.has(member.full_name.toLowerCase())) {
                                    return;
                                }
                                processedNames.add(member.full_name.toLowerCase());
                                
                                // Extract title/position
                                const titleSelectors = ['.title', '.position', '.role', '.job-title', 'p', 'span'];
                                for (const titleSelector of titleSelectors) {
                                    const titleElement = element.querySelector(titleSelector);
                                    if (titleElement && titleElement.innerText.trim() && 
                                        titleElement.innerText.trim() !== member.full_name) {
                                        member.title = titleElement.innerText.trim();
                                        break;
                                    }
                                }
                                
                                // Extract email if present
                                const emailMatch = element.innerText.match(/[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}/);
                                if (emailMatch) {
                                    member.email = emailMatch[0];
                                }
                                
                                // Extract LinkedIn if present
                                const linkedinLink = element.querySelector('a[href*="linkedin.com"]');
                                if (linkedinLink) {
                                    member.linkedin_url = linkedinLink.href;
                                }
                                
                                // Parse name into first/last
                                const nameParts = member.full_name.split(' ');
                                if (nameParts.length >= 2) {
                                    member.first_name = nameParts[0];
                                    member.last_name = nameParts.slice(-1)[0];
                                }
                                
                                // Determine seniority level
                                if (member.title) {
                                    const title = member.title.toLowerCase();
                                    if (title.includes('ceo') || title.includes('founder') || title.includes('president')) {
                                        member.seniority_level = 'C-level';
                                    } else if (title.includes('vp') || title.includes('vice president')) {
                                        member.seniority_level = 'VP';
                                    } else if (title.includes('director') || title.includes('head of')) {
                                        member.seniority_level = 'Director';
                                    } else if (title.includes('manager') || title.includes('lead')) {
                                        member.seniority_level = 'Manager';
                                    }
                                }
                                
                                teamMembers.push(member);
                            });
                        });
                        
                        return teamMembers;
                    }
                """)
                
                contacts.extend(team_data)
                
                # If we found team members, no need to check other URLs
                if team_data:
                    break
                    
            except Exception as e:
                print(f"Error scraping team page {url}: {e}")
                continue
        
        return contacts[:10]  # Limit to top 10 contacts
    
    async def _scrape_linkedin_company(self, company_name: str) -> Optional[Dict]:
        """Scrape LinkedIn company page (simplified due to anti-scraping measures)"""
        # Note: LinkedIn has strong anti-scraping measures
        # In production, use LinkedIn API or specialized services
        
        search_url = f"https://www.linkedin.com/search/results/companies/?keywords={company_name.replace(' ', '%20')}"
        
        async with async_playwright() as p:
            page = await self.stealth_browser.create_stealth_page(p)
            
            try:
                await page.goto(search_url, wait_until='domcontentloaded', timeout=30000)
                await page.wait_for_timeout(5000)  # Wait for dynamic content
                
                # Check if we're blocked or redirected to login
                current_url = page.url
                if 'login' in current_url or 'challenge' in current_url:
                    print("LinkedIn access blocked - login required")
                    return None
                
                # Extract basic company information
                company_data = await page.evaluate("""
                    () => {
                        const companies = [];
                        
                        // Look for company search results
                        const resultElements = document.querySelectorAll('[data-test-id="search-result"]');
                        
                        resultElements.forEach((element, index) => {
                            if (index < 3) { // Top 3 results
                                const company = {};
                                
                                // Company name
                                const nameElement = element.querySelector('h3 a span[aria-hidden="true"]');
                                if (nameElement) {
                                    company.name = nameElement.innerText.trim();
                                }
                                
                                // Company description
                                const descElement = element.querySelector('.entity-result__summary');
                                if (descElement) {
                                    company.description = descElement.innerText.trim();
                                }
                                
                                // Employee count
                                const statsElement = element.querySelector('.entity-result__summary .t-black--light');
                                if (statsElement) {
                                    const statsText = statsElement.innerText;
                                    const employeeMatch = statsText.match(/([0-9,]+)\\s*employees/);
                                    if (employeeMatch) {
                                        company.employee_count = parseInt(employeeMatch[1].replace(/,/g, ''));
                                    }
                                }
                                
                                // Company URL
                                const linkElement = element.querySelector('h3 a');
                                if (linkElement) {
                                    company.linkedin_url = linkElement.href;
                                }
                                
                                companies.push(company);
                            }
                        });
                        
                        return companies;
                    }
                """)
                
                # Find the best match
                best_match = None
                for company in company_data:
                    company_name_lower = company_name.lower()
                    result_name_lower = company.get('name', '').lower()
                    
                    if company_name_lower in result_name_lower or result_name_lower in company_name_lower:
                        best_match = company
                        break
                
                return {
                    'company_info': best_match,
                    'linkedin_data': company_data,
                    'source': 'linkedin'
                } if best_match else None
                
            except Exception as e:
                print(f"Error scraping LinkedIn for {company_name}: {e}")
                return None
            finally:
                await page.close()
    
    async def _scrape_crunchbase(self, company_name: str) -> Optional[Dict]:
        """Scrape Crunchbase for company information"""
        
        search_url = f"https://www.crunchbase.com/discover/organizations?query={company_name.replace(' ', '%20')}"
        
        async with async_playwright() as p:
            page = await self.stealth_browser.create_stealth_page(p)
            
            try:
                await page.goto(search_url, wait_until='domcontentloaded', timeout=30000)
                await page.wait_for_timeout(3000)
                
                # Extract company information from search results
                company_data = await page.evaluate("""
                    () => {
                        const companies = [];
                        
                        // Look for search result cards
                        const cardSelectors = [
                            '.search-results-list .list-card',
                            '.search-results .result-card',
                            '[data-testid="search-result"]'
                        ];
                        
                        for (const selector of cardSelectors) {
                            const cards = document.querySelectorAll(selector);
                            
                            cards.forEach((card, index) => {
                                if (index < 3) { // Top 3 results
                                    const company = {};
                                    
                                    // Company name
                                    const nameElement = card.querySelector('h3 a, .name a, .title a');
                                    if (nameElement) {
                                        company.name = nameElement.innerText.trim();
                                        company.crunchbase_url = nameElement.href;
                                    }
                                    
                                    // Description
                                    const descElement = card.querySelector('.description, .summary, p');
                                    if (descElement) {
                                        company.description = descElement.innerText.trim();
                                    }
                                    
                                    // Funding information
                                    const fundingElement = card.querySelector('.funding, .total-funding');
                                    if (fundingElement) {
                                        company.total_funding = fundingElement.innerText.trim();
                                    }
                                    
                                    // Industry
                                    const industryElement = card.querySelector('.industry, .categories');
                                    if (industryElement) {
                                        company.industry = industryElement.innerText.trim();
                                    }
                                    
                                    // Location
                                    const locationElement = card.querySelector('.location, .headquarters');
                                    if (locationElement) {
                                        company.location = locationElement.innerText.trim();
                                    }
                                    
                                    if (company.name) {
                                        companies.push(company);
                                    }
                                }
                            });
                            
                            if (companies.length > 0) break;
                        }
                        
                        return companies;
                    }
                """)
                
                # Find best match
                best_match = None
                for company in company_data:
                    company_name_lower = company_name.lower()
                    result_name_lower = company.get('name', '').lower()
                    
                    if company_name_lower in result_name_lower or result_name_lower in company_name_lower:
                        best_match = company
                        break
                
                return {
                    'company_info': best_match,
                    'funding_data': company_data,
                    'source': 'crunchbase'
                } if best_match else None
                
            except Exception as e:
                print(f"Error scraping Crunchbase for {company_name}: {e}")
                return None
            finally:
                await page.close()
    
    async def _scrape_duckduckgo_search(self, company_name: str) -> Optional[Dict]:
        """Scrape company information from DuckDuckGo search (alternative to Google)"""
        
        search_query = f'"{company_name}" official website contact'
        search_url = f"https://duckduckgo.com/?q={search_query.replace(' ', '+')}&t=h_&ia=web"
        
        async with async_playwright() as p:
            page = await self.stealth_browser.create_stealth_page(p)
            
            try:
                await page.goto(search_url, wait_until='domcontentloaded', timeout=30000)
                await page.wait_for_timeout(3000)
                
                # Extract search results
                results = await page.evaluate("""
                    () => {
                        const results = [];
                        const searchResults = document.querySelectorAll('[data-testid="result"]');
                        
                        searchResults.forEach((result, index) => {
                            if (index < 5) { // Top 5 results
                                const titleElement = result.querySelector('h2 a');
                                const snippetElement = result.querySelector('[data-result="snippet"]');
                                
                                if (titleElement) {
                                    results.push({
                                        title: titleElement.innerText,
                                        url: titleElement.href,
                                        snippet: snippetElement ? snippetElement.innerText : ''
                                    });
                                }
                            }
                        });
                        
                        return results;
                    }
                """)
                
                # Find the most likely official website
                domain = None
                for result in results:
                    url = result.get('url', '')
                    title = result.get('title', '').lower()
                    
                    # Look for official indicators
                    if any(word in title for word in ['official', 'home', company_name.lower()]):
                        try:
                            from urllib.parse import urlparse
                            parsed = urlparse(url)
                            domain = parsed.netloc.replace('www.', '')
                            break
                        except:
                            continue
                
                if not domain and results:
                    # Fallback to first result
                    try:
                        from urllib.parse import urlparse
                        parsed = urlparse(results[0]['url'])
                        domain = parsed.netloc.replace('www.', '')
                    except:
                        pass
                
                return {
                    'domain': domain,
                    'search_results': results,
                    'source': 'duckduckgo_search'
                } if domain else None
                
            except Exception as e:
                print(f"Error in DuckDuckGo search for {company_name}: {e}")
                return None
            finally:
                await page.close()
    
    async def _scrape_apollo_io(self, company_name: str) -> Optional[Dict]:
        """Simplified Apollo.io scraping (free tier approach)"""
        # Note: This is a simplified version without API access
        # In practice, you'd use their free tier API or public search
        try:
            search_url = f"https://app.apollo.io/companies?q={company_name.replace(' ', '%20')}"
            
            async with httpx.AsyncClient() as client:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                }
                
                response = await client.get(search_url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    # Basic HTML parsing (Apollo.io requires login for most features)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Look for any publicly available company information
                    # This is very limited without API access
                    company_links = soup.find_all('a', href=True)
                    
                    for link in company_links:
                        if company_name.lower() in link.get_text().lower():
                            return {
                                'company_info': {
                                    'name': company_name,
                                    'apollo_url': link.get('href')
                                },
                                'source': 'apollo_basic'
                            }
                
        except Exception as e:
            print(f"Error with basic Apollo search for {company_name}: {e}")
        
        return None
    
    async def _scrape_clearbit(self, company_name: str) -> Optional[Dict]:
        """Basic company enrichment without API (removed premium dependency)"""
        # This method now focuses on publicly available information only
        return None
    
    def _merge_company_data(self, existing_data: Dict, new_data: Dict) -> Dict:
        """Intelligently merge data from multiple sources"""
        
        # Source priority: company_website > linkedin > crunchbase > apollo > google_search
        source_priority = {
            'company_website': 5,
            'linkedin': 4,
            'crunchbase': 3,
            'apollo': 2,
            'google_search': 1
        }
        
        new_source = new_data.get('source', '')
        new_priority = source_priority.get(new_source, 0)
        
        # Merge company_info
        if 'company_info' in new_data:
            existing_info = existing_data.get('company_info', {})
            new_info = new_data['company_info']
            
            for key, value in new_info.items():
                if value and (not existing_info.get(key) or 
                             source_priority.get(existing_data.get('sources', [])[-1] if existing_data.get('sources') else '', 0) <= new_priority):
                    existing_info[key] = value
            
            existing_data['company_info'] = existing_info
        
        # Merge domain
        if new_data.get('domain') and (not existing_data.get('domain') or new_priority >= 3):
            existing_data['domain'] = new_data['domain']
        
        # Merge contacts (append all, deduplicate later)
        if 'contacts' in new_data:
            existing_contacts = existing_data.get('contacts', [])
            new_contacts = new_data['contacts']
            
            # Simple deduplication by name
            existing_names = {contact.get('full_name', '').lower() for contact in existing_contacts}
            
            for contact in new_contacts:
                contact_name = contact.get('full_name', '').lower()
                if contact_name and contact_name not in existing_names:
                    existing_contacts.append(contact)
                    existing_names.add(contact_name)
            
            existing_data['contacts'] = existing_contacts
        
        # Merge emails
        if new_data.get('website_emails'):
            existing_emails = set(existing_data.get('website_emails', []))
            new_emails = set(new_data['website_emails'])
            existing_data['website_emails'] = list(existing_emails.union(new_emails))
        
        return existing_data

# ============================================================================
# SENTIMENT ANALYSIS SERVICE
# ============================================================================

# app/services/sentiment_service.py
import asyncio
import aiohttp
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json
import re
from textblob import TextBlob
import openai
from transformers import pipeline
import yfinance as yf
from bs4 import BeautifulSoup
import feedparser

from app.core.config import settings

# app/services/sentiment_service.py
import asyncio
import aiohttp
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json
import re
from textblob import TextBlob
from bs4 import BeautifulSoup
import feedparser

from app.core.config import settings

class SentimentAnalysisService:
    def __init__(self):
        # Initialize free sentiment models only
        try:
            from transformers import pipeline
            self.finbert = pipeline(
                "sentiment-analysis", 
                model="ProsusAI/finbert",
                return_all_scores=True
            )
        except Exception:
            print("FinBERT not available, using TextBlob only")
            self.finbert = None
            
        try:
            from transformers import pipeline
            self.general_sentiment = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
        except Exception:
            print("RoBERTa not available, using TextBlob only")
            self.general_sentiment = None
    
    async def analyze_company_sentiment(self, company_name: str, days_back: int = 30) -> Dict:
        """Comprehensive sentiment analysis using free tools only"""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Gather news data from free sources
        news_data = await self._gather_company_news(company_name, start_date, end_date)
        
        if not news_data:
            return {
                'company_name': company_name,
                'error': 'No news data found',
                'confidence_score': 0.0
            }
        
        # Analyze sentiment using available free methods
        sentiment_results = []
        
        # TextBlob analysis (always available)
        textblob_results = self._analyze_with_textblob(news_data)
        sentiment_results.append(textblob_results)
        
        # FinBERT analysis (if available)
        if self.finbert:
            finbert_results = await self._analyze_with_finbert(news_data)
            sentiment_results.append(finbert_results)
        
        # General RoBERTa analysis (if available)
        if self.general_sentiment:
            roberta_results = await self._analyze_with_roberta(news_data)
            sentiment_results.append(roberta_results)
        
        # Simple keyword-based analysis (backup)
        keyword_results = self._analyze_with_keywords(news_data)
        sentiment_results.append(keyword_results)
        
        # Aggregate results
        final_analysis = self._aggregate_sentiment_results(
            company_name, sentiment_results, news_data
        )
        
        return final_analysis
    
    async def _gather_company_news(self, company_name: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Gather news articles about the company from multiple sources"""
        
        news_sources = [
            self._scrape_google_news,
            self._scrape_yahoo_finance,
            self._scrape_bing_news,
            self._scrape_reuters,
            self._scrape_bloomberg_search,
        ]
        
        all_articles = []
        
        for source_func in news_sources:
            try:
                articles = await source_func(company_name, start_date, end_date)
                all_articles.extend(articles)
            except Exception as e:
                print(f"Error with news source {source_func.__name__}: {e}")
                continue
        
        # Deduplicate articles by title similarity
        deduplicated = self._deduplicate_articles(all_articles)
        
        # Sort by date (newest first) and limit
        deduplicated.sort(key=lambda x: x.get('published_date', datetime.min), reverse=True)
        
        return deduplicated[:20]  # Limit to 20 most recent articles
    
    async def _scrape_google_news(self, company_name: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Scrape Google News for company mentions"""
        
        articles = []
        search_query = f'"{company_name}" OR {company_name.replace(" ", " OR ")}'
        
        # Google News RSS feed
        news_url = f"https://news.google.com/rss/search?q={search_query}&hl=en-US&gl=US&ceid=US:en"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(news_url, timeout=10) as response:
                    if response.status == 200:
                        content = await response.text()
                        feed = feedparser.parse(content)
                        
                        for entry in feed.entries[:10]:  # Limit to 10 articles
                            try:
                                pub_date = datetime(*entry.published_parsed[:6])
                                
                                if start_date <= pub_date <= end_date:
                                    # Fetch full article content
                                    article_content = await self._fetch_article_content(entry.link)
                                    
                                    articles.append({
                                        'title': entry.title,
                                        'url': entry.link,
                                        'content': article_content,
                                        'published_date': pub_date,
                                        'source': 'Google News',
                                        'summary': entry.summary if hasattr(entry, 'summary') else ''
                                    })
                            except Exception:
                                continue
        
        except Exception as e:
            print(f"Error scraping Google News: {e}")
        
        return articles
    
    async def _scrape_yahoo_finance(self, company_name: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Scrape Yahoo Finance for company news"""
        
        articles = []
        
        try:
            # Try to get ticker symbol first
            ticker_symbol = await self._get_ticker_symbol(company_name)
            
            if ticker_symbol:
                # Use yfinance to get news
                try:
                    ticker = yf.Ticker(ticker_symbol)
                    news = ticker.news
                    
                    for item in news[:5]:  # Limit to 5 articles
                        try:
                            pub_date = datetime.fromtimestamp(item.get('providerPublishTime', 0))
                            
                            if start_date <= pub_date <= end_date:
                                # Fetch full content
                                content = await self._fetch_article_content(item.get('link', ''))
                                
                                articles.append({
                                    'title': item.get('title', ''),
                                    'url': item.get('link', ''),
                                    'content': content,
                                    'published_date': pub_date,
                                    'source': 'Yahoo Finance',
                                    'summary': item.get('summary', '')
                                })
                        except Exception:
                            continue
                            
                except Exception as e:
                    print(f"Error with yfinance for {ticker_symbol}: {e}")
            
            # Fallback: search Yahoo Finance directly
            search_url = f"https://finance.yahoo.com/search?p={company_name.replace(' ', '+')}"
            # Additional scraping logic here...
            
        except Exception as e:
            print(f"Error scraping Yahoo Finance: {e}")
        
        return articles
    
    async def _scrape_bing_news(self, company_name: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Scrape Bing News for company mentions"""
        
        articles = []
        search_query = f"{company_name} news"
        search_url = f"https://www.bing.com/news/search?q={search_query.replace(' ', '+')}&qft=interval%3d%227%22"
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                async with session.get(search_url, headers=headers, timeout=10) as response:
                    if response.status == 200:
                        content = await response.text()
                        soup = BeautifulSoup(content, 'html.parser')
                        
                        # Extract news articles
                        news_cards = soup.find_all('div', class_='news-card')
                        
                        for card in news_cards[:5]:  # Limit to 5 articles
                            try:
                                title_elem = card.find('a', class_='title')
                                if not title_elem:
                                    continue
                                
                                title = title_elem.get_text(strip=True)
                                url = title_elem.get('href', '')
                                
                                # Extract date
                                date_elem = card.find('span', class_='time')
                                if date_elem:
                                    date_text = date_elem.get_text(strip=True)
                                    # Parse relative dates like "2 hours ago", "1 day ago"
                                    pub_date = self._parse_relative_date(date_text)
                                else:
                                    pub_date = datetime.now()
                                
                                if start_date <= pub_date <= end_date:
                                    # Fetch content
                                    content = await self._fetch_article_content(url)
                                    
                                    articles.append({
                                        'title': title,
                                        'url': url,
                                        'content': content,
                                        'published_date': pub_date,
                                        'source': 'Bing News',
                                        'summary': ''
                                    })
                            except Exception:
                                continue
        
        except Exception as e:
            print(f"Error scraping Bing News: {e}")
        
        return articles
    
    async def _scrape_reuters(self, company_name: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Scrape Reuters for company news"""
        
        articles = []
        search_url = f"https://www.reuters.com/site-search/?query={company_name.replace(' ', '%20')}"
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                async with session.get(search_url, headers=headers, timeout=10) as response:
                    if response.status == 200:
                        content = await response.text()
                        soup = BeautifulSoup(content, 'html.parser')
                        
                        # Extract articles
                        article_links = soup.find_all('a', href=re.compile(r'/\d{4}/\d{2}/\d{2}/'))
                        
                        for link in article_links[:3]:  # Limit to 3 articles
                            try:
                                article_url = f"https://www.reuters.com{link.get('href')}"
                                title = link.get_text(strip=True)
                                
                                # Fetch full article
                                article_data = await self._fetch_reuters_article(article_url)
                                
                                if article_data:
                                    pub_date = article_data.get('published_date', datetime.now())
                                    
                                    if start_date <= pub_date <= end_date:
                                        articles.append({
                                            'title': title,
                                            'url': article_url,
                                            'content': article_data.get('content', ''),
                                            'published_date': pub_date,
                                            'source': 'Reuters',
                                            'summary': article_data.get('summary', '')
                                        })
                            except Exception:
                                continue
        
        except Exception as e:
            print(f"Error scraping Reuters: {e}")
        
        return articles
    
    async def _scrape_bloomberg_search(self, company_name: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Scrape Bloomberg search results"""
        
        articles = []
        search_url = f"https://www.bloomberg.com/search?query={company_name.replace(' ', '%20')}"
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                async with session.get(search_url, headers=headers, timeout=10) as response:
                    if response.status == 200:
                        content = await response.text()
                        # Bloomberg has heavy anti-scraping measures
                        # This is a simplified version
                        soup = BeautifulSoup(content, 'html.parser')
                        
                        # Look for article links
                        links = soup.find_all('a', href=re.compile(r'/news/articles/'))
                        
                        for link in links[:2]:  # Limit to 2 articles
                            try:
                                article_url = f"https://www.bloomberg.com{link.get('href')}"
                                title = link.get_text(strip=True)
                                
                                articles.append({
                                    'title': title,
                                    'url': article_url,
                                    'content': '',  # Bloomberg content is paywall protected
                                    'published_date': datetime.now(),
                                    'source': 'Bloomberg',
                                    'summary': title  # Use title as summary
                                })
                            except Exception:
                                continue
        
        except Exception as e:
            print(f"Error scraping Bloomberg: {e}")
        
        return articles
    
    async def _fetch_article_content(self, url: str) -> str:
        """Fetch full content of an article"""
        
        if not url:
            return ""
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                async with session.get(url, headers=headers, timeout=10) as response:
                    if response.status == 200:
                        content = await response.text()
                        soup = BeautifulSoup(content, 'html.parser')
                        
                        # Remove unwanted elements
                        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                            element.decompose()
                        
                        # Extract main content
                        content_selectors = [
                            'article', '.article-content', '.story-content',
                            '.entry-content', '.post-content', '.content',
                            'main', '.main-content'
                        ]
                        
                        for selector in content_selectors:
                            content_elem = soup.select_one(selector)
                            if content_elem:
                                text = content_elem.get_text(separator=' ', strip=True)
                                if len(text) > 100:  # Ensure meaningful content
                                    return text[:5000]  # Limit to 5000 chars
                        
                        # Fallback: get all text
                        text = soup.get_text(separator=' ', strip=True)
                        return text[:5000] if text else ""
        
        except Exception as e:
            print(f"Error fetching article content from {url}: {e}")
            return ""
    
    async def _fetch_reuters_article(self, url: str) -> Optional[Dict]:
        """Fetch Reuters article with specific parsing"""
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                async with session.get(url, headers=headers, timeout=10) as response:
                    if response.status == 200:
                        content = await response.text()
                        soup = BeautifulSoup(content, 'html.parser')
                        
                        # Extract content
                        article_content = ""
                        content_divs = soup.find_all('div', {'data-module': 'ArticleBody'})
                        
                        for div in content_divs:
                            paragraphs = div.find_all('p')
                            for p in paragraphs:
                                article_content += p.get_text(strip=True) + " "
                        
                        # Extract date
                        date_elem = soup.find('time')
                        pub_date = datetime.now()
                        if date_elem and date_elem.get('datetime'):
                            try:
                                pub_date = datetime.fromisoformat(date_elem['datetime'].replace('Z', '+00:00'))
                            except:
                                pass
                        
                        return {
                            'content': article_content.strip(),
                            'published_date': pub_date,
                            'summary': article_content[:300] + "..." if len(article_content) > 300 else article_content
                        }
        
        except Exception as e:
            print(f"Error fetching Reuters article: {e}")
            return None
    
    def _analyze_with_textblob(self, news_data: List[Dict]) -> Dict:
        """Analyze sentiment using TextBlob"""
        
        all_text = " ".join([article.get('content', '') + " " + article.get('title', '') 
                           for article in news_data])
        
        if not all_text.strip():
            return {'method': 'textblob', 'sentiment_score': 0.0, 'confidence': 0.0}
        
        blob = TextBlob(all_text)
        
        return {
            'method': 'textblob',
            'sentiment_score': blob.sentiment.polarity,  # -1 to 1
            'subjectivity': blob.sentiment.subjectivity,  # 0 to 1
            'confidence': 0.6,  # TextBlob has moderate confidence
            'classification': 'positive' if blob.sentiment.polarity > 0.1 else 'negative' if blob.sentiment.polarity < -0.1 else 'neutral'
        }
    
    async def _analyze_with_finbert(self, news_data: List[Dict]) -> Dict:
        """Analyze sentiment using FinBERT (financial sentiment)"""
        
        if not self.finbert:
            return {'method': 'finbert', 'error': 'FinBERT not available'}
        
        # Prepare text chunks (FinBERT has token limits)
        texts = []
        for article in news_data:
            text = (article.get('title', '') + " " + article.get('content', ''))[:500]  # Limit to 500 chars
            if text.strip():
                texts.append(text)
        
        if not texts:
            return {'method': 'finbert', 'sentiment_score': 0.0, 'confidence': 0.0}
        
        try:
            # Analyze each text chunk
            results = []
            for text in texts[:10]:  # Limit to 10 chunks for performance
                result = self.finbert(text)
                if result:
                    results.append(result[0])  # FinBERT returns list of all scores
            
            if not results:
                return {'method': 'finbert', 'sentiment_score': 0.0, 'confidence': 0.0}
            
            # Aggregate results
            positive_scores = []
            negative_scores = []
            neutral_scores = []
            
            for result in results:
                for item in result:
                    if item['label'] == 'positive':
                        positive_scores.append(item['score'])
                    elif item['label'] == 'negative':
                        negative_scores.append(item['score'])
                    else:
                        neutral_scores.append(item['score'])
            
            avg_positive = sum(positive_scores) / len(positive_scores) if positive_scores else 0
            avg_negative = sum(negative_scores) / len(negative_scores) if negative_scores else 0
            avg_neutral = sum(neutral_scores) / len(neutral_scores) if neutral_scores else 0
            
            # Convert to -1 to 1 scale
            sentiment_score = avg_positive - avg_negative
            confidence = max(avg_positive, avg_negative, avg_neutral)
            
            classification = 'positive' if sentiment_score > 0.1 else 'negative' if sentiment_score < -0.1 else 'neutral'
            
            return {
                'method': 'finbert',
                'sentiment_score': sentiment_score,
                'confidence': confidence,
                'classification': classification,
                'positive_score': avg_positive,
                'negative_score': avg_negative,
                'neutral_score': avg_neutral
            }
        
        except Exception as e:
            print(f"Error with FinBERT analysis: {e}")
            return {'method': 'finbert', 'error': str(e)}
    
    async def _analyze_with_roberta(self, news_data: List[Dict]) -> Dict:
        """Analyze sentiment using RoBERTa model"""
        
        if not self.general_sentiment:
            return {'method': 'roberta', 'error': 'RoBERTa not available'}
        
        # Prepare text chunks
        texts = []
        for article in news_data:
            text = (article.get('title', '') + " " + article.get('content', ''))[:500]
            if text.strip():
                texts.append(text)
        
        if not texts:
            return {'method': 'roberta', 'sentiment_score': 0.0, 'confidence': 0.0}
        
        try:
            # Analyze each text chunk
            results = []
            for text in texts[:10]:  # Limit for performance
                result = self.general_sentiment(text)
                if result:
                    results.append(result[0])
            
            if not results:
                return {'method': 'roberta', 'sentiment_score': 0.0, 'confidence': 0.0}
            
            # Aggregate results (RoBERTa uses LABEL_0=negative, LABEL_1=neutral, LABEL_2=positive)
            positive_scores = []
            negative_scores = []
            neutral_scores = []
            
            for result in results:
                for item in result:
                    if 'LABEL_2' in item['label'] or 'positive' in item['label'].lower():
                        positive_scores.append(item['score'])
                    elif 'LABEL_0' in item['label'] or 'negative' in item['label'].lower():
                        negative_scores.append(item['score'])
                    else:
                        neutral_scores.append(item['score'])
            
            avg_positive = sum(positive_scores) / len(positive_scores) if positive_scores else 0
            avg_negative = sum(negative_scores) / len(negative_scores) if negative_scores else 0
            avg_neutral = sum(neutral_scores) / len(neutral_scores) if neutral_scores else 0
            
            # Convert to -1 to 1 scale
            sentiment_score = avg_positive - avg_negative
            confidence = max(avg_positive, avg_negative, avg_neutral)
            
            classification = 'positive' if sentiment_score > 0.1 else 'negative' if sentiment_score < -0.1 else 'neutral'
            
            return {
                'method': 'roberta',
                'sentiment_score': sentiment_score,
                'confidence': confidence,
                'classification': classification,
                'positive_score': avg_positive,
                'negative_score': avg_negative,
                'neutral_score': avg_neutral
            }
        
        except Exception as e:
            print(f"Error with RoBERTa analysis: {e}")
            return {'method': 'roberta', 'error': str(e)}
    
    def _analyze_with_keywords(self, news_data: List[Dict]) -> Dict:
        """Analyze sentiment using keyword-based approach (free backup method)"""
        
        # Combine all text
        all_text = " ".join([
            article.get('content', '') + " " + article.get('title', '') 
            for article in news_data
        ]).lower()
        
        if not all_text.strip():
            return {'method': 'keywords', 'sentiment_score': 0.0, 'confidence': 0.0}
        
        # Define keyword lists
        positive_keywords = [
            'growth', 'profit', 'revenue', 'success', 'expansion', 'breakthrough',
            'innovation', 'partnership', 'acquisition', 'funding', 'investment',
            'increase', 'rise', 'gain', 'positive', 'strong', 'excellent',
            'achievement', 'milestone', 'record', 'best', 'leading', 'winning'
        ]
        
        negative_keywords = [
            'loss', 'decline', 'bankruptcy', 'lawsuit', 'scandal', 'crisis',
            'problem', 'issue', 'concern', 'drop', 'fall', 'decrease',
            'layoff', 'closure', 'debt', 'risk', 'challenge', 'struggle',
            'poor', 'bad', 'weak', 'negative', 'worst', 'failing'
        ]
        
        neutral_keywords = [
            'stable', 'maintain', 'continue', 'steady', 'consistent',
            'regular', 'normal', 'standard', 'typical', 'average'
        ]
        
        # Count keywords
        positive_count = sum([1 for word in positive_keywords if word in all_text])
        negative_count = sum([1 for word in negative_keywords if word in all_text])
        neutral_count = sum([1 for word in neutral_keywords if word in all_text])
        
        total_count = positive_count + negative_count + neutral_count
        
        if total_count == 0:
            return {'method': 'keywords', 'sentiment_score': 0.0, 'confidence': 0.1}
        
        # Calculate sentiment score
        sentiment_score = (positive_count - negative_count) / max(total_count, 1)
        confidence = min(total_count / 10, 0.7)  # Max confidence 0.7 for keyword method
        
        classification = 'positive' if sentiment_score > 0.1 else 'negative' if sentiment_score < -0.1 else 'neutral'
        
        return {
            'method': 'keywords',
            'sentiment_score': sentiment_score,
            'confidence': confidence,
            'classification': classification,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count
        }
    
    def _aggregate_sentiment_results(self, company_name: str, sentiment_results: List[Dict], news_data: List[Dict]) -> Dict:
        """Aggregate sentiment results from multiple free methods"""
        
        valid_results = [r for r in sentiment_results if 'error' not in r and r.get('sentiment_score') is not None]
        
        if not valid_results:
            return {
                'company_name': company_name,
                'overall_sentiment_score': 0.0,
                'overall_sentiment': 'neutral',
                'confidence_score': 0.0,
                'error': 'No valid sentiment analysis results'
            }
        
        # Weight different methods (free alternatives only)
        method_weights = {
            'finbert': 0.5,     # Highest weight for financial sentiment (if available)
            'roberta': 0.3,     # High weight for general sentiment (if available)
            'textblob': 0.15,   # Medium weight for TextBlob
            'keywords': 0.05    # Lowest weight for keyword method
        }
        
        # Calculate weighted average
        total_score = 0.0
        total_weight = 0.0
        total_confidence = 0.0
        
        method_scores = {}
        
        for result in valid_results:
            method = result.get('method', '')
            score = result.get('sentiment_score', 0.0)
            confidence = result.get('confidence', 0.5)
            weight = method_weights.get(method, 0.1)
            
            total_score += score * weight * confidence
            total_weight += weight * confidence
            total_confidence += confidence
            
            method_scores[method] = {
                'score': score,
                'confidence': confidence,
                'classification': result.get('classification', 'neutral')
            }
        
        # Final aggregated score
        if total_weight > 0:
            final_score = total_score / total_weight
            avg_confidence = total_confidence / len(valid_results)
        else:
            final_score = 0.0
            avg_confidence = 0.0
        
        # Determine overall classification
        if final_score > 0.15:
            classification = 'positive'
        elif final_score < -0.15:
            classification = 'negative'
        else:
            classification = 'neutral'
        
        # Extract key topics
        key_topics = self._extract_key_topics(news_data)
        
        # Determine trend
        trend = self._determine_sentiment_trend(news_data, valid_results)
        
        return {
            'company_name': company_name,
            'overall_sentiment_score': round(final_score, 3),
            'overall_sentiment': classification,
            'confidence_score': round(avg_confidence, 3),
            'method_breakdown': method_scores,
            'key_topics': key_topics,
            'sentiment_trend': trend,
            'news_count': len(news_data),
            'analysis_date': datetime.utcnow().isoformat(),
            'detailed_analysis': self._generate_detailed_analysis(company_name, classification, final_score, key_topics),
            'news_sources': [article.get('source', 'Unknown') for article in news_data],
            'analysis_methods_used': list(method_scores.keys())
        }
    
    def _extract_key_topics(self, news_data: List[Dict]) -> List[str]:
        """Extract key topics from news articles"""
        
        # Combine all text
        all_text = " ".join([
            article.get('title', '') + " " + article.get('content', '') 
            for article in news_data
        ]).lower()
        
        # Key business/financial terms to look for
        key_terms = {
            'earnings': ['earnings', 'revenue', 'profit', 'income', 'quarterly'],
            'growth': ['growth', 'expansion', 'scaling', 'increasing'],
            'acquisition': ['acquisition', 'merger', 'buyout', 'acquired'],
            'funding': ['funding', 'investment', 'capital', 'raised', 'series'],
            'product': ['product', 'launch', 'release', 'innovation'],
            'partnership': ['partnership', 'collaboration', 'alliance'],
            'legal': ['lawsuit', 'legal', 'court', 'settlement'],
            'leadership': ['ceo', 'executive', 'leadership', 'management'],
            'market': ['market', 'industry', 'competition', 'competitors'],
            'technology': ['technology', 'ai', 'digital', 'platform']
        }
        
        found_topics = []
        for topic, terms in key_terms.items():
            if any(term in all_text for term in terms):
                found_topics.append(topic)
        
        return found_topics[:5]  # Return top 5 topics
    
    def _determine_sentiment_trend(self, news_data: List[Dict], sentiment_results: List[Dict]) -> str:
        """Determine if sentiment is improving, declining, or stable"""
        
        if len(news_data) < 3:
            return 'stable'
        
        # Sort articles by date
        sorted_articles = sorted(news_data, key=lambda x: x.get('published_date', datetime.min))
        
        # Analyze sentiment of recent vs older articles
        recent_articles = sorted_articles[-len(sorted_articles)//2:]  # Latest half
        older_articles = sorted_articles[:len(sorted_articles)//2]   # Earlier half
        
        # Simple sentiment comparison
        recent_sentiment = sum([1 if 'positive' in article.get('title', '').lower() else -1 if 'negative' in article.get('title', '').lower() else 0 for article in recent_articles])
        older_sentiment = sum([1 if 'positive' in article.get('title', '').lower() else -1 if 'negative' in article.get('title', '').lower() else 0 for article in older_articles])
        
        if recent_sentiment > older_sentiment + 1:
            return 'improving'
        elif recent_sentiment < older_sentiment - 1:
            return 'declining'
        else:
            return 'stable'
    
    def _generate_detailed_analysis(self, company_name: str, classification: str, score: float, topics: List[str]) -> str:
        """Generate a detailed analysis summary"""
        
        analysis = f"Sentiment analysis for {company_name} shows "
        
        if classification == 'positive':
            analysis += f"positive sentiment (score: {score:.2f}). "
        elif classification == 'negative':
            analysis += f"negative sentiment (score: {score:.2f}). "
        else:
            analysis += f"neutral sentiment (score: {score:.2f}). "
        
        if topics:
            analysis += f"Key topics in recent news include: {', '.join(topics)}. "
        
        if abs(score) > 0.5:
            analysis += "This represents a strong sentiment signal that investors should consider."
        elif abs(score) > 0.2:
            analysis += "This represents a moderate sentiment signal worth monitoring."
        else:
            analysis += "Sentiment appears relatively neutral with no strong directional signals."
        
        return analysis
    
    def _extract_score_from_text(self, text: str) -> float:
        """Extract sentiment score from text response"""
        
        # Look for score patterns
        score_patterns = [
            r'sentiment score[:\s]*(-?\d+\.?\d*)',
            r'score[:\s]*(-?\d+\.?\d*)',
            r'(-?\d+\.?\d*)\s*(?:out of|/)\s*1',
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    score = float(match.group(1))
                    # Ensure score is in -1 to 1 range
                    return max(-1.0, min(1.0, score))
                except ValueError:
                    continue
        
        # Fallback: look for positive/negative keywords
        positive_words = ['positive', 'bullish', 'optimistic', 'good', 'strong']
        negative_words = ['negative', 'bearish', 'pessimistic', 'bad', 'weak']
        
        text_lower = text.lower()
        positive_count = sum([1 for word in positive_words if word in text_lower])
        negative_count = sum([1 for word in negative_words if word in text_lower])
        
        if positive_count > negative_count:
            return 0.3
        elif negative_count > positive_count:
            return -0.3
        else:
            return 0.0
    
    async def _get_ticker_symbol(self, company_name: str) -> Optional[str]:
        """Get stock ticker symbol for a company"""
        
        try:
            # Use a simple search approach
            search_url = f"https://query1.finance.yahoo.com/v1/finance/search?q={company_name.replace(' ', '+')}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        quotes = data.get('quotes', [])
                        
                        for quote in quotes:
                            if quote.get('typeDisp') == 'Equity' and quote.get('symbol'):
                                return quote['symbol']
        
        except Exception as e:
            print(f"Error getting ticker symbol: {e}")
        
        return None
    
    def _parse_relative_date(self, date_text: str) -> datetime:
        """Parse relative date strings like '2 hours ago', '1 day ago'"""
        
        now = datetime.now()
        date_text = date_text.lower()
        
        # Extract number and unit
        match = re.search(r'(\d+)\s*(hour|day|week|month|year)s?\s*ago', date_text)
        
        if match:
            number = int(match.group(1))
            unit = match.group(2)
            
            if unit == 'hour':
                return now - timedelta(hours=number)
            elif unit == 'day':
                return now - timedelta(days=number)
            elif unit == 'week':
                return now - timedelta(weeks=number)
            elif unit == 'month':
                return now - timedelta(days=number * 30)
            elif unit == 'year':
                return now - timedelta(days=number * 365)
        
        return now
    
    def _deduplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate articles based on title similarity"""
        
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            title = article.get('title', '').lower()
            
            # Simple deduplication by checking if title is too similar
            is_duplicate = False
            for seen_title in seen_titles:
                if self._calculate_similarity(title, seen_title) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate and title:
                unique_articles.append(article)
                seen_titles.add(title)
        
        return unique_articles
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings"""
        
        # Simple Jaccard similarity
        set1 = set(str1.split())
        set2 = set(str2.split())
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0

# ============================================================================
# API ENDPOINTS - SCRAPING
# ============================================================================

# app/api/endpoints/scraping.py
from typing import List
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
import uuid
from datetime import datetime, timedelta

from app.core.database import get_db, get_redis
from app.api.deps import get_current_active_user
from app.models.user import User
from app.schemas.leads import ScrapeRequest, ScrapeResponse
from app.services.scraping_service import ScrapingService
from app.tasks.scraping_tasks import process_scraping_job
import redis.asyncio as aioredis

router = APIRouter()

@router.post("/start", response_model=ScrapeResponse)
async def start_scraping_job(
    scrape_request: ScrapeRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    redis_client: aioredis.Redis = Depends(get_redis),
    current_user: User = Depends(get_current_active_user)
):
    """Start a new scraping job"""
    
    # Check user's rate limits
    today = datetime.now().strftime("%Y-%m-%d")
    user_jobs_key = f"scraping_jobs:{current_user.id}:{today}"
    
    daily_jobs = await redis_client.get(user_jobs_key)
    if daily_jobs and int(daily_jobs) >= 10:  # Limit to 10 jobs per day
        raise HTTPException(
            status_code=429,
            detail="Daily scraping limit reached. Please try again tomorrow."
        )
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Estimate completion time based on number of companies and priority
    base_time_per_company = 30  # seconds
    if scrape_request.priority == "high":
        multiplier = 0.5
    elif scrape_request.priority == "low":
        multiplier = 2.0
    else:
        multiplier = 1.0
    
    estimated_duration = len(scrape_request.companies) * base_time_per_company * multiplier
    estimated_completion = datetime.utcnow() + timedelta(seconds=estimated_duration)
    
    # Store job metadata in Redis
    job_data = {
        "job_id": job_id,
        "user_id": current_user.id,
        "companies": scrape_request.companies,
        "status": "queued",
        "priority": scrape_request.priority,
        "include_contacts": scrape_request.include_contacts,
        "include_sentiment": scrape_request.include_sentiment,
        "include_technologies": scrape_request.include_technologies,
        "webhook_url": scrape_request.webhook_url,
        "created_at": datetime.utcnow().isoformat(),
        "estimated_completion": estimated_completion.isoformat()
    }
    
    await redis_client.setex(
        f"scraping_job:{job_id}",
        3600,  # 1 hour expiration
        json.dumps(job_data)
    )
    
    # Increment daily job count
    await redis_client.incr(user_jobs_key)
    await redis_client.expire(user_jobs_key, 86400)  # 24 hours
    
    # Queue background task
    background_tasks.add_task(
        process_scraping_job,
        job_id,
        scrape_request.companies,
        current_user.id,
        scrape_request.dict()
    )
    
    return ScrapeResponse(
        job_id=job_id,
        status="queued",
        estimated_completion=estimated_completion,
        companies_count=len(scrape_request.companies),
        priority=scrape_request.priority
    )

@router.get("/status/{job_id}")
async def get_scraping_status(
    job_id: str,
    redis_client: aioredis.Redis = Depends(get_redis),
    current_user: User = Depends(get_current_active_user)
):
    """Get the status of a scraping job"""
    
    job_data = await redis_client.get(f"scraping_job:{job_id}")
    
    if not job_data:
        raise HTTPException(
            status_code=404,
            detail="Job not found"
        )
    
    job = json.loads(job_data)
    
    # Check if user owns this job
    if job.get("user_id") != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="Access denied"
        )
    
    # Get progress information
    progress_data = await redis_client.get(f"scraping_progress:{job_id}")
    if progress_data:
        progress = json.loads(progress_data)
        job.update(progress)
    
    return job

@router.get("/results/{job_id}")
async def get_scraping_results(
    job_id: str,
    redis_client: aioredis.Redis = Depends(get_redis),
    current_user: User = Depends(get_current_active_user)
):
    """Get the results of a completed scraping job"""
    
    job_data = await redis_client.get(f"scraping_job:{job_id}")
    
    if not job_data:
        raise HTTPException(
            status_code=404,
            detail="Job not found"
        )
    
    job = json.loads(job_data)
    
    # Check if user owns this job
    if job.get("user_id") != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="Access denied"
        )
    
    # Check if job is completed
    if job.get("status") != "completed":
        raise HTTPException(
            status_code=400,
            detail="Job not completed yet"
        )
    
    # Get results
    results_data = await redis_client.get(f"scraping_results:{job_id}")
    if not results_data:
        raise HTTPException(
            status_code=404,
            detail="Results not found"
        )
    
    return json.loads(results_data)

@router.delete("/cancel/{job_id}")
async def cancel_scraping_job(
    job_id: str,
    redis_client: aioredis.Redis = Depends(get_redis),
    current_user: User = Depends(get_current_active_user)
):
    """Cancel a running scraping job"""
    
    job_data = await redis_client.get(f"scraping_job:{job_id}")
    
    if not job_data:
        raise HTTPException(
            status_code=404,
            detail="Job not found"
        )
    
    job = json.loads(job_data)
    
    # Check if user owns this job
    if job.get("user_id") != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="Access denied"
        )
    
    # Update job status
    job["status"] = "cancelled"
    job["cancelled_at"] = datetime.utcnow().isoformat()
    
    await redis_client.setex(
        f"scraping_job:{job_id}",
        3600,
        json.dumps(job)
    )
    
    return {"message": "Job cancelled successfully"}

@router.get("/history")
async def get_scraping_history(
    skip: int = 0,
    limit: int = 20,
    redis_client: aioredis.Redis = Depends(get_redis),
    current_user: User = Depends(get_current_active_user)
):
    """Get user's scraping job history"""
    
    # Get all job keys for this user
    pattern = f"scraping_job:*"
    job_keys = []
    
    async for key in redis_client.scan_iter(match=pattern):
        job_data = await redis_client.get(key)
        if job_data:
            job = json.loads(job_data)
            if job.get("user_id") == current_user.id:
                job_keys.append((key, job))
    
    # Sort by creation date
    job_keys.sort(key=lambda x: x[1].get("created_at", ""), reverse=True)
    
    # Apply pagination
    paginated_jobs = job_keys[skip:skip + limit]
    
    return {
        "jobs": [job for _, job in paginated_jobs],
        "total": len(job_keys),
        "skip": skip,
        "limit": limit
    }

# ============================================================================
# API ENDPOINTS - SENTIMENT
# ============================================================================

# app/api/endpoints/sentiment.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from app.core.database import get_db, get_redis
from app.api.deps import get_current_active_user
from app.models.user import User
from app.schemas.sentiment import SentimentRequest, SentimentResponse, BulkSentimentResponse
from app.services.sentiment_service import SentimentAnalysisService
import redis.asyncio as aioredis
import json

router = APIRouter()

@router.post("/analyze", response_model=BulkSentimentResponse)
async def analyze_sentiment(
    request: SentimentRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    redis_client: aioredis.Redis = Depends(get_redis),
    current_user: User = Depends(get_current_active_user)
):
    """Analyze sentiment for multiple companies"""
    
    sentiment_service = SentimentAnalysisService()
    results = []
    
    for company_name in request.company_names:
        # Check cache first (unless force_refresh is True)
        cache_key = f"sentiment:{company_name}:{request.days_back}"
        
        if not request.force_refresh:
            cached_result = await redis_client.get(cache_key)
            if cached_result:
                result = json.loads(cached_result)
                results.append(SentimentResponse(**result))
                continue
        
        # Perform sentiment analysis
        try:
            analysis = await sentiment_service.analyze_company_sentiment(
                company_name, request.days_back
            )
            
            if 'error' not in analysis:
                # Cache result for 1 hour
                await redis_client.setex(cache_key, 3600, json.dumps(analysis))
                
                results.append(SentimentResponse(
                    company_name=analysis['company_name'],
                    overall_sentiment_score=analysis['overall_sentiment_score'],
                    overall_sentiment=analysis['overall_sentiment'],
                    confidence_score=analysis['confidence_score'],
                    detailed_analysis=analysis['detailed_analysis'],
                    key_topics=analysis['key_topics'],
                    news_count=analysis['news_count'],
                    last_updated=datetime.fromisoformat(analysis['analysis_date'])
                ))
            
        except Exception as e:
            print(f"Error analyzing sentiment for {company_name}: {e}")
            continue
    
    return BulkSentimentResponse(
        results=results,
        total_companies=len(request.company_names),
        successful_analyses=len(results),
        failed_analyses=len(request.company_names) - len(results)
    )

@router.get("/company/{company_name}")
async def get_company_sentiment(
    company_name: str,
    days_back: int = 30,
    force_refresh: bool = False,
    redis_client: aioredis.Redis = Depends(get_redis),
    current_user: User = Depends(get_current_active_user)
):
    """Get sentiment analysis for a specific company"""
    
    cache_key = f"sentiment:{company_name}:{days_back}"
    
    # Check cache first
    if not force_refresh:
        cached_result = await redis_client.get(cache_key)
        if cached_result:
            return json.loads(cached_result)
    
    # Perform fresh analysis
    sentiment_service = SentimentAnalysisService()
    analysis = await sentiment_service.analyze_company_sentiment(company_name, days_back)
    
    if 'error' not in analysis:
        # Cache result
        await redis_client.setex(cache_key, 3600, json.dumps(analysis))
    
    return analysis

@router.delete("/cache/{company_name}")
async def clear_sentiment_cache(
    company_name: str,
    redis_client: aioredis.Redis = Depends(get_redis),
    current_user: User = Depends(get_current_active_user)
):
    """Clear sentiment cache for a company"""
    
    # Delete all cache entries for this company
    pattern = f"sentiment:{company_name}:*"
    deleted_keys = 0
    
    async for key in redis_client.scan_iter(match=pattern):
        await redis_client.delete(key)
        deleted_keys += 1
    
    return {
        "message": f"Cleared sentiment cache for {company_name}",
        "deleted_keys": deleted_keys
    }

# ============================================================================
# BACKGROUND TASKS
# ============================================================================

# app/tasks/scraping_tasks.py
import asyncio
import json
from typing import List, Dict
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal, get_redis
from app.services.scraping_service import ScrapingService
from app.services.sentiment_service import SentimentAnalysisService
from app.models.leads import Lead, Contact
from app.models.sentiment import SentimentAnalysis
from app.utils.data_validator import DataValidator

async def process_scraping_job(
    job_id: str,
    company_names: List[str],
    user_id: int,
    options: Dict
):
    """Background task to process scraping job"""
    
    redis_client = await get_redis()
    scraping_service = ScrapingService()
    sentiment_service = SentimentAnalysisService()
    data_validator = DataValidator()
    
    try:
        # Update job status
        await update_job_status(redis_client, job_id, "running", {
            "started_at": datetime.utcnow().isoformat(),
            "processed_companies": 0,
            "total_companies": len(company_names),
            "current_company": company_names[0] if company_names else ""
        })
        
        # Process companies
        results = []
        
        for i, company_name in enumerate(company_names):
            try:
                # Update progress
                await update_job_progress(redis_client, job_id, {
                    "processed_companies": i + 1,
                    "current_company": company_name,
                    "progress_percentage": ((i + 1) / len(company_names)) * 100
                })
                
                # Scrape company data
                company_data = await scraping_service._scrape_single_company(company_name, user_id)
                
                if company_data:
                    # Save to database
                    async with AsyncSessionLocal() as db:
                        lead = await save_lead_to_db(db, company_data)
                        
                        # Add sentiment analysis if requested
                        if options.get('include_sentiment', False) and company_data.get('domain'):
                            sentiment_data = await sentiment_service.analyze_company_sentiment(company_name)
                            if 'error' not in sentiment_data:
                                await save_sentiment_to_db(db, lead.id, sentiment_data)
                        
                        await db.commit()
                    
                    results.append({
                        "company_name": company_name,
                        "status": "success",
                        "lead_id": lead.id if 'lead' in locals() else None,
                        "data_quality_score": company_data.get('data_quality_score', 0.0),
                        "contacts_found": len(company_data.get('contacts', [])),
                        "sources_used": company_data.get('sources', [])
                    })
                else:
                    results.append({
                        "company_name": company_name,
                        "status": "failed",
                        "error": "No data found"
                    })
                
            except Exception as e:
                results.append({
                    "company_name": company_name,
                    "status": "error",
                    "error": str(e)
                })
        
        # Store final results
        final_results = {
            "job_id": job_id,
            "results": results,
            "summary": {
                "total_companies": len(company_names),
                "successful": len([r for r in results if r["status"] == "success"]),
                "failed": len([r for r in results if r["status"] in ["failed", "error"]]),
                "completion_time": datetime.utcnow().isoformat()
            }
        }
        
        await redis_client.setex(
            f"scraping_results:{job_id}",
            86400,  # 24 hours
            json.dumps(final_results)
        )
        
        # Update job status to completed
        await update_job_status(redis_client, job_id, "completed", {
            "completed_at": datetime.utcnow().isoformat(),
            "results_summary": final_results["summary"]
        })
        
        # Send webhook notification if provided
        webhook_url = options.get('webhook_url')
        if webhook_url:
            await send_webhook_notification(webhook_url, job_id, final_results)
    
    except Exception as e:
        # Update job status to failed
        await update_job_status(redis_client, job_id, "failed", {
            "failed_at": datetime.utcnow().isoformat(),
            "error": str(e)
        })
        raise

async def update_job_status(redis_client, job_id: str, status: str, additional_data: Dict = None):
    """Update job status in Redis"""
    
    job_data = await redis_client.get(f"scraping_job:{job_id}")
    if job_data:
        job = json.loads(job_data)
        job["status"] = status
        
        if additional_data:
            job.update(additional_data)
        
        await redis_client.setex(
            f"scraping_job:{job_id}",
            3600,
            json.dumps(job)
        )

async def update_job_progress(redis_client, job_id: str, progress_data: Dict):
    """Update job progress in Redis"""
    
    await redis_client.setex(
        f"scraping_progress:{job_id}",
        3600,
        json.dumps(progress_data)
    )

async def save_lead_to_db(db: AsyncSession, company_data: Dict) -> Lead:
    """Save lead data to database"""
    
    # Create lead
    lead = Lead(
        user_id=company_data['user_id'],
        company_name=company_data['company_name'],
        domain=company_data.get('domain'),
        **company_data.get('company_info', {}),
        data_quality_score=company_data.get('data_quality_score', 0.0),
        source_urls=json.dumps(company_data.get('sources', [])),
        last_scraped=datetime.utcnow()
    )
    
    db.add(lead)
    await db.flush()  # Get lead ID
    
    # Add contacts
    for contact_data in company_data.get('contacts', []):
        contact = Contact(
            lead_id=lead.id,
            **contact_data
        )
        db.add(contact)
    
    return lead

async def save_sentiment_to_db(db: AsyncSession, lead_id: int, sentiment_data: Dict) -> SentimentAnalysis:
    """Save sentiment analysis to database"""
    
    sentiment = SentimentAnalysis(
        lead_id=lead_id,
        company_name=sentiment_data['company_name'],
        overall_sentiment_score=sentiment_data['overall_sentiment_score'],
        overall_sentiment=sentiment_data['overall_sentiment'],
        confidence_score=sentiment_data['confidence_score'],
        detailed_analysis=sentiment_data['detailed_analysis'],
        key_topics=json.dumps(sentiment_data['key_topics']),
        news_sources=json.dumps(sentiment_data.get('news_sources', [])),
        news_count=sentiment_data.get('news_count', 0),
        expires_at=datetime.utcnow() + timedelta(hours=24)
    )
    
    db.add(sentiment)
    return sentiment

async def send_webhook_notification(webhook_url: str, job_id: str, results: Dict):
    """Send webhook notification when job completes"""
    
    try:
        import aiohttp
        
        payload = {
            "job_id": job_id,
            "status": "completed",
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                webhook_url,
                json=payload,
                timeout=10
            ) as response:
                if response.status == 200:
                    print(f"Webhook notification sent successfully for job {job_id}")
                else:
                    print(f"Webhook notification failed for job {job_id}: {response.status}")
    
    except Exception as e:
        print(f"Error sending webhook notification: {e}")

# ============================================================================
# DATABASE SETUP & MIGRATION COMMANDS
# ============================================================================

"""
# Database Setup Instructions

## 1. Create Database Tables
Since the Azure PostgreSQL database is empty, you need to create the tables first.

### Using Alembic (Recommended):
```bash
# Install alembic
pip install alembic

# Initialize alembic (only if not done before)
alembic init alembic

# Create initial migration
alembic revision --autogenerate -m "Initial migration"

# Apply migrations to Azure database
alembic upgrade head
```

### Using SQLAlchemy directly (Quick setup):
```python
# Run this script to create tables directly
import asyncio
from app.core.database import engine
from app.models import user, leads, contacts, sentiment

async def create_tables():
    async with engine.begin() as conn:
        # Import all models to ensure they're registered
        from app.models.user import User
        from app.models.leads import Lead
        from app.models.contacts import Contact
        from app.models.sentiment import SentimentAnalysis
        
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
        print("All tables created successfully!")

if __name__ == "__main__":
    asyncio.run(create_tables())
```

## 2. Create First Admin User
```python
# Run this script to create an admin user
import asyncio
from app.core.database import AsyncSessionLocal
from app.models.user import User
from app.core.security import get_password_hash

async def create_admin_user():
    async with AsyncSessionLocal() as db:
        admin_user = User(
            email="admin@leadgen.com",
            hashed_password=get_password_hash("admin123"),
            full_name="Lead Gen Admin",
            is_active=True,
            is_superuser=True
        )
        db.add(admin_user)
        await db.commit()
        print("Admin user created: admin@leadgen.com / admin123")

if __name__ == "__main__":
    asyncio.run(create_admin_user())
```

## 3. Test Database Connection
```python
# test_db_connection.py
import asyncio
from app.core.database import AsyncSessionLocal
from sqlalchemy import text

async def test_connection():
    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(text("SELECT version()"))
            version = result.scalar()
            print(f"Connected to PostgreSQL: {version}")
            
            # Test table existence
            result = await db.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'"))
            tables = result.fetchall()
            print(f"Tables in database: {[table[0] for table in tables]}")
            
    except Exception as e:
        print(f"Database connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_connection())
```

## 4. Environment Setup
```bash
# Create .env file with your Azure database credentials
cp .env.example .env

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium

# Start the application
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 5. Development vs Production Settings

### Development:
- Use local Redis: redis://localhost:6379/0
- Conservative scraping limits
- Debug logging enabled

### Production:
- Use Azure Redis Cache
- Production security settings
- Optimized scraping parameters
"""

# ============================================================================
# MVP IMPLEMENTATION STRATEGY (5-Hour Challenge)
# ============================================================================

"""
# 5-Hour MVP Implementation Strategy

## Core Focus Areas (Priority Order):

### 1. Basic Company Search & Domain Finding (1.5 hours)
- Implement _scrape_google_search() method
- Basic domain extraction from search results
- Simple company information parsing

### 2. Company Website Scraping (1.5 hours)
- Implement _scrape_company_website() method
- Extract basic contact info (emails, phones)
- Parse company description and basic details

### 3. Database Integration (1 hour)
- Set up basic Lead and Contact models
- Implement save_lead_to_db() function
- Basic CRUD operations

### 4. Simple Sentiment Analysis (1 hour)
- TextBlob-based sentiment only (skip FinBERT/RoBERTa)
- Basic news scraping from Google News
- Simple positive/negative classification

### Demo Script for 5-Hour Version:

```python
# mvp_demo.py - Simplified version for demo
import asyncio
from app.services.scraping_service import ScrapingService

async def demo_lead_generation():
    scraper = ScrapingService()
    
    # Test companies for demo
    test_companies = ["Tesla", "OpenAI", "Stripe"]
    
    print("🚀 Starting Lead Generation Demo...")
    
    for company in test_companies:
        print(f"\n📊 Analyzing: {company}")
        
        # Basic scraping
        result = await scraper._scrape_single_company(company, user_id=1)
        
        if result:
            print(f"✅ Found domain: {result.get('domain', 'N/A')}")
            print(f"📧 Contacts found: {len(result.get('contacts', []))}")
            print(f"⭐ Quality score: {result.get('data_quality_score', 0):.2f}")
        else:
            print("❌ No data found")
    
    print("\n🎉 Demo completed!")

if __name__ == "__main__":
    asyncio.run(demo_lead_generation())
```

## What to Show in Interview:

1. **Architecture Understanding**: Explain the full system design
2. **MVP Implementation**: Demo the basic working version
3. **Scaling Strategy**: Explain how to expand to production
4. **Business Value**: Focus on ROI and competitive advantage
5. **Technical Challenges**: Discuss anti-bot measures, data quality

## Key Talking Points:

- "This is a 5-hour MVP. Production would include FinBERT sentiment analysis, premium proxies, and advanced email verification"
- "The architecture is designed to scale - we can easily add more data sources and AI models"
- "Focus on data quality over quantity - better to have 10 high-quality leads than 100 poor ones"
- "The system is designed with rate limiting and respectful scraping practices"
"""

# Dockerfile
"""
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers
RUN playwright install chromium firefox
RUN playwright install-deps

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p logs

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

# docker-compose.yml
"""
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      # Using Azure PostgreSQL (no local DB needed)
      - DATABASE_URL=postgresql+asyncpg://lead_gen_admin:VFBZ$dPcrI)QyAag@leadgen-mvp-db.postgres.database.azure.com:5432/postgres
      - REDIS_URL=redis://redis:6379/0
      - CELERY_BROKER_URL=redis://redis:6379/1
      - CELERY_RESULT_BACKEND=redis://redis:6379/2
      - SECRET_KEY=your-production-secret-key
      - LOG_LEVEL=INFO
      - MAX_CONCURRENT_REQUESTS=3
      - REQUEST_DELAY_MIN=2.0
      - REQUEST_DELAY_MAX=5.0
    depends_on:
      - redis
    volumes:
      - ./logs:/app/logs
      - ./uploads:/app/uploads

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  worker:
    build: .
    command: celery -A app.tasks.celery_app worker --loglevel=info --concurrency=2
    environment:
      # Using Azure PostgreSQL
      - DATABASE_URL=postgresql+asyncpg://lead_gen_admin:VFBZ$dPcrI)QyAag@leadgen-mvp-db.postgres.database.azure.com:5432/postgres
      - REDIS_URL=redis://redis:6379/0
      - CELERY_BROKER_URL=redis://redis:6379/1
      - CELERY_RESULT_BACKEND=redis://redis:6379/2
      - SECRET_KEY=your-production-secret-key
      - LOG_LEVEL=INFO
    depends_on:
      - redis
    volumes:
      - ./logs:/app/logs

  # Optional: Redis Commander for Redis management
  redis-commander:
    image: rediscommander/redis-commander:latest
    environment:
      - REDIS_HOSTS=local:redis:6379
    ports:
      - "8081:8081"
    depends_on:
      - redis

volumes:
  redis_data:

# For development with local database (optional)
# Uncomment the following if you want to use local PostgreSQL for development
#
#  db:
#    image: postgres:15
#    environment:
#      - POSTGRES_DB=leadgen_dev
#      - POSTGRES_USER=leadgen_dev
#      - POSTGRES_PASSWORD=dev_password
#    volumes:
#      - postgres_dev_data:/var/lib/postgresql/data
#    ports:
#      - "5433:5432"  # Different port to avoid conflicts
#
#volumes:
#  postgres_dev_data:
"""

# .env.example
"""
# Application
PROJECT_NAME=Lead Generation API
VERSION=1.0.0
SECRET_KEY=your-super-secret-key-here
LOG_LEVEL=INFO

# Database (Azure PostgreSQL)
DATABASE_URL=postgresql+asyncpg://lead_gen_admin:VFBZ$dPcrI)QyAag@leadgen-mvp-db.postgres.database.azure.com:5432/postgres

# Redis (Local for development, Azure Redis for production)
REDIS_URL=redis://localhost:6379/0

# External APIs (Free tier options)
HUNTER_IO_API_KEY=your-hunter-io-api-key  # Free tier: 25 searches/month
CLEARBIT_API_KEY=your-clearbit-api-key    # Free tier: 20 requests/month

# CORS
BACKEND_CORS_ORIGINS=["http://localhost:3000","https://yourdomain.com"]

# Scraping Configuration (Conservative for free approach)
MAX_CONCURRENT_REQUESTS=3
REQUEST_DELAY_MIN=2.0
REQUEST_DELAY_MAX=5.0

# Rate Limiting (Conservative)
RATE_LIMIT_PER_MINUTE=30
RATE_LIMIT_PER_HOUR=200

# Monitoring
SENTRY_DSN=your-sentry-dsn

# Background Tasks
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2

# Optional: Advanced Sentiment Analysis
# Set to 'true' to enable FinBERT and RoBERTa models (requires more compute)
ENABLE_ADVANCED_SENTIMENT=false
"""