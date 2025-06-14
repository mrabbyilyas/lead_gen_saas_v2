# app/models/user.py
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from app.models.base import Base, TimestampMixin, SoftDeleteMixin

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