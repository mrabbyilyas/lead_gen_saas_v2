# app/models/leads.py
from sqlalchemy import Column, Integer, String, Text, Float, JSON, ForeignKey, Boolean, DateTime
from app.models.base import Base, TimestampMixin, SoftDeleteMixin

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
    
    # Quality Metrics
    data_quality_score = Column(Float, default=0.0)
    verification_status = Column(String(20), default="unverified")
    
    # Scraping Metadata
    source_urls = Column(JSON)  # List of source URLs
    last_scraped = Column(DateTime(timezone=True))
    
    # Note: Relationships removed for now - will add later