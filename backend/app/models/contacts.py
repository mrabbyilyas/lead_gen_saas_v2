# app/models/contacts.py
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Float
from app.models.base import Base, TimestampMixin, SoftDeleteMixin

class Contact(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "contacts"
    
    id = Column(Integer, primary_key=True, index=True)
    lead_id = Column(Integer, ForeignKey("leads.id"), nullable=False)
    
    # Personal Information
    first_name = Column(String(100))
    last_name = Column(String(100))
    full_name = Column(String(255), index=True)
    title = Column(String(255))
    email = Column(String(255), index=True)
    phone = Column(String(50))
    
    # Verification Status
    email_verified = Column(Boolean, default=False)
    phone_verified = Column(Boolean, default=False)
    
    # Quality Metrics
    contact_quality_score = Column(Float, default=0.0)
    
    # Note: Relationships removed for now - will add later