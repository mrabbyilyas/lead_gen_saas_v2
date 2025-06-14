# app/models/sentiment.py
from sqlalchemy import Column, Integer, String, Text, Float, JSON, ForeignKey, DateTime
from app.models.base import Base, TimestampMixin  # Fix import

class SentimentAnalysis(Base, TimestampMixin):
    __tablename__ = "sentiment_analysis"
    
    id = Column(Integer, primary_key=True, index=True)
    lead_id = Column(Integer, ForeignKey("leads.id"), nullable=False)
    company_name = Column(String(255), nullable=False, index=True)
    
    # Sentiment Scores (-1 to 1)
    overall_sentiment_score = Column(Float)
    confidence_score = Column(Float)
    
    # Classifications
    overall_sentiment = Column(String(20))  # positive, negative, neutral
    sentiment_trend = Column(String(20))    # improving, declining, stable
    
    # Detailed Analysis
    detailed_analysis = Column(Text)  # AI-generated summary
    key_topics = Column(JSON)         # List of key topics found
    
    # News Sources
    news_sources = Column(JSON)       # List of news sources analyzed
    news_count = Column(Integer)      # Number of articles analyzed
    
    # Expiration
    expires_at = Column(DateTime(timezone=True))  # Cache expiration
    
    # Note: Removed relationship to avoid circular imports