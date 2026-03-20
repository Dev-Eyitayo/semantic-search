from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from db.base import Base

class TestLead(Base):
    __tablename__ = "test_leads"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    interest_area = Column(String(100)) 
    created_at = Column(DateTime(timezone=True), server_default=func.now())