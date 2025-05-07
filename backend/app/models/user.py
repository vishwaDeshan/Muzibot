from sqlalchemy import Column, Integer, String, JSON, DateTime
from sqlalchemy.sql import func
from app.database import Base
from sqlalchemy.orm import relationship

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    preferences = Column(JSON, default={})
    created_at = Column(DateTime, server_default=func.now())
    music_preferences = Column(JSON, default={})

     # One-to-many relationship
    settings = relationship("UserSettings", back_populates="user", cascade="all, delete-orphan")