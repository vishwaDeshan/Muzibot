from sqlalchemy import Column, Integer, String, JSON, DateTime, Float, ForeignKey # type: ignore
from sqlalchemy.sql import func # type: ignore
from sqlalchemy.orm import relationship # type: ignore
from app.database import Base

class UserSettings(Base):
    __tablename__ = "user_settings"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    current_mood = Column(String, nullable=False)
    reward_score = Column(Float, default=1.0)
    optimal_point = Column(Float, nullable=True)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    user = relationship("User", back_populates="settings")