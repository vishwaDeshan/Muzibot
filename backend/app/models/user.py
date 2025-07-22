from sqlalchemy import Column, Integer, String, JSON, DateTime # type: ignore
from sqlalchemy.sql import func # type: ignore
from app.database import Base
from sqlalchemy.orm import relationship # type: ignore

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    desired_mood = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    favourite_music_genres = Column(JSON, default='[]')
    user_fav_artists = Column(JSON, default='[]')

    # One-to-many relationships
    q_table_entries = relationship("RLQTable", back_populates="user", cascade="all, delete-orphan")
    weights = relationship("RLWeights", back_populates="user", cascade="all, delete-orphan")
    song_ratings = relationship("SongRating", back_populates="user", cascade="all, delete-orphan")