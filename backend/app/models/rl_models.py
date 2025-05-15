from sqlalchemy import Column, Integer, Float, String, ForeignKey, DateTime, CheckConstraint, UniqueConstraint # type: ignore
from sqlalchemy.sql import func # type: ignore
from sqlalchemy.orm import relationship # type: ignore
from app.database import Base

class RLQTable(Base):
    __tablename__ = "rl_q_table"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True)
    mood = Column(String, nullable=False)  # e.g."Happy"
    prev_rating = Column(Integer, nullable=False)
    weight_similar_users_music_prefs_idx = Column(Integer, nullable=False)  
    weight_current_user_mood_idx = Column(Integer, nullable=False)  
    weight_desired_mood_after_listening_idx = Column(Integer, nullable=False)
    q_value = Column(Float, default=0.0)  # Q-value for this state-action pair
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="q_table_entries")

class RLWeights(Base):
    __tablename__ = "rl_weights"
    __table_args__ = (
        UniqueConstraint('user_id', 'mood', name='uq_rl_weights_user_id_mood'),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    mood = Column(String, nullable=False)
    weight_similar_users_music_prefs = Column(Float, nullable=False)
    weight_current_user_mood = Column(Float, nullable=False)
    weight_desired_mood_after_listening = Column(Float, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="weights")

class SongRating(Base):
    __tablename__ = "song_ratings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    song_id = Column(String, nullable=False)  # Spotify song ID
    rating = Column(Integer, CheckConstraint('rating >= 1 AND rating <= 5'), nullable=False)  # 1 to 5
    mood_at_rating = Column(String, nullable=False)  # Mood at the time of rating
    created_at = Column(DateTime, server_default=func.now())

    user = relationship("User", back_populates="song_ratings")