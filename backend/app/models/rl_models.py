from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, ForeignKey, DateTime, CheckConstraint, UniqueConstraint # type: ignore
from sqlalchemy.sql import func # type: ignore
from sqlalchemy.orm import relationship # type: ignore
from app.database import Base

class RLQTable(Base):
    __tablename__ = "rl_q_table"
    __table_args__ = (
        CheckConstraint('arousal >= -1 AND arousal <= 1', name='check_arousal_range'),
        CheckConstraint('valence >= -1 AND valence <= 1', name='check_valence_range'),
        CheckConstraint('weight_similar_users_music_prefs_idx >= 0 AND weight_similar_users_music_prefs_idx <= 4', name='check_similar_users_idx'),
        CheckConstraint('weight_current_user_mood_idx >= 0 AND weight_current_user_mood_idx <= 4', name='check_current_mood_idx'),
        CheckConstraint('weight_desired_mood_after_listening_idx >= 0 AND weight_desired_mood_after_listening_idx <= 4', name='check_desired_mood_idx'),
    )

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    song_id = Column(String, nullable=False, index=True)
    mood = Column(String, nullable=False)
    prev_rating = Column(Integer, CheckConstraint('prev_rating >= 1 AND prev_rating <= 5'), nullable=False)
    arousal = Column(Float, nullable=True)  # Arousal coordinate [-1, 1]
    valence = Column(Float, nullable=True)  # Valence coordinate [-1, 1]
    weight_similar_users_music_prefs_idx = Column(Integer, nullable=False)  # Index for weight_values [0, 4]
    weight_current_user_mood_idx = Column(Integer, nullable=False)  # Index for weight_values [0, 4]
    weight_desired_mood_after_listening_idx = Column(Integer, nullable=False)  # Index for weight_values [0, 4]
    q_value = Column(Float, default=0.0)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="q_table_entries")

class RLWeights(Base):
    __tablename__ = "rl_weights"
    __table_args__ = (
        UniqueConstraint('user_id', 'mood', name='uq_rl_weights_user_id_mood'),
        CheckConstraint('weight_similar_users_music_prefs >= 0', name='check_similar_users_non_negative'),
        CheckConstraint('weight_current_user_mood >= 0', name='check_current_mood_non_negative'),
        CheckConstraint('weight_desired_mood_after_listening >= 0', name='check_desired_mood_non_negative'),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)  # Use autoincrement=True
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
    __table_args__ = (
        CheckConstraint('arousal >= -1 AND arousal <= 1', name='check_song_arousal_range'),
        CheckConstraint('valence >= -1 AND valence <= 1', name='check_song_valence_range'),
    )

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    song_id = Column(String, nullable=False)  # Spotify song ID
    rating = Column(Integer, CheckConstraint('rating >= 1 AND rating <= 5'), nullable=False)  # 1 to 5
    mood_at_rating = Column(String, nullable=False)  # Mood at the time of rating
    arousal = Column(Float, nullable=True)  # Arousal coordinate [0, 1]
    valence = Column(Float, nullable=True)  # Valence coordinate [0, 1]
    context = Column(String, nullable=True)
    danceability = Column(Float, nullable=True)
    energy = Column(Float, nullable=True)
    acousticness = Column(Float,nullable=True)
    instrumentalness = Column(Float, nullable=True)
    speechiness = Column(Float, nullable=True)
    liveness = Column(Float, nullable=True)
    tempo = Column(Float, nullable=True)
    loudness = Column(Float, nullable=True)
    prev_rating = Column(Integer, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="song_ratings")

class RLTrainingLog(Base):
    __tablename__ = 'rl_training_logs'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    song_id = Column(String, index=True)
    mood = Column(String)
    reward = Column(Float)
    actual_rating = Column(Integer)
    episode = Column(Integer)
    created_at = Column(DateTime, server_default=func.now())