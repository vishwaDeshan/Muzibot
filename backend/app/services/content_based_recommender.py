from typing import List, Dict, Any
from sqlalchemy.orm import Session  # type: ignore
from sqlalchemy import func # type: ignore
from app.models.rl_models import SongRating
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
import numpy as np  # type: ignore

FEATURE_KEYS = [
    'danceability', 'energy', 'valence', 'acousticness',
    'instrumentalness', 'speechiness', 'liveness', 'tempo', 'loudness'
]

# Extract feature vector from a song dictionary
def get_feature_vector(song: Dict[str, Any], min_loudness: float, max_loudness: float) -> List[float]:
    vector = []
    for key in FEATURE_KEYS:
        val = song.get(key, 0.0) or 0.0
        if key == 'loudness':
            val = (val - min_loudness) / (max_loudness - min_loudness + 1e-8)  # avoid divide by zero
        vector.append(float(val))
    return vector

# Get liked songs (rating >= 4) for a specific mood
def get_user_liked_feature_matrix(user_id: int, mood: str, db: Session) -> List[List[float]]:
    liked_songs = db.query(SongRating).filter(
        SongRating.user_id == user_id,
        SongRating.mood_at_rating == mood,
        SongRating.rating >= 4
    ).all()

    feature_matrix = []
    for song in liked_songs:
        vector = [getattr(song, key, 0.0) or 0.0 for key in FEATURE_KEYS]
        feature_matrix.append(vector)

    return feature_matrix

# Recommend top 5 songs based on cosine similarity
def get_best_match_songs(
    songs: List[Dict[str, Any]], 
    mood: str, 
    user_id: int, 
    db: Session
) -> List[Dict[str, Any]]:
    user_feature_matrix = get_user_liked_feature_matrix(user_id, mood, db)
    min_loudness, max_loudness = get_loudness_bounds(db)

    if not user_feature_matrix:
        sorted_by_distance = sorted(songs, key=lambda x: x.get("distance", float("inf")))
        return sorted_by_distance[:5]

    user_pref_vector = np.mean(user_feature_matrix, axis=0).reshape(1, -1)

    scored_songs = []
    for song in songs:
        song_vector = np.array(get_feature_vector(song, min_loudness, max_loudness)).reshape(1, -1)
        similarity = cosine_similarity(user_pref_vector, song_vector)[0][0]
        scored_songs.append((similarity, song))

    scored_songs.sort(key=lambda x: -x[0])
    top_songs = [song for _, song in scored_songs[:5]]

    return top_songs

def get_loudness_bounds(db: Session) -> tuple[float, float]:
    min_loudness = db.query(SongRating).with_entities(func.min(SongRating.loudness)).scalar() or -60.0
    max_loudness = db.query(SongRating).with_entities(func.max(SongRating.loudness)).scalar() or 0.0
    return min_loudness, max_loudness
