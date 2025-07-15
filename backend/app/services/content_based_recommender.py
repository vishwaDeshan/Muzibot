from typing import List, Dict
from sqlalchemy.orm import Session # type: ignore
from app.models.rl_models import SongRating
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
import numpy as np # type: ignore

FEATURE_KEYS = [
    'danceability', 'energy', 'valence', 'acousticness',
    'instrumentalness', 'speechiness', 'liveness', 'tempo'
]

def get_feature_vector(song: Dict[str, any]) -> List[float]:
    return [song.get(k, 0.0) for k in FEATURE_KEYS]

def get_user_liked_feature_matrix(user_id: int, mood: str, db: Session) -> List[List[float]]:
    liked_songs = db.query(SongRating).filter(
        SongRating.user_id == user_id,
        SongRating.mood_at_rating == mood,
        SongRating.rating >= 4
    ).all()
    
    feature_matrix = []
    for song in liked_songs:
        vector = [getattr(song, key, 0.0) for key in FEATURE_KEYS]
        feature_matrix.append(vector)
    return feature_matrix

def get_best_match_songs(songs: List[Dict[str, any]], mood: str, user_id: int, db: Session) -> List[Dict[str, any]]:
    # Step 1: Get user's liked songs for the mood
    user_feature_matrix = get_user_liked_feature_matrix(user_id, mood, db)
    
    if not user_feature_matrix:
        # Fallback: use songs with lowest distance
        sorted_by_distance = sorted(songs, key=lambda x: x.get("distance", float("inf")))
        return sorted_by_distance[:5]
    
    # Step 2: Compute user preference vector (mean of liked songs)
    user_pref_vector = np.mean(user_feature_matrix, axis=0).reshape(1, -1)

    # Step 3: Compute similarity with each available song
    scored_songs = []
    for song in songs:
        song_vector = np.array(get_feature_vector(song)).reshape(1, -1)
        similarity = cosine_similarity(user_pref_vector, song_vector)[0][0]
        scored_songs.append((similarity, song))

    # Step 4: Sort and return top 5
    scored_songs.sort(key=lambda x: -x[0])
    top_songs = [song for _, song in scored_songs[:5]]
    
    return top_songs
