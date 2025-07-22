from typing import List, Dict, Any
from sqlalchemy.orm import Session  # type: ignore
from sqlalchemy import func # type: ignore
from app.models.rl_models import SongRating
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
from app.models.user import User
from fuzzywuzzy import fuzz # type: ignore
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

def get_user_already_seen_songs(user_id: int,mood:str, db: Session) -> set:
    all_suggested_songs = db.query(SongRating).filter(
        SongRating.user_id == user_id,
        SongRating.mood_at_rating == mood
    ).all()
    return {song.song_id for song in all_suggested_songs}

# Recommend top 5 songs based on cosine similarity
def get_best_match_songs(
    songs: List[Dict[str, Any]], 
    mood: str, 
    user_id: int, 
    db: Session
) -> List[Dict[str, Any]]:
    user_feature_matrix = get_user_liked_feature_matrix(user_id, mood, db)
    min_loudness, max_loudness = get_loudness_bounds(db)

    user = db.query(User).filter(User.id == user_id).first()
    fav_artists = set(user.user_fav_artists or [])

    # Initialize already_seen for use in both branches
    already_seen = get_user_already_seen_songs(user_id, mood, db)

    if user_feature_matrix:
        user_pref_vector = np.mean(user_feature_matrix, axis=0).reshape(1, -1)

        scored_songs = []

        for song in songs:
            song_vector = np.array(get_feature_vector(song, min_loudness, max_loudness)).reshape(1, -1)
            similarity = cosine_similarity(user_pref_vector, song_vector)[0][0]

            # Boost score if artist matches
            if fav_artists and artist_matches(song.get("track_artist", ""), fav_artists):
                similarity += 0.05  # boost score slightly

            if already_seen and song.get("track_id") in already_seen:
                similarity -= 0.05  # penalize already seen songs

            scored_songs.append((similarity, song))

        scored_songs.sort(key=lambda x: -x[0])
        top_songs = [song for _, song in scored_songs[:5]]
        return top_songs

    # If no liked songs, rely on favorite artists
    matching_artists_songs = [song for song in songs 
                      if artist_matches(song.get("track_artist", ""), fav_artists) ]

    if matching_artists_songs:
        matching_artists_songs.sort(key=lambda x: x.get("distance", float("inf")))

        if len(matching_artists_songs) >= 5:
            return matching_artists_songs[:5]

        remaining_needed = 5 - len(matching_artists_songs)
        matching_ids = {song["track_id"] for song in matching_artists_songs}

        if already_seen:
            additional_songs = [
                song for song in sorted(songs, key=lambda x: x.get("distance", float("inf")))
                if song["track_id"] not in matching_ids and song["track_id"] not in already_seen
            ][:remaining_needed]
        else:
            additional_songs = [
                song for song in sorted(songs, key=lambda x: x.get("distance", float("inf")))
                if song["track_id"] not in matching_ids
            ][:remaining_needed]

        return matching_artists_songs + additional_songs

    # If no matching artists, fallback to distance (new users with no spotify link)
    sorted_by_distance = sorted(songs, key=lambda x: x.get("distance", float("inf")))
    return sorted_by_distance[:5]

def get_loudness_bounds(db: Session) -> tuple[float, float]:
    min_loudness = db.query(SongRating).with_entities(func.min(SongRating.loudness)).scalar() or -60.0
    max_loudness = db.query(SongRating).with_entities(func.max(SongRating.loudness)).scalar() or 0.0
    return min_loudness, max_loudness

def artist_matches(song_artist: str, fav_artists: List[str], threshold: int = 70) -> bool:
    for fav_artist in fav_artists:
        similarity = fuzz.ratio(song_artist.lower(), fav_artist.lower())
        if similarity >= threshold:
            return True
    return False
