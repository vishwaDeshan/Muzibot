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
    seen_track_ids = set()  # Track unique song IDs to avoid duplicates
    result_songs = []  # Store final 5 unique songs

    if user_feature_matrix:
        user_pref_vector = np.mean(user_feature_matrix, axis=0).reshape(1, -1)

        scored_songs = []
        for song in songs:
            if song.get("track_id") in seen_track_ids:
                continue  # Skip duplicates
            song_vector = np.array(get_feature_vector(song, min_loudness, max_loudness)).reshape(1, -1)
            similarity = cosine_similarity(user_pref_vector, song_vector)[0][0]

            # Boost score if artist matches
            if fav_artists and artist_matches(song.get("track_artist", ""), fav_artists):
                similarity += 0.05  # Boost score slightly

            if already_seen and song.get("track_id") in already_seen:
                similarity -= 0.05  # Penalize already seen songs

            scored_songs.append((similarity, song))

        # Sort by similarity (descending) and collect up to 5 unique songs
        scored_songs.sort(key=lambda x: -x[0])
        for _, song in scored_songs:
            if song["track_id"] not in seen_track_ids:
                result_songs.append(song)
                seen_track_ids.add(song["track_id"])
                if len(result_songs) == 5:
                    return result_songs

    # If no liked songs or not enough from cosine similarity, rely on favorite artists
    matching_artists_songs = [
        song for song in songs 
        if artist_matches(song.get("track_artist", ""), fav_artists) and song["track_id"] not in seen_track_ids
    ]
    matching_artists_songs.sort(key=lambda x: x.get("distance", float("inf")))

    # Add unique songs from matching artists
    for song in matching_artists_songs:
        if song["track_id"] not in seen_track_ids:
            result_songs.append(song)
            seen_track_ids.add(song["track_id"])
            if len(result_songs) == 5:
                return result_songs

    # If still need more songs, add from remaining songs sorted by distance
    remaining_needed = 5 - len(result_songs)
    additional_songs = [
        song for song in sorted(songs, key=lambda x: x.get("distance", float("inf")))
        if song["track_id"] not in seen_track_ids and (not already_seen or song["track_id"] not in already_seen)
    ][:remaining_needed]

    # Add unique additional songs
    for song in additional_songs:
        if song["track_id"] not in seen_track_ids:
            result_songs.append(song)
            seen_track_ids.add(song["track_id"])

    # If still fewer than 5 songs (e.g., input list too small), pad with remaining unique songs
    if len(result_songs) < 5:
        remaining_songs = [
            song for song in songs
            if song["track_id"] not in seen_track_ids and (not already_seen or song["track_id"] not in already_seen)
        ]
        for song in remaining_songs[:5 - len(result_songs)]:
            result_songs.append(song)
            seen_track_ids.add(song["track_id"])

    return result_songs[:5]

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
