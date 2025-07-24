from typing import List, Dict, Any
from sqlalchemy.orm import Session  # type: ignore
from sqlalchemy import func  # type: ignore
from app.models.rl_models import SongRating
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
from app.models.user import User
from fuzzywuzzy import fuzz  # type: ignore
import numpy as np  # type: ignore

FEATURE_KEYS = [
    'danceability', 'energy', 'valence', 'acousticness',
    'instrumentalness', 'speechiness', 'liveness', 'tempo', 'loudness'
]

FEATURE_WEIGHTS = {
    'valence': 2.0,  # Critical for mood
    'energy': 1.5,   # Important for mood
    'danceability': 1.2,
    'acousticness': 1.0,
    'instrumentalness': 0.8,
    'speechiness': 0.8,
    'liveness': 0.8,
    'tempo': 1.0,
    'loudness': 1.0
}

def get_feature_vector(song: Dict[str, Any], min_loudness: float, max_loudness: float, min_tempo: float, max_tempo: float) -> List[float]:
    vector = []
    for key in FEATURE_KEYS:
        val = song.get(key, 0.0)
        if val is None or val == '':
            val = 0.0
            print(f"Warning: Missing {key} for song {song.get('track_id', 'unknown')}")
        if key == 'loudness' and max_loudness != min_loudness:
            val = (val - min_loudness) / (max_loudness - min_loudness + 1e-8)  # Normalize loudness
        elif key == 'tempo' and max_tempo != min_tempo:
            val = (val - min_tempo) / (max_tempo - min_tempo + 1e-8)  # Normalize tempo
        elif key in ['loudness', 'tempo']:
            val = 0.0
            print(f"Warning: Invalid bounds for {key} in song {song.get('track_id', 'unknown')}, using 0.0")
        vector.append(float(val) * FEATURE_WEIGHTS[key])
    if not any(vector):
        print(f"Error: Zero feature vector for song {song.get('track_id', 'unknown')}")
    return vector

def get_user_liked_feature_matrix(user_id: int, mood: str, db: Session) -> List[List[float]]:
    liked_songs = db.query(SongRating).filter(
        SongRating.user_id == user_id,
        SongRating.mood_at_rating == mood,
        SongRating.rating >= 4
    ).all()

    min_loudness, max_loudness = get_loudness_bounds(db)
    min_tempo, max_tempo = get_tempo_bounds(db)

    feature_matrix = []
    for song in liked_songs:
        vector = []
        for key in FEATURE_KEYS:
            val = getattr(song, key, 0.0) or 0.0
            if key == 'loudness' and max_loudness != min_loudness:
                val = (val - min_loudness) / (max_loudness - min_loudness + 1e-8)
            elif key == 'tempo' and max_tempo != min_tempo:
                val = (val - min_tempo) / (max_tempo - min_tempo + 1e-8)
            elif key in ['loudness', 'tempo']:
                val = 0.0
                print(f"Warning: Invalid bounds for {key} in liked song {song.song_id}, using 0.0")
            vector.append(float(val) * FEATURE_WEIGHTS[key])
        if any(vector):
            feature_matrix.append(vector)
        else:
            print(f"Warning: Zero feature vector for liked song {song.song_id}")

    if not feature_matrix:
        print(f"Warning: Empty feature matrix for user {user_id}, mood {mood}")
    return feature_matrix

def get_user_already_seen_songs(user_id: int, mood: str, db: Session) -> set:
    all_suggested_songs = db.query(SongRating).filter(
        SongRating.user_id == user_id,
        SongRating.mood_at_rating == mood
    ).all()
    seen = set()
    for song in all_suggested_songs:
        if song.song_id:
            seen.add(song.song_id)
        track_artist = getattr(song, 'track_artist', '') or ''
        track_name = getattr(song, 'track_name', '') or ''
        # Ensure track_artist and track_name are strings
        if not isinstance(track_artist, str):
            print(f"Warning: Invalid track_artist type for song {song.song_id}: {track_artist}")
            track_artist = ""
        if not isinstance(track_name, str):
            print(f"Warning: Invalid track_name type for song {song.song_id}: {track_name}")
            track_name = ""
        if track_artist and track_name:
            seen.add((track_artist.lower(), track_name.lower()))
    if not seen:
        print(f"Warning: No already seen songs for user {user_id}, mood {mood}")
    return seen

def get_best_match_songs(
    songs: List[Dict[str, Any]], 
    mood: str, 
    user_id: int, 
    db: Session
) -> List[Dict[str, Any]]:
    if not songs:
        print("Error: Empty songs list provided")
        return []

    # Remove duplicates and sanitize inputs
    unique_songs = []
    seen_identifiers = set()
    seen_song_keys = set()
    for song in songs:
        track_id = song.get("track_id")
        track_artist = song.get("track_artist", "")
        track_name = song.get("track_name", "")
        
        # Ensure track_artist and track_name are strings
        if not isinstance(track_artist, str):
            print(f"Warning: Invalid track_artist type for song {track_id or 'unknown'}: {track_artist}")
            track_artist = ""
        if not isinstance(track_name, str):
            print(f"Warning: Invalid track_name type for song {track_id or 'unknown'}: {track_name}")
            track_name = ""
        
        song_key = (track_artist.lower(), track_name.lower()) if track_artist and track_name else None

        if not track_id and not song_key:
            print(f"Warning: Skipping song with missing identifier: {track_name or 'Unknown'} by {track_artist or 'Unknown'}")
            continue

        if song_key and song_key in seen_song_keys:
            print(f"Warning: Duplicate song found and skipped: {track_name} by {track_artist}")
            continue
        if track_id and track_id in seen_identifiers:
            print(f"Warning: Duplicate track_id found and skipped: {track_id}")
            continue

        unique_songs.append(song)
        if track_id:
            seen_identifiers.add(track_id)
        if song_key:
            seen_song_keys.add(song_key)

    songs = unique_songs
    print(f"Processing {len(songs)} unique songs")

    user_feature_matrix = get_user_liked_feature_matrix(user_id, mood, db)
    min_loudness, max_loudness = get_loudness_bounds(db)
    min_tempo, max_tempo = get_tempo_bounds(db)

    user = db.query(User).filter(User.id == user_id).first()
    # Sanitize fav_artists to ensure it's a list of strings
    fav_artists = set()
    if user and user.user_fav_artists:
        for artist in user.user_fav_artists:
            if isinstance(artist, str) and artist.strip():
                fav_artists.add(artist)
            else:
                print(f"Warning: Invalid favorite artist entry for user {user_id}: {artist}")
    if not fav_artists:
        print(f"Warning: No valid favorite artists for user {user_id}")

    already_seen = get_user_already_seen_songs(user_id, mood, db)
    seen_identifiers = set()
    seen_song_keys = set()
    result_songs = []

    if user_feature_matrix:
        user_pref_vector = np.mean(user_feature_matrix, axis=0).reshape(1, -1)
        if not np.any(user_pref_vector):
            print(f"Error: User preference vector is zero for user {user_id}, mood {mood}")
            user_feature_matrix = []

    if user_feature_matrix:
        scored_songs = []
        for song in songs:
            track_id = song.get("track_id")
            track_artist = song.get("track_artist", "")
            track_name = song.get("track_name", "")
            # Ensure track_artist and track_name are strings
            if not isinstance(track_artist, str):
                print(f"Warning: Invalid track_artist type for song {track_id or 'unknown'}: {track_artist}")
                track_artist = ""
            if not isinstance(track_name, str):
                print(f"Warning: Invalid track_name type for song {track_id or 'unknown'}: {track_name}")
                track_name = ""
            song_key = (track_artist.lower(), track_name.lower()) if track_artist and track_name else None

            if not track_id and not song_key:
                print(f"Warning: Skipping song with missing identifier: {track_name or 'Unknown'} by {track_artist or 'Unknown'}")
                continue
            if song_key and song_key in seen_song_keys:
                continue
            if track_id and track_id in seen_identifiers:
                continue

            if not track_name:
                print(f"Warning: Missing track_name for song {track_id or f'{track_artist} - {track_name}'}")

            song_vector = np.array(get_feature_vector(song, min_loudness, max_loudness, min_tempo, max_tempo)).reshape(1, -1)
            if not np.any(song_vector):
                print(f"Skipping song {track_id or f'{track_artist} - {track_name}'}: Zero feature vector")
                continue

            base_similarity = cosine_similarity(user_pref_vector, song_vector)[0][0]
            if not np.isfinite(base_similarity):
                print(f"Skipping song {track_id or f'{track_artist} - {track_name}'}: Invalid similarity {base_similarity}")
                base_similarity = 0.0

            similarity = base_similarity
            boost = 0.0
            penalty = 0.0

            if fav_artists:
                if artist_matches(track_artist, fav_artists):
                    artist_similarity = max(fuzz.ratio(track_artist.lower(), fav_artist.lower()) for fav_artist in fav_artists if isinstance(fav_artist, str))
                    boost = 0.2 * (artist_similarity / 100.0)
                    similarity += boost
                    print(f"Artist match: {track_artist} with similarity {artist_similarity}%")

            if already_seen and (track_id in already_seen or (song_key and song_key in already_seen)):
                penalty = 0.3
                similarity -= penalty
                print(f"Penalty applied: Song {track_id or f'{track_artist} - {track_name}'} already seen")

            tiebreaker = -song.get("distance", float("inf"))
            if "popularity" in song:
                tiebreaker += song.get("popularity", 0) * 0.005

            # print(f"Song ID: {track_id or f'{track_artist} - {track_name}'}, Track: {track_name or 'Unknown'}, "
            #       f"Base similarity: {base_similarity:.4f}, Boost: +{boost:.4f}, "
            #       f"Penalty: -{penalty:.4f}, Final similarity: {similarity:.4f}, "
            #       f"Tiebreaker: {tiebreaker:.4f}, Features: {get_feature_vector(song, min_loudness, max_loudness, min_tempo, max_tempo)}")

            scored_songs.append((similarity, tiebreaker, song))

        scored_songs.sort(key=lambda x: (-x[0], -x[1]))
        for _, _, song in scored_songs:
            track_id = song.get("track_id")
            track_artist = song.get("track_artist", "")
            track_name = song.get("track_name", "")
            # Ensure track_artist and track_name are strings
            if not isinstance(track_artist, str):
                print(f"Warning: Invalid track_artist type for result song {track_id or 'unknown'}: {track_artist}")
                track_artist = ""
            if not isinstance(track_name, str):
                print(f"Warning: Invalid track_name type for result song {track_id or 'unknown'}: {track_name}")
                track_name = ""
            song_key = (track_artist.lower(), track_name.lower()) if track_artist and track_name else None
            if (song_key and song_key not in seen_song_keys) or (track_id and track_id not in seen_identifiers):
                result_songs.append(song)
                if track_id:
                    seen_identifiers.add(track_id)
                if song_key:
                    seen_song_keys.add(song_key)
                if len(result_songs) == 5:
                    print(f"Returning {len(result_songs)} unique songs from cosine similarity")
                    return result_songs

    # Fallback to artist matching
    matching_artists_songs = [
        song for song in songs 
        if (song.get("track_id") or (song.get("track_artist") and song.get("track_name")))
        and (song.get("track_id") not in seen_identifiers if song.get("track_id") 
             else (song.get("track_artist", "").lower(), song.get("track_name", "").lower()) not in seen_song_keys)
        and artist_matches(song.get("track_artist", ""), fav_artists)
    ]
    matching_artists_songs.sort(key=lambda x: (
        x.get("distance", float("inf")),
        -x.get("popularity", 0) if "popularity" in x else 0
    ))

    for song in matching_artists_songs:
        track_id = song.get("track_id")
        track_artist = song.get("track_artist", "")
        track_name = song.get("track_name", "")
        # Ensure track_artist and track_name are strings
        if not isinstance(track_artist, str):
            print(f"Warning: Invalid track_artist type for artist match song {track_id or 'unknown'}: {track_artist}")
            track_artist = ""
        if not isinstance(track_name, str):
            print(f"Warning: Invalid track_name type for artist match song {track_id or 'unknown'}: {track_name}")
            track_name = ""
        song_key = (track_artist.lower(), track_name.lower()) if track_artist and track_name else None
        if (song_key and song_key not in seen_song_keys) or (track_id and track_id not in seen_identifiers):
            result_songs.append(song)
            if track_id:
                seen_identifiers.add(track_id)
            if song_key:
                seen_song_keys.add(song_key)
            if len(result_songs) == 5:
                print(f"Returning {len(result_songs)} unique songs from artist matching")
                return result_songs

    # Fallback to remaining songs
    remaining_needed = 5 - len(result_songs)
    additional_songs = [
        song for song in sorted(
            songs, 
            key=lambda x: (x.get("distance", float("inf")), -x.get("popularity", 0) if "popularity" in x else 0)
        )
        if (song.get("track_id") or (song.get("track_artist") and song.get("track_name")))
        and (song.get("track_id") not in seen_identifiers if song.get("track_id") 
             else (song.get("track_artist", "").lower(), song.get("track_name", "").lower()) not in seen_song_keys)
        and (not already_seen or (
            song.get("track_id") not in already_seen if song.get("track_id") 
            else (song.get("track_artist", "").lower(), song.get("track_name", "").lower()) not in already_seen))
    ][:remaining_needed]

    for song in additional_songs:
        track_id = song.get("track_id")
        track_artist = song.get("track_artist", "")
        track_name = song.get("track_name", "")
        # Ensure track_artist and track_name are strings
        if not isinstance(track_artist, str):
            print(f"Warning: Invalid track_artist type for additional song {track_id or 'unknown'}: {track_artist}")
            track_artist = ""
        if not isinstance(track_name, str):
            print(f"Warning: Invalid track_name type for additional song {track_id or 'unknown'}: {track_name}")
            track_name = ""
        song_key = (track_artist.lower(), track_name.lower()) if track_artist and track_name else None
        if (song_key and song_key not in seen_song_keys) or (track_id and track_id not in seen_identifiers):
            result_songs.append(song)
            if track_id:
                seen_identifiers.add(track_id)
            if song_key:
                seen_song_keys.add(song_key)

    if len(result_songs) < 5:
        print(f"Warning: Only {len(result_songs)} unique songs found for user {user_id}, mood {mood}")

    print(f"Returning {len(result_songs)} unique songs")
    return result_songs[:5]

def get_loudness_bounds(db: Session) -> tuple[float, float]:
    min_loudness = db.query(SongRating).with_entities(func.min(SongRating.loudness)).scalar() or -60.0
    max_loudness = db.query(SongRating).with_entities(func.max(SongRating.loudness)).scalar() or 0.0
    if min_loudness == max_loudness:
        print(f"Warning: Identical loudness bounds ({min_loudness})")
    return min_loudness, max_loudness

def get_tempo_bounds(db: Session) -> tuple[float, float]:
    min_tempo = db.query(SongRating).with_entities(func.min(SongRating.tempo)).scalar() or 0.0
    max_tempo = db.query(SongRating).with_entities(func.max(SongRating.tempo)).scalar() or 200.0
    if min_tempo == max_tempo:
        print(f"Warning: Identical tempo bounds ({min_tempo})")
    return min_tempo, max_tempo

def artist_matches(song_artist: Any, fav_artists: List[Any], threshold: int = 60) -> bool:
    # Ensure song_artist is a string
    if not song_artist or not isinstance(song_artist, str):
        print(f"Warning: Invalid song_artist type or value: {song_artist}")
        return False
    
    # Ensure fav_artists is a list or set of strings
    if not fav_artists or not isinstance(fav_artists, (list, set)):
        print(f"Warning: Invalid fav_artists type or value: {fav_artists}")
        return False
    
    for fav_artist in fav_artists:
        # Ensure fav_artist is a string
        if not fav_artist or not isinstance(fav_artist, str):
            print(f"Warning: Invalid fav_artist type or value: {fav_artist}")
            continue
        similarity = fuzz.ratio(song_artist.lower(), fav_artist.lower())
        if similarity >= threshold:
            print(f"Artist match found: {song_artist} vs {fav_artist}, similarity {similarity}%")
            return True
    return False