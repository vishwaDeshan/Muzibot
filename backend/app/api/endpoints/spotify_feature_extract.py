from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import spotipy  # type: ignore
from spotipy.oauth2 import SpotifyClientCredentials  # type: ignore
import os
from dotenv import load_dotenv
import json
from typing import Dict, Optional, List
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from app.database import get_db  # Adjust path as needed
from app.models.user import User  # Adjust path as needed
from collections import Counter

# Load environment variables
load_dotenv()

router = APIRouter()

# Pydantic model for request body
class PlaylistRequest(BaseModel):
    playlist_id: str
    user_id: int

# Simulated cache for audio features (replace with actual cache implementation)
def load_cached_features(track_id: str) -> Optional[Dict]:
    # Placeholder: Assume a JSON file or database with cached audio features
    try:
        with open("audio_features_cache.json", "r") as f:
            cache = json.load(f)
        return cache.get(track_id)
    except FileNotFoundError:
        return None

# Spotify API client setup
def get_spotify_client():
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        raise HTTPException(status_code=500, detail="Spotify credentials not configured")
    
    try:
        auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        return spotipy.Spotify(auth_manager=auth_manager)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize Spotify client: {str(e)}")

# Infer playlist genre from artist genres
def infer_playlist_genre(artist_ids: List[str], spotify: spotipy.Spotify) -> Dict:
    try:
        # Fetch artist data in batches (max 50 per request)
        all_genres = []
        for i in range(0, len(artist_ids), 50):
            batch = artist_ids[i:i+50]
            artists = spotify.artists(batch)
            for artist in artists["artists"]:
                all_genres.extend(artist.get("genres", []))
        
        # Count genre frequencies
        genre_counts = Counter(all_genres)
        total_genres = sum(genre_counts.values())
        
        if not genre_counts:
            return {"primary_genre": None, "top_genres": [], "note": "No genre data available for artists"}
        
        # Get primary genre (most frequent) and top genres (e.g., top 3)
        primary_genre = genre_counts.most_common(1)[0][0]
        top_genres = [
            {"genre": genre, "percentage": round((count / total_genres) * 100, 2)}
            for genre, count in genre_counts.most_common(3)
        ]
        
        return {
            "primary_genre": primary_genre,
            "top_genres": top_genres
        }
    except Exception as e:
        return {"primary_genre": None, "top_genres": [], "note": f"Error fetching genres: {str(e)}"}

@router.post("/extract-playlist-features")
async def extract_playlist_features(
    request: PlaylistRequest,
    db: Session = Depends(get_db)
):
    try:
        spotify = get_spotify_client()

        playlist_id = request.playlist_id
        if "open.spotify.com/playlist/" in playlist_id:
            playlist_id = playlist_id.split("/playlist/")[1].split("?")[0]

        playlist = spotify.playlist(playlist_id, fields="name,description,external_urls,tracks(total)")
        if not playlist:
            raise HTTPException(status_code=404, detail="Playlist not found")

        tracks = []
        artist_ids = set()
        offset = 0
        limit = 100
        while True:
            results = spotify.playlist_tracks(
                playlist_id,
                fields="items(track(id,name,artists(id,name),album(name,images),external_urls)),next",
                limit=limit,
                offset=offset
            )
            for item in results["items"]:
                tracks.append(item)
                if item["track"] and item["track"]["artists"]:
                    for artist in item["track"]["artists"]:
                        if artist.get("id"):
                            artist_ids.add(artist["id"])
            if not results["next"]:
                break
            offset += limit

        artist_details = {}
        artist_ids = list(artist_ids)
        for i in range(0, len(artist_ids), 50):
            batch = artist_ids[i:i + 50]
            artists_response = spotify.artists(batch)
            for artist in artists_response["artists"]:
                artist_details[artist["id"]] = {
                    "id": artist["id"],
                    "name": artist["name"]
                }

        user = db.query(User).filter(User.id == request.user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        user.user_fav_artists = list({artist["name"] for artist in artist_details.values()})
        db.commit()

        genre_info = infer_playlist_genre(artist_ids, spotify)

        response = {
            "playlist_name": playlist["name"],
            "playlist_url": playlist["external_urls"]["spotify"],
            "total_tracks": playlist["tracks"]["total"],
            "genre": genre_info,
            "tracks": []
        }

        for track_item in tracks:
            track = track_item["track"]
            if not track:
                continue

            track_data = {
                "track_name": track["name"],
                "album_name": track["album"]["name"],
                "album_image": track["album"]["images"][0]["url"] if track["album"]["images"] else None,
                "spotify_url": track["external_urls"]["spotify"],
                "artists": []
            }

            for artist in track["artists"]:
                artist_id = artist["id"]
                if artist_id and artist_id in artist_details:
                    track_data["artists"].append(artist_details[artist_id])

            response["tracks"].append(track_data)

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing playlist: {str(e)}")