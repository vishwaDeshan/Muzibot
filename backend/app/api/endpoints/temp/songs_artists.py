from fastapi import APIRouter
import pandas as pd
from dotenv import load_dotenv
import os
from spotipy import Spotify  # type: ignore
from spotipy.oauth2 import SpotifyClientCredentials  # type: ignore
from typing import List

# Load environment variables
load_dotenv()

router = APIRouter()

# Spotify API authentication
client_credentials_manager = SpotifyClientCredentials(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET")
)
sp = Spotify(client_credentials_manager=client_credentials_manager)

# Define input and output paths
base_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(base_dir, '..','..','..', 'datasets')
input_file = os.path.join(datasets_dir, 'temp_spotify_dataset_scaled.csv')
output_file = os.path.join(datasets_dir, 'temp_spotify_dataset_scaled_with_artists.csv')


# Extract track ID from Spotify URL
def extract_track_id(spotify_link: str) -> str:
    try:
        if "open.spotify.com" in spotify_link:
            return spotify_link.split("/track/")[1].split("?")[0]
        elif "spotify:track:" in spotify_link:
            return spotify_link.split("spotify:track:")[1]
        else:
            return None
    except Exception:
        return None


# Batch fetch artist names using Spotify's /tracks endpoint
def get_artist_names_batch(track_ids: List[str]) -> List[str]:
    try:
        response = sp.tracks(track_ids)
        artist_names = []
        for track in response["tracks"]:
            if track and track["artists"]:
                artist_names.append(track["artists"][0]["name"])
            else:
                artist_names.append("Unknown Artist")
        return artist_names
    except Exception:
        return ["Unknown Artist"] * len(track_ids)


@router.get("/process-local-dataset")
def process_local_csv():
    try:
        df = pd.read_csv(input_file)

        if "track_id" not in df.columns:
            return {"error": "Column 'track_id' not found in dataset."}

        df["track_id"] = df["track_id"].apply(extract_track_id)

        all_artist_names = []
        track_ids = df["track_id"].tolist()

        batch_size = 50
        total = len(track_ids)

        for i in range(0, total, batch_size):
            batch = track_ids[i:i + batch_size]
            valid_ids = [tid if tid else None for tid in batch]

            # Replace None with dummy string to maintain alignment
            real_ids = [tid if tid else "0" * 22 for tid in valid_ids]
            artist_names = get_artist_names_batch(real_ids)

            # Correct "dummy" IDs back to "Invalid URL"
            for j, tid in enumerate(valid_ids):
                if tid is None or tid == "0" * 22:
                    artist_names[j] = "Invalid URL"

                # Print each song and artist
                print(f"{i + j + 1}/{total} | Track ID: {tid or 'None'} | Artist: {artist_names[j]}")

            all_artist_names.extend(artist_names)

        df["track_artist"] = all_artist_names

        # Ensure output directory exists
        os.makedirs(datasets_dir, exist_ok=True)
        df.to_csv(output_file, index=False)

        return {"message": "Dataset processed successfully", "output_file": output_file}

    except Exception as e:
        return {"error": str(e)}
