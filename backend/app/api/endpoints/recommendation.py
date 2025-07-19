from fastapi import APIRouter, Depends, HTTPException # type: ignore
from sqlalchemy.orm import Session # type: ignore
from pydantic import BaseModel # type: ignore
from typing import List, Optional
from app.models.user import User
from app.database import get_db
from app.services.similar_user_finder import find_similar_users
from app.services.optimal_point_calculator import calculate_optimal_point
from app.services.find_songs_in_region import find_songs_in_region
from app.services.content_based_recommender import get_best_match_songs
from app.services.rl_service import RLRecommendationAgent
from app.services.rL_evaluation import RLEvaluation
import logging

router = APIRouter()

# Pydantic models for input validation
class UserInput(BaseModel):
    age: int
    sex: str
    profession: str
    fav_music_genres: List[str]
    user_current_mood: str

class RecommendationInput(BaseModel):
    user_id: int
    age: int
    sex: str
    profession: str
    fav_music_genres: List[str]
    desired_mood: str
    current_mood: str

class SongRatingInput(BaseModel):
    user_id: int
    song_id: str
    rating: int
    current_mood: str
    previous_rating: int
    next_mood: str
    arousal: float
    valence: float
    danceability: float
    energy: float
    acousticness: float
    instrumentalness: float
    speechiness: float
    liveness: float
    tempo: float
    loudness: float
    context: Optional[str] = None

@router.post("/rate-song")
async def rate_song(input_data: SongRatingInput, db: Session = Depends(get_db)):
    try:
        # 1. Validate user
        try:
            user = db.query(User).filter(User.id == input_data.user_id).first()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
        except Exception as e:
            logging.exception(f"Error validating user: {str(e)}")
            raise HTTPException(status_code=500, detail="Database error while validating user")

        # 2. Validate rating
        try:
            if input_data.rating < 1 or input_data.rating > 5:
                raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
        except Exception as e:
            logging.exception(f"Rating validation error: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid rating value")

        # 3. Initialize RL agent
        try:
            rl_agent = RLRecommendationAgent(db, input_data.user_id)
        except Exception as e:
            logging.exception(f"Error initializing RL agent: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to initialize recommendation agent")

        # 4. Save the rating
        try:
            rl_agent.save_rating(
                song_id=input_data.song_id,
                rating=input_data.rating,
                mood=input_data.current_mood,
                arousal=input_data.arousal,
                valence=input_data.valence,
                danceability=input_data.danceability,
                energy=input_data.energy,
                acousticness=input_data.acousticness,
                instrumentalness=input_data.instrumentalness,
                speechiness=input_data.speechiness,
                liveness=input_data.liveness,
                tempo=input_data.tempo,
                loudness = input_data.loudness,
                context=input_data.context
            )
        except Exception as e:
            logging.exception(f"Error saving song rating: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to save song rating")

        # 5. Train the RL model
        try:
            rl_agent.train(
                song_id=input_data.song_id,
                mood=input_data.current_mood,
                prev_rating=input_data.previous_rating,
                new_rating=input_data.rating,
                next_mood=input_data.next_mood
            )
        except Exception as e:
            logging.exception(f"Error training RL model: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to train RL model")

        # 6. Success response
        return {"message": "Rating saved and RL model trained successfully"}

    except HTTPException as http_exc:
        raise http_exc  # Allow known HTTP errors to bubble up
    except ValueError as ve:
        logging.exception(f"Value error in rate_song: {str(ve)}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(ve)}")
    except Exception as e:
        logging.exception(f"Unexpected error in rate_song: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error while rating song")

@router.post("/recommend-songs")
async def recommend_songs(input_data: RecommendationInput, db: Session = Depends(get_db)):
    try:
        # 1. Validate user
        try:
            user = db.query(User).filter(User.id == input_data.user_id).first()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
        except Exception as e:
            logging.error(f"Error validating user: {str(e)}")
            raise HTTPException(status_code=500, detail="Error validating user")

        # 2. Initialize RL agent
        try:
            rl_agent = RLRecommendationAgent(db, input_data.user_id)
        except Exception as e:
            logging.error(f"Error initializing RL agent: {str(e)}")
            raise HTTPException(status_code=500, detail="Error initializing recommendation agent")

        # 3. Get generalized RL weights
        try:
            rl_weights = rl_agent.get_generalized_weights(mood=input_data.current_mood)
        except Exception as e:
            logging.error(f"Failed to get generalized weights: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to compute generalized weights")

        # 4. Find similar users
        try:
            similar_users_music_prefs = find_similar_users(
                input_age=input_data.age,
                input_sex=input_data.sex,
                input_profession=input_data.profession,
                input_fav_music_genres=input_data.fav_music_genres,
                user_current_mood=input_data.current_mood
            )
            if similar_users_music_prefs is None:
                raise HTTPException(status_code=404, detail="No similar songs found")
        except HTTPException as he:
            raise he
        except Exception as e:
            logging.error(f"Error finding similar users: {str(e)}")
            raise HTTPException(status_code=500, detail="Error finding similar users")

        # 5. Calculate the optimal point
        try:
            optimal_point = calculate_optimal_point(
                similar_users_music_prefs=similar_users_music_prefs,
                current_mood=input_data.current_mood,
                desired_mood_after_listening=input_data.desired_mood,
                rl_weights=rl_weights
            )
            optimal_point_tuple = (optimal_point["valence"], optimal_point["arousal"])
        except Exception as e:
            logging.error(f"Error calculating optimal point: {str(e)}")
            raise HTTPException(status_code=500, detail="Error calculating optimal point")

        # 6. Find songs in region
        try:
            songs_in_region = find_songs_in_region(optimal_point=optimal_point_tuple, radius=0.15)
            if not songs_in_region:
                raise HTTPException(status_code=404, detail="No songs found in optimal region")
        except HTTPException as he:
            raise he
        except Exception as e:
            logging.error(f"Error finding songs in region: {str(e)}")
            raise HTTPException(status_code=500, detail="Error finding songs in optimal region")

        # 7. Get top matching songs
        try:
            selected_top_songs = get_best_match_songs(
                songs_in_region,
                input_data.current_mood,
                input_data.user_id,
                db
            )
        except Exception as e:
            logging.error(f"Error getting best match songs: {str(e)}")
            raise HTTPException(status_code=500, detail="Error selecting top songs")

        # 8. Return result
        return {
            "selected_top_songs": selected_top_songs
        }

    except HTTPException as http_exc:
        raise http_exc  # Already handled HTTP exceptions
    except Exception as e:
        logging.error(f"Unhandled error in recommend_songs: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during song recommendation")

@router.post("/find-similar-users")
async def find_similar_users_route(input_data: UserInput, db: Session = Depends(get_db)):
    try:
        similar_users_music_prefs = find_similar_users(
            input_age=input_data.age,
            input_sex=input_data.sex,
            input_profession=input_data.profession,
            input_fav_music_genres=input_data.fav_music_genres,
            user_current_mood=input_data.user_current_mood
        )
        if similar_users_music_prefs is None:
            raise HTTPException(status_code=400, detail="No similar users music preferences found.")
        
        return {"similar_users_music_prefs": similar_users_music_prefs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@router.post("/evaluate-rl-accuracy")
async def test_rl_accuracy_all(db: Session = Depends(get_db)):
    try:
        result = RLEvaluation.evaluate_all_users(db)
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except Exception as e:
        logging.error(f"Unexpected error in test_rl_accuracy_all: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during RL accuracy test for all users")