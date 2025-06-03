from fastapi import APIRouter, Depends, HTTPException # type: ignore
from sqlalchemy.orm import Session # type: ignore
from pydantic import BaseModel # type: ignore
from typing import List, Dict
from app.models.user import User
from app.database import get_db
from app.services.text_generation import text_generator
from app.services.similar_user_finder import find_similar_users
from app.services.optimal_point_calculator import calculate_optimal_point
from app.services.rl_service import RLRecommendationAgent
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
    previous_rating: int

class SongRatingInput(BaseModel):
    user_id: int
    song_id: str
    rating: int
    current_mood: str
    previous_rating: int
    next_mood: str

@router.post("/rate-song")
async def rate_song(input_data: SongRatingInput, db: Session = Depends(get_db)):
    try:
        # Validate user
        user = db.query(User).filter(User.id == input_data.user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Validate rating
        if input_data.rating < 1 or input_data.rating > 5:
            raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")

        # Initialize RL agent
        rl_agent = RLRecommendationAgent(db, input_data.user_id)

        # Save the rating
        rl_agent.save_rating(input_data.song_id, input_data.rating, input_data.current_mood)

        # Train the RL model
        rl_agent.train(
            song_id=input_data.song_id,
            mood=input_data.current_mood,
            prev_rating=input_data.previous_rating,
            new_rating=input_data.rating,
            next_mood=input_data.next_mood
        )

        return {"message": "Rating saved and RL model trained successfully"}
    except Exception as e:
        logging.error(f"Error in rate_song: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@router.post("/recommend-songs")
async def recommend_songs(input_data: RecommendationInput, db: Session = Depends(get_db)):
    try:
        # Validate user
        user = db.query(User).filter(User.id == input_data.user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Initialize RL agent
        rl_agent = RLRecommendationAgent(db, input_data.user_id)

        # Get generalized weights from RL model
        try:
            rl_weights = rl_agent.get_generalized_weights(input_data.current_mood, input_data.previous_rating)
        except Exception as e:
            logging.error(f"Failed to get generalized weights: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to compute generalized weights")

        # Find similar users
        similar_users_music_prefs = find_similar_users(
            input_age=input_data.age,
            input_sex=input_data.sex,
            input_profession=input_data.profession,
            input_fav_music_genres=input_data.fav_music_genres,  
            user_current_mood=input_data.current_mood            
        )
        if similar_users_music_prefs is None:
            raise HTTPException(status_code=400, detail="No similar users music preferences found.")

        # Calculate the optimal point using RL weights
        optimal_point = calculate_optimal_point(
            similar_users_music_prefs=similar_users_music_prefs,
            current_mood=input_data.current_mood,
            desired_mood_after_listening=input_data.desired_mood,
            rl_weights=rl_weights
        )

        # For now, return the optimal point and weights
        return {
            "optimal_point": optimal_point,
            "rl_weights": rl_weights,
            "current_mood": input_data.current_mood
        }
    except Exception as e:
        logging.error(f"Error in recommend_songs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

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