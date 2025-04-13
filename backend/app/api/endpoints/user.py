from fastapi import APIRouter, Depends, HTTPException # type: ignore
from sqlalchemy.orm import Session # type: ignore
from pydantic import BaseModel # type: ignore
from typing import List
from app.models.user import User
from app.database import get_db
from app.services.text_generation import text_generator
from app.services.similar_user_finder import find_similar_users

router = APIRouter()

# Pydantic model for input validation
class UserInput(BaseModel):
    age: int
    sex: str
    profession: str
    music: List[str]
    target_mood_prefs: str

@router.post("/recommend")
async def recommend_songs(input_data: dict, db: Session = Depends(get_db)):
    user_id = input_data.get("user_id")
    text_input = input_data.get("text_input")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return {"error": "User not found"}

    # Step 3: Generate a chat-like response
    generated_response = text_generator.generate_response("joy", text_input)

    return {
        "emotion": "joy",
        "generated_response": generated_response,
    }

@router.get("/users")
async def get_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return [{"id": user.id, "username": user.username, "email": user.email} for user in users]

@router.post("/find-similar-users")
async def find_similar_users_route(input_data: UserInput, db: Session = Depends(get_db)):
    try:
        # Directly calling the function like in main.py
        result, mood = find_similar_users(
            input_age=input_data.age,
            input_sex=input_data.sex,
            input_profession=input_data.profession,
            input_music=input_data.music,
            target_mood_prfes=input_data.target_mood_prefs
        )
        return {
            "mood": mood,
            "preferences": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

