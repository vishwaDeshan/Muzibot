from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.models.user import User
from app.database import get_db
from app.services.text_generation import text_generator

router = APIRouter()

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