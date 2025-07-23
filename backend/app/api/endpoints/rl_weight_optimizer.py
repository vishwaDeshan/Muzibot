from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.rl_models import SongRating
from app.services.rl_service import RLRecommendationAgent
import numpy as np
import itertools

router = APIRouter()

@router.post("/optimize-weights")
def optimize_weights_all_users(db: Session = Depends(get_db)):
    step = 0.1
    weight_combinations = []

    # Generate all valid combinations that sum to 1
    for s, c, d in itertools.product(np.arange(0.1, 1.0, step), repeat=3):
        if abs(s + c + d - 1.0) < 1e-3:
            weight_combinations.append({'similar': round(s, 2), 'current': round(c, 2), 'desired': round(d, 2)})

    users = db.query(User).all()  # <--- Use User class here
    user_results = {}

    for user in users:
        user_id = user.id
        rated_songs = db.query(SongRating).filter(SongRating.user_id == user_id).all()
        if not rated_songs:
            continue

        results = []
        for weights in weight_combinations:
            total_reward = 0
            count = 0

            for song in rated_songs:
                try:
                    agent = RLRecommendationAgent(db, user_id)

                    # Override weights
                    def dummy_weights(*args, **kwargs):
                        return {
                            'similar_users_music_prefs': weights['similar'],
                            'current_user_mood': weights['current'],
                            'desired_mood_after_listening': weights['desired']
                        }
                    agent.get_optimized_weights = dummy_weights

                    reward = agent._get_reward(song.rating, song.song_id, song.mood_at_rating)
                    total_reward += reward
                    count += 1

                except Exception:
                    continue

            avg_reward = total_reward / count if count else -999
            results.append((weights, avg_reward))

        best_weights = max(results, key=lambda x: x[1])[0]
        user_results[user_id] = best_weights

    return {"optimized_weights_per_user": user_results}