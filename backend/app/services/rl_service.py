import numpy as np
from sqlalchemy.orm import Session
from app.models.rl_models import RLQTable, RLWeights, SongRating
from datetime import datetime

class RLRecommendationAgent:
    def __init__(self, db: Session, user_id: int):
        self.db = db
        self.user_id = user_id
        self.moods = ['Happy', 'Sad', 'Angry', 'Relaxed']
        self.ratings = [1, 2, 3, 4, 5]
        self.weight_values = np.linspace(0.1, 0.5, 5)  # [0.1, 0.2, 0.3, 0.4, 0.5]
        self.num_weights = len(self.weight_values)
        
        # Learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1

        # Load or initialize Q-table from database
        self.q_table = self._load_q_table()

    def _load_q_table(self):
        # Initialize or load Q-table as a numpy array
        q_table = np.zeros((len(self.moods), len(self.ratings), self.num_weights, self.num_weights, self.num_weights))
        entries = self.db.query(RLQTable).filter(RLQTable.user_id == self.user_id).all()
        
        for entry in entries:
            mood_idx = self.moods.index(entry.mood) if entry.mood in self.moods else 0
            rating_idx = self.ratings.index(entry.prev_rating) if entry.prev_rating in self.ratings else 0
            q_table[mood_idx, rating_idx, entry.weight_similar_users_music_prefs_idx, entry.weight_current_user_mood_idx, entry.weight_desired_mood_after_listening_idx] = entry.q_value
        
        return q_table

    def _save_q_table(self):
        # Save the Q-table to the database
        for mood_idx, mood in enumerate(self.moods):
            for rating_idx, rating in enumerate(self.ratings):
                for w_similar_idx in range(self.num_weights):
                    for w_current_idx in range(self.num_weights):
                        for w_desired_idx in range(self.num_weights):
                            q_value = self.q_table[mood_idx, rating_idx, w_similar_idx, w_current_idx, w_desired_idx]
                            # Check if entry exists
                            entry = self.db.query(RLQTable).filter(
                                RLQTable.user_id == self.user_id,
                                RLQTable.mood == mood,
                                RLQTable.prev_rating == rating,
                                RLQTable.weight_similar_users_music_prefs_idx == w_similar_idx,
                                RLQTable.weight_current_user_mood_idx == w_current_idx,
                                RLQTable.weight_desired_mood_after_listening_idx == w_desired_idx
                            ).first()
                            if entry:
                                entry.q_value = q_value
                                entry.updated_at = datetime.utcnow()
                            else:
                                new_entry = RLQTable(
                                    user_id=self.user_id,
                                    mood=mood,
                                    prev_rating=rating,
                                    weight_similar_users_music_prefs_idx=w_similar_idx,
                                    weight_current_user_mood_idx=w_current_idx,
                                    weight_desired_mood_after_listening_idx=w_desired_idx,
                                    q_value=q_value
                                )
                                self.db.add(new_entry)
        self.db.commit()

    def _get_state_index(self, mood: str, prev_rating: int):
        mood_idx = self.moods.index(mood) if mood in self.moods else 0
        rating_idx = self.ratings.index(prev_rating) if prev_rating in self.ratings else 0
        return mood_idx, rating_idx

    def _get_reward(self, rating: int):
        # Map rating to reward
        reward_map = {5: 1.0, 4: 0.5, 3: 0.0, 2: -0.5, 1: -1.0}
        return reward_map.get(rating, 0.0)

    def _choose_action(self, mood_idx: int, rating_idx: int):
        if np.random.random() < self.epsilon:
            # Explore: random action
            w_similar_idx = np.random.randint(self.num_weights)
            w_current_idx = np.random.randint(self.num_weights)
            w_desired_idx = np.random.randint(self.num_weights)
        else:
            # Exploit: best action from Q-table
            slice = self.q_table[mood_idx, rating_idx, :, :, :]
            w_similar_idx, w_current_idx, w_desired_idx = np.unravel_index(np.argmax(slice), slice.shape)
        return w_similar_idx, w_current_idx, w_desired_idx

    def _update_q_table(self, mood_idx: int, rating_idx: int, w_similar_idx: int, w_current_idx: int, w_desired_idx: int,
                        reward: float, next_mood_idx: int, next_rating_idx: int):
        current_q = self.q_table[mood_idx, rating_idx, w_similar_idx, w_current_idx, w_desired_idx]
        next_max_q = np.max(self.q_table[next_mood_idx, next_rating_idx, :, :, :])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max_q - current_q)
        self.q_table[mood_idx, rating_idx, w_similar_idx, w_current_idx, w_desired_idx] = new_q

    def get_optimized_weights(self, mood: str, prev_rating: int):
        mood_idx, rating_idx = self._get_state_index(mood, prev_rating)
        w_similar_idx, w_current_idx, w_desired_idx = self._choose_action(mood_idx, rating_idx)
        
        w_similar_users = self.weight_values[w_similar_idx]
        w_current_mood = self.weight_values[w_current_idx]
        w_desired_mood = self.weight_values[w_desired_idx]
        
        total = w_similar_users + w_current_mood + w_desired_mood
        if total == 0:
            w_similar_users = w_current_mood = w_desired_mood = 1/3
        else:
            w_similar_users /= total
            w_current_mood /= total
            w_desired_mood /= total
        
        weights = {
            'similar_users_music_prefs': w_similar_users,
            'current_user_mood': w_current_mood,
            'desired_mood_after_listening': w_desired_mood
        }
        
        # Check for existing weights for this user_id and mood
        existing_weights = self.db.query(RLWeights).filter(
            RLWeights.user_id == self.user_id,
            RLWeights.mood == mood
        ).first()
        if existing_weights:
            # Update existing record
            existing_weights.weight_similar_users_music_prefs = w_similar_users
            existing_weights.weight_current_user_mood = w_current_mood
            existing_weights.weight_desired_mood_after_listening = w_desired_mood
            existing_weights.updated_at = datetime.utcnow()
        else:
            # Create new record
            new_weights = RLWeights(
                user_id=self.user_id,
                mood=mood,
                weight_similar_users_music_prefs=w_similar_users,
                weight_current_user_mood=w_current_mood,
                weight_desired_mood_after_listening=w_desired_mood
            )
            self.db.add(new_weights)
        
        self.db.commit()
        return weights

    def train(self, mood: str, prev_rating: int, new_rating: int, next_mood: str):
        # Get current and next state indices
        mood_idx, rating_idx = self._get_state_index(mood, prev_rating)
        next_mood_idx, next_rating_idx = self._get_state_index(next_mood, new_rating)
        
        # Get reward
        reward = self._get_reward(new_rating)
        
        # Choose action based on current state
        w_similar_idx, w_current_idx, w_desired_idx = self._choose_action(mood_idx, rating_idx)
        
        # Update Q-table
        self._update_q_table(mood_idx, rating_idx, w_similar_idx, w_current_idx, w_desired_idx, reward, next_mood_idx, next_rating_idx)
        
        # Save updated Q-table to database
        self._save_q_table()

    def save_rating(self, song_id: str, rating: int, mood: str):
        # Save the rating to the database
        new_rating = SongRating(
            user_id=self.user_id,
            song_id=song_id,
            rating=rating,
            mood=mood
        )
        self.db.add(new_rating)
        self.db.commit()