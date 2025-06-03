import numpy as np # type: ignore
from sqlalchemy.orm import Session # type: ignore
from sqlalchemy import func # type: ignore
from app.models.rl_models import RLQTable, SongRating
from datetime import datetime
import logging

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

    def _init_q_table(self):
        return np.zeros((len(self.moods), len(self.ratings), self.num_weights, self.num_weights, self.num_weights))

    def _load_q_table_for_song(self, song_id: str):
        q_table = self._init_q_table()
        entries = self.db.query(RLQTable).filter(
            RLQTable.user_id == self.user_id,
            RLQTable.song_id == song_id
        ).all()
        
        for entry in entries:
            mood_idx = self.moods.index(entry.mood) if entry.mood in self.moods else 0
            rating_idx = self.ratings.index(entry.prev_rating) if entry.prev_rating in self.ratings else 0
            q_table[mood_idx, rating_idx, entry.weight_similar_users_music_prefs_idx, 
                    entry.weight_current_user_mood_idx, entry.weight_desired_mood_after_listening_idx] = entry.q_value
        
        return q_table

    def _save_q_table_for_song(self, song_id: str, q_table: np.ndarray):
        for mood_idx, mood in enumerate(self.moods):
            for rating_idx, rating in enumerate(self.ratings):
                for w_similar_idx in range(self.num_weights):
                    for w_current_idx in range(self.num_weights):
                        for w_desired_idx in range(self.num_weights):
                            q_value = q_table[mood_idx, rating_idx, w_similar_idx, w_current_idx, w_desired_idx]
                            entry = self.db.query(RLQTable).filter(
                                RLQTable.user_id == self.user_id,
                                RLQTable.song_id == song_id,
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
                                    song_id=song_id,
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
        reward_map = {5: 1.0, 4: 0.5, 3: 0.0, 2: -0.5, 1: -1.0}
        return reward_map.get(rating, 0.0)

    def _choose_action(self, q_table: np.ndarray, mood_idx: int, rating_idx: int):
        if np.random.random() < self.epsilon:
            w_similar_idx = np.random.randint(self.num_weights)
            w_current_idx = np.random.randint(self.num_weights)
            w_desired_idx = np.random.randint(self.num_weights)
        else:
            slice = q_table[mood_idx, rating_idx, :, :, :]
            w_similar_idx, w_current_idx, w_desired_idx = np.unravel_index(np.argmax(slice), slice.shape)
        return w_similar_idx, w_current_idx, w_desired_idx

    def _update_q_table(self, q_table: np.ndarray, mood_idx: int, rating_idx: int, w_similar_idx: int, w_current_idx: int, w_desired_idx: int,
                        reward: float, next_mood_idx: int, next_rating_idx: int):
        current_q = q_table[mood_idx, rating_idx, w_similar_idx, w_current_idx, w_desired_idx]
        next_max_q = np.max(q_table[next_mood_idx, next_rating_idx, :, :, :])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max_q - current_q)
        q_table[mood_idx, rating_idx, w_similar_idx, w_current_idx, w_desired_idx] = new_q

    def get_optimized_weights(self, song_id: str, mood: str, prev_rating: int):
        q_table = self._load_q_table_for_song(song_id)
        mood_idx, rating_idx = self._get_state_index(mood, prev_rating)
        q_slice = q_table[mood_idx, rating_idx, :, :, :]

        if np.sum(q_slice) == 0:
            return {
                'similar_users_music_prefs': 0.5,
                'current_user_mood': 0.3,
                'desired_mood_after_listening': 0.2
            }

        w_similar_idx, w_current_idx, w_desired_idx = np.unravel_index(np.argmax(q_slice), q_slice.shape)
        return {
            'similar_users_music_prefs': self.weight_values[w_similar_idx],
            'current_user_mood': self.weight_values[w_current_idx],
            'desired_mood_after_listening': self.weight_values[w_desired_idx]
        }

    def get_generalized_weights(self, mood: str, prev_rating: int):
        try:
            # Validate inputs
            if mood not in self.moods:
                logging.warning(f"Invalid mood '{mood}', defaulting to 'Happy'")
                mood = 'Happy'
            if prev_rating not in self.ratings:
                logging.warning(f"Invalid prev_rating '{prev_rating}', defaulting to 3")
                prev_rating = 3

            mood_idx, rating_idx = self._get_state_index(mood, prev_rating)
            q_sum = np.zeros((self.num_weights, self.num_weights, self.num_weights))
            count = 0

            # Query Q-table entries
            entries = self.db.query(RLQTable).filter(
                RLQTable.user_id == self.user_id,
                RLQTable.mood == mood,
                RLQTable.prev_rating == prev_rating
            ).all()

            for entry in entries:
                w_similar_idx = entry.weight_similar_users_music_prefs_idx
                w_current_idx = entry.weight_current_user_mood_idx
                w_desired_idx = entry.weight_desired_mood_after_listening_idx
                # Validate indices
                if (0 <= w_similar_idx < self.num_weights and 
                    0 <= w_current_idx < self.num_weights and 
                    0 <= w_desired_idx < self.num_weights):
                    q_sum[w_similar_idx, w_current_idx, w_desired_idx] += entry.q_value
                    count += 1
                else:
                    logging.warning(f"Invalid indices for song {entry.song_id}: "
                                    f"({w_similar_idx}, {w_current_idx}, {w_desired_idx})")

            if count == 0:
                logging.info(f"No Q-table entries found for user {self.user_id}, mood {mood}, "
                             f"prev_rating {prev_rating}. Using default weights.")
                return {
                    'similar_users_music_prefs': 0.5,
                    'current_user_mood': 0.3,
                    'desired_mood_after_listening': 0.2
                }

            q_avg = q_sum / count
            w_similar_idx, w_current_idx, w_desired_idx = np.unravel_index(np.argmax(q_avg), q_avg.shape)
            weights = {
                'similar_users_music_prefs': self.weight_values[w_similar_idx],
                'current_user_mood': self.weight_values[w_current_idx],
                'desired_mood_after_listening': self.weight_values[w_desired_idx]
            }
            logging.info(f"Computed generalized weights for mood {mood}, prev_rating {prev_rating}: {weights}")
            return weights

        except Exception as e:
            logging.error(f"Error in get_generalized_weights: {str(e)}")
            return {
                'similar_users_music_prefs': 0.5,
                'current_user_mood': 0.3,
                'desired_mood_after_listening': 0.2
            }

    def train(self, song_id: str, mood: str, prev_rating: int, new_rating: int, next_mood: str):
        q_table = self._load_q_table_for_song(song_id)
        mood_idx, rating_idx = self._get_state_index(mood, prev_rating)
        next_mood_idx, next_rating_idx = self._get_state_index(next_mood, new_rating)
        
        reward = self._get_reward(new_rating)
        w_similar_idx, w_current_idx, w_desired_idx = self._choose_action(q_table, mood_idx, rating_idx)
        self._update_q_table(q_table, mood_idx, rating_idx, w_similar_idx, w_current_idx, w_desired_idx, 
                             reward, next_mood_idx, next_rating_idx)
        self._save_q_table_for_song(song_id, q_table)

    def save_rating(self, song_id: str, rating: int, mood: str, context: str = None):
        new_rating = SongRating(
            user_id=self.user_id,
            song_id=song_id,
            rating=rating,
            mood_at_rating=mood,
            context=context
        )
        self.db.add(new_rating)
        self.db.commit()

    def is_low_rated(self, song_id: str, mood: str):
        rating = self.db.query(SongRating).filter(
            SongRating.user_id == self.user_id,
            SongRating.song_id == song_id,
            SongRating.mood_at_rating == mood
        ).first()
        return rating and rating.rating < 3