import numpy as np # type: ignore
from sqlalchemy.orm import Session # type: ignore
from sqlalchemy import func # type: ignore
from app.models.rl_models import RLQTable, RLWeights, SongRating
from sqlalchemy.exc import SQLAlchemyError  # type: ignore # Added import
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
        self.arousal_bins = np.linspace(0, 1, 5)  # Discretize arousal (0 to 1)
        self.valence_bins = np.linspace(0, 1, 5)  # Discretize valence (0 to 1)
        
        # Learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.similarity_penalty = 0.05  # Penalty for inconsistent ratings

    def _init_q_table(self):
        return np.zeros((len(self.moods), len(self.ratings), len(self.arousal_bins), len(self.valence_bins),
                         self.num_weights, self.num_weights, self.num_weights))

    def _load_q_table_for_song(self, song_id: str):
        q_table = self._init_q_table()
        song = self.db.query(SongRating).filter(SongRating.song_id == song_id).first()
        arousal = song.arousal
        valence = song.valence
        arousal_idx = np.digitize(arousal, self.arousal_bins, right=True) - 1
        valence_idx = np.digitize(valence, self.valence_bins, right=True) - 1
        arousal_idx = max(0, min(arousal_idx, len(self.arousal_bins) - 1))
        valence_idx = max(0, min(valence_idx, len(self.valence_bins) - 1))

        entries = self.db.query(RLQTable).filter(
            RLQTable.user_id == self.user_id,
            RLQTable.song_id == song_id
        ).all()
        
        for entry in entries:
            mood_idx = self.moods.index(entry.mood) if entry.mood in self.moods else 0
            rating_idx = self.ratings.index(entry.prev_rating) if entry.prev_rating in self.ratings else 0
            q_table[mood_idx, rating_idx, arousal_idx, valence_idx,
                    entry.weight_similar_users_music_prefs_idx, 
                    entry.weight_current_user_mood_idx, 
                    entry.weight_desired_mood_after_listening_idx] = entry.q_value
        
        return q_table, arousal_idx, valence_idx

    def _save_q_table_for_song(self, song_id: str, q_table: np.ndarray, arousal_idx: int, valence_idx: int):
        for mood_idx, mood in enumerate(self.moods):
            for rating_idx, rating in enumerate(self.ratings):
                for w_similar_idx in range(self.num_weights):
                    for w_current_idx in range(self.num_weights):
                        for w_desired_idx in range(self.num_weights):
                            q_value = q_table[mood_idx, rating_idx, arousal_idx, valence_idx,
                                              w_similar_idx, w_current_idx, w_desired_idx]
                            entry = self.db.query(RLQTable).filter(
                                RLQTable.user_id == self.user_id,
                                RLQTable.song_id == song_id,
                                RLQTable.mood == mood,
                                RLQTable.prev_rating == rating,
                                RLQTable.arousal == self.arousal_bins[arousal_idx],
                                RLQTable.valence == self.valence_bins[valence_idx],
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
                                    arousal=self.arousal_bins[arousal_idx],
                                    valence=self.valence_bins[valence_idx],
                                    weight_similar_users_music_prefs_idx=w_similar_idx,
                                    weight_current_user_mood_idx=w_current_idx,
                                    weight_desired_mood_after_listening_idx=w_desired_idx,
                                    q_value=q_value
                                )
                                self.db.add(new_entry)
        self.db.commit()

    def _get_state_index(self, mood: str, prev_rating: int, arousal: float, valence: float):
        mood_idx = self.moods.index(mood) if mood in self.moods else 0
        rating_idx = self.ratings.index(prev_rating) if prev_rating in self.ratings else 0
        arousal_idx = np.digitize(arousal, self.arousal_bins, right=True) - 1
        valence_idx = np.digitize(valence, self.valence_bins, right=True) - 1
        arousal_idx = max(0, min(arousal_idx, len(self.arousal_bins) - 1))
        valence_idx = max(0, min(valence_idx, len(self.valence_bins) - 1))
        return mood_idx, rating_idx, arousal_idx, valence_idx

    def _get_reward(self, rating: int, song_id: str, mood: str):
        base_reward = {5: 1.0, 4: 0.5, 3: 0.0, 2: -0.5, 1: -1.0}.get(rating, 0.0)
        
        similar_songs = self.db.query(SongRating).filter(
            SongRating.user_id == self.user_id,
            SongRating.mood_at_rating == mood
        ).all()
        current_song = self.db.query(SongRating).filter(SongRating.song_id == song_id).first()
        if not current_song or current_song.arousal is None or current_song.valence is None:
            return base_reward

        penalty = 0.0
        for song in similar_songs:
            if song.song_id != song_id and song.arousal is not None and song.valence is not None:
                distance = np.sqrt((current_song.arousal - song.arousal)**2 + (current_song.valence - song.valence)**2)
                if distance < 0.3:
                    rating_diff = abs(rating - song.rating)
                    penalty += self.similarity_penalty * rating_diff / (distance + 1e-3)  # Increased constant for stability
        
        return base_reward - penalty

    def _choose_action(self, q_table: np.ndarray, mood_idx: int, rating_idx: int, arousal_idx: int, valence_idx: int):
        self.epsilon = max(0.01, self.epsilon * 0.995)  # Epsilon decay
        if np.random.random() < self.epsilon:
            w_similar_idx = np.random.randint(self.num_weights)
            w_current_idx = np.random.randint(self.num_weights)
            w_desired_idx = np.random.randint(self.num_weights)
        else:
            slice = q_table[mood_idx, rating_idx, arousal_idx, valence_idx, :, :, :]
            w_similar_idx, w_current_idx, w_desired_idx = np.unravel_index(np.argmax(slice), slice.shape)
        return w_similar_idx, w_current_idx, w_desired_idx

    def _update_q_table(self, q_table: np.ndarray, mood_idx: int, rating_idx: int, arousal_idx: int, valence_idx: int,
                        w_similar_idx: int, w_current_idx: int, w_desired_idx: int,
                        reward: float, next_mood_idx: int, next_rating_idx: int, next_arousal_idx: int, next_valence_idx: int):
        current_q = q_table[mood_idx, rating_idx, arousal_idx, valence_idx, w_similar_idx, w_current_idx, w_desired_idx]
        next_max_q = np.max(q_table[next_mood_idx, next_rating_idx, next_arousal_idx, next_valence_idx, :, :, :])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max_q - current_q)
        q_table[mood_idx, rating_idx, arousal_idx, valence_idx, w_similar_idx, w_current_idx, w_desired_idx] = new_q

    def get_optimized_weights(self, song_id: str, mood: str, prev_rating: int):
        q_table, arousal_idx, valence_idx = self._load_q_table_for_song(song_id)
        mood_idx = self.moods.index(mood) if mood in self.moods else 0
        rating_idx = self.ratings.index(prev_rating) if prev_rating in self.ratings else 0
        slice = q_table[mood_idx, rating_idx, arousal_idx, valence_idx, :, :, :]
        w_similar_idx, w_current_idx, w_desired_idx = np.unravel_index(np.argmax(slice), slice.shape)
        return {
            'similar_users_music_prefs': self.weight_values[w_similar_idx],
            'current_user_mood': self.weight_values[w_current_idx],
            'desired_mood_after_listening': self.weight_values[w_desired_idx]
        }

    def get_generalized_weights(self, mood: str, prev_rating: int = None):

        # Check RLWeights for existing weights
        weights_entry = self.db.query(RLWeights).filter(
            RLWeights.user_id == self.user_id,
            RLWeights.mood == mood
        ).first()

        # Check if the user has rated songs for this mood
        rated_songs = self.db.query(SongRating).filter(
            SongRating.user_id == self.user_id,
            SongRating.mood_at_rating == mood
        ).all()

        # If weights exist and no new ratings exist, return stored weights
        if weights_entry and not rated_songs:
            return {
                'similar_users_music_prefs': weights_entry.weight_similar_users_music_prefs,
                'current_user_mood': weights_entry.weight_current_user_mood,
                'desired_mood_after_listening': weights_entry.weight_desired_mood_after_listening
            }

        # If no ratings exist (first time), use default weights
        if not rated_songs:
            weights = {
                'similar_users_music_prefs': 0.5,
                'current_user_mood': 0.3,
                'desired_mood_after_listening': 0.2
            }
            self._save_weights(mood, weights)
            return weights

        # Average weights from rated songs
        total_weights = {
            'similar_users_music_prefs': 0.0,
            'current_user_mood': 0.0,
            'desired_mood_after_listening': 0.0
        }
        
        count = 0
        for song in rated_songs:
            try:
                song_weights = self.get_optimized_weights(song.song_id, mood, prev_rating)
                for key in total_weights:
                    total_weights[key] += song_weights[key]
                count += 1
            except Exception as e:
                logging.warning(f"Error computing weights for song {song.song_id}: {str(e)}")
                continue

        if count == 0:
            # Fallback to default weights if no valid song weights
            weights = {
                'similar_users_music_prefs': 0.5,
                'current_user_mood': 0.3,
                'desired_mood_after_listening': 0.2
            }
        else:
            weights = {key: value / count for key, value in total_weights.items()}
        
        self._save_weights(mood, weights)
        return weights

    def _save_weights(self, mood: str, weights: dict):
        total = sum(weights.values())
        if total == 0:
            total = 1
        normalized_weights = {k: v / total for k, v in weights.items()}
        entry = self.db.query(RLWeights).filter(
            RLWeights.user_id == self.user_id,
            RLWeights.mood == mood
        ).first()

        if entry:
            entry.weight_similar_users_music_prefs = normalized_weights['similar_users_music_prefs']
            entry.weight_current_user_mood = normalized_weights['current_user_mood']
            entry.weight_desired_mood_after_listening = normalized_weights['desired_mood_after_listening']
            entry.updated_at = datetime.utcnow()
        else:
            new_entry = RLWeights(
                user_id=self.user_id,
                mood=mood,
                weight_similar_users_music_prefs=normalized_weights['similar_users_music_prefs'],
                weight_current_user_mood=normalized_weights['current_user_mood'],
                weight_desired_mood_after_listening=normalized_weights['desired_mood_after_listening']
            )
            self.db.add(new_entry)
        self.db.commit()
        logging.info(f"Saved normalized weights for user {self.user_id}, mood {mood}: {normalized_weights}")

    def train(self, song_id: str, mood: str, prev_rating: int, new_rating: int, next_mood: str):
        song = self.db.query(SongRating).filter(SongRating.song_id == song_id).first()
        arousal = song.arousal if song and song.arousal is not None else 0.5
        valence = song.valence if song and song.valence is not None else 0.5
        q_table, arousal_idx, valence_idx = self._load_q_table_for_song(song_id)
        mood_idx, rating_idx, arousal_idx, valence_idx = self._get_state_index(mood, prev_rating, arousal, valence)
        next_mood_idx, next_rating_idx, next_arousal_idx, next_valence_idx = self._get_state_index(next_mood, new_rating, arousal, valence)
        
        reward = self._get_reward(new_rating, song_id, mood)
        w_similar_idx, w_current_idx, w_desired_idx = self._choose_action(q_table, mood_idx, rating_idx, arousal_idx, valence_idx)
        self._update_q_table(q_table, mood_idx, rating_idx, arousal_idx, valence_idx, w_similar_idx, w_current_idx, w_desired_idx,
                             reward, next_mood_idx, next_rating_idx, next_arousal_idx, next_valence_idx)
        self._save_q_table_for_song(song_id, q_table, arousal_idx, valence_idx)

    def save_rating(self, song_id: str, rating: int, mood: str, arousal: float, valence: float, context: str = None):
        # Check for existing rating
        existing_rating = self.db.query(SongRating).filter(
            SongRating.user_id == self.user_id,
            SongRating.song_id == song_id
        ).first()
        if existing_rating:
            existing_rating.rating = rating
            existing_rating.mood_at_rating = mood
            existing_rating.arousal = arousal
            existing_rating.valence = valence
            existing_rating.context = context
            existing_rating.updated_at = datetime.utcnow()
        else:
            new_rating = SongRating(
                user_id=self.user_id,
                song_id=song_id,
                rating=rating,
                mood_at_rating=mood,
                arousal=arousal,
                valence=valence,
                context=context,
                # Explicitly set nullable fields to None
                danceability=None,
                energy=None,
                acousticness=None,
                instrumentalness=None,
                speechiness=None,
                liveness=None,
                tempo=None
            )
            self.db.add(new_rating)
        try:
            self.db.commit()
        except SQLAlchemyError as e:
            self.db.rollback()
            logging.exception(f"Failed to save rating for song {song_id}: {str(e)}")
            raise

    def is_low_rated(self, song_id: str, mood: str):
        rating = self.db.query(SongRating).filter(
            SongRating.user_id == self.user_id,
            SongRating.song_id == song_id,
            SongRating.mood_at_rating == mood
        ).first()
        return rating and rating.rating < 3