import sys
import os
import logging
import matplotlib.pyplot as plt # type: ignore
from typing import Dict, List, Tuple
from sqlalchemy.orm import Session # type: ignore
from sqlalchemy.exc import SQLAlchemyError # type: ignore
from app.models.rl_models import SongRating
from app.services.rl_service import RLRecommendationAgent
from app.database import get_db

# Fix ModuleNotFoundError
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

logging.basicConfig(level=logging.INFO)

class RLEvaluation:

    def __init__(self, agent: RLRecommendationAgent):
        self.agent = agent
        self.db = agent.db

    def prepare_test_data(self, user_id: int) -> List[Tuple[str, str, int, int]]:
        test_data = []
        try:
            ratings = self.db.query(SongRating).filter(SongRating.user_id == user_id).all()
            if not ratings:
                logging.warning(f"No ratings found for user {user_id}")
                return test_data
            for rating in ratings:
                prev_rating = rating.rating if rating.rating is not None else 3
                test_data.append((rating.song_id, rating.mood_at_rating, prev_rating, rating.rating))
        except SQLAlchemyError as e:
            logging.exception(f"Failed to prepare test data for user {user_id}: {str(e)}")
        return test_data

    def evaluate_rating_accuracy(self, test_data: List[Tuple[str, str, int, int]]) -> float:
        correct = 0
        total = 0
        for song_id, mood, prev_rating, actual_rating in test_data:
            if actual_rating == 3:
                continue  # Skip rating 3
            try:
                weights = self.agent.get_optimized_weights(song_id, mood, prev_rating)
                weight_sum = sum(weights.values())
                if weight_sum > 0.9:
                    if actual_rating >= 4:
                        correct += 1
                    total += 1
            except Exception as e:
                logging.warning(f"Error evaluating song {song_id}: {str(e)}")
        accuracy = correct / total if total > 0 else 0.0
        logging.info(f"Rating Accuracy: {accuracy:.4f} ({correct}/{total} correct)")
        return accuracy


    def _visualize_accuracy(self, results: Dict[str, float], chart_title: str = "Rating Prediction Accuracy by Mood",
                           output_path: str = "accuracy_chart.png", primary_color: str = "#0B560A",
                           border_color: str = "#094D0E") -> str:
        if not results:
            logging.error("No results for visualization")
            return ""

        # Extract labels and data
        labels = list(results.keys())
        data = list(results.values())

        # Create bar chart
        plt.figure(figsize=(6, 4))
        bars = plt.bar(labels, data, color=primary_color, edgecolor=border_color, linewidth=1)

        # Add percentage labels above bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,  # Center the text horizontally
                height,  # Position text just above the bar
                f'{height * 100:.1f}%',  # Format as percentage with one decimal
                ha='center', va='bottom', fontsize=8
            )

        # Customize chart
        plt.title(chart_title, fontsize=12, fontweight='bold')
        plt.xlabel("Mood", fontsize=10)
        plt.ylabel("Accuracy", fontsize=10)
        plt.ylim(0, 1)  # Set y-axis range from 0 to 1
        plt.grid(True, axis='y', color='#e0e0e0', linestyle='-', linewidth=0.1)
        plt.grid(False, axis='x')  # Disable x-axis grid

        # Format y-axis as percentages
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x * 100)}%"))

        # Add legend
        plt.legend([bars], ["Rating Accuracy (Rating ≥ 4)"], loc='upper center', fontsize=8)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the chart as PNG
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()

        logging.info(f"Chart saved to {output_path}")
        return output_path

def evaluate_all_users(db: Session = None) -> Dict:

    # Define output directory
    output_dir = "./charts"
    os.makedirs(output_dir, exist_ok=True)

    if db is None:
        db = next(get_db())

    try:
        user_ids = db.query(SongRating.user_id).distinct().all()
        user_ids = [uid[0] for uid in user_ids]
    except Exception as e:
        logging.error(f"Failed to fetch user IDs: {str(e)}")
        return {"error": f"Failed to fetch user IDs: {str(e)}"}

    if not user_ids:
        logging.warning("No users found in the SongRating table.")
        return {"error": "No users found in the SongRating table."}

    try:
        agent = RLRecommendationAgent(db=db, user_id=user_ids[0])
        mood_data = {mood: [] for mood in agent.moods}

        for user_id in user_ids:
            evaluator = RLEvaluation(agent=RLRecommendationAgent(db=db, user_id=user_id))
            test_data = evaluator.prepare_test_data(user_id=user_id)
            for data in test_data:
                mood = data[1]
                if mood in mood_data:
                    mood_data[mood].append(data)

        evaluator = RLEvaluation(agent=agent)
        results = {}
        overall_correct = 0
        overall_total = 0

        for mood in agent.moods:
            if mood_data[mood]:
                accuracy = evaluator.evaluate_rating_accuracy(mood_data[mood])
                results[mood] = accuracy
                correct = int(accuracy * len(mood_data[mood]))
                overall_correct += correct
                overall_total += len(mood_data[mood])
            else:
                results[mood] = 0.0

        results['Overall'] = overall_correct / overall_total if overall_total > 0 else 0.0
        chart_path = evaluator._visualize_accuracy(
            results,
            chart_title="Rating Prediction Accuracy by All Users",
            output_path=os.path.join(output_dir, "accuracy_rl_model.png")
        )

        db.close()

        return {
            "accuracy_by_mood": {mood: f"{acc:.2%}" for mood, acc in results.items()},
        }

    except Exception as e:
        logging.error(f"Unexpected error in evaluate_all_users: {str(e)}")
        db.close()
        return {"error": f"Internal server error: {str(e)}"}