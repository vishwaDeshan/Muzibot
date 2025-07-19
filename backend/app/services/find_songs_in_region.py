import pandas as pd # type: ignore
import numpy as np # type: ignore
from typing import Tuple, List, Dict
import os
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from matplotlib.patches import Circle # type: ignore

def find_songs_in_region(optimal_point: Tuple[float, float], radius: float = 0.15) -> List[Dict]:
    # Define input and output paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(base_dir, '..', 'datasets')
    input_file = os.path.join(datasets_dir, 'spotify_full_with_predictions.csv')

    # Define output path
    output_dir = os.path.join(base_dir, '..', '..', 'plots')
    output_file = os.path.join(output_dir, "arousal_valence_song_region_plot.png")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    print(f"Loading data from: {input_file}")
    df = pd.read_csv(input_file, low_memory=False)

    # Convert arousal_final and valence_final to numeric (handle non-numeric values)
    df['valence_final'] = pd.to_numeric(df['valence_final'], errors='coerce')
    df['arousal_final'] = pd.to_numeric(df['arousal_final'], errors='coerce')

    # Extract optimal point coordinates
    opt_valence, opt_energy = optimal_point

    # Calculate Euclidean distance from optimal point
    df['distance'] = np.sqrt(
        (df['valence_final'] - opt_valence) ** 2 +
        (df['arousal_final'] - opt_energy) ** 2
    )

    # Filter out non-finite distances
    df = df[np.isfinite(df['distance'])]
    print(f"Dropped {len(df[df['distance'].isna()])} rows with non-finite distance values.")

    # Filter songs within the radius
    songs_in_region = df[df['distance'] <= radius][[
        'track_name', 'playlist_genre', 'arousal_final', 'valence_final', 'distance', 'track_id', 
        'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 
        'liveness', 'valence', 'tempo'
    ]]

    # Convert to list of dicts sorted by distance
    result = songs_in_region.to_dict(orient='records')
    result = sorted(result, key=lambda x: x['distance'] if pd.notna(x['distance']) else float('inf'))

    # Ensure all values in result are JSON-compliant
    for record in result:
        for key in ['arousal_final', 'valence_final', 'distance']:
            if not np.isfinite(record[key]):
                record[key] = None  # Replace non-finite values with None
        # Ensure track_name and playlist_genre are strings (handle potential None or non-string types)
        record['track_name'] = str(record['track_name']) if pd.notna(record['track_name']) else "Unknown"
        record['playlist_genre'] = str(record['playlist_genre']) if pd.notna(record['playlist_genre']) else "Unknown"
        record ['track_id'] = str(record['track_id'] if pd.notna(record['track_id']) else "Unknown")

    # **** Create scatter plot ****

    plt.figure(figsize=(10, 8))

    # All songs in green
    sns.scatterplot(data=df, x='valence_final', y='arousal_final',
                    color='green', s=20, alpha=0.6, label='All Songs')

    # Recommended songs in red
    sns.scatterplot(data=songs_in_region, x='valence_final', y='arousal_final',
                    color='red', s=30, alpha=0.8, label='Recommended Songs')

    # Plot optimal point
    plt.scatter(opt_valence, opt_energy, color='blue', s=100, marker='+',
                label='Optimal Point')

    # Draw circular recommendation region
    circle = Circle(optimal_point, radius, color='blue', fill=False, linestyle='--',
                    linewidth=2, label='Recommendation Region')
    plt.gca().add_patch(circle)

    # Customize plot
    plt.title("Songs in Energy-Valence Plane with Recommendation Region", fontsize=14)
    plt.xlabel("Valence (Negative to Positive)", fontsize=12)
    plt.ylabel("Energy (Calm to Energetic)", fontsize=12)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)

    # Add quadrant labels
    plt.text(0.5, 0.5, "Happy", fontsize=12, ha='center', va='center')
    plt.text(-0.5, 0.5, "Angry", fontsize=12, ha='center', va='center')
    plt.text(-0.5, -0.5, "Sad", fontsize=12, ha='center', va='center')
    plt.text(0.5, -0.5, "Relaxed", fontsize=12, ha='center', va='center')

    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save plot
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"Plot saved to: {os.path.abspath(output_file)}")

    plt.close()

    return result

# Example usage
# if __name__ == "__main__":
#     sample_optimal_point = (0.5, 0.5)
#     radius = 0.15
    
#     try:
#         # Call the function
#         recommended_songs = find_songs_in_region(sample_optimal_point, radius)
        
#         # Print results
#         print(f"\nRecommended songs within radius {radius} of optimal point {sample_optimal_point}:")
#         if recommended_songs:
#             for song in recommended_songs:
#                 print(f"Song: {song['track_name']}, playlist_genre: {song['playlist_genre']}, "
#                       f"Valence: {song['valence_final']:.3f}, Energy: {song['arousal_final']:.3f}, "
#                       f"Distance: {song['distance']:.3f}")
#         else:
#             print("No songs found within the specified radius.")
#     except Exception as e:
#         print(f"Error: {str(e)}")