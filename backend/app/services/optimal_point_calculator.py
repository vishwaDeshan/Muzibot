import numpy as np # type: ignore

def calculate_optimal_point(flattened_mood_preferences, target_mood, preferred_mood='calm', reward_score=1.0):
    print("Calculating optimal point...")

    # Step 1: Define the inputs
    current_emotion = 'Happy' if target_mood == 'Happy_Prefs' else 'Sad' if target_mood == 'Sad_Prefs' else 'Angry' if target_mood == 'Angry_Prefs' else 'Relaxed'

    # Step 2: Map moods to valence and arousal values
    mood_to_valence_arousal = {
        'Joyful music': {'valence': 0.8, 'arousal': 0.7},
        'Relaxing music': {'valence': 0.6, 'arousal': 0.3},
        'Sad music': {'valence': 0.2, 'arousal': 0.4},
        'Aggressive music': {'valence': 0.3, 'arousal': 0.9}
    }

    emotion_to_valence_arousal = {
        'Happy': {'valence': 0.9, 'arousal': 0.6},
        'Sad': {'valence': 0.2, 'arousal': 0.3},
        'Angry': {'valence': 0.3, 'arousal': 0.8},
        'Relaxed': {'valence': 0.7, 'arousal': 0.2}
    }

    preferred_mood_to_valence_arousal = {
        'calm': {'valence': 0.6, 'arousal': 0.2},
        'happy': {'valence': 0.9, 'arousal': 0.6},
        'energetic': {'valence': 0.7, 'arousal': 0.8}
    }

    # Step 3: Calculate valence and arousal for each component
    # Component 1: Similar users' mood preferences
    mood_valence_values = []
    mood_arousal_values = []
    for pref in flattened_mood_preferences:
        if pref in mood_to_valence_arousal:
            mood_valence_values.append(mood_to_valence_arousal[pref]['valence'])
            mood_arousal_values.append(mood_to_valence_arousal[pref]['arousal'])
        else:
            print(f"Warning: '{pref}' not found in mood mapping, skipping.")

    if not mood_valence_values:
        print("Error: No valid mood preferences found.")
        return None

    avg_mood_valence = np.mean(mood_valence_values)
    avg_mood_arousal = np.mean(mood_arousal_values)
    print(f"Average mood preferences (valence, arousal): ({avg_mood_valence:.2f}, {avg_mood_arousal:.2f})")

    # Component 2: User's current emotion
    if current_emotion in emotion_to_valence_arousal:
        emotion_valence = emotion_to_valence_arousal[current_emotion]['valence']
        emotion_arousal = emotion_to_valence_arousal[current_emotion]['arousal']
        print(f"Current emotion (valence, arousal): ({emotion_valence:.2f}, {emotion_arousal:.2f})")
    else:
        print(f"Error: Current emotion '{current_emotion}' not found in mapping.")
        return None

    # Component 3: User's preferred mood
    if preferred_mood in preferred_mood_to_valence_arousal:
        preferred_valence = preferred_mood_to_valence_arousal[preferred_mood]['valence']
        preferred_arousal = preferred_mood_to_valence_arousal[preferred_mood]['arousal']
        print(f"Preferred mood (valence, arousal): ({preferred_valence:.2f}, {preferred_arousal:.2f})")
    else:
        print(f"Error: Preferred mood '{preferred_mood}' not found in mapping.")
        return None

    # Step 4: Use the center of gravity method to calculate the optimal point
    weight_mood = 0.5
    weight_emotion = 0.3
    weight_preferred = 0.2

    adjusted_weight_mood = weight_mood * reward_score
    adjusted_weight_emotion = weight_emotion * reward_score
    adjusted_weight_preferred = weight_preferred * reward_score

    total_weight = adjusted_weight_mood + adjusted_weight_emotion + adjusted_weight_preferred
    if total_weight == 0:
        print("Error: Total weight is 0, cannot calculate optimal point.")
        return None

    adjusted_weight_mood /= total_weight
    adjusted_weight_emotion /= total_weight
    adjusted_weight_preferred /= total_weight

    optimal_valence = (adjusted_weight_mood * avg_mood_valence +
                       adjusted_weight_emotion * emotion_valence +
                       adjusted_weight_preferred * preferred_valence)

    optimal_arousal = (adjusted_weight_mood * avg_mood_arousal +
                       adjusted_weight_emotion * emotion_arousal +
                       adjusted_weight_preferred * preferred_arousal)

    print(f"\nOptimal Point (Valence, Arousal): ({optimal_valence:.2f}, {optimal_arousal:.2f})")
    print(f"Optimal Point: {{'valence': {optimal_valence:.2f}, 'arousal': {optimal_arousal:.2f}}}")

    return {'valence': optimal_valence, 'arousal': optimal_arousal}


# Example usage
if __name__ == "__main__":
    # Example input
    flattened_mood_preferences = ['Joyful music', 'Relaxing music', 'Sad music']
    target_mood = 'Sad_Prefs'  # Could be: 'Happy_Prefs', 'Sad_Prefs', 'Angry_Prefs', 'Relaxed_Prefs'
    preferred_mood = 'calm'     # Could be: 'calm', 'happy', 'energetic'
    reward_score = 1.0          # Float value

    result = calculate_optimal_point(flattened_mood_preferences, target_mood, preferred_mood, reward_score)

    if result:
        print("\n✅ Final Optimal Point Result:")
        print(result)
