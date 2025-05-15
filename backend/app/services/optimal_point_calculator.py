import numpy as np # type: ignore

def transform_to_range(valence, arousal):
    transformed_valence = (valence - 0.5) * 2
    transformed_arousal = (arousal - 0.5) * 2
    return transformed_valence, transformed_arousal

def calculate_optimal_point(similar_users_music_prefs, current_mood, desired_mood_after_listening='calm', 
                           rl_weights=None):
    print("Calculating optimal point...")

    # Step 1: Determine the user's current mood based on the mood preferences
    current_user_mood = (
        'Happy' if current_mood == 'Happy_Prefs' else
        'Sad' if current_mood == 'Sad_Prefs' else
        'Angry' if current_mood == 'Angry_Prefs' else
        'Relaxed'
    )

    # Step 2: Map music preferences and moods to valence and arousal values
    music_prefs_to_valence_arousal = {
        'Joyful music': {'valence': 0.8, 'arousal': 0.7},
        'Relaxing music': {'valence': 0.6, 'arousal': 0.3},
        'Sad music': {'valence': 0.2, 'arousal': 0.4},
        'Aggressive music': {'valence': 0.3, 'arousal': 0.9}
    }

    user_current_mood_to_valence_arousal = {
        'Happy': {'valence': 0.9, 'arousal': 0.6},
        'Sad': {'valence': 0.2, 'arousal': 0.3},
        'Angry': {'valence': 0.3, 'arousal': 0.8},
        'Relaxed': {'valence': 0.7, 'arousal': 0.2}
    }

    desired_mood_to_valence_arousal = {
        'calm': {'valence': 0.6, 'arousal': 0.2},
        'happy': {'valence': 0.9, 'arousal': 0.6},
        'energetic': {'valence': 0.7, 'arousal': 0.8}
    }

    # Step 3: Calculate valence and arousal for each component
    # Component 1: Similar users' music preferences based on the current user mood
    similar_users_prefs_valence_values = []
    similar_users_prefs_arousal_values = []
    
    for pref in similar_users_music_prefs:
        if pref in music_prefs_to_valence_arousal:
            valence, arousal = (
                music_prefs_to_valence_arousal[pref]['valence'],
                music_prefs_to_valence_arousal[pref]['arousal']
            )
            transformed_valence, transformed_arousal = transform_to_range(valence, arousal)
            similar_users_prefs_valence_values.append(transformed_valence)
            similar_users_prefs_arousal_values.append(transformed_arousal)
        else:
            print(f"Warning: '{pref}' not found in music preferences mapping, skipping.")

    if not similar_users_prefs_valence_values:
        print("Error: No valid music preferences from similar users found.")
        return None

    avg_similar_users_prefs_valence = np.mean(similar_users_prefs_valence_values)
    avg_similar_users_prefs_arousal = np.mean(similar_users_prefs_arousal_values)
    print(f"Average similar users' music preferences (valence, arousal): "
          f"({avg_similar_users_prefs_valence:.2f}, {avg_similar_users_prefs_arousal:.2f})")

    # Component 2: Current user's mood
    if current_user_mood in user_current_mood_to_valence_arousal:
        valence, arousal = (
            user_current_mood_to_valence_arousal[current_user_mood]['valence'],
            user_current_mood_to_valence_arousal[current_user_mood]['arousal']
        )
        current_mood_valence, current_mood_arousal = transform_to_range(valence, arousal)
        print(f"Current user mood (valence, arousal): "
              f"({current_mood_valence:.2f}, {current_mood_arousal:.2f})")
    else:
        print(f"Error: Current user mood '{current_user_mood}' not found in mapping.")
        return None

    # Component 3: Desired user mood after listening to a song
    if desired_mood_after_listening in desired_mood_to_valence_arousal:
        valence, arousal = (
            desired_mood_to_valence_arousal[desired_mood_after_listening]['valence'],
            desired_mood_to_valence_arousal[desired_mood_after_listening]['arousal']
        )
        desired_mood_valence, desired_mood_arousal = transform_to_range(valence, arousal)
        print(f"Desired user mood after listening (valence, arousal): "
              f"({desired_mood_valence:.2f}, {desired_mood_arousal:.2f})")
    else:
        print(f"Error: Desired user mood after listening '{desired_mood_after_listening}' not found in mapping.")
        return None

    # Step 4: Use the center of gravity method to calculate the optimal point
    # Use RL weights if provided, otherwise use defaults
    default_weights = {
        'similar_users_music_prefs': 0.5,
        'current_user_mood': 0.3,
        'desired_mood_after_listening': 0.2
    }
    weights = rl_weights if rl_weights is not None else default_weights

    # Ensure weights sum to 1
    total_weight = (weights['similar_users_music_prefs'] + 
                    weights['current_user_mood'] + 
                    weights['desired_mood_after_listening'])
    if total_weight == 0:
        print("Error: Total weight is 0, cannot calculate optimal point.")
        return None

    adjusted_weight_similar_users_prefs = weights['similar_users_music_prefs'] / total_weight
    adjusted_weight_current_mood = weights['current_user_mood'] / total_weight
    adjusted_weight_desired_mood = weights['desired_mood_after_listening'] / total_weight

    optimal_valence = (
        adjusted_weight_similar_users_prefs * avg_similar_users_prefs_valence +
        adjusted_weight_current_mood * current_mood_valence +
        adjusted_weight_desired_mood * desired_mood_valence
    )

    optimal_arousal = (
        adjusted_weight_similar_users_prefs * avg_similar_users_prefs_arousal +
        adjusted_weight_current_mood * current_mood_arousal +
        adjusted_weight_desired_mood * desired_mood_arousal
    )

    # Round to 2 decimal points
    optimal_valence = round(optimal_valence, 2)
    optimal_arousal = round(optimal_arousal, 2)

    print(f"\nOptimal Point (Valence, Arousal): ({optimal_valence:.2f}, {optimal_arousal:.2f})")
    print(f"Optimal Point: {{'valence': {optimal_valence:.2f}, 'arousal': {optimal_arousal:.2f}}}")

    return {'valence': optimal_valence, 'arousal': optimal_arousal}

# Example usage
# if __name__ == "__main__":
#     # Example inputs
#     similar_users_music_prefs = ['Relaxing music', 'Sad music', 'Sad music', 'Relaxing music', 'Aggressive music']
#     current_mood = 'Sad_Prefs'
#     desired_mood_after_listening = 'calm'
#     rl_weights = {
#         'similar_users_music_prefs': 0.5,
#         'current_user_mood': 0.3,
#         'desired_mood_after_listening': 0.2
#     }

#     # Run the function
#     result = calculate_optimal_point(
#         similar_users_music_prefs=similar_users_music_prefs,
#         current_mood=current_mood,
#         desired_mood_after_listening=desired_mood_after_listening,
#         rl_weights=rl_weights
#     )