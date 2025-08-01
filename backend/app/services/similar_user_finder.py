import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler # type: ignore
from sklearn.preprocessing import MultiLabelBinarizer # type: ignore
from sklearn.metrics.pairwise import euclidean_distances # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.utils import get_custom_objects # type: ignore
import tensorflow as tf # type: ignore
import pickle
import os
import warnings

def find_similar_users(input_age, input_sex, input_profession, input_fav_music_genres, user_current_mood):
    # Get the directory of the current script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(base_dir, '..', 'artifacts', 'siamese_network')
    datasets_dir = os.path.join(base_dir, '..', 'datasets')

    # Step 1: Load the trained model and preprocessing objects
    with open(os.path.join(artifacts_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    with open(os.path.join(artifacts_dir, 'encoder.pkl'), 'rb') as f:
        encoder = pickle.load(f)

    with open(os.path.join(artifacts_dir, 'mlb.pkl'), 'rb') as f:
        mlb = pickle.load(f)

    model_path = os.path.join(artifacts_dir, 'base_network_model.h5')
    try:
        base_network = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    # Step 2: Load the original users and create their embeddings
    data_path = os.path.join(datasets_dir, 'user_profile.csv')
    print(f"Loading data from: {data_path}")
    data = pd.read_csv(data_path)
    original_data = data.copy()
    data = data[['Age', 'Sex', 'Profession', 'Type of music you like to listen?']].copy()

    # Preprocess user data
    age_mean = data['Age'].mean()
    data['Age'] = data['Age'].fillna(age_mean)
    data['Sex'] = data['Sex'].fillna('Unknown')
    data['Profession'] = data['Profession'].fillna('Unknown')
    data['Profession'] = data['Profession'].replace('No', 'Unknown')
    data['Type of music you like to listen?'] = data['Type of music you like to listen?'].fillna('Missing')
    data['Type of music you like to listen?'] = data['Type of music you like to listen?'].apply(
        lambda x: x.strip(',').split(', ') if x != 'Missing' else ['Missing']
    )
    data['Type of music you like to listen?'] = data['Type of music you like to listen?'].apply(
        lambda x: [] if x == [''] or x == ['Missing'] or len(x) == 0 else x
    )

    # Encode data
    data['Age'] = scaler.transform(data[['Age']])
    encoded_data = encoder.transform(data[['Sex', 'Profession']])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['Sex', 'Profession']))
    music_encoded = pd.DataFrame(mlb.transform(data['Type of music you like to listen?']), columns=mlb.classes_)
    feature_vectors = pd.concat([data[['Age']], encoded_df, music_encoded], axis=1).values

    # Create embeddings for original users
    user_embeddings = base_network.predict(feature_vectors)

    # Step 3: Process the target user
    target_user_df = pd.DataFrame({
        'Age': [input_age],
        'Sex': [input_sex],
        'Profession': [input_profession],
        'Type of music you like to listen?': [input_fav_music_genres]
    })

    target_user_df['Age'] = scaler.transform(target_user_df[['Age']])
    encoded_target = encoder.transform(target_user_df[['Sex', 'Profession']])
    encoded_target_df = pd.DataFrame(encoded_target, columns=encoder.get_feature_names_out(['Sex', 'Profession']))

    # Step 3.1: Normalize and filter music genres
    genre_correction_map = {
        'Classical': 'Classic',
        'EDM': 'Electronic Dance Music (EDM)',
        'Hip Hop': 'Hip-Hop',
        'RnB': 'R&B',
        'Orchestra': 'Orchestral ',
        'Afro': 'Trap music and Afro music',
        'Metallic': 'Metal',
        'Country Music': 'Country ',
        'Jazz Music': 'Jazz'
    }

    corrected_genres = [genre_correction_map.get(g.strip(), g.strip()) for g in input_fav_music_genres]
    known_music = [genre for genre in corrected_genres if genre in mlb.classes_]
    unknown_music = [genre for genre in corrected_genres if genre not in mlb.classes_]

    if unknown_music:
        print(f"Unknown genres ignored: {unknown_music}")

    if not known_music:
        raise ValueError("No valid music genres found. Please check input genres.")

    music_encoded_target = pd.DataFrame(mlb.transform([known_music]), columns=mlb.classes_)

    target_feature_vector = pd.concat([target_user_df[['Age']], encoded_target_df, music_encoded_target], axis=1).values

    # Step 4: Create embedding for the target user
    target_embedding = base_network.predict(target_feature_vector)

    # Step 5: Find similar users using Euclidean distance
    distances = euclidean_distances(target_embedding, user_embeddings)[0]
    K = 5
    similar_indices = np.argsort(distances)[:K]

    # Step 6: Extract mood preferences
    mood_data = original_data[[
        'Id', 'Name', 'Age', 'Sex', 'Profession',
        'Type of music you like to listen?',
        'What type of music do you prefer to listen to when you\'re in a happy mood?',
        'What type of music do you prefer to listen to when you\'re sad?',
        'What type of music do you prefer to listen to when you\'re angry?',
        'What type of music do you prefer to listen to when you\'re in a relaxed mood?'
    ]]

    mood_data = mood_data.rename(columns={
        'Type of music you like to listen?': 'Genres',
        'What type of music do you prefer to listen to when you\'re in a happy mood?': 'Happy',
        'What type of music do you prefer to listen to when you\'re sad?': 'Sad',
        'What type of music do you prefer to listen to when you\'re angry?': 'Angry',
        'What type of music do you prefer to listen to when you\'re in a relaxed mood?': 'Relaxed'
    })

    mood_columns = ['Happy', 'Sad', 'Angry', 'Relaxed']
    for col in mood_columns:
        mood_data[col] = mood_data[col].fillna('').str.strip(',').str.split(', ')
        mood_data[col] = mood_data[col].apply(lambda x: [] if x == [''] else x)

    similar_users_with_moods = mood_data.iloc[similar_indices]

    # Step 7: Extract the target preferences
    mood_preferences = similar_users_with_moods[user_current_mood].tolist()
    flattened_mood_preferences = [pref for user_prefs in mood_preferences for pref in user_prefs]
    print("Flattened mood preferences:", flattened_mood_preferences)

    return flattened_mood_preferences


# Example usage
# if __name__ == "__main__":
#     input_age = 22
#     input_sex = "Male"
#     input_profession = "Undergraduate"
#     input_fav_music_genres = ["Pop", "Classical"]
#     user_current_mood = "Sad"

#     result = find_similar_users(input_age, input_sex, input_profession, input_fav_music_genres, user_current_mood)
#     print(result)