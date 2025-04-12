import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler # type: ignore
from sklearn.preprocessing import MultiLabelBinarizer # type: ignore
from sklearn.metrics.pairwise import euclidean_distances # type: ignore
import tensorflow as tf # type: ignore
import pickle

def find_similar_users(input_age, input_sex, input_profession, input_music, target_mood_prfes):
    # Step 1: Load the trained model and tools
    with open('../artifacts/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('../artifacts/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)

    with open('../artifacts/mlb.pkl', 'rb') as f:
        mlb = pickle.load(f)

    base_network = tf.keras.models.load_model('../artifacts/base_network_model.h5')

    # Step 2: Load the original users and create their embeddings
    data = pd.read_csv('../datasets/user_profile.csv')
    original_data = data.copy()
    data = data[['Age', 'Sex', 'Profession', 'Type of music you like to listen?']].copy()

    # Fill missing values
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

    # Turn the data into numbers
    data['Age'] = scaler.transform(data[['Age']])
    encoded_data = encoder.transform(data[['Sex', 'Profession']])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['Sex', 'Profession']))
    music_encoded = pd.DataFrame(mlb.transform(data['Type of music you like to listen?']), columns=mlb.classes_)
    feature_vectors = pd.concat([data[['Age']], encoded_df, music_encoded], axis=1).values

    # Create embeddings for the original users
    user_embeddings = base_network.predict(feature_vectors)

    # Step 3: Process the target user
    target_user_df = pd.DataFrame({
        'Age': [input_age],
        'Sex': [input_sex],
        'Profession': [input_profession],
        'Type of music you like to listen?': [input_music]
    })

    target_user_df['Age'] = scaler.transform(target_user_df[['Age']])
    encoded_target = encoder.transform(target_user_df[['Sex', 'Profession']])
    encoded_target_df = pd.DataFrame(encoded_target, columns=encoder.get_feature_names_out(['Sex', 'Profession']))
    music_encoded_target = pd.DataFrame(mlb.transform(target_user_df['Type of music you like to listen?']), columns=mlb.classes_)
    target_feature_vector = pd.concat([target_user_df[['Age']], encoded_target_df, music_encoded_target], axis=1).values

    # Create embedding for the target user
    target_embedding = base_network.predict(target_feature_vector)

    # Step 4: Find similar users using Euclidean distance
    distances = euclidean_distances(target_embedding, user_embeddings)[0]
    K = 5
    similar_indices = np.argsort(distances)[:K]

    # Step 5: Extract mood preferences
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
        'What type of music do you prefer to listen to when you\'re in a happy mood?': 'Happy_Prefs',
        'What type of music do you prefer to listen to when you\'re sad?': 'Sad_Prefs',
        'What type of music do you prefer to listen to when you\'re angry?': 'Angry_Prefs',
        'What type of music do you prefer to listen to when you\'re in a relaxed mood?': 'Relaxed_Prefs'
    })

    mood_columns = ['Happy_Prefs', 'Sad_Prefs', 'Angry_Prefs', 'Relaxed_Prefs']
    for col in mood_columns:
        mood_data[col] = mood_data[col].fillna('').str.strip(',').str.split(', ')
        mood_data[col] = mood_data[col].apply(lambda x: [] if x == [''] else x)

    similar_users_with_moods = mood_data.iloc[similar_indices]

    # Step 6: Extract the target preferences
    mood_preferences = similar_users_with_moods[target_mood_prfes].tolist()
    flattened_mood_preferences = [pref for user_prefs in mood_preferences for pref in user_prefs]

    return flattened_mood_preferences, target_mood_prfes


# Example usage
# if __name__ == "__main__":
#     input_age = 22
#     input_sex = "Male"
#     input_profession = "Undergraduate"
#     input_music = ["Pop", "Classical"]  # should be a list
#     target_mood_prfes = "Sad_Prefs"  # can be Happy_Prefs, Sad_Prefs, Angry_Prefs, Relaxed_Prefs

#     result, mood = find_similar_users(input_age, input_sex, input_profession, input_music, target_mood_prfes)
#     print(f"\nMost common preferences for mood '{mood}':")
#     print(result)