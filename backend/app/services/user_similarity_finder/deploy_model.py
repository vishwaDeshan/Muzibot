import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import euclidean_distances
import tensorflow as tf
import pickle

# Load the preprocessing objects
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("Loaded scaler")

with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
print("Loaded encoder")

with open('mlb.pkl', 'rb') as f:
    mlb = pickle.load(f)
print("Loaded MultiLabelBinarizer")

# Load the base network model
base_network = tf.keras.models.load_model('base_network_model.h5')
print("Loaded base network model")

# Load the user data (to get embeddings for all users)
data = pd.read_csv('../../datasets/user_profile.csv')

# Select relevant columns and create a copy to avoid SettingWithCopyWarning
data = data[['Age', 'Sex', 'Profession', 'Type of music you like to listen?']].copy()

# --- Preprocess the Data (same as training) ---

# Impute Missing Values
age_mean = data['Age'].mean()
data['Age'] = data['Age'].fillna(age_mean)
print(f"Imputed missing Age values with mean: {age_mean}")

data['Sex'] = data['Sex'].fillna('Unknown')
data['Profession'] = data['Profession'].fillna('Unknown')
data['Profession'] = data['Profession'].replace('No', 'Unknown')
print("Imputed missing Sex and Profession values with 'Unknown'")

data['Type of music you like to listen?'] = data['Type of music you like to listen?'].fillna('Missing')
data['Type of music you like to listen?'] = data['Type of music you like to listen?'].apply(
    lambda x: x.strip(',').split(', ') if x != 'Missing' else ['Missing']
)
data['Type of music you like to listen?'] = data['Type of music you like to listen?'].apply(
    lambda x: [] if x == [''] or x == ['Missing'] or len(x) == 0 else x
)
print("Imputed missing Type of music you like to listen? values with empty list")

# Normalize Age
data['Age'] = scaler.transform(data[['Age']])

# Encode Sex and Profession
encoded_data = encoder.transform(data[['Sex', 'Profession']])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['Sex', 'Profession']))

# Encode music genres
music_encoded = pd.DataFrame(
    mlb.transform(data['Type of music you like to listen?']),
    columns=mlb.classes_
)

# Combine features
feature_vectors = pd.concat([data[['Age']], encoded_df, music_encoded], axis=1).values
print("Feature vectors shape:", feature_vectors.shape)

# Generate embeddings for all users
user_embeddings = base_network.predict(feature_vectors)
print("User embeddings shape:", user_embeddings.shape)

# --- Process the Target User ---

# Hardcoded input for the target user
input_age = 26
input_sex = 'Female'
input_profession = 'Undergraduate'  # Will map to 'Student'
input_music = ['Rock', 'Classical', 'Pop']  # 'Classical' will be mapped to 'Classic'

# Create a DataFrame for the target user
target_user_df = pd.DataFrame({
    'Age': [input_age],
    'Sex': [input_sex],
    'Profession': ['Student'],
    'Type of music you like to listen?': [['Rock', 'Classic', 'Pop']]
})

# Preprocess the target user
target_user_df['Age'] = scaler.transform(target_user_df[['Age']])
encoded_target = encoder.transform(target_user_df[['Sex', 'Profession']])
encoded_target_df = pd.DataFrame(encoded_target, columns=encoder.get_feature_names_out(['Sex', 'Profession']))
music_encoded_target = pd.DataFrame(
    mlb.transform(target_user_df['Type of music you like to listen?']),
    columns=mlb.classes_
)
target_feature_vector = pd.concat([target_user_df[['Age']], encoded_target_df, music_encoded_target], axis=1).values

# Generate embedding for the target user
target_embedding = base_network.predict(target_feature_vector)
print("Target user embedding shape:", target_embedding.shape)

# --- Find Similar Users ---

# Compute distances between the target user and all other users
distances = euclidean_distances(target_embedding, user_embeddings)[0]

# Find the indices of the K nearest neighbors
K = 5
similar_indices = np.argsort(distances)[:K]
similar_distances = distances[similar_indices]

# Get the similar users' details
similar_users = data.iloc[similar_indices]
print("Top 5 Similar Users:")
print(similar_users[['Age', 'Sex', 'Profession', 'Type of music you like to listen?']])
print("Distances to similar users:", similar_distances)