import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Dense # type: ignore
import matplotlib.pyplot as plt
import pickle
import os

# Define output directory
output_dir = "../artifacts/autoencoder"
os.makedirs(output_dir, exist_ok=True)
print("Output directory:", os.path.abspath(output_dir))

# Load the data
data = pd.read_csv('../datasets/user_profile.csv')

# Select relevant columns and create a copy
data = data[['Age', 'Sex', 'Profession', 'Type of music you like to listen?']].copy()

# --- Impute Missing Values ---
# Numerical: Impute missing Age with the mean
age_mean = data['Age'].mean()
data['Age'] = data['Age'].fillna(age_mean)
print(f"Imputed missing Age values with mean: {age_mean}")

# Categorical: Impute missing Sex and Profession with 'Unknown'
data['Sex'] = data['Sex'].fillna('Unknown')
data['Profession'] = data['Profession'].fillna('Unknown')
data['Profession'] = data['Profession'].replace('No', 'Unknown')
print("Imputed missing Sex and Profession values with 'Unknown'")

# Multi-label: Impute missing Type of music with a placeholder
data['Type of music you like to listen?'] = data['Type of music you like to listen?'].fillna('Missing')
data['Type of music you like to listen?'] = data['Type of music you like to listen?'].apply(
    lambda x: x.strip(',').split(', ') if x != 'Missing' else ['Missing']
)
data['Type of music you like to listen?'] = data['Type of music you like to listen?'].apply(
    lambda x: [] if x == [''] or x == ['Missing'] or len(x) == 0 else x
)
print("Imputed missing Type of music values with empty list")

# Reset index
data = data.reset_index(drop=True)

# --- Encode Categorical Variables ---
# OneHotEncoder for 'Sex' and 'Profession'
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
categorical_cols = ['Sex', 'Profession']
encoded_data = encoder.fit_transform(data[categorical_cols])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))

# MultiLabelBinarizer for 'Type of music you like to listen?'
mlb = MultiLabelBinarizer()
music_encoded = pd.DataFrame(
    mlb.fit_transform(data['Type of music you like to listen?']),
    columns=mlb.classes_
)

# Normalize Age
scaler = MinMaxScaler()
data['Age'] = scaler.fit_transform(data[['Age']])

# Combine features
feature_vectors = pd.concat([data[['Age']], encoded_df, music_encoded], axis=1).values
print("Feature vectors shape:", feature_vectors.shape)

# Define the create_pairs function
def create_pairs(feature_vectors, data):
    pairs = []
    labels = []
    n = len(feature_vectors)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append([feature_vectors[i], feature_vectors[j]])
            same_prof = data.iloc[i]['Profession'] == data.iloc[j]['Profession']
            age_diff = abs(data.iloc[i]['Age'] - data.iloc[j]['Age']) < 0.2
            labels.append(1 if same_prof and age_diff else 0)
    return np.array(pairs), np.array(labels)

# Define the autoencoder
input_shape = (feature_vectors.shape[1],)
input_layer = Input(shape=input_shape)
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(16, activation='relu')(encoded)  # Embedding layer
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(input_shape[0], activation='sigmoid')(decoded)
autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
history = autoencoder.fit(
    feature_vectors, feature_vectors,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
    verbose=1
)

# Generate embeddings
user_embeddings = encoder.predict(feature_vectors)
print("User embeddings shape:", user_embeddings.shape)

# Generate pairs from embeddings
pairs, labels = create_pairs(user_embeddings, data)
print("Pairs shape:", pairs.shape, "Labels shape:", labels.shape)

# Split data
pairs_train, pairs_val, labels_train, labels_val = train_test_split(
    pairs, labels, test_size=0.2, random_state=42
)

# Evaluate using Euclidean distance
distances = np.sqrt(np.sum(np.square(pairs_val[:, 0] - pairs_val[:, 1]), axis=1))
threshold = np.percentile(distances, 50)  # Median distance as threshold
val_pred = (distances < threshold).astype(int)
accuracy = accuracy_score(labels_val, val_pred)
precision, recall, f1, _ = precision_recall_fscore_support(labels_val, val_pred, average='binary')
print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Validation Precision: {precision:.4f}")
print(f"Validation Recall: {recall:.4f}")
print(f"Validation F1-Score: {f1:.4f}")

# Save metrics
metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
with open(os.path.join(output_dir, 'metrics.pkl'), 'wb') as f:
    pickle.dump(metrics, f)
print("Saved metrics")

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(output_dir, 'training_history.png'))
plt.close()

# Save outputs
np.save(os.path.join(output_dir, 'user_embeddings.npy'), user_embeddings)
np.save(os.path.join(output_dir, 'pairs.npy'), pairs)
np.save(os.path.join(output_dir, 'labels.npy'), labels)
data.to_csv(os.path.join(output_dir, 'processed_user_data.csv'), index=False)
print("Saved user embeddings and processed data")

# Save the encoder model
encoder.save(os.path.join(output_dir, 'encoder_model.h5'))
print("Saved encoder model in HDF5 (.h5) format")

# Save preprocessing objects
with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
print("Saved scaler")

with open(os.path.join(output_dir, 'encoder.pkl'), 'wb') as f:
    pickle.dump(encoder, f)
print("Saved encoder")

with open(os.path.join(output_dir, 'mlb.pkl'), 'wb') as f:
    pickle.dump(mlb, f)
print("Saved MultiLabelBinarizer")