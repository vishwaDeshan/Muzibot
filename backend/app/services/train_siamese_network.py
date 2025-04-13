import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import MultiLabelBinarizer # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Dense, Lambda, Dropout # type: ignore
from tensorflow.keras import backend as K # type: ignore
import matplotlib.pyplot as plt # type: ignore
import pickle
import os

# Define output directory
output_dir = "../artifacts/siamese_network"
os.makedirs(output_dir, exist_ok=True)
print("Output directory:", os.path.abspath(output_dir))

# Load the data
data = pd.read_csv('../datasets/user_profile.csv')

# Select relevant columns and create a copy to avoid SettingWithCopyWarning
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

# Multi-label: Impute missing Type of music you like to listen? with a placeholder
data['Type of music you like to listen?'] = data['Type of music you like to listen?'].fillna('Missing')
data['Type of music you like to listen?'] = data['Type of music you like to listen?'].apply(
    lambda x: x.strip(',').split(', ') if x != 'Missing' else ['Missing']
)
data['Type of music you like to listen?'] = data['Type of music you like to listen?'].apply(
    lambda x: [] if x == [''] or x == ['Missing'] or len(x) == 0 else x
)
print("Imputed missing Type of music you like to listen? values with empty list")

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

# Generate pairs
pairs, labels = create_pairs(feature_vectors, data)
print("Pairs shape:", pairs.shape, "Labels shape:", labels.shape)

# Check label balance
print("Number of positive pairs:", np.sum(labels))
print("Number of negative pairs:", len(labels) - np.sum(labels))

# Split data
pairs_train, pairs_val, labels_train, labels_val = train_test_split(
    pairs, labels, test_size=0.2, random_state=42
)

# Define the base network
input_shape = (feature_vectors.shape[1],)
input_layer = Input(shape=input_shape)  # Correct input shape
x = Dense(128, activation='relu')(input_layer)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
embedding = Dense(16)(x)
base_network = Model(inputs=input_layer, outputs=embedding)

# Define the Siamese Network
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)
processed_a = base_network(input_a)
processed_b = base_network(input_b)

# Calculate Euclidean distance
epsilon = 1e-6
distance = Lambda(lambda tensors: K.sqrt(K.sum(K.square(tensors[0] - tensors[1]), axis=1, keepdims=True) + epsilon))([processed_a, processed_b])
siamese_network = Model(inputs=[input_a, input_b], outputs=distance)

# Define contrastive loss
def contrastive_loss(y_true, y_pred):
    margin = 1.0
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

# Compile the model
siamese_network.compile(optimizer='adam', loss=contrastive_loss)

# Add early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train the model
history = siamese_network.fit(
    [pairs_train[:, 0], pairs_train[:, 1]], labels_train,
    validation_data=([pairs_val[:, 0], pairs_val[:, 1]], labels_val),
    epochs=30,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(output_dir, 'training_history.png'))
plt.close()

# Generate embeddings
user_embeddings = base_network.predict(feature_vectors)
print("User embeddings shape:", user_embeddings.shape)

# Save outputs
np.save(os.path.join(output_dir, 'user_embeddings.npy'), user_embeddings)
data.to_csv(os.path.join(output_dir, 'processed_user_data.csv'), index=False)
print("Saved user embeddings and processed data")

# Save the base network model using HDF5 format
base_network.save(os.path.join(output_dir, 'base_network_model.h5'))
print("Saved base_network model in HDF5 (.h5) format")

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
