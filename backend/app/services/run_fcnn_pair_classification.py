import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Dense, Dropout # type: ignore
import matplotlib.pyplot as plt
import pickle
import os

# Define output directory
output_dir = "../artifacts/fcnn"
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

# Define the create_pairs function (concatenated features)
def create_pairs_concat(feature_vectors, data):
    pairs = []
    labels = []
    n = len(feature_vectors)
    for i in range(n):
        for j in range(i + 1, n):
            concat_features = np.concatenate([feature_vectors[i], feature_vectors[j]])
            pairs.append(concat_features)
            same_prof = data.iloc[i]['Profession'] == data.iloc[j]['Profession']
            age_diff = abs(data.iloc[i]['Age'] - data.iloc[j]['Age']) < 0.2
            labels.append(1 if same_prof and age_diff else 0)
    return np.array(pairs), np.array(labels)

# Generate pairs
pairs, labels = create_pairs_concat(feature_vectors, data)
print("Pairs shape:", pairs.shape, "Labels shape:", labels.shape)

# Check label balance
print("Number of positive pairs:", np.sum(labels))
print("Number of negative pairs:", len(labels) - np.sum(labels))

# Save pairs and labels
np.save(os.path.join(output_dir, 'pairs.npy'), pairs)
np.save(os.path.join(output_dir, 'labels.npy'), labels)
print("Saved pairs and labels arrays")

# Split data
pairs_train, pairs_val, labels_train, labels_val = train_test_split(
    pairs, labels, test_size=0.2, random_state=42
)

# Define the FCNN model
input_shape = (feature_vectors.shape[1] * 2,)
input_layer = Input(shape=input_shape)
x = Dense(128, activation='relu')(input_layer)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)
fcnn_model = Model(inputs=input_layer, outputs=output)

# Compile the model
fcnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Add early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train the model
history = fcnn_model.fit(
    pairs_train, labels_train,
    validation_data=(pairs_val, labels_val),
    epochs=30,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model
val_pred = (fcnn_model.predict(pairs_val) > 0.5).astype(int)
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

# Save the model
fcnn_model.save(os.path.join(output_dir, 'fcnn_model.h5'))
print("Saved FCNN model in HDF5 (.h5) format")

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

# Save processed data
data.to_csv(os.path.join(output_dir, 'processed_user_data.csv'), index=False)
print("Saved processed data")