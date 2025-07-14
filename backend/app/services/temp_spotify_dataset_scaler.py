import os
import pandas as pd # type: ignore

# Define output directory
output_dir = "../datasets"
os.makedirs(output_dir, exist_ok=True)
print("Output directory:", os.path.abspath(output_dir))

# Define input and output file paths
input_file = "../datasets/spotify_dataset.csv"
output_file = os.path.join(output_dir, "spotify_dataset_scaled.csv")

# Load the dataset
df = pd.read_csv(input_file)

# Scale energy and valence to [-1, 1]
df['arousal_scaled'] = 2 * df['energy'] - 1
df['valence_scaled'] = 2 * df['valence'] - 1

# Save the modified dataset
df.to_csv(output_file, index=False)
print(f"Modified dataset saved to: {os.path.abspath(output_file)}")