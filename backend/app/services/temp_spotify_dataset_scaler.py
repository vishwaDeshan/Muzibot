import os
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

# Define output directory
output_dir = "../datasets"
os.makedirs(output_dir, exist_ok=True)
print("Output directory:", os.path.abspath(output_dir))

# Define input and output file paths
input_file = "../datasets/temp_spotify_dataset.csv"
output_file = os.path.join(output_dir, "temp_spotify_dataset_scaled.csv")

# Load the dataset
df = pd.read_csv(input_file)

# Scale energy and valence to [-1, 1]
df['energy_scaled'] = 2 * df['energy'] - 1
df['valence_scaled'] = 2 * df['valence'] - 1

# Save the modified dataset
df.to_csv(output_file, index=False)
print(f"Modified dataset saved to: {os.path.abspath(output_file)}")

# Scale energy and valence to [-1, 1]
df['energy_scaled'] = 2 * df['energy'] - 1
df['valence_scaled'] = 2 * df['valence'] - 1

# Create scatter plot
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='valence_scaled', y='energy_scaled', hue='genre', size=1, sizes=(20, 20), alpha=0.6)

# Customize plot
plt.title("Songs Plotted on Energy-Valence Plane", fontsize=14)
plt.xlabel("Valence (Negative to Positive)", fontsize=12)
plt.ylabel("Energy (Calm to Energetic)", fontsize=12)
plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
plt.legend(title="Genre", bbox_to_anchor=(1.05, 1), loc='upper left')

# Add quadrant labels
plt.text(0.5, 0.5, "Happy", fontsize=12, ha='center', va='center')
plt.text(-0.5, 0.5, "Angry", fontsize=12, ha='center', va='center')
plt.text(-0.5, -0.5, "Sad", fontsize=12, ha='center', va='center')
plt.text(0.5, -0.5, "Relaxed", fontsize=12, ha='center', va='center')

# Adjust layout to prevent legend cutoff
plt.tight_layout()

# Save the plot
output_dir = "../artifacts/plots"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "energy_valence_scatter.png")
plt.savefig(output_file, bbox_inches='tight', dpi=300)
print(f"Plot saved to: {os.path.abspath(output_file)}")

# Show the plot
plt.show()