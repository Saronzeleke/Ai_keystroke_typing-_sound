import os

# Define base path
base_path = "training_data"

# Define labels (folder names)
labels = ["a", "b","1", "space"]

# Create base directory and subdirectories
for label in labels:
    folder_path = os.path.join(base_path, label)
    os.makedirs(folder_path, exist_ok=True)
    print(f"Created: {folder_path}")

print("\n✅ Folder structure created! Now you can add your .wav files manually.")
