import os
import shutil
import random

source_dir = 'training_data'       #  original dataset
val_dir = 'validation_data'        # New directory for validation data
val_split = 0.2                    # 20% validation split
random.seed(42)                    # For reproducibility

# Create validation_data directory
os.makedirs(val_dir, exist_ok=True)

# Go through each class in training_data
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    
    if not os.path.isdir(class_path):
        continue  # Skip non-directory files

    # List and shuffle files
    files = os.listdir(class_path)
    random.shuffle(files)
    
    # Calculate number of validation samples
    val_count = int(len(files) * val_split)

    # Create class folder inside validation_data
    val_class_path = os.path.join(val_dir, class_name)
    os.makedirs(val_class_path, exist_ok=True)

    # Copy the selected files
    for file_name in files[:val_count]:
        src = os.path.join(class_path, file_name)
        dst = os.path.join(val_class_path, file_name)
        shutil.copy2(src, dst)

    print(f"Copied {val_count} files from '{class_name}' to validation set.")

print("\n✅ Validation data created in 'validation_data' folder without deleting original files.")
