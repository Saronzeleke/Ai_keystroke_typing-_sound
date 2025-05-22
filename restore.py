import os
import shutil
source_dir = 'validation_data'
dest_dir = 'training_data'
os.makedirs(dest_dir, exist_ok=True)
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    dest_class_path = os.path.join(dest_dir, class_name)
    os.makedirs(dest_class_path, exist_ok=True)
    files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
    for file_name in files:
        src = os.path.join(class_path, file_name)
        dst = os.path.join(dest_class_path, file_name)
        shutil.move(src, dst)
        print(f"Moved {file_name} from validation_data/{class_name} to training_data/{class_name}")

    print(f"Moved {len(files)} files for class '{class_name}'")
