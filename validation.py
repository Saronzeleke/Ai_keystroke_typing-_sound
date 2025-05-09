import os
import shutil
import random

source_dir = 'training_data'      
val_dir = 'validation_data'        
val_split = 0.4                   
random.seed(42)                    

os.makedirs(val_dir, exist_ok=True)

for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    
    if not os.path.isdir(class_path):
        continue  

   
    files = os.listdir(class_path)
    random.shuffle(files)
    
   
    val_count = int(len(files) * val_split)

  
    val_class_path = os.path.join(val_dir, class_name)
    os.makedirs(val_class_path, exist_ok=True)

   
    for file_name in files[:val_count]:
        src = os.path.join(class_path, file_name)
        dst = os.path.join(val_class_path, file_name)
        shutil.copy2(src, dst)

    print(f"Copied {val_count} files from '{class_name}' to validation set.")

