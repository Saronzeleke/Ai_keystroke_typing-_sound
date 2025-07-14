import os
import shutil

def merge_wav_datasets(dataset1_path, dataset2_path, output_path):
    os.makedirs(output_path, exist_ok=True)

    def copy_wav_files(source_path):
        for filename in os.listdir(source_path):
            if filename.lower().endswith('.wav'):
                src_file = os.path.join(source_path, filename)
                base, ext = os.path.splitext(filename)
                dest_file = os.path.join(output_path, filename)
                counter = 1

                # Avoid overwriting existing files
                while os.path.exists(dest_file):
                    new_filename = f"{base}_{counter}{ext}"
                    dest_file = os.path.join(output_path, new_filename)
                    counter += 1

                shutil.copy2(src_file, dest_file)
                print(f"Copied: {src_file} → {dest_file}")

    print("🔁 Merging dataset 1...")
    copy_wav_files(dataset1_path)

    print("🔁 Merging dataset 2...")
    copy_wav_files(dataset2_path)

    print(f"\n✅ Merged all .wav files into '{output_path}' successfully!")

# Example usage
dataset1 =r'C:\Users\admin\Ai_keystroke_typing-_sound\training_data'         
dataset2 =r'C:\Users\admin\Ai_keystroke_typing-_sound\training_data2'         
output_folder = r'C:\Users\admin\Ai_keystroke_typing-_sound\merged_wav'


merge_wav_datasets(dataset1, dataset2, output_folder)
