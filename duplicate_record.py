import os
import numpy as np
import librosa
import soundfile as sf
import random

# Parameters
SAMPLE_RATE = 44100
DURATION = 0.1
ORIGINAL_SAMPLES_PER_CLASS = 50
TARGET_SAMPLES_PER_CLASS = 200
DATA_DIR = "./training_data"  # Original data folder
AUGMENTED_DATA_DIR = "./augmented_training_data"  # Output folder
CLASSES = [str(i) for i in range(10)] + [chr(i) for i in range(97, 123)] + ["space", "enter", "noise"]

def augment_audio(audio, sr=SAMPLE_RATE):
    """Apply pitch shift, time stretch, and noise injection."""
    augments = []

    # Pitch shifting
    for shift in [-0.3, 0.3]:
        shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=shift)
        augments.append(shifted)

    # Time stretching
    for rate in [0.95, 1.05]:
        stretched = librosa.effects.time_stretch(audio, rate=rate)
        augments.append(stretched)

    # Noise injection
    noise = np.random.normal(0, 0.003, audio.shape)
    noisy = audio + noise
    augments.append(noisy)

    return augments

def pad_or_trim(audio, sr=SAMPLE_RATE):
    target_length = int(DURATION * sr)
    if len(audio) > target_length:
        return audio[:target_length]
    else:
        return np.pad(audio, (0, target_length - len(audio)), mode="constant")

def process_class(label, label_dir):
    output_dir = os.path.join(AUGMENTED_DATA_DIR, label)
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(label_dir) if f.endswith(".wav")]
    current_count = len(files)
    print(f"Processing class '{label}': {current_count} samples found.")

    # Copy original files
    idx = 0
    for file in files:
        src_path = os.path.join(label_dir, file)
        dst_path = os.path.join(output_dir, file)
        audio, _ = librosa.load(src_path, sr=SAMPLE_RATE, mono=True)
        audio = pad_or_trim(audio)
        sf.write(dst_path, audio, SAMPLE_RATE)
        idx += 1

    # Generate augmented samples until we reach 200
    while idx < TARGET_SAMPLES_PER_CLASS:
        src_file = random.choice(files)
        src_path = os.path.join(label_dir, src_file)
        try:
            audio, _ = librosa.load(src_path, sr=SAMPLE_RATE, mono=True)
            audio = pad_or_trim(audio)
            augmented_audios = augment_audio(audio)

            for aug_audio in augmented_audios:
                if idx >= TARGET_SAMPLES_PER_CLASS:
                    break
                dst_path = os.path.join(output_dir, f"aug_{idx}.wav")
                sf.write(dst_path, aug_audio, SAMPLE_RATE)
                idx += 1
        except Exception as e:
            print(f"Error augmenting {src_path}: {e}")

    print(f"Class '{label}' balanced to {idx} samples.")

def main():
    os.makedirs(AUGMENTED_DATA_DIR, exist_ok=True)
    for label in CLASSES:
        label_dir = os.path.join(DATA_DIR, label)
        if not os.path.isdir(label_dir):
            continue
        process_class(label, label_dir)

    print("✅ Dataset augmentation completed.")
    print(f"Augmented dataset saved to: {AUGMENTED_DATA_DIR}")

if __name__ == "__main__":
    main()