import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import time
import random

SAMPLE_RATE = 44100   
DURATION = 0.1
TRAINING_DATA_DIR = "./training_data"

def record_sample(class_name, sample_idx):
    print(f"\nRecording sample {sample_idx} for '{class_name}'")
    time.sleep(2)
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    if np.max(np.abs(audio)) < 0.01:
        print("Warning: Too quiet, try again.")
        return False
    filepath = os.path.join(TRAINING_DATA_DIR, class_name, f"sample_{sample_idx}.wav")
    sf.write(filepath, audio, SAMPLE_RATE)
    print(f"Saved: {filepath}")
    return True

def ensure_dirs():
    for label in ['space', 'enter', 'noise']:
        os.makedirs(os.path.join(TRAINING_DATA_DIR, label), exist_ok=True)

def main():
    ensure_dirs()

    # Generate randomized order of sample indices for space and enter
    sample_indices = list(range(20))
    random.shuffle(sample_indices)

    for i in sample_indices:
        record_sample("space", i)
        time.sleep(1)
        record_sample("enter", i)
        time.sleep(1)

    # Noise: only record sample 13
    record_sample("noise", 13)

if __name__ == "__main__":
    main()
    print("Recording Completed.")
