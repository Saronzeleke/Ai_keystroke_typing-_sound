import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import time

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
        return
    filepath = os.path.join(TRAINING_DATA_DIR, class_name, f"sample_{sample_idx}.wav")
    sf.write(filepath, audio, SAMPLE_RATE)
    print(f"Saved: {filepath}")

def ensure_dirs():
    for label in ['space', 'enter', 'noise']:
        os.makedirs(os.path.join(TRAINING_DATA_DIR, label), exist_ok=True)

def main():
    ensure_dirs()
    for i in range(20):  # space & enter: full range
        record_sample("space", i)
        time.sleep(1)
        record_sample("enter", i)
        time.sleep(1)

    # noise: only sample 13
    record_sample("noise", 13)

if __name__ == "__main__":
    main()
    print("Recording Complted.")
    
