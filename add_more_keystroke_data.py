import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import time


SAMPLE_RATE = 44100
DURATION = 0.1 
TARGET_SAMPLES_PER_CLASS = 50  # total desired per class
TRAINING_DATA_DIR = "./training_data"
CLASSES = [str(i) for i in range(10)] + [chr(i) for i in range(97, 123)] + ['space', 'enter', 'noise']
class AudioRecorder:
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.recording = []
        self.is_recording = False
        self.stream = None

    def start_recording(self, max_duration=10):
        self.recording = []
        self.is_recording = True
        start_time = time.time()

        def callback(indata, frames, time, status):
            if status:
                print(f"Audio status: {status}")
            if self.is_recording:
                self.recording.append(indata.copy())

        self.stream = sd.InputStream(
            callback=callback, channels=1, samplerate=self.sample_rate, blocksize=1024
        )
        self.stream.start()
        print("Recording started...")
        while self.is_recording and (time.time() - start_time < max_duration):
            sd.sleep(100)

    def stop_recording(self):
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        return np.concatenate(self.recording).flatten() if self.recording else np.array([])


# === Directory Setup ===
def create_directory_structure():
    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
    for class_name in CLASSES:
        class_dir = os.path.join(TRAINING_DATA_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
        print(f"Directory ensured: {class_dir}")


# === Real-Time Recording ===
def record_realtime(class_name, sample_idx):
    print(f"\nRecording sample {sample_idx + 1} for class '{class_name}'...")
    if class_name == "noise":
        print("Make some background noise (talk, rustle, etc.) in 2 seconds...")
    else:
        print(f"Press the '{class_name}' key in 2 seconds...")
    time.sleep(3)

    recorder = AudioRecorder()
    recorder.start_recording(max_duration=DURATION)
    time.sleep(DURATION)
    audio = recorder.stop_recording()

    # Skip if silent
    if len(audio) == 0 or np.max(np.abs(audio)) < 0.01:
        print(f"⚠️  Too quiet or empty for {class_name}, sample {sample_idx}")
        return None

    # Save to class directory
    class_dir = os.path.join(TRAINING_DATA_DIR, class_name)
    filename = f"sample_{sample_idx}.wav"
    filepath = os.path.join(class_dir, filename)
    sf.write(filepath, audio, SAMPLE_RATE)
    print(f"✅ Saved: {filepath}")
    return filepath


# === Main Recording Loop ===
def main():
    print("📣 Make sure your mic is working and close to the keyboard.")
    create_directory_structure()
    total_new_samples = 0

    for class_name in CLASSES:
        print(f"\n=== Processing class '{class_name}' ===")
        class_dir = os.path.join(TRAINING_DATA_DIR, class_name)
        existing_files = [f for f in os.listdir(class_dir) if f.endswith(".wav")]
        existing_count = len(existing_files)
        remaining = TARGET_SAMPLES_PER_CLASS - existing_count

        print(f"🗂️  Found {existing_count} samples. Need {remaining} more...")

        for idx in range(existing_count, existing_count + remaining):
            filepath = None
            attempts = 0
            max_attempts = 3
            while filepath is None and attempts < max_attempts:
                filepath = record_realtime(class_name, idx)
                attempts += 1
                if filepath is None:
                    print(f"Retry {attempts}/{max_attempts} for '{class_name}', sample {idx}")
                    time.sleep(3)
            if filepath:
                total_new_samples += 1
            else:
                print(f"❌ Failed to record '{class_name}', sample {idx}")
            time.sleep(3)

        print(f"✅ Finished {TARGET_SAMPLES_PER_CLASS} total samples for '{class_name}'")

    print("\n🎉 Dataset Update Complete!")
    print(f"Total new samples recorded: {total_new_samples}")
    print(f"Check your data in: {TRAINING_DATA_DIR}")


if __name__ == "__main__":
    main()
