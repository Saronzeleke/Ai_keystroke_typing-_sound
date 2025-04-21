
import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import time

SAMPLE_RATE = 44100
DURATION = 0.1  # 0.1 seconds per keystroke
CLASSES = [str(i) for i in range(10)] + [chr(i) for i in range(97, 123)] + ['space', 'enter', 'noise']
TRAINING_DATA_DIR = "./training_data"  
SAMPLES_PER_CLASS = 10  

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

def create_directory_structure():
    
    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
    for class_name in CLASSES:
        class_dir = os.path.join(TRAINING_DATA_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
        print(f"Created directory: {class_dir}")

def record_realtime(class_name, sample_idx):
    # Record a 5-second live audio sample in real-time .
    print(f"\nRecording sample {sample_idx + 1}/{SAMPLES_PER_CLASS} for class '{class_name}'...")
    if class_name == "noise":
        print("Make a noise (e.g., rustle paper, human voice, walking) in 2 seconds...")
    else:
        print(f"Press the '{class_name}' key in 2 seconds...")
    time.sleep(2)  # Allow time to prepare

    recorder = AudioRecorder()
    recorder.start_recording(max_duration=DURATION)
    time.sleep(DURATION)  
    audio = recorder.stop_recording()
     # Check for silent recordings
    if len(audio) == 0 or np.max(np.abs(audio)) < 0.01:  
        print(f"Warning: No audio or too quiet for {class_name}, sample {sample_idx}")
        return None

    # Save to class directory
    class_dir = os.path.join(TRAINING_DATA_DIR, class_name)
    filename = f"sample_{sample_idx}.wav"
    filepath = os.path.join(class_dir, filename)
    sf.write(filepath, audio, SAMPLE_RATE)
    print(f"Saved: {filepath}")
    return filepath

def main():
    # Create training data with real-time live recordings for each class.
 
    print("Place your phone 10-20 cm from the keyboard for clear recordings.")
    create_directory_structure()
    total_samples = 0

    for class_name in CLASSES:
        print(f"\n=== Recording for class '{class_name}' ===")
        for idx in range(SAMPLES_PER_CLASS):
            filepath = None
            attempts = 0
            max_attempts = 3
            while filepath is None and attempts < max_attempts:
                filepath = record_realtime(class_name, idx)
                attempts += 1
                if filepath is None:
                    print(f"Retry {attempts}/{max_attempts} for {class_name}, sample {idx}")
                    time.sleep(1)
            if filepath:
                total_samples += 1
            else:
                print(f"Failed to record {class_name}, sample {idx} after {max_attempts} attempts")
            time.sleep(1)  # Pause between recordings
        print(f"Completed {SAMPLES_PER_CLASS} samples for '{class_name}'")

    print(f"\nTraining data created in {TRAINING_DATA_DIR}")
    print(f"Total samples recorded: {total_samples} across {len(CLASSES)} classes")
    if total_samples < len(CLASSES) * SAMPLES_PER_CLASS:
        print("Warning: Some samples failed to record. Consider retrying or increasing SAMPLES_PER_CLASS.")

if __name__ == "__main__":
    main()
