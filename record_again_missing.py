import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import time

SAMPLE_RATE = 44100
DURATION = 0.1  
TRAINING_DATA_DIR = "./training_data"


MISSING_SAMPLES = {
    'space': [1, 3, 6, 15],
    'enter': [3, 5, 6, 7, 11],
    'noise': [13]
}
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

def record_missing_samples():
    print("Recording missing samples only...")
    print("Place your phone near the keyboard for clear recordings.\n")

    for class_name, sample_indices in MISSING_SAMPLES.items():
        class_dir = os.path.join(TRAINING_DATA_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)

        print(f"\n=== Recording {class_name.upper()} samples ===")
        for sample_idx in sample_indices:
            print(f"\nPreparing to record {class_name} sample #{sample_idx}...")
            if class_name == "noise":
                print("Make background noise (e.g., keyboard rustling, breath) in 2 seconds...")
            else:
                print(f"Press the '{class_name}' key in 2 seconds...")
            
            time.sleep(3)  

            recorder = AudioRecorder()
            recorder.start_recording(max_duration=DURATION)
            time.sleep(DURATION)
            audio = recorder.stop_recording()

            # Check for silent recordings
            if len(audio) == 0 or np.max(np.abs(audio)) < 0.01:
                print(f"Warning: No audio detected for {class_name} sample #{sample_idx}. Skipping...")
                continue

            # Save the file
            filename = f"sample_{sample_idx}.wav"
            filepath = os.path.join(class_dir, filename)
            sf.write(filepath, audio, SAMPLE_RATE)
            print(f"Saved: {filepath}")
            
            time.sleep(3)  # Pause between recordings

    print("\n=== Missing samples recording complete ===")

if __name__ == "__main__":
    record_missing_samples()   