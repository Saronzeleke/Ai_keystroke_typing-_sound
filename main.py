import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse 
from contextlib import asynccontextmanager
import sounddevice as sd
import soundfile as sf
from scipy import signal 
import noisereduce as nr
import tempfile
from fastapi.middleware.cors import CORSMiddleware
from sklearn.model_selection import train_test_split
import asyncio
import uvicorn
import time
from threading import Lock
import traceback
import os


SAMPLE_RATE = 44100
DURATION = 0.1
N_MELS = 128
FFT_WINDOW = 1024
HOP_LENGTH = 512
CLASSES = [str(i) for i in range(10)] + [chr(i) for i in range(97, 123)] + ['space', 'enter', 'noise']
NUM_CLASSES = len(CLASSES)
EXPECTED_INPUT_SHAPE = (N_MELS, 87, 1)
TRAINING_DATA_DIR = r"C:\Users\USER\Desktop\Ai_keystroke_typing-_sound-1\training_data"
MODEL_PATH = r"C:\Users\USER\Desktop\Ai_keystroke_typing-_sound-1\keystroke_model.h5"
model = None
model_lock = Lock()

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        with model_lock:
            if os.path.exists(MODEL_PATH):
                print(f"Loading pre-trained model from {MODEL_PATH}...")
                model = KeystrokeCNN(input_shape=EXPECTED_INPUT_SHAPE)
                model.load_model(MODEL_PATH)
            else:
                print(f"No pre-trained model found at {MODEL_PATH}. Initializing untrained model...")
                model = KeystrokeCNN(input_shape=EXPECTED_INPUT_SHAPE)
        yield
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        raise
    finally:
        print("Shutting down...")

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

    def save_recording(self, filename):
        if self.recording:
            sf.write(filename, np.concatenate(self.recording).flatten(), self.sample_rate)

class AudioPreprocessor:
    @staticmethod
    def load_audio(file_path, sr=SAMPLE_RATE):
        try:
            audio, _ = librosa.load(file_path, sr=sr, mono=True)
            print(f"Loaded audio from {file_path}, length: {len(audio)} samples")
            return audio
        except Exception as e:
            print(f"Error loading audio {file_path}: {e}")
            raise

    @staticmethod
    def reduce_noise(audio, sr=SAMPLE_RATE):
        try:
            noise_samples = []
            segment_length = int(0.1 * sr)
            for i in range(0, len(audio), segment_length):
                segment = audio[i:i + segment_length]
                if np.std(segment) < 0.01:
                    noise_samples.append(segment)
            if not noise_samples:
                print("No noise samples found, returning original audio")
                return audio
            noise_profile = np.concatenate(noise_samples)
            return nr.reduce_noise(y=audio, y_noise=noise_profile, sr=sr)
        except Exception as e:
            print(f"Error reducing noise: {e}")
            return audio

    @staticmethod
    def detect_keystrokes(audio, threshold=0.05, min_silence=0.05, sr=SAMPLE_RATE):
        try:
            audio = audio / np.max(np.abs(audio))
            peaks, _ = signal.find_peaks(np.abs(audio), height=threshold, distance=int(min_silence * sr))
            print(f"Detected {len(peaks)} peaks")
            keystrokes = []
            for peak in peaks:
                start = max(0, peak - int(0.03 * sr))
                end = min(len(audio), peak + int(0.07 * sr))
                segment = audio[start:end]
                if len(segment) > 0:
                    target_length = int(DURATION * sr)
                    if len(segment) < target_length:
                        segment = np.pad(segment, (0, target_length - len(segment)), mode='constant')
                    elif len(segment) > target_length:
                        segment = segment[:target_length]
                    keystrokes.append(segment)
            return keystrokes
        except Exception as e:
            print(f"Error detecting keystrokes: {e}")
            return []

    @staticmethod
    def extract_keystroke(audio, sr=SAMPLE_RATE, threshold=0.05, min_silence=0.05):
        try:
            audio = audio / np.max(np.abs(audio))
            peaks, _ = signal.find_peaks(np.abs(audio), height=threshold, distance=int(min_silence * sr))
            if not peaks.size:
                print("No keystrokes detected in audio")
                return None
            peak = peaks[0]
            start = max(0, peak - int(0.03 * sr))
            end = min(len(audio), peak + int(0.07 * sr))
            segment = audio[start:end]
            target_length = int(DURATION * sr)
            if len(segment) < target_length:
                segment = np.pad(segment, (0, target_length - len(segment)), mode='constant')
            elif len(segment) > target_length:
                segment = segment[:target_length]
            print(f"Extracted keystroke, length: {len(segment)} samples")
            return segment
        except Exception as e:
            print(f"Error extracting keystroke: {e}")
            return None

    @staticmethod
    def create_mel_spectrogram(audio, sr=SAMPLE_RATE, n_mels=N_MELS):
        try:
            target_length = int(DURATION * sr)
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            elif len(audio) > target_length:
                audio = audio[:target_length]
            S = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=n_mels, n_fft=FFT_WINDOW, hop_length=HOP_LENGTH
            )
            S_dB = librosa.power_to_db(S, ref=np.max)
            target_time_steps = 87
            if S_dB.shape[1] < target_time_steps:
                S_dB = np.pad(S_dB, ((0, 0), (0, target_time_steps - S_dB.shape[1])), mode='constant')
            elif S_dB.shape[1] > target_time_steps:
                S_dB = S_dB[:, :target_time_steps]
            print(f"Spectrogram shape: {S_dB.shape}")
            return np.expand_dims(S_dB, axis=-1)
        except Exception as e:
            print(f"Error creating spectrogram: {e}")
            return None

    @staticmethod 
    def plot_mel_spectrogram(audio, title="Mel Spectrogram", sr=SAMPLE_RATE, n_mels=N_MELS):
        
        try:
            target_length = int(DURATION * sr)
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            elif len(audio) > target_length:
                audio = audio[:target_length]
            S = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=n_mels, n_fft=FFT_WINDOW, hop_length=HOP_LENGTH, fmax=8000
            )
            S_dB = librosa.power_to_db(S, ref=np.max)
            target_time_steps = 87
            if S_dB.shape[1] < target_time_steps:
                S_dB = np.pad(S_dB, ((0, 0), (0, target_time_steps - S_dB.shape[1])), mode='constant')
            elif S_dB.shape[1] > target_time_steps:
                S_dB = S_dB[:, :target_time_steps]

            plt.figure(figsize=(6, 4))
            librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
            plt.colorbar(format='%+2.0f dB')
            plt.title(title)
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(2)
            plt.close()
        except Exception as e:
            print(f"Error plotting spectrogram: {e}")

class KeystrokeCNN:
    def __init__(self, input_shape=EXPECTED_INPUT_SHAPE, num_classes=NUM_CLASSES):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=70, batch_size=32):
        history = self.model.fit(
            X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size
        )
        return history

    def save_model(self, filepath):
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        self.model = models.load_model(filepath)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.predict(np.zeros((1, *self.input_shape)))   

    def predict(self, spectrogram):
        try:
            spectrogram = np.expand_dims(spectrogram, axis=0)
            print(f"Predicting on spectrogram shape: {spectrogram.shape}")
            predictions = self.model.predict(spectrogram, verbose=0)
            return np.argmax(predictions[0]), predictions[0]
        except Exception as e:
            print(f"Error in predict: {e}")
            raise

@app.post("/record/")
async def record_keystrokes(duration: float = Query(5.0, gt=0)):
    try:
        recorder = AudioRecorder()
        recorder.start_recording(max_duration=duration)
        await asyncio.sleep(duration)
        audio = recorder.stop_recording()
        if len(audio) == 0:
            return JSONResponse(status_code=400, content={"error": "No audio recorded"})
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, SAMPLE_RATE)
            return JSONResponse(content={"filename": tmp.name})
    except Exception as e:
        print(f"Error in /record/: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/process/")
async def process_audio(file: UploadFile = File(...)):
    with model_lock:
        if model is None:
            return JSONResponse(status_code=500, content={"error": "Model not loaded"})
    try:
        print(f"Received file: {file.filename}")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
            print(f"Saved temp file: {tmp_path}")
        try:
            audio = AudioPreprocessor.load_audio(tmp_path)
            audio = AudioPreprocessor.reduce_noise(audio)
            keystrokes = AudioPreprocessor.detect_keystrokes(audio)
            if not keystrokes:
                print("No keystrokes detected")
                return {
                    "summary": {
                        "characters": 0,
                        "spaces": 0,
                        "enters": 0,
                        "numbers": 0
                    }
                }
            results = []
            for i, ks in enumerate(keystrokes):
                print(f"Processing keystroke {i}, length: {len(ks)} samples")
                spectrogram = AudioPreprocessor.create_mel_spectrogram(ks)
             
                AudioPreprocessor.plot_mel_spectrogram(ks, title=f"Keystroke {i}: {file.filename}")
                pred_idx, confidences = model.predict(spectrogram)
                prediction = CLASSES[pred_idx]
                confidence = float(confidences[pred_idx])
                results.append({
                    "keystroke": i,
                    "prediction": prediction,
                    "confidence": confidence,
                    "is_low_confidence": bool(confidence < 0.5)
                })
            char_count = len([r for r in results if r['prediction'] not in ['noise', 'space', 'enter']])
            space_count = len([r for r in results if r['prediction'] == 'space'])
            enter_count = len([r for r in results if r['prediction'] == 'enter'])
            number_count = len([r for r in results if r['prediction'] in [str(i) for i in range(10)]])
            print(f"Results: {len(results)} keystrokes processed, {char_count} characters")
            if char_count == 0 and space_count == 0 and enter_count == 0 and number_count == 0:
                return {
                    "summary": {
                        "characters": 0,
                        "spaces": 0,
                        "enters": 0,
                        "numbers": 0
                    }
                }
            return {
                "keystrokes": results,
                "summary": {
                    "characters": char_count,
                    "spaces": space_count,
                    "enters": enter_count,
                    "numbers": number_count
                }
            }
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            print(traceback.format_exc())
            return JSONResponse(status_code=500, content={"error": f"Processing failed: {str(e)}"})
        finally:
            try:
                os.unlink(tmp_path)
                print(f"Deleted temp file: {tmp_path}")
            except Exception as e:
                print(f"Failed to delete temp file {tmp_path}: {e}")
    except Exception as e:
        print(f"Error handling file upload: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": f"File upload failed: {str(e)}"})

@app.post("/stream/")
async def stream_audio(duration: float = Query(30.0, gt=0)):
    with model_lock:
        if model is None:
            return JSONResponse(status_code=500, content={"error": "Model not loaded"})
    try:
        print("Starting real-time audio stream from phone mic...")
        recorder = AudioRecorder(sample_rate=SAMPLE_RATE)
        recorder.start_recording(max_duration=duration)
        start_time = time.time()
        chunk_duration = 0.5
        chunk_samples = int(chunk_duration * SAMPLE_RATE)
        results = []
        start_time
        silent_count = 0

        while time.time() - start_time < duration:
            await asyncio.sleep(chunk_duration)
            if not recorder.recording or len(recorder.recording) < chunk_samples:
                continue
            audio_chunk = np.concatenate(recorder.recording[-chunk_samples:]).flatten()[:chunk_samples]
            audio_chunk = AudioPreprocessor.reduce_noise(audio_chunk)
            keystrokes = AudioPreprocessor.detect_keystrokes(audio_chunk, threshold=0.05, min_silence=0.05)

            chunk_results = []
            for i, ks in enumerate(keystrokes):
                print(f"Processing keystroke {i} in chunk, length: {len(ks)} samples")
                spectrogram = AudioPreprocessor.create_mel_spectrogram(ks)
            
                AudioPreprocessor.plot_mel_spectrogram(ks, title=f"Stream Keystroke {len(results) + i}")
                pred_idx, confidences = model.predict(spectrogram)
                prediction = CLASSES[pred_idx]
                confidence = float(confidences[pred_idx])
                if prediction != "noise" or confidence < 0.9:
                    chunk_results.append({
                        "keystroke": len(results) + i,
                        "prediction": prediction,
                        "confidence": confidence,
                        "is_low_confidence": bool(confidence < 0.5)
                    })

            if chunk_results:
                results.extend(chunk_results)
                silent_count = 0
                for res in chunk_results:
                    print(f"🔑 Keystroke {res['keystroke']}: {res['prediction']} (Confidence: {res['confidence']:.2f})")
            else:
                silent_count += 1

            if silent_count >= 4:
                print("🔇 Summary: {'characters': 0, 'spaces': 0, 'enters': 0, 'numbers': 0}")
                time.time()
                silent_count = 0

        recorder.stop_recording()
        print("Stream ended.")
        if not results:
            print("No valid keystrokes detected in stream")
            return {
                "summary": {
                    "characters": 0,
                    "spaces": 0,
                    "enters": 0,
                    "numbers": 0
                }
            }
        char_count = len([r for r in results if r['prediction'] not in ['noise', 'space', 'enter']])
        space_count = len([r for r in results if r['prediction'] == 'space'])
        enter_count = len([r for r in results if r['prediction'] == 'enter'])
        number_count = len([r for r in results if r['prediction'] in [str(i) for i in range(10)]])
        print(f"Final results: {len(results)} keystrokes, {char_count} characters")
        if char_count == 0 and space_count == 0 and enter_count == 0 and number_count == 0:
            return {
                "summary": {
                    "characters": 0,
                    "spaces": 0,
                    "enters": 0,
                    "numbers": 0
                }
            }
        return {
            "keystrokes": results,
            "summary": {
                "characters": char_count,
                "spaces": space_count,
                "enters": enter_count,
                "numbers": number_count
            }
        }
    except Exception as e:
        print(f"Error in /stream/: {str(e)}")
        print(traceback.format_exc())
        recorder.stop_recording()
        return JSONResponse(status_code=500, content={"error": f"Streaming failed: {str(e)}"})

@app.post("/train/")
async def train_model():
    global model
    try:
        print("Starting model training...")
        X, y = prepare_training_data(TRAINING_DATA_DIR)
        if len(X) == 0:
            return JSONResponse(status_code=400, content={"error": "No valid training data found"})
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
     
        for label in CLASSES:
            label_dir = os.path.join(TRAINING_DATA_DIR, label)
            if not os.path.isdir(label_dir):
                continue
            for file in os.listdir(label_dir):
                if file.endswith('.wav'):
                    file_path = os.path.join(label_dir, file)
                    audio = AudioPreprocessor.load_audio(file_path)
                    audio = AudioPreprocessor.reduce_noise(audio)
                    keystroke = AudioPreprocessor.extract_keystroke(audio)
                    if keystroke is not None:
                        AudioPreprocessor.plot_mel_spectrogram(keystroke, title=f"Training Sample: {label}")
                    break  
        
        with model_lock:
            model = KeystrokeCNN(input_shape=EXPECTED_INPUT_SHAPE)
            history = model.train(X_train, y_train, X_val, y_val, epochs=70, batch_size=32)
            model.save_model(MODEL_PATH)
            print(f"Model trained and saved as {MODEL_PATH}")
        
        return {
            "message": "Model trained and saved successfully",
            "history": {
                "loss": history.history.get("loss", []),
                "accuracy": history.history.get("accuracy", []),
                "val_loss": history.history.get("val_loss", []),
                "val_accuracy": history.history.get("val_accuracy", [])
            }
        }
    except Exception as e:
        print(f"Error in /train/: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse(status_code=400, content={"error": f"Training failed: {str(e)}"})

def prepare_training_data(data_dir):
    X = []
    y = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        if label not in CLASSES:
            print(f"Skipping unknown label: {label}")
            continue
        class_idx = CLASSES.index(label)
        for file in os.listdir(label_dir):
            if not file.endswith('.wav'):
                continue
            file_path = os.path.join(label_dir, file)
            try:
                audio = AudioPreprocessor.load_audio(file_path)
                audio = AudioPreprocessor.reduce_noise(audio)
                keystroke = AudioPreprocessor.extract_keystroke(audio)
                if keystroke is None:
                    print(f"Skipping {file_path}: no keystroke detected")
                    continue
                spectrogram = AudioPreprocessor.create_mel_spectrogram(keystroke)
                if spectrogram is None:
                    continue
                X.append(spectrogram)
                y.append(class_idx)
                print(f"Processed {file_path}, label: {label}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
    X = np.array(X)
    y = np.array(y)
    if len(X) == 0 or len(y) == 0:
        print("No valid audio files processed")
        return X, y
    print(f"Loaded {len(X)} samples across {len(np.unique(y))} classes")
    return X, y

if __name__ == "__main__":
    print("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
print("Server stopped!")
