import os
import numpy as np
import librosa
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from scipy import signal
import noisereduce as nr
import matplotlib.pyplot as plt
SAMPLE_RATE = 44100
DURATION = 0.1
N_MELS = 128
FFT_WINDOW = 1024
HOP_LENGTH = 512
CLASSES = [str(i) for i in range(10)] + [chr(i) for i in range(97, 123)] + ['space', 'enter', 'noise']
NUM_CLASSES = len(CLASSES)
EXPECTED_INPUT_SHAPE = (N_MELS, 87, 1)
TRAINING_DATA_DIR = "./training_data"
MODEL_PATH = "keystroke_model.keras"
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
            if np.random.rand() < 0.5:  # 50% chance of augmentation
                audio += np.random.normal(0, 0.01, audio.shape)
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=np.random.uniform(-1, 1))
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
class KeystrokeCNN:
    def __init__(self, input_shape=EXPECTED_INPUT_SHAPE, num_classes=NUM_CLASSES):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
    def build_model(self):
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.6),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=0.0001), 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
        return model
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        checkpoint_callback = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss', mode='min')
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = self.model.fit(
            X_train, y_train, 
            validation_data=(X_val, y_val), 
            epochs=epochs, 
            batch_size=batch_size,
            callbacks=[checkpoint_callback, early_stopping]
        )
        return history
    def save_model(self, filepath):
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
def prepare_training_data(data_dir):
    class_counts = {label: 0 for label in CLASSES}
    X, y = [], []
    for label in os.listdir(data_dir):
        if label not in CLASSES:
            print(f"Skipping unknown label: {label}")
            continue
        class_idx = CLASSES.index(label)
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        files = [f for f in os.listdir(label_dir) if f.endswith('.wav')]
        class_counts[label] = len(files)
        for file in files:
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
    print("Samples per class:", class_counts)
    if len(X) == 0 or len(y) == 0:
        print("No valid audio files processed")
        return X, y
    print(f"Loaded {len(X)} samples across {len(np.unique(y))} classes")
    return X, y
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()
def main():
    print("Loading training data...")
    X, y = prepare_training_data(TRAINING_DATA_DIR)
    if len(X) == 0:
        print("Error: No valid training data found.")
        return
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    print(f"Training spectrogram mean: {np.mean(X_train):.4f}, std: {np.std(X_train):.4f}")
    print(f"Validation spectrogram mean: {np.mean(X_val):.4f}, std: {np.std(X_val):.4f}")
    print("Training model...")
    model = KeystrokeCNN(input_shape=EXPECTED_INPUT_SHAPE)
    history = model.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)
    model.save_model(MODEL_PATH)
    print(f"Training completed. Model saved as {MODEL_PATH}")
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    print(f"Final training accuracy: {final_train_acc:.4f}")
    print(f"Final validation accuracy: {final_val_acc:.4f}")
    print("Plotting training history...")
    plot_training_history(history)
    print("Training history plot saved as 'training_history.png'")
if __name__ == "__main__":
    main()