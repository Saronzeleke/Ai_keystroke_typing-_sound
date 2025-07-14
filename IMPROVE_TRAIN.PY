import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from scipy import signal
import noisereduce as nr
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
import seaborn as sns
from spec_augument import SpecAugment

SAMPLE_RATE = 44100
DURATION = 0.2
N_MELS = 128
FFT_WINDOW = 1024
HOP_LENGTH = 128
EXPECTED_INPUT_SHAPE = (N_MELS + 20, 86, 1)  # e.g., (148, 86, 1)
CLASSES = [str(i) for i in range(10)] + [chr(i) for i in range(97, 123)] + ["space", "enter", "noise"]
NUM_CLASSES = len(CLASSES)
EXPECTED_INPUT_SHAPE = (N_MELS + 40, 173, 1)
TRAINING_DATA_DIR =r"C:/Users/admin/Ai_keystroke_typing-_sound/training_data2"
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
            start = max(0, peak - int(0.07 * sr))
            end = min(len(audio), peak + int(0.13 * sr))
            segment = audio[start:end]
            target_length = int(DURATION * sr)
            if len(segment) < target_length:
                segment = np.pad(segment, (0, target_length - len(segment)), mode="constant")
            elif len(segment) > target_length:
                segment = segment[:target_length]
            print(f"Extracted keystroke, length: {len(segment)} samples")
            return segment
        except Exception as e:
            print(f"Error extracting keystroke: {e}")
            return None

    @staticmethod
    def create_mel_spectrogram(audio, sr=SAMPLE_RATE, n_mels=N_MELS, apply_augmentation=True):
        try:
            if apply_augmentation:
                pitch_shift = np.random.uniform(-0.5, 0.5)
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift)
                stretch_rate = np.random.uniform(0.95, 1.05)
                audio = librosa.effects.time_stretch(audio, rate=stretch_rate)
                if np.random.rand() < 0.2:
                    audio += np.random.normal(0, 0.003, audio.shape)
            target_length = int(DURATION * sr)
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)), mode="constant")
            elif len(audio) > target_length:
                audio = audio[:target_length]
            S = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=n_mels, n_fft=FFT_WINDOW, hop_length=HOP_LENGTH
            )
            S_dB = librosa.power_to_db(S, ref=np.max)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40, n_fft=FFT_WINDOW, hop_length=HOP_LENGTH)
            S_dB = (S_dB - np.mean(S_dB)) / (np.std(S_dB) + 1e-10)
            mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-10)
            combined = np.concatenate((S_dB, mfcc), axis=0)
            if apply_augmentation:
                spec = tf.convert_to_tensor(combined, dtype=tf.float32)
                augmenter = SpecAugment(freq_mask_param=10, time_mask_param=10)
                spec = augmenter(spec)
                combined = spec.numpy()
            target_time_steps = 173
            if combined.shape[1] < target_time_steps:
                combined = np.pad(combined, ((0, 0), (0, target_time_steps - combined.shape[1])), mode="constant")
            elif combined.shape[1] > target_time_steps:
                combined = combined[:, :target_time_steps]
            print(f"Feature shape: {combined.shape}")
            return np.expand_dims(combined, axis=-1)
        except Exception as e:
            print(f"Error creating features: {e}")
            return None

class KeystrokeCNN:
    def __init__(self, input_shape=EXPECTED_INPUT_SHAPE, num_classes=NUM_CLASSES):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape)
        
        def residual_block(x, filters, kernel_size=(3, 3)):
            shortcut = x
            x = layers.Conv2D(filters, kernel_size, activation='relu', padding='same', kernel_regularizer=l2(1e-5))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(filters, kernel_size, activation='relu', padding='same', kernel_regularizer=l2(1e-5))(x)
            x = layers.BatchNormalization()(x)
            shortcut = layers.Conv2D(filters, (1, 1), padding='same')(shortcut)
            x = layers.Add()([shortcut, x])
            x = layers.Activation('relu')(x)
            return x

        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-5))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.3)(x)
        
        x = residual_block(x, 64)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.3)(x)
        
        x = residual_block(x, 128)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-5))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.GlobalAveragePooling2D()(x)
        
        x = layers.Dense(2048, activation='relu', kernel_regularizer=l2(1e-5))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(1024, activation='relu', kernel_regularizer=l2(1e-5))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=0.0005, clipnorm=1.0), 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=200, batch_size=16):
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights_dict = dict(enumerate(class_weights))
        
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=1e-6)
        checkpoint_callback = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss', mode='min')
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        
        history = self.model.fit(
            X_train, y_train, 
            validation_data=(X_val, y_val), 
            epochs=epochs, 
            batch_size=batch_size,
            callbacks=[checkpoint_callback, early_stopping, lr_scheduler],
            class_weight=class_weights_dict
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
        files = [f for f in os.listdir(label_dir) if f.endswith(".wav")]
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
                features = AudioPreprocessor.create_mel_spectrogram(keystroke)
                if features is None:
                    continue
                X.append(features)
                y.append(class_idx)
                print(f"Processed {file_path}, label: {label}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
    X = np.array(X)
    y = np.array(y)
    print("Samples per class before balancing:", class_counts)
    if len(X) == 0 or len(y) == 0:
        print("No valid audio files processed")
        return X, y
    
    # Check for data imbalance
    max_samples = max(class_counts.values())
    min_samples_threshold = 100
    imbalance_threshold = 0.5 * max_samples
    imbalanced = False
    imbalanced_classes = []
    for label, count in class_counts.items():
        if count < imbalance_threshold or count < min_samples_threshold:
            imbalanced = True
            imbalanced_classes.append((label, count))
    
    if imbalanced:
        
        print("Data imbalance detected. Imbalanced classes:", imbalanced_classes)
        print("Balancing dataset...")
        target_samples = max(max_samples, min_samples_threshold)
        X_balanced, y_balanced = [], []
        for class_idx in np.unique(y):
            X_class = X[y == class_idx]
            y_class = y[y == class_idx]
            if len(X_class) < target_samples:
                X_class_resampled, y_class_resampled = resample(
                    X_class, y_class, replace=True, n_samples=target_samples, random_state=42
                )
                X_balanced.append(X_class_resampled)
                y_balanced.append(y_class_resampled)
            else:
                X_balanced.append(X_class)
                y_balanced.append(y_class)
        X = np.concatenate(X_balanced)
        y = np.concatenate(y_balanced)
        print(f"Balanced dataset: {len(X)} samples across {len(np.unique(y))} classes")
    else:
        print("No significant data imbalance detected. Proceeding with original dataset.")
    
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
    print(f"Training feature mean: {np.mean(X_train):.4f}, std: {np.std(X_train):.4f}")
    print(f"Validation feature mean: {np.mean(X_val):.4f}, std: {np.std(X_val):.4f}")
    
    print("Training model...")
    model = KeystrokeCNN(input_shape=EXPECTED_INPUT_SHAPE)
    history = model.train(X_train, y_train, X_val, y_val, epochs=200, batch_size=16)
    model.save_model(MODEL_PATH)
    
    print(f"Training completed. Model saved as {MODEL_PATH}")
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    print(f"Final training accuracy: {final_train_acc:.4f}")
    print(f"Final validation accuracy: {final_val_acc:.4f}")
    
    print("Plotting training history...")
    plot_training_history(history)
    print("Training history plot saved as 'training_history.png'")
    
    y_pred = model.model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_val, y_pred_classes)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("Confusion matrix saved as 'confusion_matrix.png'")
    
    per_class_acc = np.diag(cm) / np.sum(cm, axis=1)
    print("Per-class accuracy:")
    for i, acc in enumerate(per_class_acc):
        if np.isnan(acc):
            acc = 0.0
        print(f"Class {CLASSES[i]}: {acc:.4f}")

if __name__ == "__main__":
    main()