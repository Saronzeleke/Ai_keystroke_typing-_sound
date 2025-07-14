import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import noisereduce as nr
import seaborn as sns

# -----------------------------
# Configuration
# -----------------------------
SAMPLE_RATE = 44100
DURATION = 0.2
N_MELS = 128
HOP_LENGTH = 256
FFT_WINDOW = 1024
CLASSES = [str(i) for i in range(10)] + [chr(i) for i in range(97, 123)] + ["space", "enter", "noise"]
NUM_CLASSES = len(CLASSES)
INPUT_SHAPE = (N_MELS, 173, 1)
DATA_DIR = r"C:/Users/admin/Ai_keystroke_typing-_sound/training_data2"
MODEL_SAVE_PATH = "keystroke_model.keras"

# -----------------------------
# Audio Preprocessing
# -----------------------------
def load_audio(file_path):
    try:
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        return audio
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def reduce_noise(audio):
    try:
        noise_samples = []
        segment_length = int(0.1 * SAMPLE_RATE)
        for i in range(0, len(audio), segment_length):
            segment = audio[i:i+segment_length]
            if np.std(segment) < 0.01:
                noise_samples.append(segment)
        if not noise_samples:
            return audio
        noise_profile = np.concatenate(noise_samples)
        return nr.reduce_noise(y=audio, y_noise=noise_profile, sr=SAMPLE_RATE)
    except:
        return audio

def extract_keystroke(audio):
    try:
        audio = audio / np.max(np.abs(audio))
        peaks = librosa.util.peak_pick(np.abs(audio), pre_max=20, post_max=20,
                                       pre_avg=50, post_avg=50,
                                       delta=0.05, wait=10)
        if not len(peaks):
            return None
        peak = peaks[0]
        start = max(0, peak - int(0.07 * SAMPLE_RATE))
        end = min(len(audio), peak + int(0.13 * SAMPLE_RATE))
        segment = audio[start:end]
        target_length = int(DURATION * SAMPLE_RATE)
        if len(segment) < target_length:
            segment = np.pad(segment, (0, target_length - len(segment)), mode="constant")
        else:
            segment = segment[:target_length]
        return segment
    except:
        return None

def create_mel_spectrogram(audio):
    try:
        S = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS,
                                          n_fft=FFT_WINDOW, hop_length=HOP_LENGTH)
        S_dB = librosa.power_to_db(S, ref=np.max)
        S_dB = (S_dB - np.mean(S_dB)) / (np.std(S_dB) + 1e-8)
        target_time_steps = 173
        if S_dB.shape[1] < target_time_steps:
            S_dB = np.pad(S_dB, ((0, 0), (0, target_time_steps - S_dB.shape[1])), mode="constant")
        elif S_dB.shape[1] > target_time_steps:
            S_dB = S_dB[:, :target_time_steps]
        return np.expand_dims(S_dB, axis=-1)
    except:
        return None

# -----------------------------
# Data Preparation
# -----------------------------
def prepare_training_data(data_dir):
    X, y = [], []
    class_counts = {cls: 0 for cls in CLASSES}
    
    for label in os.listdir(data_dir):
        if label not in CLASSES:
            continue
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        files = [f for f in os.listdir(label_dir) if f.endswith(".wav")]
        for file in files:
            file_path = os.path.join(label_dir, file)
            try:
                audio = load_audio(file_path)
                if audio is None:
                    continue
                audio = reduce_noise(audio)
                keystroke = extract_keystroke(audio)
                if keystroke is None:
                    continue
                features = create_mel_spectrogram(keystroke)
                if features is None:
                    continue
                X.append(features)
                y.append(CLASSES.index(label))
                class_counts[label] += 1
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    print("Before balancing:", class_counts)

    # Balance classes
    unique_labels = np.unique(y)
    max_count = max([class_counts[CLASSES[i]] for i in unique_labels])
    X_balanced, y_balanced = [], []

    for idx in unique_labels:
        mask = np.array(y) == idx
        X_class = np.array(X)[mask]
        y_class = np.array(y)[mask]
        count = len(X_class)
        if count < 10:
            continue
        if count < max_count:
            X_resampled, y_resampled = resample(X_class, y_class,
                                               replace=True,
                                               n_samples=max_count,
                                               random_state=42)
            X_balanced.extend(X_resampled)
            y_balanced.extend(y_resampled)
        else:
            X_balanced.extend(X_class)
            y_balanced.extend(y_class)

    X = np.array(X_balanced)
    y = np.array(y_balanced)
    print(f"Balanced dataset: {len(X)} samples across {len(np.unique(y))} classes")

    return X, y

# -----------------------------
# Model Definition
# -----------------------------
def build_simple_cnn(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(1./255)(inputs)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# -----------------------------
# Train the Model
# -----------------------------
def main():
    print("Loading data...")
    X, y = prepare_training_data(DATA_DIR)
    if len(X) == 0:
        print("No valid data found.")
        return

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print(f"Training shape: {X_train.shape}, Validation shape: {X_val.shape}")

    print("Building model...")
    model = build_simple_cnn(INPUT_SHAPE, NUM_CLASSES)

    print("Training model...")
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=16,
        callbacks=[
            ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss'),
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        ],
        class_weight=class_weights_dict
    )

    print("Saving model...")
    model.save(MODEL_SAVE_PATH)

    print("Plotting training history...")
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()

    print("Generating confusion matrix...")
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_val, y_pred_classes)

    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

    print("Per-class accuracy:")
    per_class_acc = np.diag(cm) / np.sum(cm, axis=1)
    for i, acc in enumerate(per_class_acc):
        print(f"{CLASSES[i]}: {acc:.4f}")

if __name__ == "__main__":
    main()