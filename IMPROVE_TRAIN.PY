
import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy import signal
import noisereduce as nr
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import joblib

SAMPLE_RATE = 44100
DURATION = 0.1
N_MELS = 64  # Reduced from 128
FFT_WINDOW = 1024
HOP_LENGTH = 512
CLASSES = [str(i) for i in range(10)] + [chr(i) for i in range(97, 123)] + ["space", "enter", "noise"]
NUM_CLASSES = len(CLASSES)
EXPECTED_INPUT_SHAPE = (N_MELS, 44, 1)  # Updated for new dimensions
TRAINING_DATA_DIR = "./training_data"
MODEL_PATH = "keystroke_model_improved.keras"

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
            # Simplified noise reduction
            if len(audio) > sr * 0.5:  # Only if audio is long enough
                noise_segment = audio[:int(0.1 * sr)]  # Use first 0.1s as noise
                if np.std(noise_segment) < 0.01:
                    return nr.reduce_noise(y=audio, y_noise=noise_segment, sr=sr)
            return audio
        except Exception as e:
            print(f"Error reducing noise: {e}")
            return audio

    @staticmethod
    def extract_keystroke(audio, sr=SAMPLE_RATE, threshold=0.02):  # Lower threshold
        try:
            # Normalize audio
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            # Find the loudest part (likely the keystroke)
            energy = np.abs(audio)
            smoothed_energy = np.convolve(energy, np.ones(int(0.01 * sr))/int(0.01 * sr), mode='same')
            
            # Find peak
            peak_idx = np.argmax(smoothed_energy)
            
            # Extract around peak
            start = max(0, peak_idx - int(0.02 * sr))
            end = min(len(audio), peak_idx + int(0.08 * sr))
            segment = audio[start:end]
            
            # Pad or trim to exact duration
            target_length = int(DURATION * sr)
            if len(segment) < target_length:
                segment = np.pad(segment, (0, target_length - len(segment)), mode="constant")
            elif len(segment) > target_length:
                segment = segment[:target_length]
            
            print(f"Extracted keystroke, length: {len(segment)} samples, max amplitude: {np.max(np.abs(segment)):.4f}")
            return segment
        except Exception as e:
            print(f"Error extracting keystroke: {e}")
            return None

    @staticmethod
    def create_mel_spectrogram(audio, sr=SAMPLE_RATE, n_mels=N_MELS, apply_augmentation=False):
        try:
            # Minimal augmentation only during training
            if apply_augmentation and np.random.rand() < 0.3:  # Only 30% chance
                # Light noise addition
                audio += np.random.normal(0, 0.005, audio.shape)
            
            # Ensure correct length
            target_length = int(DURATION * sr)
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)), mode="constant")
            elif len(audio) > target_length:
                audio = audio[:target_length]

            # Create mel spectrogram
            S = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=n_mels, n_fft=FFT_WINDOW, hop_length=HOP_LENGTH
            )
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            print(f"Raw spectrogram shape: {S_dB.shape}")
            
            # Pad/trim time dimension to consistent size
            target_time_steps = 44  # For 0.1s at 44100 Hz with hop_length=512
            if S_dB.shape[1] < target_time_steps:
                S_dB = np.pad(S_dB, ((0, 0), (0, target_time_steps - S_dB.shape[1])), mode="constant")
            elif S_dB.shape[1] > target_time_steps:
                S_dB = S_dB[:, :target_time_steps]

            print(f"Final spectrogram shape: {S_dB.shape}")
            return np.expand_dims(S_dB, axis=-1)
        except Exception as e:
            print(f"Error creating spectrogram: {e}")
            return None

class SimpleCNN:
    def __init__(self, input_shape=EXPECTED_INPUT_SHAPE, num_classes=NUM_CLASSES):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        # Much simpler model
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),

            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),

            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation="softmax")
        ])
        
        # Higher learning rate
        model.compile(
            optimizer=Adam(learning_rate=0.001),  # Increased from 0.0001
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        checkpoint_callback = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy", mode="max")
        early_stopping = EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=0.00001)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint_callback, early_stopping, reduce_lr],
            verbose=1
        )
        return history

def train_optimized_random_forest(X_train, X_val, y_train, y_val):
    print("\n" + "="*50)
    print("TRAINING OPTIMIZED RANDOM FOREST MODEL")
    print("="*50)
    
    # Flatten spectrograms for RF
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    
    print(f"Training RF with {X_train_flat.shape[0]} samples, {X_train_flat.shape[1]} features")
    
    # Feature selection - remove low variance features
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=0.1)
    X_train_selected = selector.fit_transform(X_train_flat)
    X_val_selected = selector.transform(X_val_flat)
    
    print(f"After feature selection: {X_train_selected.shape[1]} features (removed {X_train_flat.shape[1] - X_train_selected.shape[1]})")
    
    # Optimized Random Forest with better parameters
    rf_optimized = RandomForestClassifier(
        n_estimators=200,          # More trees
        max_depth=15,              # Deeper trees
        min_samples_split=5,       # More samples required to split
        min_samples_leaf=2,        # Minimum samples in leaf
        max_features='sqrt',       # Use sqrt of features per tree
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'    # Handle class imbalance
    )
    
    rf_optimized.fit(X_train_selected, y_train)
    
    # Predict
    y_pred = rf_optimized.predict(X_val_selected)
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"\nOPTIMIZED RANDOM FOREST RESULTS:")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Detailed performance analysis
    from sklearn.metrics import classification_report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_val, y_pred, target_names=[CLASSES[i] for i in sorted(set(y_val))], zero_division=0))
    
    # Feature importance analysis
    print(f"\nTop 15 most important features:")
    feature_importance = rf_optimized.feature_importances_
    top_features = np.argsort(feature_importance)[-15:]
    for i, feat_idx in enumerate(top_features):
        mel_bin = feat_idx // 44  # Assuming 44 time steps
        time_step = feat_idx % 44
        print(f"  Feature {feat_idx} (Mel bin {mel_bin}, Time {time_step}): {feature_importance[feat_idx]:.4f}")
    
    # Class-wise accuracy
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_val, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    print(f"\nPer-class accuracies (top 10):")
    best_classes = np.argsort(class_accuracies)[-10:]
    for class_idx in best_classes:
        if class_idx < len(CLASSES):
            print(f"  {CLASSES[class_idx]}: {class_accuracies[class_idx]:.4f}")
    
    return rf_optimized, selector, accuracy

def prepare_training_data(data_dir):
    class_counts = {label: 0 for label in CLASSES}
    X, y = [], []
    
    print("Loading and processing audio files...")
    
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
        
        print(f"\nProcessing class '{label}' with {len(files)} files...")
        
        for i, file in enumerate(files):
            if i % 10 == 0:
                print(f"  Processing file {i+1}/{len(files)}")
                
            file_path = os.path.join(label_dir, file)
            try:
                audio = AudioPreprocessor.load_audio(file_path)
                audio = AudioPreprocessor.reduce_noise(audio)
                keystroke = AudioPreprocessor.extract_keystroke(audio)
                
                if keystroke is None:
                    print(f"    Skipping {file}: no keystroke detected")
                    continue
                    
                # Only apply augmentation to training samples (not validation)
                spectrogram = AudioPreprocessor.create_mel_spectrogram(keystroke, apply_augmentation=True)
                
                if spectrogram is None:
                    print(f"    Skipping {file}: spectrogram creation failed")
                    continue
                    
                X.append(spectrogram)
                y.append(class_idx)
                
            except Exception as e:
                print(f"    Error processing {file}: {e}")
                continue
    
    if len(X) == 0:
        print("ERROR: No valid audio files processed!")
        return np.array([]), np.array([])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nSUMMARY:")
    print(f"Samples per class: {class_counts}")
    print(f"Total samples loaded: {len(X)}")
    print(f"Classes found: {len(np.unique(y))}")
    print(f"Data shape: {X.shape}")
    print(f"Data range: [{np.min(X):.2f}, {np.max(X):.2f}]")
    
    return X, y

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("training_history_improved.png", dpi=150)
    plt.close()
    print("Training history plot saved as 'training_history_improved.png'")

def main():
    print("="*60)
    print("IMPROVED KEYSTROKE CLASSIFICATION TRAINING")
    print("="*60)
    
    # Load data
    print("Loading training data...")
    X, y = prepare_training_data(TRAINING_DATA_DIR)
    
    if len(X) == 0:
        print("ERROR: No valid training data found. Check your audio files!")
        return
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Training data mean: {np.mean(X_train):.4f}, std: {np.std(X_train):.4f}")
    print(f"Validation data mean: {np.mean(X_val):.4f}, std: {np.std(X_val):.4f}")
    
    # Train Optimized Random Forest as primary model
    rf_model, selector, rf_accuracy = train_optimized_random_forest(X_train, X_val, y_train, y_val)
    
    # Save the Random Forest model and feature selector
    import joblib
    joblib.dump(rf_model, "keystroke_rf_model.pkl")
    joblib.dump(selector, "feature_selector.pkl")
    print(f"\nRandom Forest model saved as: keystroke_rf_model.pkl")
    print(f"Feature selector saved as: feature_selector.pkl")
    
    print(f"\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Optimized Random Forest Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
    
    # Analysis and recommendations
    if rf_accuracy > 0.25:
        print("\n✅ EXCELLENT: >25% accuracy - Model is working well!")
        print("   Ready for deployment and real-world testing.")
    elif rf_accuracy > 0.15:
        print("\n✅ GOOD: 15-25% accuracy - Decent performance for 39 classes.")
        print("   Consider collecting more data for challenging classes.")
    elif rf_accuracy > 0.10:
        print("\n⚠️  MODERATE: 10-15% accuracy - Some signal but needs improvement.")
        print("   Check audio quality and consider feature engineering.")
    else:
        print("\n❌ POOR: <10% accuracy - Major issues need addressing.")
        print("   Review audio preprocessing and data collection process.")
    
    print("\n" + "="*50)
    print("RECOMMENDATIONS:")
    if rf_accuracy < 0.05:
        print("1. Check your audio files - make sure they contain clear keystrokes")
        print("2. Verify the training_data folder structure")
        print("3. Try recording new samples with less background noise")
    elif rf_accuracy > 0.15:
        print("1. Your audio preprocessing is working!")
        print("2. Consider using Random Forest as your final model")
        print("3. Or try a more sophisticated CNN architecture")
    print("="*50)

if __name__ == "__main__":
    main()
