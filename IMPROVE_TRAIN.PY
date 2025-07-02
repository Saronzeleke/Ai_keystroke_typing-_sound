import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import noisereduce as nr
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Enhanced parameters for better accuracy
SAMPLE_RATE = 44100
DURATION = 0.15  # Increased from 0.1s to capture more keystroke context
N_MELS = 128     # Increased mel bins for better frequency resolution
FFT_WINDOW = 2048  # Larger window for better frequency resolution
HOP_LENGTH = 256   # Smaller hop for better temporal resolution
CLASSES = [str(i) for i in range(10)] + [chr(i) for i in range(97, 123)] + ["space", "enter", "noise"]
NUM_CLASSES = len(CLASSES)
TRAINING_DATA_DIR = "./training_data"

class AdvancedAudioPreprocessor:
    @staticmethod
    def load_audio(file_path, sr=SAMPLE_RATE):
        try:
            audio, _ = librosa.load(file_path, sr=sr, mono=True)
            # Remove silence from beginning and end
            audio, _ = librosa.effects.trim(audio, top_db=20)
            return audio
        except Exception as e:
            print(f"Error loading audio {file_path}: {e}")
            raise

    @staticmethod
    def advanced_noise_reduction(audio, sr=SAMPLE_RATE):
        try:
            # Multi-stage noise reduction
            # Stage 1: Spectral subtraction
            audio = nr.reduce_noise(y=audio, sr=sr, stationary=False, prop_decrease=0.8)
            
            # Stage 2: High-pass filter to remove low-frequency noise
            from scipy import signal as scipy_signal
            b, a = scipy_signal.butter(5, 100, btype='high', fs=sr)
            audio = scipy_signal.filtfilt(b, a, audio)
            
            return audio
        except Exception as e:
            print(f"Error in noise reduction: {e}")
            return audio

    @staticmethod
    def extract_enhanced_keystroke(audio, sr=SAMPLE_RATE):
        try:
            # Normalize
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            # Enhanced peak detection
            from scipy import signal as scipy_signal
            
            # Calculate energy envelope
            hop_length = 512
            frame_length = 2048
            energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Find the main keystroke peak
            peaks, properties = scipy_signal.find_peaks(energy, 
                                                       height=np.max(energy) * 0.3,
                                                       distance=int(0.05 * sr / hop_length))
            
            if len(peaks) == 0:
                # If no clear peak, use the loudest section
                peak_frame = np.argmax(energy)
            else:
                # Use the highest peak
                peak_frame = peaks[np.argmax(properties['peak_heights'])]
            
            # Convert frame to sample
            peak_sample = peak_frame * hop_length
            
            # Extract with more context
            start = max(0, peak_sample - int(0.05 * sr))
            end = min(len(audio), peak_sample + int(0.10 * sr))
            segment = audio[start:end]
            
            # Pad or trim to exact duration
            target_length = int(DURATION * sr)
            if len(segment) < target_length:
                segment = np.pad(segment, (0, target_length - len(segment)), mode="constant")
            elif len(segment) > target_length:
                segment = segment[:target_length]
            
            return segment
        except Exception as e:
            print(f"Error extracting keystroke: {e}")
            return None

    @staticmethod
    def create_rich_features(audio, sr=SAMPLE_RATE):
        try:
            if audio is None or len(audio) == 0:
                return None
                
            # Ensure correct length
            target_length = int(DURATION * sr)
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)), mode="constant")
            elif len(audio) > target_length:
                audio = audio[:target_length]

            features = []
            
            # 1. Enhanced Mel Spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=N_MELS, n_fft=FFT_WINDOW, hop_length=HOP_LENGTH
            )
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize mel spectrogram
            mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-8)
            features.append(mel_db.flatten())
            
            # 2. MFCC features
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=FFT_WINDOW, hop_length=HOP_LENGTH)
            mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
            features.append(mfcc.flatten())
            
            # 3. Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=HOP_LENGTH)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=HOP_LENGTH)[0]
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, hop_length=HOP_LENGTH)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=HOP_LENGTH)[0]
            
            features.extend([
                spectral_centroids,
                spectral_bandwidth, 
                spectral_contrast.flatten(),
                spectral_rolloff
            ])
            
            # 4. Rhythm features
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr, hop_length=HOP_LENGTH)
            features.append([tempo])
            
            # 5. Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LENGTH)[0]
            features.append(zcr)
            
            # 6. Energy and dynamics
            rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)[0]
            features.append(rms)
            
            # Combine all features
            combined_features = np.concatenate([np.array(f).flatten() for f in features])
            
            print(f"Generated {len(combined_features)} features")
            return combined_features
            
        except Exception as e:
            print(f"Error creating features: {e}")
            return None

class EnsembleModel:
    def __init__(self):
        self.models = {}
        self.scaler = RobustScaler()
        self.feature_selector = None
        self.pca = None
        
    def prepare_features(self, X, fit=False):
        """Advanced feature preparation pipeline"""
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Feature selection
        if fit:
            self.feature_selector = SelectKBest(score_func=f_classif, k=min(1000, X_scaled.shape[1]//2))
            X_selected = self.feature_selector.fit_transform(X_scaled, self.y_train)
        else:
            X_selected = self.feature_selector.transform(X_scaled)
        
        # PCA for dimensionality reduction
        if fit:
            self.pca = PCA(n_components=min(300, X_selected.shape[1], X_selected.shape[0]//2))
            X_final = self.pca.fit_transform(X_selected)
        else:
            X_final = self.pca.transform(X_selected)
            
        return X_final
    
    def train(self, X_train, y_train, X_val, y_val):
        self.y_train = y_train
        
        print("Preparing enhanced features...")
        X_train_prep = self.prepare_features(X_train, fit=True)
        X_val_prep = self.prepare_features(X_val, fit=False)
        
        print(f"Final feature dimensions: {X_train_prep.shape}")
        
        # Train multiple models
        self.models['rf'] = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        self.models['gb'] = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        print("Training Random Forest...")
        self.models['rf'].fit(X_train_prep, y_train)
        
        print("Training Gradient Boosting...")
        self.models['gb'].fit(X_train_prep, y_train)
        
        # Evaluate each model
        results = {}
        for name, model in self.models.items():
            y_pred = model.predict(X_val_prep)
            accuracy = accuracy_score(y_val, y_pred)
            results[name] = accuracy
            print(f"{name.upper()} Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Ensemble prediction
        rf_pred = self.models['rf'].predict_proba(X_val_prep)
        gb_pred = self.models['gb'].predict_proba(X_val_prep)
        
        # Weighted ensemble (RF gets more weight as it's usually better)
        ensemble_pred = 0.7 * rf_pred + 0.3 * gb_pred
        ensemble_labels = np.argmax(ensemble_pred, axis=1)
        ensemble_accuracy = accuracy_score(y_val, ensemble_labels)
        
        print(f"\nENSEMBLE Accuracy: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")
        
        # Detailed analysis for best model
        best_model_name = max(results, key=results.get)
        best_model = self.models[best_model_name]
        best_pred = best_model.predict(X_val_prep)
        
        print(f"\nDetailed report for best model ({best_model_name.upper()}):")
        print(classification_report(y_val, best_pred, 
                                  target_names=[CLASSES[i] for i in sorted(set(y_val))], 
                                  zero_division=0))
        
        return max(ensemble_accuracy, max(results.values()))

def prepare_enhanced_training_data(data_dir):
    print("Loading and processing audio files with advanced preprocessing...")
    
    class_counts = {label: 0 for label in CLASSES}
    X, y = [], []
    
    for label in os.listdir(data_dir):
        if label not in CLASSES:
            continue
            
        class_idx = CLASSES.index(label)
        label_dir = os.path.join(data_dir, label)
        
        if not os.path.isdir(label_dir):
            continue
            
        files = [f for f in os.listdir(label_dir) if f.endswith(".wav")]
        class_counts[label] = len(files)
        
        print(f"\nProcessing class '{label}' with {len(files)} files...")
        
        valid_samples = 0
        for i, file in enumerate(files):
            if i % 10 == 0:
                print(f"  Processing file {i+1}/{len(files)}")
                
            file_path = os.path.join(label_dir, file)
            try:
                # Advanced audio processing
                audio = AdvancedAudioPreprocessor.load_audio(file_path)
                audio = AdvancedAudioPreprocessor.advanced_noise_reduction(audio)
                keystroke = AdvancedAudioPreprocessor.extract_enhanced_keystroke(audio)
                
                if keystroke is None:
                    continue
                    
                # Create rich feature set
                features = AdvancedAudioPreprocessor.create_rich_features(keystroke)
                
                if features is None or len(features) == 0:
                    continue
                    
                X.append(features)
                y.append(class_idx)
                valid_samples += 1
                
            except Exception as e:
                print(f"    Error processing {file}: {e}")
                continue
        
        print(f"  Successfully processed {valid_samples}/{len(files)} files")
    
    if len(X) == 0:
        print("ERROR: No valid audio files processed!")
        return np.array([]), np.array([])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nENHANCED PROCESSING SUMMARY:")
    print(f"Samples per class: {class_counts}")
    print(f"Total samples loaded: {len(X)}")
    print(f"Feature dimensions: {X.shape[1] if len(X) > 0 else 0}")
    print(f"Classes found: {len(np.unique(y))}")
    
    return X, y

def main():
    print("="*70)
    print("ULTRA-IMPROVED KEYSTROKE CLASSIFICATION TRAINING")
    print("TARGET: 70-90% ACCURACY")
    print("="*70)
    
    # Load data with enhanced preprocessing
    X, y = prepare_enhanced_training_data(TRAINING_DATA_DIR)
    
    if len(X) == 0:
        print("ERROR: No valid training data found!")
        print("\nRECOMMENDATIONS:")
        print("1. Collect more data (200+ samples per class minimum)")
        print("2. Record in quieter environment")
        print("3. Use consistent recording setup")
        print("4. Check audio file quality")
        return
    
    # Check data quality
    if len(X) < 200 * NUM_CLASSES:
        print(f"\n⚠️  WARNING: You have {len(X)} samples but need {200 * NUM_CLASSES}+ for 70-90% accuracy")
        print("Current data is insufficient for target accuracy!")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Train ensemble model
    ensemble = EnsembleModel()
    final_accuracy = ensemble.train(X_train, y_train, X_val, y_val)
    
    # Save models
    joblib.dump(ensemble, "keystroke_ensemble_model.pkl")
    print(f"\nEnsemble model saved as: keystroke_ensemble_model.pkl")
    
    print(f"\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Best Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    
    # Realistic expectations
    if final_accuracy > 0.60:
        print("\n🎉 EXCELLENT: >60% - Great performance!")
    elif final_accuracy > 0.40:
        print("\n✅ VERY GOOD: 40-60% - Solid improvement!")
    elif final_accuracy > 0.25:
        print("\n👍 GOOD: 25-40% - Decent progress!")
    else:
        print("\n⚠️  NEEDS WORK: <25% - More data required!")
    
    print(f"\n" + "="*50)
    print("TO REACH 70-90% ACCURACY, YOU NEED:")
    print("="*50)
    print("1. 📊 MORE DATA: 200-500 samples per class (you have ~50)")
    print("2. 🔧 BETTER RECORDING: Consistent environment, less noise")
    print("3. 🎯 FOCUSED CLASSES: Start with fewer, more distinct classes")
    print("4. ⚡ BETTER HARDWARE: Higher quality microphone")
    print("5. 🧹 DATA CLEANING: Remove poor quality samples")
    print("="*50)

if __name__ == "__main__":
    main()