
import numpy as np
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, WebSocket
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import sounddevice as sd
import soundfile as sf
import scipy.signal as signal
import noisereduce as nr
import time
import tempfile
import os
import asyncio
import traceback
import io
from threading import Lock
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

SAMPLE_RATE = 44100
DURATION = 0.1
N_MELS = 128
FFT_WINDOW = 1024
HOP_LENGTH = 512
CLASSES = [str(i) for i in range(10)] + [chr(i) for i in range(97, 123)] + ['space', 'enter', 'noise']
model = None
model_lock = Lock()
last_keystrokes = {}  # Store keystrokes for /spectrogram/

# [AudioPreprocessor, AudioRecorder, KeystrokeCNN classes unchanged]

@app.on_event("startup")
async def startup_event():
    global model
    with model_lock:
        try:
            model = KeystrokeCNN()
            if os.path.exists("keystroke_model.h5"):
                model.model = tf.keras.models.load_model("keystroke_model.h5")
                print("Loaded existing model")
            else:
                print("Initialized new model")
        except Exception as e:
            print(f"Error loading model: {e}")

@app.post("/record/")
async def record_audio(duration: float = Query(5.0, gt=0)):
    # [Unchanged from previous]
    with model_lock:
        if model is None:
            return JSONResponse(status_code=500, content={"error": "Model not loaded"})
    try:
        print("Starting recording...")
        recorder = AudioRecorder(sample_rate=SAMPLE_RATE)
        recorder.start_recording(max_duration=duration)
        await asyncio.sleep(duration)
        recorder.stop_recording()
        audio = np.concatenate(recorder.recording).flatten()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(audio, tmp.name, SAMPLE_RATE)
            tmp_path = tmp.name
            print(f"Recorded audio saved to {tmp_path}")
        return {"filename": tmp_path}
    except Exception as e:
        print(f"Error in /record/: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": f"Recording failed: {str(e)}"})

@app.post("/process/")
async def process_audio(file: UploadFile = File(...)):
    global last_keystrokes
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
            last_keystrokes["process"] = keystrokes  # Store for /spectrogram/
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
            # Anomaly detection
            anomaly_count = len([r for r in results if r['prediction'] == 'noise' and r['confidence'] > 0.95])
            if anomaly_count > 5:
                results.append({"alert": "Possible unauthorized activity detected"})
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
    # [Unchanged from previous, includes anomaly detection below]
    global last_keystrokes
    with model_lock:
        if model is None:
            return JSONResponse(status_code=500, content={"error": "Model not loaded"})
    try:
        print("Starting real-time audio stream...")
        recorder = AudioRecorder(sample_rate=SAMPLE_RATE)
        recorder.start_recording(max_duration=duration)
        start_time = time.time()
        chunk_duration = 0.5
        chunk_samples = int(chunk_duration * SAMPLE_RATE)
        results = []
        last_print_time = start_time
        silent_count = 0
        last_keystrokes["stream"] = []

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
                    last_keystrokes["stream"].append(ks)  # Store for /spectrogram/
            if chunk_results:
                results.extend(chunk_results)
                silent_count = 0
                for res in chunk_results:
                    print(f"🔑 Keystroke {res['keystroke']}: {res['prediction']} (Confidence: {res['confidence']:.2f})")
            else:
                silent_count += 1

            if silent_count >= 4:
                print("🔇 Summary: {'characters': 0, 'spaces': 0, 'enters': 0, 'numbers': 0}")
                last_print_time = time.time()
                silent_count = 0

        recorder.stop_recording()
        # Anomaly detection
        anomaly_count = len([r for r in results if r['prediction'] == 'noise' and r['confidence'] > 0.95])
        if anomaly_count > 5:
            results.append({"alert": "Possible unauthorized activity detected"})
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

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket, duration: float = 30.0):
    global last_keystrokes
    await websocket.accept()
    try:
        print("Starting WebSocket audio stream...")
        recorder = AudioRecorder(sample_rate=SAMPLE_RATE)
        recorder.start_recording(max_duration=duration)
        start_time = time.time()
        chunk_duration = 0.5
        chunk_samples = int(chunk_duration * SAMPLE_RATE)
        results = []
        silent_count = 0
        last_keystrokes["ws_stream"] = []

        while time.time() - start_time < duration and websocket.client_state == WebSocket.CONNECTED:
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
                    chunk_result = {
                        "keystroke": len(results) + i,
                        "prediction": prediction,
                        "confidence": confidence,
                        "is_low_confidence": bool(confidence < 0.5)
                    }
                    chunk_results.append(chunk_result)
                    last_keystrokes["ws_stream"].append(ks)  # Store for /spectrogram/
                    await websocket.send_json(chunk_result)
            if chunk_results:
                results.extend(chunk_results)
                silent_count = 0
                for res in chunk_results:
                    print(f"🔑 Keystroke {res['keystroke']}: {res['prediction']} (Confidence: {res['confidence']:.2f})")
            else:
                silent_count += 1

            if silent_count >= 4:
                await websocket.send_json({
                    "summary": {
                        "characters": 0,
                        "spaces": 0,
                        "enters": 0,
                        "numbers": 0
                    }
                })
                silent_count = 0

        recorder.stop_recording()
        anomaly_count = len([r for r in results if r['prediction'] == 'noise' and r['confidence'] > 0.95])
        if anomaly_count > 5:
            results.append({"alert": "Possible unauthorized activity detected"})
        char_count = len([r for r in results if r['prediction'] not in ['noise', 'space', 'enter']])
        space_count = len([r for r in results if r['prediction'] == 'space'])
        enter_count = len([r for r in results if r['prediction'] == 'enter'])
        number_count = len([r for r in results if r['prediction'] in [str(i) for i in range(10)]])
        await websocket.send_json({
            "keystrokes": results,
            "summary": {
                "characters": char_count,
                "spaces": space_count,
                "enters": enter_count,
                "numbers": number_count
            }
        })
        print("WebSocket stream ended.")
    except Exception as e:
        print(f"Error in /ws/stream: {str(e)}")
        print(traceback.format_exc())
        await websocket.send_json({"error": f"Streaming failed: {str(e)}"})
    finally:
        recorder.stop_recording()
        await websocket.close()

@app.get("/spectrogram/{source}/{keystroke_idx}")
async def get_spectrogram(source: str, keystroke_idx: int):
    try:
        if source not in last_keystrokes or keystroke_idx >= len(last_keystrokes[source]):
            return JSONResponse(status_code=404, content={"error": "Keystroke not found"})
        audio = last_keystrokes[source][keystroke_idx]
        S = AudioPreprocessor.create_mel_spectrogram(audio)
        fig, ax = plt.subplots(figsize=(4, 3))
        librosa.display.specshow(S.squeeze(), sr=SAMPLE_RATE, x_axis='time', y_axis='mel', fmax=8000)
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"Spectrogram {keystroke_idx}")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        print(f"Error generating spectrogram: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Spectrogram generation failed: {str(e)}"})

@app.post("/train/")
async def train_model():
    # [Unchanged from previous]
    with model_lock:
        if model is None:
            return JSONResponse(status_code=500, content={"error": "Model not loaded"})
    try:
        print("Starting model training...")
        X, y = [], []
        label_encoder = LabelEncoder()
        label_encoder.fit(CLASSES)

        for label in CLASSES:
            folder = f"training_data/{label}"
            if not os.path.exists(folder):
                print(f"Folder {folder} does not exist")
                continue
            for file in os.listdir(folder):
                if file.endswith(".wav"):
                    file_path = os.path.join(folder, file)
                    audio = AudioPreprocessor.load_audio(file_path)
                    spectrogram = AudioPreprocessor.create_mel_spectrogram(audio)
                    if spectrogram is not None:
                        X.append(spectrogram)
                        y.append(label)
                        AudioPreprocessor.plot_mel_spectrogram(audio, title=f"Training Sample: {label}")

        if not X:
            print("No training data found")
            return JSONResponse(status_code=400, content={"error": "No training data found"})

        X = np.array(X)
        y = label_encoder.transform(y)
        y = tf.keras.utils.to_categorical(y, num_classes=len(CLASSES))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)
        model.model.save("keystroke_model.h5")
        print("Model training completed and saved as keystroke_model.h5")
        return {"status": "Model trained successfully"}
    except Exception as e:
        print(f"Error in /train/: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": f"Training failed: {str(e)}"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
