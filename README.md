# AI Keystroke Typing Sound Detection — Detailed Documentation

This documentation provides an in-depth explanation of the main components, code structure, and usage for the AI Keystroke Typing Sound Detection project.

---

## Table of Contents

- [System Overview](#system-overview)
- [Core Components](#core-components)
  - [1. Backend (main.py)](#1-backend-mainpy)
  - [2. Data Recorder (create_traingin_data_realtime.py)](#2-data-recorder-createtraingin_data_realtimepy)
  - [3. Frontend (index.html)](#3-frontend-indexhtml)
  - [4. Tests (test_keystroke.py)](#4-tests-test_keystrokepy)
- [How to Run](#how-to-run)
- [Endpoints & API](#endpoints--api)
- [Model Architecture](#model-architecture)
- [Extending & Customizing](#extending--customizing)

---

## System Overview

This project detects keyboard keystrokes from audio using a deep learning model (CNN with mel spectrogram input). It features:

- Real-time audio recording and keystroke detection.
- FastAPI backend with REST and WebSocket endpoints.
- A web-based frontend for user interaction.
- Scripts and endpoints for training, testing, and data collection.

---

## Core Components

### 1. Backend (`main.py`)

#### Major Classes & Functions

- **AudioRecorder**: 
  - Handles real-time microphone recording using `sounddevice`.
  - Methods: `start_recording`, `stop_recording`, `save_recording`.

- **AudioPreprocessor**:
  - Loads audio, performs noise reduction, detects keystroke events, and creates mel spectrograms.
  - Key static methods:
    - `load_audio(path, sr)`
    - `reduce_noise(audio, sr)`
    - `detect_keystrokes(audio, threshold, min_silence, sr)`
    - `extract_keystroke(audio, sr, ...)`
    - `create_mel_spectrogram(audio, sr, n_mels)`
    - `plot_mel_spectrogram(audio, title, ...)`

- **KeystrokeCNN**:
  - Defines and manages the Convolutional Neural Network for keystroke classification.
  - Methods: `build_model`, `train`, `save_model`, `load_model`, `predict`.

- **prepare_training_data(data_dir)**:
  - Prepares and preprocesses all training data for model training.

#### FastAPI Endpoints

- **POST `/record/`**  
  Record audio from microphone for a given duration.

- **POST `/process/`**  
  Process uploaded audio, detect and classify keystrokes, return predictions.

- **POST `/stream/`**  
  Real-time audio stream for keystroke prediction.

- **POST `/train/`**  
  Triggers model training using the available labeled data.

#### Server Startup

- Run `main.py` directly to start the FastAPI server:
  ```bash
  python main.py
  ```
  The server will listen on port 8000 by default.

---

### 2. Data Recorder (`create_traingin_data_realtime.py`)

*Note: File contents not found, but by naming convention and references:*

- This script is used to record labeled keystroke sounds for data collection.
- Likely records audio samples and saves them to disk with the appropriate label (e.g., for each key).

---

### 3. Frontend (`index.html`)

- **Technology**: HTML + Tailwind CSS + Vanilla JS.
- **Main Features**:
  - Record audio via browser.
  - Upload audio files for classification.
  - Real-time streaming with WebSocket.
  - Trigger model training.
  - Display results with confidence bars, labels, and summary statistics.
  - Light/Dark theme toggle.

- **JavaScript Highlights**:
  - Functions for recording audio, uploading files, streaming, and model training.
  - UI helpers for displaying results and notifications.
  - WebSocket logic for real-time updates.

---

### 4. Tests (`test_keystroke.py`)

- **Framework**: unittest (Python)
- **Coverage**:
  - Audio recording and file I/O tests.
  - Audio preprocessing (load, noise reduction, keystroke detection).
  - Model construction and prediction.
  - FastAPI endpoint integration tests (`/record/`, `/process/`, `/spectrogram/`, WebSocket).
- **How to Run**:
  ```bash
  pytest
  ```
  or
  ```bash
  python test_keystroke.py
  ```

---

## How to Run

### 1. Install Dependencies

```bash
python -m venv venv
.\venv\Scripts\activate    # On Windows
pip install -r requiremnt.txt
```

### 2. Start the Backend Server

```bash
python main.py
# FastAPI server starts on http://localhost:8000
```

### 3. Start the Frontend

```bash
python -m http.server 8080
# Then open http://localhost:8080/index.html in your browser
```

### 4. Use the Web Interface

- Record, upload, or stream audio.
- View predictions and confidence.
- Trigger model training from the web UI.

---

## Endpoints & API

| Endpoint             | Method | Description                               |
|----------------------|--------|-------------------------------------------|
| `/record/`           | POST   | Record audio from mic (duration param)    |
| `/process/`          | POST   | Upload audio file for keystroke detection |
| `/stream/`           | POST   | Real-time audio stream prediction         |
| `/train/`            | POST   | Train the model on all labeled data       |
| `/ws/stream`         | WS     | WebSocket for real-time predictions       |

---

## Model Architecture

- **Input**: Mel spectrogram (128 bins x 87 time steps x 1 channel)
- **Layers**: 
  - Multiple Conv2D layers with BatchNorm, MaxPooling, and Dropout.
  - Flatten + Dense layers ending with softmax for classification.
- **Output**: Class probabilities for keystroke labels.

---

## Extending & Customizing

- **Add new keys**: Add label folders in your data directory, retrain the model.
- **Change audio parameters**: Modify `SAMPLE_RATE`, `DURATION`, `N_MELS`, etc. in `main.py`.
- **Front-end customization**: Edit `index.html` for UI changes.
- **Model improvement**: Edit `KeystrokeCNN` for deeper or different architectures.

---

## Testing

- Unit and integration tests are in `test_keystroke.py`.
- Use `pytest` or `unittest` as shown above.

---

## Troubleshooting

- **No audio device**: Ensure your microphone is enabled and accessible.
- **Model not loaded**: Train a model first via `/train/` if it does not exist.
- **WebSocket errors**: Ensure backend is running and CORS is correctly set up.

---

## Credits

**Author:** Saron Zeleke  
Please credit the original author if you use or extend this work.
