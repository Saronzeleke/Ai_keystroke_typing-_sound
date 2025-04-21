# Ai_keystroke_typing-_sound
Keystroke Recognition System
Overview
The Keystroke Recognition System is a deep learning-based cybersecurity application that predicts keystrokes from audio recordings. It uses a Convolutional Neural Network (CNN) to classify 38 keystroke classes (0-9, a-z, space, enter, noise) captured via a microphone. The system provides real-time predictions through a WebSocket, displays spectrograms for each keystroke, detects anomalies (e.g., excessive noise), and offers a web interface for user interaction. It is designed to monitor typing patterns for cybersecurity applications.
Key Features

Audio-based keystroke classification using a CNN with mel spectrograms.
Real-time predictions via WebSocket (/ws/stream).
Spectrogram visualization for keystroke analysis.
Anomaly detection for potential tampering (e.g., >5 high-confidence noise predictions).
Web interface with file upload, streaming, and training capabilities.
Unit and integration tests for reliability.

# Prerequisites
Software:

Python 3.8+
Web Browser (e.g., Chrome, Firefox)
IDE (e.g., VS Code)

Hardware:

Microphone (DroidCam recommended as virtual mic)
PC with Windows 10/11, 8GB+ RAM

Dependencies
Listed in requirements.txt:

numpy==1.24.3
librosa==0.10.0
matplotlib==3.7.1
tensorflow==2.10.0
fastapi==0.95.1
uvicorn==0.22.0
sounddevice==0.4.6
soundfile==0.12.1
scipy==1.10.1
noisereduce==2.0.1
scikit-learn==1.2.2
pytest
websocket-client

# Setup Instructions:

Place Files:Ensure the following files are in C:\Users\USER\Desktop\Ai keystroke:

create_training_data_corrected.py
keystroke_app_with_spectrogram.py
index.html
requirements.txt
test_keystroke_app.py


# Set Up Virtual Environment:
cd C:\Users\USER\Desktop\Ai keystroke
python -m venv venv
.\venv\Scripts\activate


# Install Dependencies:
pip install -r requirements.txt
pip install pytest websocket-client


# Configure DroidCam:

Install DroidCam on PC and phone (DroidCam website).
Set DroidCam as default microphone:
Windows: Right-click speaker icon > Sounds > Recording > Select “DroidCam Virtual Audio” > Set as Default.


Test audio:python -c "import sounddevice as sd; print(sd.query_devices())"





How to Run
# Step 1: Generate Training Data

Script: create_training_data_corrected.py
Purpose: Records 10 audio samples (0.1s each) for 38 classes, saving WAVs in training_data/.
Run:python create_training_data_corrected.py


Process:
Prompts to press Enter and type each key (or make noise for noise).
Saves 380 WAVs (e.g., training_data/a/a_0.wav).


Verify:dir training_data


Should show 38 folders with 10 WAVs each.



# Step 2: Run the Main Application

Script: keystroke_app_with_spectrogram.py
Purpose: Hosts FastAPI server for training, prediction, and spectrogram display.
Run:python keystroke_app_with_spectrogram.py


Endpoints:
/record/?duration=5: Records 5s audio.
/process/: Processes uploaded WAV.
/stream/?duration=30: Streams 30s audio.
/ws/stream: WebSocket for real-time predictions.
/spectrogram/{source}/{keystroke_idx}: Returns spectrogram PNG.
/train/: Trains CNN, saves keystroke_model.h5.



# Step 3: Access the Frontend

File: index.html
Purpose: Provides UI for recording, processing, streaming, and training.
Run:python -m http.server 8080


Access: Open http://localhost:8080 in a browser.
Actions:
Record Audio: Calls /record/.
Process Audio: Uploads WAV, shows predictions and spectrograms.
Start Streaming: Uses /ws/stream for live predictions.
Train Model: Calls /train/.



# Step 4: Run Tests

Script: test_keystroke_app.py
Purpose: Tests audio processing, CNN, and endpoints.
Run:pytest test_keystroke_app.py -v


 # Tests:
Unit tests: Audio preprocessing, CNN prediction.
Integration tests: API endpoints, WebSocket.


# Output: Should pass 11 tests.

Troubleshooting

 # IDE Errors:
Re-select interpreter: VS Code > Ctrl+Shift+P > “Python: Select Interpreter” > .\venv\Scripts\python.exe.
Install dependencies:pip install -r requirements.txt




 # No Audio:
Set DroidCam as default mic (Windows Sound settings).
Test: python -c "import sounddevice as sd; print(sd.query_devices())".


# WebSocket Issues:
Check browser console (F12 > Console).
Ensure server runs (http://localhost:8000), firewall allows port 8000.


# Training Fails:
Verify training_data/ has 38 folders, 10 WAVs each.
Check keystroke_model.h5 path.


# Test Fails:
Ensure FastAPI server is running.
Share pytest output for debugging.



# License
MIT License. Saron Zeleke
#Contact me 
 Email: sharonkuye369@gmail.com

