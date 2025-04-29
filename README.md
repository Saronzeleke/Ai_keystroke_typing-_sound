# 🎧 AI Keystroke Typing Sound Detection

A deep learning-based system that predicts **keystrokes from sound** using mel spectrograms and a Convolutional Neural Network (CNN). This project supports **real-time keystroke classification**, **anomaly detection**, and **visual feedback** via a web interface and WebSocket communication.

---

## 🎯 Purpose

This project aims to recognize individual keyboard keystrokes using only the **sound** they produce. By capturing audio through a microphone and processing it with a CNN trained on mel spectrograms, the system can identify pressed keys such as letters, numbers, space, and enter — all in real time.

### Key Objectives:
- 🔊 Detect and classify keystrokes from audio.
- 🧠 Use a CNN model trained on mel spectrogram features.
- ⚡ Enable real-time prediction using WebSockets.
- 🖥️ Provide visual feedback via a browser-based frontend.

### Potential Applications:
- 🔐 Acoustic side-channel attack demonstration.
- 🧪 Audio-based input classification research.
- 🖱️ Sound-controlled UI or accessibility tools.

---

## 🚀 Features

- 🎙️ Audio-based keystroke classification (`0–9`, `a–z`, `space`, `enter`, `noise`)
- 🧠 CNN model using mel spectrogram input
- 🌐 Real-time predictions over WebSocket (`/ws/stream`)
- 📈 Spectrogram visualization for each keypress
- ⚠️ Anomaly detection (e.g., noisy input)
- 🖥️ Web interface for live or uploaded audio
- ✅ Unit and integration tests included

---

## 🛠️ Requirements

### Software:
- Python 3.8+
- Packages: FastAPI, TensorFlow, librosa, etc. (see `requiremnt.txt`)
- Modern browser (Chrome/Firefox)

### Hardware:
- Microphone (e.g., DroidCam, built-in)
- Windows 10/11 with 8GB+ RAM

---

## 📦 Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Saronzeleke/Ai_keystroke_typing-_sound.git
   cd Ai_keystroke_typing-_sound
2.**Set up a virtual environment:**
    python -m venv venv
  .\venv\Scripts\activate
3.**Install dependencies:**
   pip install -r requiremnt.txt
4.**(Optional)Configure DroidCam:**
   Install DroidCam on both PC and phone
   Set "DroidCam Virtual Audio" as your default microphone
🎙️ Usage
Step 1: Run the App
 python main.py
Starts a FastAPI backend server with the following endpoints:

**. /record/**

**. /process/**

**. /stream/**

**. /ws/stream (WebSocket for real-time predictions)**

**./train/ (Model training)**

Step 2: Open the Frontend
Start a local server:
python -m http.server 8080
Then open:
http://localhost:8080/index.html
Use the browser interface to upload or stream audio and see predicted keystrokes.
🧪 Testing
Run the tests using:
pytest
📂 File Structure

| File                         | Description                          |
|------------------------------|--------------------------------------|
| `create_traingin_data_realtime.py` | Training data recorder             |
| `main.py`                    | FastAPI backend and WebSocket server |
| `index.html`                 | Frontend interface                   |
| `test_keystroke.py`          | Unit tests                           |
| `requiremnt.txt`             | Dependency list                      |

# 👨‍💻 Author
Saron Zeleke
# GitHub: github.com/Saronzeleke

# 📄 License
This project is licensed under the MIT License.

You are free to:

✅ Use this code commercially or privately

✅ Modify and redistribute the project

✅ Fork it and build your own version
# Please credit the original author Saron Zeleke if you use or adapt this work.
🔗 Repo: github.com/Saronzeleke/Ai_keystroke_typing-_sound

