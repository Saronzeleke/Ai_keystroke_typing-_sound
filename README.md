# 🎧 AI Keystroke Typing Sound Detection

A deep learning-based system that predicts keystrokes from audio using mel spectrograms and a CNN. This project enables real-time keypress classification, anomaly detection, and visual feedback via a web interface and WebSocket communication.

## 🚀 Features
- 🎙️ Audio-based keystroke classification (0-9, a-z, space, enter, noise)
- 🧠 CNN model with mel spectrogram input
- 🌐 Real-time prediction over WebSocket (`/ws/stream`)
- 📈 Spectrogram visualization for each keystroke
- ⚠️ Anomaly detection (e.g., excessive noise)
- 🖥️ Web interface with audio streaming and upload
- ✅ Unit and integration tests included

## 🛠 Requirements

**Software**
- Python 3.8+
- FastAPI, TensorFlow, librosa, etc. (see `requiremnt.txt`)
- Modern browser (Chrome/Firefox)

**Hardware**
- Microphone (e.g., DroidCam)
- Windows 10/11 with 8GB+ RAM

# 📦 Setup

#. **Clone the repo**
   ```bash
   git clone https://github.com/Saronzeleke/Ai_keystroke_typing-_sound.git
   cd Ai_keystroke_typing-_sound
2. Set up virtual environment
python -m venv venv
.\venv\Scripts\activate
3.Install dependencies
pip install -r requiremnt.txt
4. (Optional) Configure DroidCam
   * Install on PC and phone

 # Set "DroidCam Virtual Audio" as default mic
 #  🎙️ Usage
Step 1: Generate Training Data
python create_traingin_data_realtime.py
Prompts you to record each key.

Saves samples under training_data/

 # Step 2: Run the App
  python main.py
  Starts a FastAPI server with endpoints for:

  * /record/

  *  /process/

  *  /stream/

  * /ws/stream (WebSocket)

  *   /train/ (model training)
 # Step 3: Open index.html
  python -m http.server 8080
then when it start type http://localhost:8080/index.html
in your defualt browser
Upload audio, view predictions, see spectrograms

 # 🧪 Testing
pytest
  # 📂 File Structure
create_traingin_data_realtime.py: Data collection

main.py: FastAPI backend

index.html: Frontend interface

test_keystroke.py: Test scripts

requiremnt.txt: Dependencies list

🤖 Author
# Saron Zeleke
📄 License
MIT License

🔗 Repo: github.com/Saronzeleke/Ai_keystroke_typing-_sound






