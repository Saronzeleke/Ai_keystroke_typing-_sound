
import unittest
import numpy as np
import librosa
import os
import tempfile
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import soundfile as sf
import websocket
import json
import asyncio
from keystroke_app_with_spectrogram import app, AudioPreprocessor, AudioRecorder, KeystrokeCNN, SAMPLE_RATE, CLASSES
from create_training_data_corrected import record_audio

class TestKeystrokeRecognition(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.sample_rate = SAMPLE_RATE
        self.classes = CLASSES
        self.temp_dir = tempfile.mkdtemp()
        self.audio_data = np.random.randn(int(0.1 * self.sample_rate))  # 0.1s random audio
        self.wav_file = os.path.join(self.temp_dir, "test.wav")
        sf.write(self.audio_data, self.wav_file, self.sample_rate)

    def tearDown(self):
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)

    # Unit Tests for create_training_data_corrected.py
    def test_record_audio(self):
        output_file = os.path.join(self.temp_dir, "recorded.wav")
        with patch("sounddevice.rec", return_value=self.audio_data), patch("sounddevice.wait"):
            record_audio(0.1, self.sample_rate, output_file)
        self.assertTrue(os.path.exists(output_file))
        audio, sr = librosa.load(output_file, sr=self.sample_rate)
        self.assertEqual(sr, self.sample_rate)
        self.assertEqual(len(audio), int(0.1 * self.sample_rate))

    # Unit Tests for AudioPreprocessor
    def test_load_audio(self):
        audio = AudioPreprocessor.load_audio(self.wav_file, sr=self.sample_rate)
        self.assertIsInstance(audio, np.ndarray)
        self.assertEqual(len(audio), int(0.1 * self.sample_rate))

    def test_reduce_noise(self):
        audio = AudioPreprocessor.reduce_noise(self.audio_data, sr=self.sample_rate)
        self.assertIsInstance(audio, np.ndarray)
        self.assertEqual(len(audio), len(self.audio_data))

    def test_detect_keystrokes(self):
        # Simulate a single peak
        audio = np.zeros(int(0.1 * self.sample_rate))
        audio[2000:2500] = 0.1  # Simulated keystroke
        keystrokes = AudioPreprocessor.detect_keystrokes(audio, threshold=0.05, min_silence=0.05, sr=self.sample_rate)
        self.assertIsInstance(keystrokes, list)
        self.assertGreaterEqual(len(keystrokes), 1)
        self.assertEqual(len(keystrokes[0]), int(0.1 * self.sample_rate))

    def test_create_mel_spectrogram(self):
        spectrogram = AudioPreprocessor.create_mel_spectrogram(self.audio_data, sr=self.sample_rate)
        self.assertIsInstance(spectrogram, np.ndarray)
        self.assertEqual(spectrogram.shape, (128, 87, 1))

    # Unit Tests for KeystrokeCNN
    def test_build_model(self):
        cnn = KeystrokeCNN()
        model = cnn.build_model()
        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape, (None, 128, 87, 1))
        self.assertEqual(model.output_shape, (None, len(self.classes)))

    def test_predict(self):
        cnn = KeystrokeCNN()
        spectrogram = np.random.randn(128, 87, 1)
        with patch.object(cnn.model, "predict", return_value=np.ones((1, len(self.classes)))):
            pred_idx, confidences = cnn.predict(spectrogram)
            self.assertIsInstance(pred_idx, np.int64)
            self.assertIsInstance(confidences, np.ndarray)
            self.assertEqual(len(confidences), len(self.classes))

    # Integration Tests for FastAPI Endpoints
    def test_record_endpoint(self):
        with patch("sounddevice.rec", return_value=self.audio_data), patch("sounddevice.wait"):
            response = self.client.post("/record/?duration=0.1")
            self.assertEqual(response.status_code, 200)
            self.assertIn("filename", response.json())
            self.assertTrue(os.path.exists(response.json()["filename"]))

    def test_process_endpoint(self):
        with open(self.wav_file, "rb") as f:
            response = self.client.post("/process/", files={"file": ("test.wav", f, "audio/wav")})
            self.assertEqual(response.status_code, 200)
            result = response.json()
            self.assertIn("summary", result)
            self.assertIn("characters", result["summary"])

    def test_spectrogram_endpoint(self):
        # Simulate stored keystroke
        global last_keystrokes
        last_keystrokes["process"] = [self.audio_data]
        response = self.client.get("/spectrogram/process/0")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")

    # Integration Test for WebSocket
    def test_websocket_stream(self):
        async def run_websocket():
            ws = websocket.WebSocket()
            ws.connect("ws://localhost:8000/ws/stream")
            message = json.loads(ws.recv())
            ws.close()
            return message

        with patch("sounddevice.rec", return_value=self.audio_data), patch("sounddevice.wait"):
            loop = asyncio.get_event_loop()
            message = loop.run_until_complete(run_websocket())
            self.assertIsInstance(message, dict)

if __name__ == "__main__":
    unittest.main()
