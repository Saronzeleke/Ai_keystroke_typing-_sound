# Example test file (test_keystroke.py)
import pytest
from your_module import AudioPreprocessor, KeystrokeCNN
import numpy as np

# 1. Test Audio Preprocessing
def test_spectrogram_shape():
    dummy_audio = np.random.rand(4410)  # 0.1 sec of audio
    spectrogram = AudioPreprocessor.create_mel_spectrogram(dummy_audio)
    assert spectrogram.shape == (128, 87, 1), "Spectrogram shape mismatch"

# 2. Test Model I/O
def test_model_prediction():
    model = KeystrokeCNN()
    dummy_input = np.random.rand(1, 128, 87, 1)
    pred_idx, _ = model.predict(dummy_input)
    assert 0 <= pred_idx < 39, "Invalid class prediction"

# 3. Integration Test (requires mock)
@pytest.mark.asyncio
async def test_api_endpoint(mock_audio_file):
    # Use FastAPI TestClient
    response = await client.post("/process/", files={"file": mock_audio_file})
    assert response.status_code == 200
    assert "summary" in response.json()