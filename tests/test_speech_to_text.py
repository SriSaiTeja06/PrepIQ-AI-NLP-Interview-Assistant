import os
import pytest
from src.speech_to_text import SpeechToText
import logging
from unittest.mock import patch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_speech_to_text_init():
    """Test SpeechToText initialization."""
    stt = SpeechToText()
    assert stt is not None
    assert stt.client is not None
    assert stt.encoding.name == "LINEAR16"
    assert stt.sample_rate_hertz == 16000
    assert stt.language_code == "en-US"

def test_custom_vocabulary():
    """Test custom vocabulary creation."""
    stt = SpeechToText()
    vocab = stt._create_custom_vocabulary()
    assert "data_science" in vocab
    assert "software_engineering" in vocab
    assert "devops" in vocab
    assert len(vocab["data_science"]) > 0

def test_transcribe_audio():
    """Test audio transcription."""
    stt = SpeechToText()
    
    # Create a simple test audio file (silence for testing)
    import wave
    import numpy as np
    
    # Create a 1-second silence audio
    duration = 1  # seconds
    sample_rate = 16000
    samples = np.zeros(int(duration * sample_rate), dtype=np.int16)
    
    # Save to WAV file
    test_file = "test_audio.wav"
    with wave.open(test_file, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())
    
    try:
        # Read the audio file
        with open(test_file, 'rb') as audio_file:
            audio_content = audio_file.read()
            
        # Mock the response
        class MockResponse:
            def __init__(self):
                self.results = [MockResult()]
        
        class MockResult:
            def __init__(self):
                self.alternatives = [MockAlternative()]
                self.is_final = True
        
        class MockAlternative:
            def __init__(self):
                self.transcript = ""
                self.confidence = 0.9
        
        # Transcribe
        with patch.object(stt.client, 'recognize', return_value=MockResponse()):
            transcript = stt.transcribe_audio(audio_content, domain="data_science")
        
        # Verify the transcription process worked
        assert transcript is not None
        # For testing, we expect silence to return an empty string
        assert transcript == ""
        
    finally:
        # Clean up test file
        if os.path.exists(test_file):
            os.remove(test_file)

def test_stream_transcribe():
    """Test streaming transcription."""
    stt = SpeechToText()
    
    # Create a simple test audio file (silence for testing)
    import wave
    import numpy as np
    
    # Create a 1-second silence audio
    duration = 1  # seconds
    sample_rate = 16000
    samples = np.zeros(int(duration * sample_rate), dtype=np.int16)
    
    # Save to WAV file
    test_file = "test_audio.wav"
    with wave.open(test_file, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())
    
    try:
        # Read the audio file in chunks
        chunk_size = 1024
        def audio_generator():
            with open(test_file, 'rb') as audio_file:
                while True:
                    chunk = audio_file.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
        
        # Mock the response
        class MockResult:
            def __init__(self):
                self.alternatives = [MockAlternative()]
                self.is_final = True
        
        class MockAlternative:
            def __init__(self):
                self.transcript = ""
                self.confidence = 0.9
        
        # Transcribe
        with patch.object(stt.client, 'streaming_recognize', return_value=[MockResult()]):
            transcript = stt.stream_transcribe(audio_generator(), domain="data_science")
        
        # Verify the transcription process worked
        assert transcript is not None
        # For testing, we expect silence to return an empty string
        assert transcript == ""
        
    finally:
        # Clean up test file
        if os.path.exists(test_file):
            os.remove(test_file)

if __name__ == "__main__":
    pytest.main(["-v", __file__])
