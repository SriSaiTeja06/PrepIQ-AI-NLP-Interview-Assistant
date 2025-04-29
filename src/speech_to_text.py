import os
import tempfile
import whisper # Import Whisper
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeechToText:
    def __init__(self, model_size: str = "base"):
        """
        Initialize the Speech-to-Text service using Whisper.

        Args:
            model_size (str): The Whisper model size to load 
                              (e.g., "tiny", "base", "small", "medium", "large").
                              Defaults to "base".
        """
        try:
            logger.info(f"Loading Whisper model: {model_size}")
            # Load the specified Whisper model
            self.model = whisper.load_model(model_size)
            logger.info(f"Whisper model '{model_size}' loaded successfully.")
        except Exception as e:
            logger.exception(f"Failed to load Whisper model '{model_size}'. Ensure ffmpeg is installed and model exists.")
            raise RuntimeError(f"Could not initialize Whisper model '{model_size}'") from e

    def transcribe_audio(self, audio_content: bytes) -> Optional[str]:
        """
        Transcribe audio content to text using the loaded Whisper model.

        Args:
            audio_content: The audio content (bytes) to transcribe.

        Returns:
            The transcribed text or None if transcription fails.
        """
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".opus") as temp_audio_file:
                temp_audio_file.write(audio_content)
                temp_file_path = temp_audio_file.name
            
            logger.info(f"Transcribing temporary audio file: {temp_file_path}")

            result = self.model.transcribe(temp_file_path, fp16=False) 

            transcript = result.get("text")

            if transcript is None:
                 logger.warning(f"Whisper transcription result was empty or invalid for {temp_file_path}.")
                 return None

            final_transcript = transcript.strip()
            logger.info(f"Transcription result: {final_transcript[:100]}...")
            return final_transcript

        except Exception as e:
            logger.exception(f"Error during Whisper transcription process: {e}")
            return None
        finally:
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.info(f"Deleted temporary audio file: {temp_file_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to delete temporary audio file {temp_file_path}: {cleanup_error}")


# Example usage (updated for Whisper)
def main():
    stt = SpeechToText(model_size="base") 
    
    # Example audio content (replace with actual audio bytes from a file)
    # e.g., with open("my_audio.opus", "rb") as f: audio_content = f.read()
    # For testing, create dummy bytes (won't produce meaningful transcript)
    audio_content = b'\x00' * 48000 * 5 # Dummy 5 seconds of silence (adjust format if needed)
    
    if not audio_content:
         print("Placeholder audio_content is empty. Replace with real audio bytes.")
         return

    # Transcribe audio
    print("Starting transcription...")
    transcript = stt.transcribe_audio(audio_content)
    
    if transcript is not None:
        print(f"Transcribed text: {transcript}")
    else:
        print("Transcription failed.")

if __name__ == "__main__":
    print("SpeechToText class defined. To test, uncomment main() and provide real audio bytes.")
