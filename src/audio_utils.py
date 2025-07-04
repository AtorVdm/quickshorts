# This file contains audio processing utility functions.
import os
from pydub import AudioSegment

# Placeholder for get_audio_duration_ms and other audio utils
# Will be populated in the next step.


def get_audio_duration_ms(audio_file_path: str) -> int:
    """
    Calculates the duration of an audio file in milliseconds.

    Args:
        audio_file_path: Path to the audio file.

    Returns:
        Duration of the audio file in milliseconds.
        Returns 0 if the file is not found or cannot be read.
    """
    if not os.path.exists(audio_file_path): # Added explicit check for pydub
        print(f"Audio file not found: {audio_file_path}")
        return 0
    try:
        return len(AudioSegment.from_file(audio_file_path))
    except FileNotFoundError: # Should be caught by os.path.exists, but good fallback
        print(f"Audio file not found (pydub): {audio_file_path}")
        return 0
    except Exception as e: # Catch other pydub errors
        print(f"Error reading audio file {audio_file_path}: {e}")
        return 0
