import os
import whisperx
import pytest

# Define the paths for the audio file, prompt file, and output file
AUDIO_FILE_PATH = "resources/background_aow.mp3"
PROMPT_FILE_PATH = "resources/sample_prompt.txt"
OUTPUT_TRANSCRIPTION_PATH = "whisperx_transcription_output.txt"

# Determine compute type based on GPU availability
DEVICE = "cuda" # "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" # "float16" if torch.cuda.is_available() else "int8"
# It's good practice to use a specific model version if possible, e.g., "large-v2"
# For this test, we'll use the default "large" model that whisperX uses.
MODEL_SIZE = "large-v2" # Using large-v2 as a robust default

def test_transcribe_audio_with_prompt():
    """
    Tests transcribing an audio file using whisperx with a prompt from a text file.
    The transcription output is saved to a file in the project root.
    """
    # Ensure the audio and prompt files exist
    assert os.path.exists(AUDIO_FILE_PATH), f"Audio file not found: {AUDIO_FILE_PATH}"
    assert os.path.exists(PROMPT_FILE_PATH), f"Prompt file not found: {PROMPT_FILE_PATH}"

    # Read the prompt text
    with open(PROMPT_FILE_PATH, 'r') as f:
        prompt_text = f.read()

    # Load the whisper model
    # Note: WhisperX automatically downloads the model if not found locally.
    # batch_size can be adjusted based on VRAM, 4 is usually a safe default for "large" models
    model = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE)

    # Load audio
    audio = whisperx.load_audio(AUDIO_FILE_PATH)

    # Transcribe with prompt (initial_prompt)
    # WhisperX's transcribe method uses `initial_prompt` for this purpose.
    result = model.transcribe(audio, batch_size=4, initial_prompt=prompt_text)

    # Save the transcription segments to the output file
    with open(OUTPUT_TRANSCRIPTION_PATH, 'w') as f:
        for segment in result["segments"]:
            f.write(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}\n")

    # Assert that the output file was created
    assert os.path.exists(OUTPUT_TRANSCRIPTION_PATH), f"Output transcription file not created: {OUTPUT_TRANSCRIPTION_PATH}"

    # Optional: Clean up the created output file after the test
    # if os.path.exists(OUTPUT_TRANSCRIPTION_PATH):
    #     os.remove(OUTPUT_TRANSCRIPTION_PATH)

if __name__ == "__main__":
    # This allows running the test directly, for example, for debugging.
    # You would typically run this with `pytest test_whisperx_transcription.py`
    pytest.main([__file__])
