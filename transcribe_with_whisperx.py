import os
import whisperx
# import torch # Implicitly used by whisperx

# --- Configuration (Hardcoded Values) ---
INPUT_AUDIO_PATH = "resources/background_aow.mp3"  # Default input audio
PROMPT_TEXT_PATH = "resources/sample_prompt.txt"   # Default prompt file
OUTPUT_TRANSCRIPTION_PATH = "transcription_output.txt" # Default output file

MODEL_SIZE = "large-v2" # WhisperX model size
BATCH_SIZE = 8         # Batch size for transcription (adjust based on VRAM if using GPU)
# For device and compute_type, we'll let whisperx/torch attempt to use CUDA if available,
# otherwise fallback to CPU. This simplifies the script by removing explicit checks here.
# whisperx.load_model default device is "cuda" if available, else "cpu".
# Default compute_type also varies based on device and model. For simplicity, we can rely on those defaults.
# Or, be explicit:
DEVICE = None # None means auto-detect: "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = None # None means auto-detect based on device: e.g. "float16" for CUDA, "int8" for CPU

def run_transcription():
    """
    Transcribes an audio file using WhisperX with an initial prompt,
    using hardcoded paths and settings.
    """
    print(f"Starting transcription for audio: {INPUT_AUDIO_PATH}")

    if not os.path.exists(INPUT_AUDIO_PATH):
        print(f"Error: Input audio file not found at: {INPUT_AUDIO_PATH}")
        return
    if not os.path.exists(PROMPT_TEXT_PATH):
        print(f"Error: Prompt text file not found at: {PROMPT_TEXT_PATH}")
        # Optionally, proceed without a prompt or return
        # For this simplified version, we'll just print an error and not use a prompt if missing.
        initial_prompt_text = None
        print("Warning: Prompt file not found. Proceeding without an initial prompt.")
    else:
        try:
            with open(PROMPT_TEXT_PATH, 'r', encoding='utf-8') as f:
                initial_prompt_text = f.read().strip()
            if not initial_prompt_text:
                print(f"Warning: Prompt file {PROMPT_TEXT_PATH} is empty. Proceeding without an initial prompt.")
                initial_prompt_text = None
        except Exception as e:
            print(f"Error reading prompt file {PROMPT_TEXT_PATH}: {e}. Proceeding without an initial prompt.")
            initial_prompt_text = None

    print(f"Loading WhisperX model: {MODEL_SIZE}...")
    try:
        # Let whisperx handle device and compute_type defaults if None
        model_device = DEVICE
        model_compute_type = COMPUTE_TYPE

        # Attempt to intelligently set defaults if not specified, mimicking previous logic
        if model_device is None:
            try:
                import torch
                if torch.cuda.is_available():
                    model_device = "cuda"
                    if model_compute_type is None: model_compute_type = "float16"
                else:
                    model_device = "cpu"
                    if model_compute_type is None: model_compute_type = "int8"
            except ImportError:
                model_device = "cpu" # Fallback if torch isn't even there
                if model_compute_type is None: model_compute_type = "int8"

        print(f"Attempting to load model with device='{model_device}' and compute_type='{model_compute_type}'")

        model = whisperx.load_model(
            MODEL_SIZE,
            device=model_device,
            compute_type=model_compute_type,
        )
        print("Model loaded.")
    except Exception as e:
        print(f"Fatal Error: Could not load WhisperX model: {e}")
        print("This could be due to missing dependencies (like PyTorch, CUDA), model download issues, or insufficient disk space/memory.")
        print("Please ensure all dependencies are installed and the environment has sufficient resources.")
        return

    print(f"Loading audio from: {INPUT_AUDIO_PATH}...")
    try:
        audio = whisperx.load_audio(INPUT_AUDIO_PATH)
        print("Audio loaded.")
    except Exception as e:
        print(f"Error loading audio: {e}")
        return

    print("Starting transcription...")
    try:
        result = model.transcribe(audio, batch_size=BATCH_SIZE, initial_prompt=initial_prompt_text)
        print("Transcription complete.")
    except Exception as e:
        print(f"Error during transcription: {e}")
        return

    print(f"Saving transcription to: {OUTPUT_TRANSCRIPTION_PATH}...")
    try:
        with open(OUTPUT_TRANSCRIPTION_PATH, 'w', encoding='utf-8') as f:
            if result and "segments" in result:
                for segment in result["segments"]:
                    f.write(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text'].strip()}\n")
            else:
                f.write("No segments transcribed or result format unexpected.\n")
        print("Transcription saved.")
    except Exception as e:
        print(f"Error saving transcription: {e}")

if __name__ == "__main__":
    print("Running WhisperX Transcription Script with hardcoded values...")
    print("---------------------------------------------------------")
    print(f"Input Audio: {INPUT_AUDIO_PATH}")
    print(f"Prompt File: {PROMPT_TEXT_PATH}")
    print(f"Output File: {OUTPUT_TRANSCRIPTION_PATH}")
    print(f"Model: {MODEL_SIZE}, Batch Size: {BATCH_SIZE}")
    print("---------------------------------------------------------")

    # Note: The script relies on 'resources/background_aow.mp3' and
    # 'resources/sample_prompt.txt' existing for default operation.
    # If these files are not present, the script will print errors or warnings.
    # The 'sample_prompt.txt' was part of a previous plan. If it doesn't exist,
    # the script will attempt to run without a prompt if the file is missing/empty.
    # Example content for resources/sample_prompt.txt:
    # "This is a game. Let's play. Victory. Defeat. Strategy and tactics. Audio from Age of Wonders."

    # Crucially, this script WILL FAIL if whisperx and its dependencies (PyTorch, etc.)
    # are not installed due to the ongoing disk space issue.
    run_transcription()
    print("\nScript finished.")
