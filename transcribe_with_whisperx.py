import os
import argparse
import whisperx
# import torch # WhisperX will import torch; ensure it's available if specifying device="cuda"

# --- Configuration ---
# Model size: "tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"
# Using large-v2 as a robust default. User might want to change this for speed/accuracy trade-off.
DEFAULT_MODEL_SIZE = "large-v2"
DEFAULT_BATCH_SIZE = 16 # Adjust based on VRAM if using GPU. 16 is often fine for large models.
DEFAULT_COMPUTE_TYPE = "int8" # "float16" for GPU (faster, more VRAM), "int8" for CPU (wider compatibility)
DEFAULT_DEVICE = "cpu" # "cuda" if GPU is available and configured, otherwise "cpu"

# --- Helper Functions ---
def _check_file_exists(filepath: str, file_description: str) -> None:
    """Checks if a file exists and raises FileNotFoundError if not."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{file_description} not found at: {filepath}")

def _determine_device_and_compute_type() -> tuple[str, str]:
    """
    Determines the device (CPU/GPU) and compute type for WhisperX.
    Prioritizes GPU if available.
    """
    device = DEFAULT_DEVICE
    compute_type = DEFAULT_COMPUTE_TYPE
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16" # Or "bfloat16" on newer GPUs
            print("CUDA is available. Using GPU (cuda) with compute_type float16.")
        else:
            print("CUDA not available. Using CPU with compute_type int8.")
    except ImportError:
        print("PyTorch not found. Defaulting to CPU with compute_type int8 for WhisperX.")
    return device, compute_type


# --- Main Transcription Logic ---
def transcribe_audio(
    audio_path: str,
    prompt_path: str,
    output_path: str,
    model_size: str = DEFAULT_MODEL_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = None, # Allow override
    compute_type: str = None # Allow override
) -> None:
    """
    Transcribes an audio file using WhisperX with an initial prompt from a text file.

    Args:
        audio_path: Path to the input audio file.
        prompt_path: Path to the text file containing the initial prompt.
        output_path: Path to save the transcription output.
        model_size: WhisperX model size (e.g., "large-v2").
        batch_size: Batch size for transcription.
        device: Device to run on ("cpu" or "cuda"). Auto-detected if None.
        compute_type: Compute type for the model (e.g., "int8", "float16"). Auto-detected if None.
    """
    print(f"Starting transcription for audio: {audio_path}")
    _check_file_exists(audio_path, "Input audio file")
    _check_file_exists(prompt_path, "Prompt text file")

    # Determine device and compute type if not overridden
    effective_device, effective_compute_type = _determine_device_and_compute_type()
    if device:
        effective_device = device
    if compute_type:
        effective_compute_type = compute_type

    print(f"Using device: {effective_device}, compute_type: {effective_compute_type}, model: {model_size}")

    # Read the prompt text
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            initial_prompt_text = f.read().strip()
        if not initial_prompt_text:
            print(f"Warning: Prompt file {prompt_path} is empty. Proceeding without an initial prompt.")
            initial_prompt_text = None # WhisperX handles None as no prompt
    except Exception as e:
        print(f"Error reading prompt file {prompt_path}: {e}. Proceeding without an initial prompt.")
        initial_prompt_text = None

    # Load the WhisperX model
    # Model is downloaded automatically by WhisperX if not found locally.
    print(f"Loading WhisperX model: {model_size}...")
    try:
        model = whisperx.load_model(
            model_size,
            device=effective_device,
            compute_type=effective_compute_type,
            # download_root="path/to/your/model/cache" # Optional: specify model cache path
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading WhisperX model: {e}")
        print("Please ensure you have a working internet connection for the first run to download models,")
        print("and that PyTorch and other dependencies are correctly installed.")
        print("If using GPU, ensure CUDA drivers and toolkit are compatible with PyTorch.")
        return # Exit if model fails to load

    # Load audio
    print(f"Loading audio from: {audio_path}...")
    try:
        audio = whisperx.load_audio(audio_path)
        print("Audio loaded successfully.")
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return

    # Perform transcription
    print("Starting transcription process...")
    try:
        result = model.transcribe(audio, batch_size=batch_size, initial_prompt=initial_prompt_text)
        print("Transcription complete.")
    except Exception as e:
        print(f"Error during transcription: {e}")
        return

    # Save the transcription segments to the output file
    print(f"Saving transcription to: {output_path}...")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            if result and "segments" in result:
                for i, segment in enumerate(result["segments"]):
                    # Output format: [start_time - end_time] text
                    f.write(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text'].strip()}\n")
            else:
                f.write("Transcription produced no segments or an unexpected result format.\n")
        print("Transcription saved successfully.")
    except Exception as e:
        print(f"Error saving transcription to {output_path}: {e}")

# --- Command-Line Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file using WhisperX with a prompt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        required=True,
        help="Path to the input audio file (e.g., input.mp3)."
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        required=True,
        help="Path to the text file containing the initial prompt."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the transcription output file (e.g., output.txt)."
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default=DEFAULT_MODEL_SIZE,
        help="WhisperX model size (e.g., 'tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3')."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for transcription."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None, # Auto-detect by default
        choices=["cpu", "cuda"],
        help="Device to run on ('cpu' or 'cuda'). Overrides auto-detection if set."
    )
    parser.add_argument(
        "--compute_type",
        type=str,
        default=None, # Auto-detect by default
        choices=["int8", "float16", "float32"], # Add float32 if supported/needed
        help="Compute type for the model (e.g., 'int8' for CPU, 'float16' for GPU). Overrides auto-detection if set."
    )

    args = parser.parse_args()

    try:
        transcribe_audio(
            audio_path=args.audio_path,
            prompt_path=args.prompt_path,
            output_path=args.output_path,
            model_size=args.model_size,
            batch_size=args.batch_size,
            device=args.device,
            compute_type=args.compute_type
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # Example usage from command line:
    # python transcribe_with_whisperx.py --audio_path resources/background_aow.mp3 --prompt_path resources/sample_prompt.txt --output_path whisperx_script_output.txt
    #
    # To use GPU (if available and PyTorch with CUDA is installed):
    # python transcribe_with_whisperx.py --audio_path ... --prompt_path ... --output_path ... --device cuda --compute_type float16
    #
    # Note: The first time running with a specific model, it will be downloaded.
    # Ensure 'resources/background_aow.mp3' and 'resources/sample_prompt.txt' exist for the example to run.
    # The `sample_prompt.txt` was created in a previous (aborted) plan. If it doesn't exist, create it or use another text file.
    # Example content for resources/sample_prompt.txt:
    # This is a game. Let's play. Victory. Defeat. Strategy and tactics. Audio from Age of Wonders.
    #
    # If the disk space issue is not resolved, installing whisperx and its dependencies (especially torch for GPU)
    # will likely fail, preventing this script from running successfully.
    # The script is written to be functional once dependencies are met.
    print("\nScript execution finished.")
