import os
import glob
import whisperx
from pydub import AudioSegment
# import torch # Implicitly used by whisperx

# --- Configuration (Hardcoded Values) ---
INPUT_FOLDER_PATH = "shorts/test_story/narrations" # Folder containing .mp3 files to merge
PROMPT_TEXT_PATH = "shorts/test_story/input.txt"      # Default prompt file
OUTPUT_TRANSCRIPTION_PATH = "merged_transcription_output.txt" # Default output file
TEMP_MERGED_AUDIO_PATH = "temp_merged_audio.mp3"      # Temporary file for merged audio

MODEL_SIZE = "large-v2" # WhisperX model size
BATCH_SIZE = 8         # Batch size for transcription
DEVICE = None          # Auto-detect: "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = None    # Auto-detect: e.g. "float16" for CUDA, "int8" for CPU

def merge_mp3_files_in_folder(folder_path: str, output_merged_path: str) -> bool:
    """
    Finds all .mp3 files in a folder, merges them, and saves to output_merged_path.
    Returns True if successful, False otherwise.
    """
    print(f"Scanning folder for .mp3 files: {folder_path}")
    if not os.path.isdir(folder_path):
        print(f"Error: Input folder not found at: {folder_path}")
        return False

    mp3_files = sorted(glob.glob(os.path.join(folder_path, "*.mp3")))

    if not mp3_files:
        print(f"No .mp3 files found in {folder_path}")
        return False

    print(f"Found {len(mp3_files)} .mp3 files to merge: {mp3_files}")

    merged_audio = AudioSegment.empty()
    for file_path in mp3_files:
        try:
            segment = AudioSegment.from_mp3(file_path)
            merged_audio += segment
            print(f"Appended {file_path} ({len(segment)}ms)")
        except Exception as e:
            print(f"Error loading or appending {file_path}: {e}. Skipping this file.")
            continue # Skip problematic files

    if len(merged_audio) == 0:
        print("No audio could be merged (all files failed or were skipped).")
        return False

    try:
        merged_audio.export(output_merged_path, format="mp3")
        print(f"Successfully merged audio files into: {output_merged_path} ({len(merged_audio)}ms)")
        return True
    except Exception as e:
        print(f"Error exporting merged audio to {output_merged_path}: {e}")
        return False

def run_transcription_on_merged_audio():
    """
    Merges MP3s from a folder, then transcribes the merged audio using WhisperX
    with an initial prompt, using hardcoded paths and settings.
    """
    print(f"Starting transcription process. Input folder: {INPUT_FOLDER_PATH}")

    # 1. Merge audio files
    if not merge_mp3_files_in_folder(INPUT_FOLDER_PATH, TEMP_MERGED_AUDIO_PATH):
        print("Audio merging failed. Aborting transcription.")
        return

    # 2. Proceed with transcription of the merged audio file
    input_audio_to_transcribe = TEMP_MERGED_AUDIO_PATH

    if not os.path.exists(PROMPT_TEXT_PATH):
        print(f"Warning: Prompt text file not found at: {PROMPT_TEXT_PATH}. Proceeding without prompt.")
        initial_prompt_text = None
    else:
        try:
            with open(PROMPT_TEXT_PATH, 'r', encoding='utf-8') as f:
                initial_prompt_text = f.read().strip()
            if not initial_prompt_text:
                print(f"Warning: Prompt file {PROMPT_TEXT_PATH} is empty. Proceeding without prompt.")
                initial_prompt_text = None
        except Exception as e:
            print(f"Error reading prompt file {PROMPT_TEXT_PATH}: {e}. Proceeding without prompt.")
            initial_prompt_text = None

    print(f"Loading WhisperX model: {MODEL_SIZE}...")
    try:
        model_device = DEVICE
        model_compute_type = COMPUTE_TYPE
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
                model_device = "cpu"
                if model_compute_type is None: model_compute_type = "int8"

        print(f"Attempting to load model with device='{model_device}' and compute_type='{model_compute_type}'")
        model = whisperx.load_model(MODEL_SIZE, device=model_device, compute_type=model_compute_type)
        print("Model loaded.")
    except Exception as e:
        print(f"Fatal Error: Could not load WhisperX model: {e}")
        if os.path.exists(TEMP_MERGED_AUDIO_PATH): os.remove(TEMP_MERGED_AUDIO_PATH) # Cleanup
        return

    print(f"Loading merged audio from: {input_audio_to_transcribe}...")
    try:
        audio = whisperx.load_audio(input_audio_to_transcribe)
        print("Merged audio loaded.")
    except Exception as e:
        print(f"Error loading merged audio: {e}")
        if os.path.exists(TEMP_MERGED_AUDIO_PATH): os.remove(TEMP_MERGED_AUDIO_PATH) # Cleanup
        return

    print("Starting transcription of merged audio...")
    try:
        result = model.transcribe(audio, batch_size=BATCH_SIZE)
        print("Transcription complete.")
    except Exception as e:
        print(f"Error during transcription: {e}")
        if os.path.exists(TEMP_MERGED_AUDIO_PATH): os.remove(TEMP_MERGED_AUDIO_PATH) # Cleanup
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
    finally:
        if os.path.exists(TEMP_MERGED_AUDIO_PATH):
            os.remove(TEMP_MERGED_AUDIO_PATH)
            print(f"Cleaned up temporary merged audio file: {TEMP_MERGED_AUDIO_PATH}")

if __name__ == "__main__":
    print("Running WhisperX Transcription Script with folder input and merging...")
    print("--------------------------------------------------------------------")
    print(f"Input Folder: {INPUT_FOLDER_PATH}")
    print(f"Prompt File: {PROMPT_TEXT_PATH}")
    print(f"Output Transcription File: {OUTPUT_TRANSCRIPTION_PATH}")
    print(f"Temporary Merged Audio: {TEMP_MERGED_AUDIO_PATH}")
    print(f"Model: {MODEL_SIZE}, Batch Size: {BATCH_SIZE}")
    print("--------------------------------------------------------------------")

    # This script will fail if whisperx, pydub, and their dependencies (PyTorch, ffmpeg for pydub)
    # are not installed, likely due to the ongoing disk space issue.

    # For the script to run:
    # 1. Ensure the INPUT_FOLDER_PATH exists and contains some .mp3 files.
    #    Example: Create `resources/narrations_for_merge/` and put `narration_1.mp3`, `narration_2.mp3` in it.
    # 2. Ensure PROMPT_TEXT_PATH exists (or handle its absence).
    #    Example content for resources/sample_prompt.txt:
    #    "This is a game. Let's play. Victory. Defeat. Strategy and tactics."

    run_transcription_on_merged_audio()
    print("\nScript finished.")
