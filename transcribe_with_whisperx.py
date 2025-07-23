import os
import glob
import whisperx
from pydub import AudioSegment
import gc
# import torch # Implicitly used by whisperx, and explicitly for cleanup if cuda

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
    result = None # Ensure result is defined in case of early exit
    model_a = None # Ensure model_a is defined for cleanup
    try:
        result = model.transcribe(audio, batch_size=BATCH_SIZE, initial_prompt=initial_prompt_text)
        print("Transcription complete.")

        # Clean up ASR model
        print("Cleaning up ASR model...")
        del model
        if model_device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                print("PyTorch not found, skipping cuda.empty_cache()")
            except Exception as e:
                print(f"Error during CUDA cache clear for ASR model: {e}")
        gc.collect()
        print("ASR model cleaned up.")

        # 2. Align whisper output
        print("Loading alignment model...")
        # Note: language_code should come from the result of transcription
        # If result is None or doesn't have 'language', this will fail.
        # Adding a check for robustness, though ideally transcription should always yield a language.
        language_code = result.get("language", "en") if result else "en"
        if result is None:
            print("Error: Transcription result is missing. Cannot proceed with alignment.")
            if os.path.exists(TEMP_MERGED_AUDIO_PATH): os.remove(TEMP_MERGED_AUDIO_PATH)
            return

        model_a, metadata = whisperx.load_align_model(language_code=language_code, device=model_device)
        print(f"Alignment model loaded for language: {language_code}")

        print("Aligning transcription...")
        # The BATCH_SIZE for alignment can be different, but not specified in original script context
        # Using a general approach; whisperx.align handles segments internally.
        aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, model_device, return_char_alignments=False)
        print("Alignment complete.")

        # Clean up alignment model
        print("Cleaning up alignment model...")
        del model_a
        if model_device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                print("PyTorch not found, skipping cuda.empty_cache()")
            except Exception as e:
                print(f"Error during CUDA cache clear for alignment model: {e}")
        gc.collect()
        print("Alignment model cleaned up.")

    except Exception as e:
        print(f"Error during transcription or alignment: {e}")
        # Cleanup models if they were loaded before the error
        if 'model' in locals() and model is not None:
            del model
            print("Cleaned up ASR model after error.")
        if model_a is not None: # model_a is already defined outside the try block
            del model_a
            print("Cleaned up alignment model after error.")
        if model_device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass # PyTorch not found
            except Exception as ec:
                print(f"Error during CUDA cache clear after main error: {ec}")
        gc.collect()
        if os.path.exists(TEMP_MERGED_AUDIO_PATH): os.remove(TEMP_MERGED_AUDIO_PATH) # Cleanup temp audio
        return

    print(f"Saving word-level transcription to: {OUTPUT_TRANSCRIPTION_PATH}...")
    try:
        with open(OUTPUT_TRANSCRIPTION_PATH, 'w', encoding='utf-8') as f:
            if aligned_result and "word_segments" in aligned_result:
                if not aligned_result["word_segments"]:
                    f.write("No words were aligned. The audio might be silent or too noisy.\n")
                for word_info in aligned_result["word_segments"]:
                    start_time = word_info.get('start', "N/A")
                    end_time = word_info.get('end', "N/A")
                    word_text = word_info.get('word', "[UNKNOWN_WORD]")
                    score = word_info.get('score', "N/A")

                    # Formatting, ensuring times are numbers before formatting
                    start_str = f"{start_time:.2f}" if isinstance(start_time, (int, float)) else str(start_time)
                    end_str = f"{end_time:.2f}" if isinstance(end_time, (int, float)) else str(end_time)
                    score_str = f"{score:.2f}" if isinstance(score, (int, float)) else str(score)

                    f.write(f"[{start_str}s - {end_str}s] {word_text} (score: {score_str})\n")
            elif result and "segments" in result and not aligned_result.get("word_segments"):
                # Fallback to segment transcription if word alignment somehow yielded empty but segments exist
                f.write("Word alignment did not produce output, but segment transcription exists:\n")
                for segment in result["segments"]:
                     f.write(f"SEGMENT: [{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text'].strip()}\n")
            else:
                f.write("No words or segments transcribed/aligned, or result format unexpected.\n")
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
