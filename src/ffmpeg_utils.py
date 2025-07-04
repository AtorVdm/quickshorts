# This file contains utility functions for interacting with FFmpeg.

import os
import subprocess
from typing import Optional

# Placeholders for FFmpeg wrapper functions.
# Will be populated in the next step.

def combine_narrations_and_add_to_video( # Renamed to be public
    narration_sentences_count: int, # Changed from list of sentences to just the count
    base_dir: str,
    input_video_path: str,
    output_video_path: str,
    audio_segment_loader: Any # Function to load AudioSegment, e.g. pydub.AudioSegment.from_file
) -> bool:
    """
    Combines individual narration audio files into a single track and adds it to the video.
    Requires a loader for AudioSegment to avoid direct pydub dependency here.
    """
    full_narration_track_len_ms = 0 # Used to check if any audio was loaded
    # Create an empty audio segment using the loader's utility if available,
    # or manage segments in a list then combine. For simplicity, let's sum lengths
    # and assume the caller (short_video) handles pydub AudioSegment objects directly.
    # This function will now focus only on the ffmpeg command.
    # The actual audio combination will be done by the caller.

    temp_narration_path = os.path.join(base_dir, "combined_narration.mp3") # Assume this is pre-created by caller

    if not os.path.exists(temp_narration_path):
        print("Combined narration file not found. Assuming no narration to add.")
        # If no narration, copy input to output and return true
        try:
            if input_video_path != output_video_path:
                subprocess.run(['cp', input_video_path, output_video_path], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error copying video when no narration: {e}")
            return False
        except FileNotFoundError:
             print("Error: 'cp' command not found. Cannot copy video.")
             return False


    ffmpeg_command = [
        'ffmpeg', '-y',
        '-i', input_video_path,
        '-i', temp_narration_path,
        '-map', '0:v', '-map', '1:a',
        '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental',
        output_video_path
    ]
    try:
        process = subprocess.run(ffmpeg_command, capture_output=True, text=True, check=True)
        if process.stderr: print(f"FFmpeg (audio merge) warnings/errors: {process.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running FFmpeg to add narration: {e.stderr}")
        return False
    except FileNotFoundError:
        print("ffmpeg command not found. Make sure ffmpeg is installed and in PATH.")
        return False
    # temp_narration_path is now expected to be cleaned up by the caller


def add_background_music( # Renamed to be public
    input_video_path: str,
    background_music_path: str,
    output_video_path: str,
    background_volume_adjust: str = "-15dB"
) -> bool:
    """Adds background music to a video file, mixing it with existing audio."""
    if not os.path.exists(input_video_path):
        print(f"Input video for background music not found: {input_video_path}")
        return False
    if not os.path.exists(background_music_path):
        print(f"Background music file not found: {background_music_path}")
        return False

    has_existing_audio = False
    try:
        ffprobe_command = [
            'ffprobe', '-v', 'error', '-select_streams', 'a',
            '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', input_video_path
        ]
        result = subprocess.run(ffprobe_command, capture_output=True, text=True, check=True)
        has_existing_audio = bool(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"ffprobe error checking audio: {e.stderr}. Assuming no existing audio.")
    except FileNotFoundError:
        print("ffprobe not found. Assuming video has existing audio for safety.")
        has_existing_audio = True # Safer to assume audio if ffprobe fails

    common_ffmpeg_options = ['-y', '-i', input_video_path, '-i', background_music_path]
    video_map = ['-map', '0:v']
    codec_options = ['-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental']

    filter_complex: List[str] = []
    audio_map: List[str] = []

    if has_existing_audio:
        filter_complex = ['-filter_complex', f"[1:a]volume={background_volume_adjust}[bg_audio];[0:a][bg_audio]amix=inputs=2:duration=longest:dropout_transition=2[out_audio]"]
        audio_map = ['-map', '[out_audio]']
    else:
        print(f"No existing audio stream in {input_video_path} or ffprobe failed. Adding background music directly.")
        filter_complex = ['-filter_complex', f"[1:a]volume={background_volume_adjust}[bg_audio]"]
        audio_map = ['-map', '[bg_audio]']

    ffmpeg_command = ['ffmpeg'] + common_ffmpeg_options + filter_complex + video_map + audio_map + codec_options + ['-shortest', output_video_path]

    try:
        process = subprocess.run(ffmpeg_command, capture_output=True, text=True, check=True)
        if process.stderr: print(f"FFmpeg (background music) warnings/errors: {process.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running FFmpeg for background music: {e.stderr}")
        return False
    except FileNotFoundError:
        print("ffmpeg command not found. Make sure ffmpeg is installed and in PATH.")
        return False
