from pydub import AudioSegment
import subprocess
import numpy as np
import captacity # Assuming this is a third-party or custom library
import math
import cv2
import os
from typing import List, Dict, Any, Optional, Tuple # Added Tuple

def get_audio_duration_ms(audio_file_path: str) -> int:
    """
    Calculates the duration of an audio file in milliseconds.

    Args:
        audio_file_path: Path to the audio file.

    Returns:
        Duration of the audio file in milliseconds.
        Returns 0 if the file is not found or cannot be read.
    """
    try:
        return len(AudioSegment.from_file(audio_file_path))
    except FileNotFoundError:
        print(f"Audio file not found: {audio_file_path}")
        return 0
    except Exception as e: # Catch other pydub errors
        print(f"Error reading audio file {audio_file_path}: {e}")
        return 0


def _combine_narrations_and_add_to_video(
    narration_sentences: List[str],
    base_dir: str, # Renamed from output_dir for clarity, as it's the base for subfolders
    input_video_path: str,
    output_video_path: str
) -> bool:
    """
    Combines individual narration audio files into a single track and adds it to the video.

    Args:
        narration_sentences: A list of narration sentences (used to count narration files).
        base_dir: The base directory containing the 'narrations' subdirectory.
        input_video_path: Path to the video file (without audio or with temporary audio).
        output_video_path: Path where the video with combined narration will be saved.

    Returns:
        True if successful, False otherwise.
    """
    full_narration_track = AudioSegment.empty()
    narration_files_dir = os.path.join(base_dir, "narrations")

    for i in range(len(narration_sentences)):
        audio_file = os.path.join(narration_files_dir, f"narration_{i+1}.mp3")
        if not os.path.exists(audio_file):
            print(f"Narration file {audio_file} not found. Skipping.")
            continue
        try:
            segment = AudioSegment.from_file(audio_file)
            full_narration_track += segment
        except Exception as e:
            print(f"Error loading narration segment {audio_file}: {e}")
            return False # Abort if a segment can't be loaded

    if len(full_narration_track) == 0:
        print("No narration segments found or loaded. Video will not have narration.")
        # Copy input video to output path if no narration, to maintain workflow
        try:
            if input_video_path != output_video_path: # Avoid copying to itself
                subprocess.run(['cp', input_video_path, output_video_path], check=True)
            return True # Technically successful, but with no narration added
        except subprocess.CalledProcessError as e:
            print(f"Error copying video when no narration: {e}")
            return False


    temp_narration_path = os.path.join(base_dir, "combined_narration.mp3")
    try:
        full_narration_track.export(temp_narration_path, format="mp3")
    except Exception as e:
        print(f"Error exporting combined narration audio: {e}")
        return False

    ffmpeg_command = [
        'ffmpeg',
        '-y',  # Overwrite output files without asking
        '-i', input_video_path,
        '-i', temp_narration_path,
        '-map', '0:v',   # Map video from the first input (input_video_path)
        '-map', '1:a',   # Map audio from the second input (temp_narration_path)
        '-c:v', 'copy',  # Copy video codec (no re-encoding)
        '-c:a', 'aac',   # Encode audio to AAC
        '-strict', 'experimental', # Needed for some AAC configurations
        output_video_path
    ]

    try:
        process = subprocess.run(ffmpeg_command, capture_output=True, text=True, check=True)
        print(f"FFmpeg (audio merge) output: {process.stdout}")
        if process.stderr:
            print(f"FFmpeg (audio merge) errors: {process.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Error running FFmpeg to add narration: {e}")
        print(f"FFmpeg stdout: {e.stdout}")
        print(f"FFmpeg stderr: {e.stderr}")
        return False
    finally:
        if os.path.exists(temp_narration_path):
            os.remove(temp_narration_path)

    return True


def _resize_image_to_fit_canvas(
    image: np.ndarray,
    canvas_width: int,
    canvas_height: int
) -> np.ndarray:
    """
    Resizes an image to fit within specified dimensions while preserving aspect ratio.
    The image will be scaled down if it's larger than the canvas in either dimension.
    If the image is smaller, it will be returned as is (no upscaling by default here,
    though cv2.resize would upscale if new_width/new_height are larger).

    Args:
        image: The input image as a NumPy array (OpenCV format).
        canvas_width: The target width for the image to fit into.
        canvas_height: The target height for the image to fit into.

    Returns:
        The resized image as a NumPy array.
    """
    img_height, img_width = image.shape[:2]
    img_aspect_ratio = img_width / img_height
    canvas_aspect_ratio = canvas_width / canvas_height

    if img_aspect_ratio > canvas_aspect_ratio:
        new_width = canvas_width
        new_height = int(new_width / img_aspect_ratio)
    else:
        new_height = canvas_height
        new_width = int(new_height * img_aspect_ratio)

    new_width, new_height = max(1, new_width), max(1, new_height)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def _create_frame_with_centered_image(
    image_resized: np.ndarray,
    video_width: int,
    video_height: int
) -> np.ndarray:
    """Creates a black frame and centers the resized image onto it."""
    frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
    # Calculate offsets to center the image
    y_offset = (video_height - image_resized.shape[0]) // 2
    x_offset = (video_width - image_resized.shape[1]) // 2
    # Place the image on the frame
    frame[y_offset:y_offset+image_resized.shape[0], x_offset:x_offset+image_resized.shape[1]] = image_resized
    return frame


def _generate_visual_frames(
    video_writer: cv2.VideoWriter,
    narration_sentences: List[str],
    image_dir: str,
    narration_dir: str, # Needed for getting audio durations
    video_width: int,
    video_height: int,
    frame_rate: int,
    fade_duration_ms: int
) -> bool:
    """
    Generates and writes all visual frames (images with transitions) to the video_writer.

    Returns:
        True if successful, False if critical errors occur (e.g., no images).
    """
    image_files = sorted([
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if f.startswith("image_") and f.endswith(".webp")
    ])
    num_images = len(image_files)
    num_narrations = len(narration_sentences)

    if num_images == 0:
        print(f"No images found in {image_dir}. Cannot generate visual frames.")
        return False

    frames_per_fade = math.floor(fade_duration_ms / 1000 * frame_rate)

    for i in range(num_narrations):
        current_image_path = image_files[i % num_images]
        next_image_idx = (i + 1) % num_images
        # For the last narration, the "next" image for fade is the current image itself (hold)
        next_image_path = image_files[next_image_idx] if i + 1 < num_narrations else current_image_path

        img1_bgr = cv2.imread(current_image_path)
        img2_bgr = cv2.imread(next_image_path)

        if img1_bgr is None:
            print(f"Error: Could not read image {current_image_path}. Skipping visual segment.")
            # Attempt to write black frames for the expected duration? Or just skip?
            # For now, skipping, which might desync audio if not handled.
            continue
        if img2_bgr is None: # Should only happen if next_image_path was different and failed
            print(f"Error: Could not read next image {next_image_path}. Using current image for fade.")
            img2_bgr = img1_bgr.copy()

        img1_resized = _resize_image_to_fit_canvas(img1_bgr, video_width, video_height)
        img2_resized = _resize_image_to_fit_canvas(img2_bgr, video_width, video_height)

        frame1 = _create_frame_with_centered_image(img1_resized, video_width, video_height)
        frame2 = _create_frame_with_centered_image(img2_resized, video_width, video_height)

        narration_audio_path = os.path.join(narration_dir, f"narration_{i+1}.mp3")
        segment_duration_ms = get_audio_duration_ms(narration_audio_path)

        static_duration_ms = segment_duration_ms
        # Adjust for fade in only if not the first segment
        if i > 0:
            static_duration_ms -= fade_duration_ms / 2
        # Adjust for fade out only if not the last segment
        if i < num_narrations - 1:
            static_duration_ms -= fade_duration_ms / 2

        static_duration_ms = max(0, static_duration_ms)
        num_static_frames = math.floor(static_duration_ms / 1000 * frame_rate)

        for _ in range(num_static_frames):
            video_writer.write(frame1)

        # Perform fade to next image only if not the last narration segment and fade is needed
        if i < num_narrations - 1 and frames_per_fade > 0:
            for alpha in np.linspace(0, 1, frames_per_fade): # type: ignore
                blended_frame = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
                video_writer.write(blended_frame)
        elif i == num_narrations - 1: # Last segment, hold its frame for a duration equivalent to a fade out
            # This ensures the video doesn't end abruptly if total audio is slightly longer
            # than image presentation time due to rounding or minimum fade time.
            hold_frames = frames_per_fade
            for _ in range(hold_frames):
                 video_writer.write(frame1)
    return True


def create_short_video(
    narration_sentences: List[str],
    base_dir: str,
    final_output_filename: str,
    caption_settings: Optional[Dict[str, Any]] = None,
    video_width: int = 1080,
    video_height: int = 1920,
    frame_rate: int = 30,
    fade_duration_ms: int = 1000
) -> None:
    """
    Creates a short video with images, narration, and captions.

    The process involves:
    1. Generating a silent video with transitioning images based on narration durations.
    2. Combining individual narration audio files and merging them into the silent video.
    3. Generating transcript segments for captions.
    4. Adding captions to the video with narration.

    Args:
        narration_sentences: A list of narration sentences. Used to determine the number
                             of images and narration segments.
        base_dir: The base output directory. Expected to contain 'images' and 'narrations'
                  subdirectories.
        final_output_filename: The filename for the final output video (e.g., "my_short.mp4").
                               This will be saved inside `base_dir`.
        caption_settings: Optional dictionary of settings for captacity.add_captions.
        video_width: Width of the output video. Defaults to 1080.
        video_height: Height of the output video. Defaults to 1920 (for vertical shorts).
        frame_rate: Frame rate of the output video. Defaults to 30.
        fade_duration_ms: Duration of the cross-fade effect between images in milliseconds.
                          Defaults to 1000ms.
    """
    if caption_settings is None:
        caption_settings = {}

    image_dir = os.path.join(base_dir, "images")
    narration_dir = os.path.join(base_dir, "narrations")

    # Temporary path for the video with visuals but no final audio/captions
    temp_visuals_video_path = os.path.join(base_dir, "temp_visuals.mp4")

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(temp_visuals_video_path, fourcc, frame_rate, (video_width, video_height))

    # Generate visual frames (images with transitions)
    if not _generate_visual_frames(
        video_writer, narration_sentences, image_dir, narration_dir,
        video_width, video_height, frame_rate, fade_duration_ms
    ):
        # Error occurred in _generate_visual_frames (e.g., no images)
        video_writer.release() # Release writer even if generation failed
        cv2.destroyAllWindows() # Clean up any OpenCV windows
        if os.path.exists(temp_visuals_video_path): # Remove potentially incomplete temp file
            os.remove(temp_visuals_video_path)
        print("Video generation aborted due to issues in visual frame generation.")
        return # Exit create_short_video

    video_writer.release()
    cv2.destroyAllWindows()

    # --- Audio Processing ---
    video_with_narration_path = os.path.join(base_dir, "video_with_narration.mp4")
    if not _combine_narrations_and_add_to_video(
        narration_sentences, base_dir, temp_visuals_video_path, video_with_narration_path
    ):
        print("Failed to add narration to video. Aborting subsequent steps.")
        if os.path.exists(temp_visuals_video_path): os.remove(temp_visuals_video_path)
        return

    # Generate segments for captions
    print("Creating segments for captions...")
    caption_segments = _create_caption_segments(narration_sentences, narration_dir)
    if not caption_segments:
        print("No caption segments generated. Final video will not have captions.")
        # If no captions, the video_with_narration is the final one. Rename it.
        final_video_full_path = os.path.join(base_dir, final_output_filename)
        os.rename(video_with_narration_path, final_video_full_path)
        print(f"Final video (no captions) saved to: {final_video_full_path}")
        if os.path.exists(temp_visuals_video_path): os.remove(temp_visuals_video_path)
        return


    # Add captions to video
    final_video_full_path = os.path.join(base_dir, final_output_filename)
    print(f"Adding captions to video, outputting to: {final_video_full_path}")
    try:
        captacity.add_captions(
            video_file=video_with_narration_path,
            output_file=final_video_full_path,
            segments=caption_segments,
            print_info=True, # Or make this configurable
            **caption_settings,
        )
        print(f"Successfully added captions. Final video at: {final_video_full_path}")
    except Exception as e:
        print(f"Error adding captions using captacity: {e}")
        print("Falling back to video with narration but without captions.")
        # If captacity fails, rename the video_with_narration to the final output name
        os.rename(video_with_narration_path, final_video_full_path)

    # --- Add Background Music ---
    video_after_captions_path = final_video_full_path # This is the video with narration and possibly captions
    video_with_background_music_path = os.path.join(base_dir, f"final_with_bgm_{final_output_filename}")
    background_music_file = "background.mp3" # Assuming it's in the root, relative to where script is run

    if not os.path.exists(background_music_file):
        print(f"Background music file '{background_music_file}' not found. Skipping adding background music.")
        # If background music is skipped, the video_after_captions_path is the final one.
        # If its name is already final_video_full_path, no rename needed.
        # If it was an intermediate name that was supposed to be replaced by video_with_background_music_path,
        # we need to ensure final_video_full_path is the correct one.
        # Current logic: final_video_full_path is already the target from captioning.
        # So, if background music is skipped, final_video_full_path is already correctly named.
    else:
        print(f"Adding background music from '{background_music_file}'...")
        if _add_background_music(video_after_captions_path, background_music_file, video_with_background_music_path):
            print(f"Successfully added background music. Final video at: {video_with_background_music_path}")
            # Clean up the video_after_captions_path if it's different and exists
            if video_after_captions_path != video_with_background_music_path and os.path.exists(video_after_captions_path):
                os.remove(video_after_captions_path)
            # Update final_video_full_path to the new video with background music
            final_video_full_path = video_with_background_music_path
        else:
            print("Failed to add background music. The video without background music will be kept.")
            # final_video_full_path remains as the video_after_captions_path

    # --- Final Cleanup ---
    # Ensure final_video_full_path (which might be video_with_background_music_path or video_after_captions_path)
    # is named as original final_output_filename if they are different.
    # This handles the case where video_with_background_music_path was created.
    desired_final_path = os.path.join(base_dir, final_output_filename)
    if os.path.exists(final_video_full_path) and final_video_full_path != desired_final_path:
        print(f"Renaming '{final_video_full_path}' to '{desired_final_path}'")
        if os.path.exists(desired_final_path): # remove if an older version exists
            os.remove(desired_final_path)
        os.rename(final_video_full_path, desired_final_path)
        final_video_full_path = desired_final_path # update variable for final message

    print(f"Final video processing complete. Output at: {final_video_full_path}")

    # Clean up other temporary files
    if os.path.exists(temp_visuals_video_path):
        os.remove(temp_visuals_video_path)
    # video_with_narration_path is either same as video_after_captions_path or was already cleaned up if captions were added.
    # If captions were skipped, video_with_narration_path would be video_after_captions_path.
    # This path should have been removed or renamed by the background music step if it was successful.
    # Let's check for video_with_narration_path specifically if it's not the final output.
    intermediate_narration_video = os.path.join(base_dir, "video_with_narration.mp4")
    if os.path.exists(intermediate_narration_video) and intermediate_narration_video != final_video_full_path:
        os.remove(intermediate_narration_video)


def _create_caption_segments(narration_texts: List[str], narration_audio_dir: str) -> List[Dict[str, Any]]:
    """
    Generates caption segments by transcribing individual narration audio files.
    Offsets are calculated based on the duration of preceding audio files.

    Args:
        narration_texts: A list of the original narration texts, used as prompts for transcription.
        narration_audio_dir: The directory containing the narration audio files (e.g., narration_1.mp3).

    Returns:
        A list of segment dictionaries compatible with captacity, or an empty list if errors occur.
    """
    all_segments: List[Dict[str, Any]] = []
    current_offset_s = 0.0

    for i, text_prompt in enumerate(narration_texts):
        audio_file_path = os.path.join(narration_audio_dir, f"narration_{i+1}.mp3")

        if not os.path.exists(audio_file_path):
            print(f"Audio file for transcription not found: {audio_file_path}. Skipping this segment.")
            # We need to account for its potential duration if we want to keep subsequent timings accurate.
            # This is tricky if we don't know the duration. For now, we'll just skip.
            # A better approach might be to have durations pre-calculated and passed in.
            continue

        try:
            # Attempt local transcription first, then API as fallback
            try:
                # Assuming transcribe_locally might raise ImportError if whisper/model not available
                # or some other specific exception for "local not possible"
                transcribed_segments = captacity.transcriber.transcribe_locally(
                    audio_file=audio_file_path,
                    prompt=text_prompt, # Use original text as prompt for better accuracy
                )
            except (ImportError, AttributeError, Exception) as e_local: # Catch broader errors for local
                print(f"Local transcription failed for {audio_file_path} (prompt: '{text_prompt}'): {e_local}. Trying API.")
                try:
                    transcribed_segments = captacity.transcriber.transcribe_with_api(
                        audio_file=audio_file_path,
                        prompt=text_prompt,
                    )
                except Exception as e_api:
                    print(f"API transcription also failed for {audio_file_path} (prompt: '{text_prompt}'): {e_api}")
                    # If API also fails, we skip this segment's transcription
                    # and need to get its duration to correctly offset the next one.
                    audio_duration_ms = get_audio_duration_ms(audio_file_path)
                    current_offset_s += audio_duration_ms / 1000.0
                    continue

            # Offset these segments by the cumulative duration of previous segments
            offset_adjusted_segments = _offset_timestamps_in_segments(transcribed_segments, current_offset_s)
            all_segments.extend(offset_adjusted_segments)

            # Add current audio's duration to offset for the next iteration
            audio_duration_ms = get_audio_duration_ms(audio_file_path)
            if audio_duration_ms == 0 and transcribed_segments: # If duration is 0 but we got segments, use segment end time
                current_offset_s = offset_adjusted_segments[-1]['end'] if offset_adjusted_segments else current_offset_s
            else:
                 current_offset_s += audio_duration_ms / 1000.0

        except Exception as e:
            print(f"An unexpected error occurred during segment creation for {audio_file_path}: {e}")
            # Fallback: try to get duration to keep subsequent segments timed correctly
            audio_duration_ms = get_audio_duration_ms(audio_file_path)
            current_offset_s += audio_duration_ms / 1000.0
            continue

    return all_segments


def _offset_timestamps_in_segments(segments: List[Dict[str, Any]], offset_s: float) -> List[Dict[str, Any]]:
    """
    Adjusts 'start' and 'end' timestamps for all words and segments by a given offset.

    Args:
        segments: A list of segment dictionaries from transcription.
                  Each segment should have 'start', 'end', and a 'words' list.
                  Each item in 'words' should have 'start' and 'end'.
        offset_s: The time offset in seconds to add to all timestamps.

    Returns:
        The list of segments with adjusted timestamps.
    """
    adjusted_segments = []
    for segment in segments:
        # Create a new segment dict to avoid modifying the original if it's reused
        new_segment = segment.copy()
        new_segment["start"] = segment.get("start", 0) + offset_s
        new_segment["end"] = segment.get("end", 0) + offset_s

        adjusted_words = []
        if "words" in segment and segment["words"] is not None: # Check for None explicitly
            for word_info in segment["words"]:
                new_word_info = word_info.copy()
                new_word_info["start"] = word_info.get("start", 0) + offset_s
                new_word_info["end"] = word_info.get("end", 0) + offset_s
                adjusted_words.append(new_word_info)
        new_segment["words"] = adjusted_words
        adjusted_segments.append(new_segment)
    return adjusted_segments


def _add_background_music(
    input_video_path: str,
    background_music_path: str,
    output_video_path: str,
    background_volume_adjust: str = "-15dB"
) -> bool:
    """
    Adds background music to a video file, mixing it with existing audio.

    Args:
        input_video_path: Path to the input video file.
        background_music_path: Path to the background music file.
        output_video_path: Path to save the video with background music.
        background_volume_adjust: FFmpeg volume filter value for the background music.
                                  Defaults to "-15dB" to make it quieter than main audio.

    Returns:
        True if successful, False otherwise.
    """
    if not os.path.exists(input_video_path):
        print(f"Input video for background music not found: {input_video_path}")
        return False
    if not os.path.exists(background_music_path):
        print(f"Background music file not found: {background_music_path}")
        return False

    # Check if input video has an audio stream
    has_audio_command = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'a',
        '-show_entries', 'stream=codec_type',
        '-of', 'csv=p=0',
        input_video_path
    ]
    try:
        result = subprocess.run(has_audio_command, capture_output=True, text=True, check=True)
        has_existing_audio = bool(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"Error checking for existing audio stream in {input_video_path}: {e.stderr}")
        # Proceed assuming no audio, ffmpeg will handle it or fail if it's critical
        has_existing_audio = False
    except FileNotFoundError:
        print("ffprobe command not found. Make sure ffmpeg is installed and in PATH.")
        print("Assuming video has existing audio for safety.") # Safer to assume audio
        has_existing_audio = True


    if has_existing_audio:
        ffmpeg_command = [
            'ffmpeg',
            '-y',  # Overwrite output files without asking
            '-i', input_video_path,
            '-i', background_music_path,
            '-filter_complex',
            # Stream 0 audio (original), Stream 1 audio (background)
            # Adjust volume of background music, then mix them.
            # amix inputs=2, duration=longest: ensures mixing continues for the duration of the longest input
            # -ac 2 forces stereo output, good for general compatibility
            f"[1:a]volume={background_volume_adjust}[bg_audio];[0:a][bg_audio]amix=inputs=2:duration=longest:dropout_transition=2[out_audio]",
            '-map', '0:v',      # Map video from the first input
            '-map', '[out_audio]', # Map the mixed audio
            '-c:v', 'copy',     # Copy video codec
            '-c:a', 'aac',      # Encode audio to AAC
            '-strict', 'experimental',
            '-shortest',        # Finish encoding when the shortest input stream ends (typically the video)
            output_video_path
        ]
    else:
        # If no existing audio, just add the background music (looping if shorter than video)
        # This also handles the case where the input video might be silent but has an audio track.
        # If it truly has no audio track, ffmpeg will map 1:a.
        # To make background music loop if it's shorter than video: add "-stream_loop -1" for the music input
        # However, to keep it simple and avoid overly long audio if music is very short,
        # we'll let it be as long as the music or video, whichever is shorter, by default.
        # For true looping of short background music, a more complex command would be needed.
        # The command below will result in audio being as long as the background music.
        # If video is longer, it will have silence at the end. If music is longer, -shortest (if added for video) would cut it.
        # Using -shortest here is important if the background music is longer than the video.
        print(f"No existing audio stream detected in {input_video_path} or ffprobe failed. Adding background music directly.")
        ffmpeg_command = [
            'ffmpeg',
            '-y',
            '-i', input_video_path,
            '-i', background_music_path,
            '-filter_complex', f"[1:a]volume={background_volume_adjust}[bg_audio]",
            '-map', '0:v',
            '-map', '[bg_audio]', # Map the background audio
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-strict', 'experimental',
            '-shortest', # Ensure audio stream doesn't make file longer than video
            output_video_path
        ]


    try:
        process = subprocess.run(ffmpeg_command, capture_output=True, text=True, check=True)
        print(f"FFmpeg (background music) output: {process.stdout}")
        if process.stderr:
            print(f"FFmpeg (background music) errors: {process.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running FFmpeg to add background music: {e}")
        print(f"FFmpeg stdout: {e.stdout}")
        print(f"FFmpeg stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("ffmpeg command not found. Make sure ffmpeg is installed and in PATH.")
        return False
