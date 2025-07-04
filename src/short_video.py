from pydub import AudioSegment
import subprocess
import numpy as np
import captacity # Assuming this is a third-party or custom library
import math
import cv2
import os
from typing import List, Dict, Any, Optional, Tuple # Added Tuple

# get_audio_duration_ms moved to audio_utils.py
from .audio_utils import get_audio_duration_ms
from .ffmpeg_utils import combine_narrations_and_add_to_video, add_background_music


# _combine_narrations_and_add_to_video is now mostly a wrapper to prepare combined_narration.mp3
# and then call the ffmpeg_utils version.
def _prepare_and_combine_audio_ffmpeg(
    narration_sentences: List[str],
    base_dir: str,
    input_video_path: str,
    output_video_path: str
) -> bool:
    """
    Prepares combined narration audio and then calls the FFmpeg utility
    to merge it with the video.
    """
    narration_files_dir = os.path.join(base_dir, "narrations")
    temp_narration_path = os.path.join(base_dir, "combined_narration.mp3")

    # Combine narration audio segments using pydub
    full_narration_track = AudioSegment.empty()
    narration_audio_exists = False
    for i in range(len(narration_sentences)):
        audio_file = os.path.join(narration_files_dir, f"narration_{i+1}.mp3")
        if not os.path.exists(audio_file):
            print(f"Narration file {audio_file} not found. Skipping.")
            continue
        try:
            segment = AudioSegment.from_file(audio_file)
            full_narration_track += segment
            narration_audio_exists = True
        except Exception as e:
            print(f"Error loading narration segment {audio_file}: {e}")
            # Decide if one bad segment should stop all audio. For now, it does.
            if os.path.exists(temp_narration_path): os.remove(temp_narration_path) # cleanup
            return False

    if not narration_audio_exists:
        print("No narration audio segments found or loaded.")
        # ffmpeg_utils.combine_narrations_and_add_to_video will handle copying video if no audio
        # but we must ensure temp_narration_path does not exist.
        if os.path.exists(temp_narration_path):
             print(f"Warning: {temp_narration_path} existed but no audio segments. Removing.")
             try: os.remove(temp_narration_path)
             except OSError as e: print(f"Error removing temp narration file: {e}")

    else: # Narration audio exists, export it
        try:
            full_narration_track.export(temp_narration_path, format="mp3")
        except Exception as e:
            print(f"Error exporting combined narration audio: {e}")
            if os.path.exists(temp_narration_path): os.remove(temp_narration_path)
            return False

    # Call the FFmpeg utility function
    # Pass narration_sentences count, or just a boolean indicating if audio is expected
    success = combine_narrations_and_add_to_video(
        narration_sentences_count=len(narration_sentences) if narration_audio_exists else 0,
        base_dir=base_dir, # used by util to find temp_narration_path
        input_video_path=input_video_path,
        output_video_path=output_video_path,
        audio_segment_loader=None # No longer needed by the util
    )

    # Clean up combined_narration.mp3
    if os.path.exists(temp_narration_path):
        try:
            os.remove(temp_narration_path)
        except OSError as e:
            print(f"Error cleaning up temporary narration file {temp_narration_path}: {e}")
            # If cleanup fails, it's not critical enough to return False for the whole op.

    return success


# Functions _resize_image_to_fit_canvas, _create_frame_with_centered_image,
# _apply_ken_burns_effect, _calculate_ken_burns_pan_coordinates,
# and _generate_single_ken_burns_frame have been moved to video_utils.py

# Import them
from .video_utils import (
    resize_image_to_fit_canvas,
    create_frame_with_centered_image,
    apply_ken_burns_effect,
    KEN_BURNS_SEQUENCE # Constant also moved
)

# ken_burns_sequence_index will be managed locally within _generate_visual_frames
# ken_burns_sequence_index = 0 # Removed global


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
    Generates and writes all visual frames (images with Ken Burns effect and transitions)
    to the video_writer.
    """
    # Manage Ken Burns sequence index locally
    ken_burns_sequence_index = 0

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
    if fade_duration_ms > 0 and frames_per_fade == 0:
        frames_per_fade = 1

    for i in range(num_narrations):
        current_movement_type = KEN_BURNS_SEQUENCE[ken_burns_sequence_index % len(KEN_BURNS_SEQUENCE)]

        next_ken_burns_sequence_index = ken_burns_sequence_index + 1
        next_image_movement_type = KEN_BURNS_SEQUENCE[next_ken_burns_sequence_index % len(KEN_BURNS_SEQUENCE)]

        _process_visual_segment(
            video_writer=video_writer,
            segment_index=i,
            num_narrations=num_narrations,
            image_files=image_files,
            num_images=num_images,
            narration_dir=narration_dir,
            video_width=video_width,
            video_height=video_height,
            frame_rate=frame_rate,
            frames_per_fade=frames_per_fade,
            current_movement_type=current_movement_type,
            next_image_movement_type=next_image_movement_type
        )
        ken_burns_sequence_index += 1 # Increment for the next segment
    return True


def _process_visual_segment(
    video_writer: cv2.VideoWriter,
    segment_index: int,
    num_narrations: int,
    image_files: List[str],
    num_images: int,
    narration_dir: str,
    video_width: int,
    video_height: int,
    frame_rate: int,
    frames_per_fade: int,
    current_movement_type: str,
    next_image_movement_type: str
):
    """
    Processes a single visual segment: displays an image with Ken Burns effect
    and handles the cross-fade to the next image.
    """
    # global ken_burns_sequence_index # No longer global

    current_image_path = image_files[segment_index % num_images]
    next_image_for_fade_path = image_files[(segment_index + 1) % num_images] if segment_index < num_narrations - 1 else current_image_path

    current_img_bgr = cv2.imread(current_image_path)
    if current_img_bgr is None:
        print(f"Error: Could not read image {current_image_path}. Skipping visual segment {segment_index + 1}.")
        return

    narration_audio_path = os.path.join(narration_dir, f"narration_{segment_index + 1}.mp3")
    segment_duration_ms = get_audio_duration_ms(narration_audio_path)
    if segment_duration_ms == 0:
        print(f"Warning: Audio duration for {narration_audio_path} is 0. Defaulting to 1s for frame calculation.")
        segment_duration_ms = 1000

    total_frames_for_segment = math.floor(segment_duration_ms / 1000 * frame_rate)
    if total_frames_for_segment == 0 and segment_duration_ms > 0: total_frames_for_segment = 1

    # current_movement_type is now passed as an argument
    print(f"Image {segment_index + 1}: Using Ken Burns effect '{current_movement_type}'")

    frames_for_A_main_display = total_frames_for_segment
    frames_for_A_fade_out = 0

    if segment_index < num_narrations - 1 and frames_per_fade > 0:
        frames_for_A_fade_out = frames_per_fade
        frames_for_A_main_display = max(0, total_frames_for_segment - frames_for_A_fade_out)
        if frames_for_A_main_display == 0 and total_frames_for_segment > 0:
            frames_for_A_fade_out = total_frames_for_segment

    if total_frames_for_segment <= 0:
        print(f"Skipping visual segment for image {segment_index + 1} as total_frames_for_segment is {total_frames_for_segment}.")
        # ken_burns_sequence_index increment is handled by the caller (_generate_visual_frames)
        return

    frame_offset_A_main = 0
    if segment_index > 0 and frames_per_fade > 0:
        frame_offset_A_main = frames_per_fade

    ken_burns_frames_A_main = []
    actual_frames_to_generate_A_main = frames_for_A_main_display
    if frame_offset_A_main + actual_frames_to_generate_A_main > total_frames_for_segment:
         actual_frames_to_generate_A_main = max(0, total_frames_for_segment - frame_offset_A_main - frames_for_A_fade_out)

    if actual_frames_to_generate_A_main > 0:
        ken_burns_frames_A_main = apply_ken_burns_effect( # Use imported version
            image_bgr=current_img_bgr, video_width=video_width, video_height=video_height,
            movement_type=current_movement_type,
            full_animation_duration_frames=total_frames_for_segment,
            generate_count=actual_frames_to_generate_A_main,
            generate_from_frame_offset=frame_offset_A_main
        )
        for frame in ken_burns_frames_A_main:
            video_writer.write(frame)

    if segment_index < num_narrations - 1 and frames_for_A_fade_out > 0:
        frame_offset_A_fade_component = frame_offset_A_main + actual_frames_to_generate_A_main
        kb_frames_A_for_fade_component = apply_ken_burns_effect( # Use imported version
            image_bgr=current_img_bgr, video_width=video_width, video_height=video_height,
            movement_type=current_movement_type,
            full_animation_duration_frames=total_frames_for_segment,
            generate_count=frames_for_A_fade_out,
            generate_from_frame_offset=frame_offset_A_fade_component
        )

        next_img_bgr = cv2.imread(next_image_for_fade_path)
        if next_img_bgr is None:
            print(f"Error reading next image {next_image_for_fade_path} for cross-fade. Holding last frame of current image.")
            last_frame_A_to_hold = ken_burns_frames_A_main[-1] if ken_burns_frames_A_main else \
                                 (kb_frames_A_for_fade_component[-1] if kb_frames_A_for_fade_component else None)
            if last_frame_A_to_hold is None:
                 temp_A_frame = apply_ken_burns_effect(current_img_bgr, video_width, video_height, current_movement_type, total_frames_for_segment, 1, frame_offset_A_fade_component) # Use imported version
                 if temp_A_frame: last_frame_A_to_hold = temp_A_frame[0]

            if last_frame_A_to_hold is not None:
                for _ in range(frames_for_A_fade_out): video_writer.write(last_frame_A_to_hold)
            else:
                black_frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
                for _ in range(frames_for_A_fade_out): video_writer.write(black_frame)
        else:
            next_narration_audio_path = os.path.join(narration_dir, f"narration_{(segment_index + 1) + 1}.mp3")
            next_segment_duration_ms = get_audio_duration_ms(next_narration_audio_path)
            if next_segment_duration_ms == 0:
                print(f"Warning: Audio duration for next image {next_image_for_fade_path} is 0 for fade. Defaulting to 1s.")
                next_segment_duration_ms = 1000

            total_frames_for_segment_B = math.floor(next_segment_duration_ms / 1000 * frame_rate)
            if total_frames_for_segment_B == 0 and next_segment_duration_ms > 0: total_frames_for_segment_B = 1
            if total_frames_for_segment_B <= 0:
                total_frames_for_segment_B = frames_for_A_fade_out

            # next_image_movement_type is now passed as an argument
            # next_image_movement_type_index = (ken_burns_sequence_index + 1) % len(KEN_BURNS_SEQUENCE)
            # next_image_movement_type = KEN_BURNS_SEQUENCE[next_image_movement_type_index]

            kb_frames_B_for_fade_component = apply_ken_burns_effect( # Use imported version
                image_bgr=next_img_bgr, video_width=video_width, video_height=video_height,
                movement_type=next_image_movement_type, # Passed as argument
                full_animation_duration_frames=total_frames_for_segment_B,
                generate_count=frames_for_A_fade_out,
                generate_from_frame_offset=0
            )

            num_blend_frames = min(len(kb_frames_A_for_fade_component), len(kb_frames_B_for_fade_component))
            if num_blend_frames < frames_for_A_fade_out:
                print(f"Warning: Mismatch in KB frames for fade. Blending {num_blend_frames} of {frames_for_A_fade_out} expected.")

            if num_blend_frames == 0:
                print(f"Error: Could not generate KB frames for fade. Holding last frame of current image.")
                # Similar fallback as when next_img_bgr is None
                last_frame_A_to_hold = ken_burns_frames_A_main[-1] if ken_burns_frames_A_main else None
                if last_frame_A_to_hold is None:
                    temp_A_frame = apply_ken_burns_effect(current_img_bgr, video_width, video_height, current_movement_type, total_frames_for_segment, 1, frame_offset_A_fade_component) # Use imported version
                    if temp_A_frame: last_frame_A_to_hold = temp_A_frame[0]
                if last_frame_A_to_hold is not None:
                    for _ in range(frames_for_A_fade_out): video_writer.write(last_frame_A_to_hold)
                else:
                    for _ in range(frames_for_A_fade_out): video_writer.write(np.zeros((video_height, video_width, 3), dtype=np.uint8))

            else:
                for k in range(num_blend_frames):
                    frame_A = kb_frames_A_for_fade_component[k]
                    frame_B = kb_frames_B_for_fade_component[k]
                    alpha = (k + 1) / frames_for_A_fade_out
                    blended_frame = cv2.addWeighted(frame_A, 1.0 - alpha, frame_B, alpha, 0)
                    video_writer.write(blended_frame)

                if num_blend_frames > 0 and num_blend_frames < frames_for_A_fade_out:
                    # Hold the last successfully blended frame for the remainder of the fade duration
                    last_blended_frame = cv2.addWeighted(
                        kb_frames_A_for_fade_component[num_blend_frames-1], 0.0, # Effectively frame_B at full alpha
                        kb_frames_B_for_fade_component[num_blend_frames-1], 1.0, 0)
                    for _ in range(frames_for_A_fade_out - num_blend_frames):
                        video_writer.write(last_blended_frame)
    else: # Not fading, or last segment
        frames_written_so_far = len(ken_burns_frames_A_main)
        remaining_frames_in_segment = total_frames_for_segment - frames_written_so_far
        if remaining_frames_in_segment > 0:
            last_frame_to_hold = None
            if ken_burns_frames_A_main:
                last_frame_to_hold = ken_burns_frames_A_main[-1]
            elif total_frames_for_segment > 0: # Generate one if none exist
                temp_A_frame = apply_ken_burns_effect(current_img_bgr, video_width, video_height, current_movement_type, total_frames_for_segment, 1, frame_offset_A_main) # Use imported version
                if temp_A_frame: last_frame_to_hold = temp_A_frame[0]

            if last_frame_to_hold is not None:
                for _ in range(remaining_frames_in_segment):
                    video_writer.write(last_frame_to_hold)
            else:
                print(f"Warning: No frames for image {segment_index + 1} in last part. Black frames for {remaining_frames_in_segment}.")
                for _ in range(remaining_frames_in_segment):
                    video_writer.write(np.zeros((video_height, video_width, 3), dtype=np.uint8))

    # ken_burns_sequence_index increment is handled by the caller (_generate_visual_frames)
    # ken_burns_sequence_index += 1


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
    if not _prepare_and_combine_audio_ffmpeg( # Updated call
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
        if add_background_music(video_after_captions_path, background_music_file, video_with_background_music_path): # Updated call
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

# _add_background_music has been moved to ffmpeg_utils.py
# and is imported above.
