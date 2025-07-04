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
    img_h, img_w = image_resized.shape[:2]
    y_offset = (video_height - img_h) // 2
    x_offset = (video_width - img_w) // 2

    # Ensure offsets are not negative (can happen if resized image is larger than canvas, though _resize should prevent)
    y_offset = max(0, y_offset)
    x_offset = max(0, x_offset)

    # Ensure the slice does not go out of bounds of the frame
    y_end = min(video_height, y_offset + img_h)
    x_end = min(video_width, x_offset + img_w)

    # Ensure the slice does not go out of bounds of the image_resized
    img_y_end = min(img_h, y_end - y_offset)
    img_x_end = min(img_w, x_end - x_offset)

    frame[y_offset:y_end, x_offset:x_end] = image_resized[0:img_y_end, 0:img_x_end]
    return frame


# Define Ken Burns movement sequence
KEN_BURNS_SEQUENCE = [
    "top_left_to_bottom_right",
    "bottom_left_to_top_right",
    "top_right_to_bottom_left",
    "bottom_right_to_top_left",
]
ken_burns_sequence_index = 0

def _apply_ken_burns_effect(
    image_bgr: np.ndarray,
    video_width: int,
    video_height: int,
    movement_type: str,
    full_animation_duration_frames: int, # Total frames the KB effect would span if played fully
    generate_count: int,                 # How many frames to generate in this call
    generate_from_frame_offset: int = 0  # Starting frame number within the full animation
) -> List[np.ndarray]:
    """
    Applies a Ken Burns effect (pan and zoom) to an image, generating a specific segment of it.
    """
    # print(f"KB: type={movement_type}, full_dur={full_animation_duration_frames}, gen_count={generate_count}, offset={generate_from_frame_offset}")

    frames = []
    img_h, img_w = image_bgr.shape[:2]

    zoom_level = 0.9 # We see 80% of the image content, so effectively zoom in by 1/0.9 = 1.1

    # crop_w_on_source and crop_h_on_source are the dimensions of the window
    # cut from the original image. This window will be resized to video_width, video_height.
    # For the effective zoom to be 1.1, this window should be 1/1.1 = 0.9 times
    # the video dimensions, if the image was already at video resolution.
    # More generally, this is the part of the *source* image that will be shown.
    # The size of this crop window on the source image needs to maintain the video's aspect ratio.

    crop_w_on_source = int(round(video_width / (1/zoom_level))) # video_width * zoom_level
    crop_h_on_source = int(round(video_height / (1/zoom_level))) # video_height * zoom_level

    # Ensure crop dimensions are at least 1x1
    crop_w_on_source = max(1, crop_w_on_source)
    crop_h_on_source = max(1, crop_h_on_source)

    # Define movement start and end points (top-left corner of the crop window)
    # These are relative to the original image_bgr
    start_x, start_y = 0, 0
    end_x, end_y = 0, 0

    start_x, start_y, end_x, end_y = _calculate_ken_burns_pan_coordinates(
        img_w, img_h, crop_w_on_source, crop_h_on_source, movement_type
    )

    # Ensure generate_count is at least 0, and if full_animation_duration_frames is 0, handle that.
    if generate_count <= 0:
        return frames # Return empty list if no frames are to be generated

    if full_animation_duration_frames <=0: # Should not happen with current calling logic
        print(f"Warning: full_animation_duration_frames is {full_animation_duration_frames}. Cannot generate Ken Burns effect.")
        for _ in range(generate_count):
            frames.append(np.zeros((video_height, video_width, 3), dtype=np.uint8))
        return frames

    for k in range(generate_count):
        current_animation_frame_index = generate_from_frame_offset + k
        frame = _generate_single_ken_burns_frame(
            image_bgr=image_bgr,
            video_width=video_width, video_height=video_height,
            img_w=img_w, img_h=img_h,
            crop_w_on_source=crop_w_on_source, crop_h_on_source=crop_h_on_source,
            start_x=start_x, start_y=start_y, end_x=end_x, end_y=end_y,
            current_animation_frame_index=current_animation_frame_index,
            full_animation_duration_frames=full_animation_duration_frames
        )
        frames.append(frame)

    if not frames: # Should have been handled by generate_count > 0 or full_animation_duration_frames > 0
        print("Warning: Ken Burns effect produced no frames (after loop). Returning a single black frame.")
        black_frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
        frames.append(black_frame)

    return frames


def _calculate_ken_burns_pan_coordinates(
    img_w: int, img_h: int,
    crop_w_on_source: int, crop_h_on_source: int,
    movement_type: str
) -> Tuple[int, int, int, int]:
    """
    Calculates the start (x,y) and end (x,y) coordinates for the Ken Burns pan effect
    based on the movement type. Handles cases where the image is smaller than the crop window.
    """
    max_crop_x = img_w - crop_w_on_source
    max_crop_y = img_h - crop_h_on_source

    start_x, start_y, end_x, end_y = 0, 0, 0, 0

    # Handle cases where image is smaller than the crop window by centering the crop
    if max_crop_x < 0:
        start_x = end_x = (img_w - crop_w_on_source) // 2
        max_crop_x = start_x # No panning range in x
    if max_crop_y < 0:
        start_y = end_y = (img_h - crop_h_on_source) // 2
        max_crop_y = start_y # No panning range in y

    if movement_type == "top_left_to_bottom_right":
        start_x, start_y = (img_w - crop_w_on_source) // 2 if max_crop_x < 0 else 0, \
                           (img_h - crop_h_on_source) // 2 if max_crop_y < 0 else 0
        end_x, end_y = max_crop_x, max_crop_y
    elif movement_type == "bottom_left_to_top_right":
        start_x, start_y = (img_w - crop_w_on_source) // 2 if max_crop_x < 0 else 0, \
                           max_crop_y
        end_x, end_y = max_crop_x, \
                       (img_h - crop_h_on_source) // 2 if max_crop_y < 0 else 0
    elif movement_type == "top_right_to_bottom_left":
        start_x, start_y = max_crop_x, \
                           (img_h - crop_h_on_source) // 2 if max_crop_y < 0 else 0
        end_x, end_y = (img_w - crop_w_on_source) // 2 if max_crop_x < 0 else 0, \
                       max_crop_y
    elif movement_type == "bottom_right_to_top_left":
        start_x, start_y = max_crop_x, max_crop_y
        end_x, end_y = (img_w - crop_w_on_source) // 2 if max_crop_x < 0 else 0, \
                       (img_h - crop_h_on_source) // 2 if max_crop_y < 0 else 0

    # Final clamping for coordinates if they were not adjusted for small images
    # If image was smaller, start_x/end_x (or y) are already set to the centered position.
    if not (img_w < crop_w_on_source): # Only adjust if image width is >= crop width
        start_x = max(0, start_x)
        end_x = max(0, end_x)
    if not (img_h < crop_h_on_source): # Only adjust if image height is >= crop height
        start_y = max(0, start_y)
        end_y = max(0, end_y)

    return start_x, start_y, end_x, end_y


def _generate_single_ken_burns_frame(
    image_bgr: np.ndarray,
    video_width: int, video_height: int,
    img_w: int, img_h: int, # Original image dimensions
    crop_w_on_source: int, crop_h_on_source: int, # Crop window size on original image
    start_x: int, start_y: int, end_x: int, end_y: int, # Pan coordinates
    current_animation_frame_index: int,
    full_animation_duration_frames: int
) -> np.ndarray:
    """Generates a single frame for the Ken Burns effect."""
    progress = 0.0
    if full_animation_duration_frames == 1:
        progress = 0.0
    elif full_animation_duration_frames > 1:
        progress = current_animation_frame_index / (full_animation_duration_frames - 1)
    progress = min(1.0, max(0.0, progress))

    current_x = int(round(start_x + (end_x - start_x) * progress))
    current_y = int(round(start_y + (end_y - start_y) * progress))

    crop_x1 = max(0, current_x)
    crop_y1 = max(0, current_y)

    actual_crop_w = crop_w_on_source
    actual_crop_h = crop_h_on_source
    paste_x_offset = 0
    paste_y_offset = 0

    if current_x < 0: # Image width is less than crop window, and crop is centered
        actual_crop_w = img_w
        crop_x1 = 0
        paste_x_offset = int(round(abs(current_x) * (video_width / crop_w_on_source)))
    else:
        actual_crop_w = min(crop_w_on_source, img_w - crop_x1)

    if current_y < 0: # Image height is less than crop window, and crop is centered
        actual_crop_h = img_h
        crop_y1 = 0
        paste_y_offset = int(round(abs(current_y) * (video_height / crop_h_on_source)))
    else:
        actual_crop_h = min(crop_h_on_source, img_h - crop_y1)

    if actual_crop_w <= 0 or actual_crop_h <= 0:
        print(f"Warning: Ken Burns crop dimension zero or negative ({actual_crop_w}x{actual_crop_h}). Black frame.")
        return np.zeros((video_height, video_width, 3), dtype=np.uint8)

    cropped_image_part = image_bgr[crop_y1 : crop_y1 + actual_crop_h, crop_x1 : crop_x1 + actual_crop_w]
    if cropped_image_part.size == 0:
        print(f"Warning: Ken Burns cropped_image_part is empty. Black frame.")
        return np.zeros((video_height, video_width, 3), dtype=np.uint8)

    target_w_for_resize = video_width
    target_h_for_resize = video_height
    if current_x < 0 or (crop_x1 + actual_crop_w < current_x + crop_w_on_source and current_x >=0) :
        target_w_for_resize = int(round(actual_crop_w * (video_width / crop_w_on_source)))
    if current_y < 0 or (crop_y1 + actual_crop_h < current_y + crop_h_on_source and current_y >=0):
        target_h_for_resize = int(round(actual_crop_h * (video_height / crop_h_on_source)))

    target_w_for_resize = max(1, target_w_for_resize)
    target_h_for_resize = max(1, target_h_for_resize)

    resized_crop = cv2.resize(cropped_image_part, (target_w_for_resize, target_h_for_resize), interpolation=cv2.INTER_LANCZOS4)
    final_frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)

    current_paste_x = paste_x_offset
    current_paste_y = paste_y_offset
    if target_w_for_resize < video_width and paste_x_offset == 0 :
        current_paste_x = (video_width - target_w_for_resize) // 2
    if target_h_for_resize < video_height and paste_y_offset == 0 :
        current_paste_y = (video_height - target_h_for_resize) // 2

    current_paste_x = max(0, current_paste_x)
    current_paste_y = max(0, current_paste_y)

    paste_h, paste_w = resized_crop.shape[:2]
    y_slice_end = min(video_height, current_paste_y + paste_h)
    x_slice_end = min(video_width, current_paste_x + paste_w)
    img_slice_h = min(paste_h, y_slice_end - current_paste_y)
    img_slice_w = min(paste_w, x_slice_end - current_paste_x)

    if img_slice_h > 0 and img_slice_w > 0:
        final_frame[current_paste_y:y_slice_end, current_paste_x:x_slice_end] = resized_crop[0:img_slice_h, 0:img_slice_w]
    return final_frame


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
    global ken_burns_sequence_index # Use the global index

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
    # Ensure frames_per_fade is at least 1 if fade_duration_ms > 0, otherwise it can be 0
    if fade_duration_ms > 0 and frames_per_fade == 0:
        frames_per_fade = 1

    for i in range(num_narrations):
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
            frames_per_fade=frames_per_fade
        )
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
    frames_per_fade: int
):
    """
    Processes a single visual segment: displays an image with Ken Burns effect
    and handles the cross-fade to the next image.
    """
    global ken_burns_sequence_index

    current_image_path = image_files[segment_index % num_images]
    next_image_for_fade_path = image_files[(segment_index + 1) % num_images] if segment_index < num_narrations - 1 else current_image_path

    current_img_bgr = cv2.imread(current_image_path)
    if current_img_bgr is None:
        print(f"Error: Could not read image {current_image_path}. Skipping visual segment {segment_index + 1}.")
        # Write black frames for the expected duration or handle error appropriately
        # For now, just returning, which means this segment will be missing.
        # A more robust solution might involve calculating expected duration and writing black frames.
        return

    narration_audio_path = os.path.join(narration_dir, f"narration_{segment_index + 1}.mp3")
    segment_duration_ms = get_audio_duration_ms(narration_audio_path)
    if segment_duration_ms == 0:
        print(f"Warning: Audio duration for {narration_audio_path} is 0. Defaulting to 1s for frame calculation.")
        segment_duration_ms = 1000

    total_frames_for_segment = math.floor(segment_duration_ms / 1000 * frame_rate)
    if total_frames_for_segment == 0 and segment_duration_ms > 0: total_frames_for_segment = 1

    current_movement_type = KEN_BURNS_SEQUENCE[ken_burns_sequence_index % len(KEN_BURNS_SEQUENCE)]
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
        ken_burns_sequence_index += 1
        return

    frame_offset_A_main = 0
    if segment_index > 0 and frames_per_fade > 0:
        frame_offset_A_main = frames_per_fade

    ken_burns_frames_A_main = []
    actual_frames_to_generate_A_main = frames_for_A_main_display
    if frame_offset_A_main + actual_frames_to_generate_A_main > total_frames_for_segment:
         actual_frames_to_generate_A_main = max(0, total_frames_for_segment - frame_offset_A_main - frames_for_A_fade_out)

    if actual_frames_to_generate_A_main > 0:
        ken_burns_frames_A_main = _apply_ken_burns_effect(
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
        kb_frames_A_for_fade_component = _apply_ken_burns_effect(
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
                 temp_A_frame = _apply_ken_burns_effect(current_img_bgr, video_width, video_height, current_movement_type, total_frames_for_segment, 1, frame_offset_A_fade_component)
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

            next_image_movement_type_index = (ken_burns_sequence_index + 1) % len(KEN_BURNS_SEQUENCE)
            next_image_movement_type = KEN_BURNS_SEQUENCE[next_image_movement_type_index]

            kb_frames_B_for_fade_component = _apply_ken_burns_effect(
                image_bgr=next_img_bgr, video_width=video_width, video_height=video_height,
                movement_type=next_image_movement_type,
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
                    temp_A_frame = _apply_ken_burns_effect(current_img_bgr, video_width, video_height, current_movement_type, total_frames_for_segment, 1, frame_offset_A_fade_component)
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
                temp_A_frame = _apply_ken_burns_effect(current_img_bgr, video_width, video_height, current_movement_type, total_frames_for_segment, 1, frame_offset_A_main)
                if temp_A_frame: last_frame_to_hold = temp_A_frame[0]

            if last_frame_to_hold is not None:
                for _ in range(remaining_frames_in_segment):
                    video_writer.write(last_frame_to_hold)
            else:
                print(f"Warning: No frames for image {segment_index + 1} in last part. Black frames for {remaining_frames_in_segment}.")
                for _ in range(remaining_frames_in_segment):
                    video_writer.write(np.zeros((video_height, video_width, 3), dtype=np.uint8))

    ken_burns_sequence_index += 1


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
