# This file contains video processing utility functions,
# including image manipulation for video frames and effects like Ken Burns.

import cv2
import numpy as np
import math
from typing import List, Tuple

# Placeholders for resize, frame creation, and Ken Burns functions.
# Will be populated in the next step.

# Define Ken Burns movement sequence constant
KEN_BURNS_SEQUENCE = [
    "top_left_to_bottom_right",
    "bottom_left_to_top_right",
    "top_right_to_bottom_left",
    "bottom_right_to_top_left",
]

def resize_image_to_fit_canvas( # Renamed to be public
    image: np.ndarray,
    canvas_width: int,
    canvas_height: int
) -> np.ndarray:
    """
    Resizes an image to fit within specified dimensions while preserving aspect ratio.
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


def create_frame_with_centered_image( # Renamed to be public
    image_resized: np.ndarray,
    video_width: int,
    video_height: int
) -> np.ndarray:
    """Creates a black frame and centers the resized image onto it."""
    frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
    img_h, img_w = image_resized.shape[:2]
    y_offset = (video_height - img_h) // 2
    x_offset = (video_width - img_w) // 2
    y_offset, x_offset = max(0, y_offset), max(0, x_offset)
    y_end, x_end = min(video_height, y_offset + img_h), min(video_width, x_offset + img_w)
    img_y_end, img_x_end = min(img_h, y_end - y_offset), min(img_w, x_end - x_offset)
    frame[y_offset:y_end, x_offset:x_end] = image_resized[0:img_y_end, 0:img_x_end]
    return frame


def _calculate_ken_burns_pan_coordinates(
    img_w: int, img_h: int,
    crop_w_on_source: int, crop_h_on_source: int,
    movement_type: str
) -> Tuple[int, int, int, int]:
    """
    Calculates the start (x,y) and end (x,y) coordinates for the Ken Burns pan effect.
    (Internal helper for apply_ken_burns_effect)
    """
    max_crop_x = img_w - crop_w_on_source
    max_crop_y = img_h - crop_h_on_source
    start_x, start_y, end_x, end_y = 0, 0, 0, 0

    if max_crop_x < 0:
        start_x = end_x = (img_w - crop_w_on_source) // 2
        max_crop_x = start_x
    if max_crop_y < 0:
        start_y = end_y = (img_h - crop_h_on_source) // 2
        max_crop_y = start_y

    if movement_type == "top_left_to_bottom_right":
        start_x, start_y = (img_w - crop_w_on_source) // 2 if max_crop_x < 0 else 0, \
                           (img_h - crop_h_on_source) // 2 if max_crop_y < 0 else 0
        end_x, end_y = max_crop_x, max_crop_y
    elif movement_type == "bottom_left_to_top_right":
        start_x, start_y = (img_w - crop_w_on_source) // 2 if max_crop_x < 0 else 0, max_crop_y
        end_x, end_y = max_crop_x, (img_h - crop_h_on_source) // 2 if max_crop_y < 0 else 0
    elif movement_type == "top_right_to_bottom_left":
        start_x, start_y = max_crop_x, (img_h - crop_h_on_source) // 2 if max_crop_y < 0 else 0
        end_x, end_y = (img_w - crop_w_on_source) // 2 if max_crop_x < 0 else 0, max_crop_y
    elif movement_type == "bottom_right_to_top_left":
        start_x, start_y = max_crop_x, max_crop_y
        end_x, end_y = (img_w - crop_w_on_source) // 2 if max_crop_x < 0 else 0, \
                       (img_h - crop_h_on_source) // 2 if max_crop_y < 0 else 0

    if not (img_w < crop_w_on_source):
        start_x, end_x = max(0, start_x), max(0, end_x)
    if not (img_h < crop_h_on_source):
        start_y, end_y = max(0, start_y), max(0, end_y)
    return start_x, start_y, end_x, end_y


def _generate_single_ken_burns_frame(
    image_bgr: np.ndarray,
    video_width: int, video_height: int,
    img_w: int, img_h: int,
    crop_w_on_source: int, crop_h_on_source: int,
    start_x: int, start_y: int, end_x: int, end_y: int,
    current_animation_frame_index: int,
    full_animation_duration_frames: int
) -> np.ndarray:
    """
    Generates a single frame for the Ken Burns effect.
    (Internal helper for apply_ken_burns_effect)
    """
    progress = 0.0
    if full_animation_duration_frames == 1: progress = 0.0
    elif full_animation_duration_frames > 1: progress = current_animation_frame_index / (full_animation_duration_frames - 1)
    progress = min(1.0, max(0.0, progress))

    current_x = int(round(start_x + (end_x - start_x) * progress))
    current_y = int(round(start_y + (end_y - start_y) * progress))
    crop_x1, crop_y1 = max(0, current_x), max(0, current_y)
    actual_crop_w, actual_crop_h = crop_w_on_source, crop_h_on_source
    paste_x_offset, paste_y_offset = 0, 0

    if current_x < 0:
        actual_crop_w, crop_x1 = img_w, 0
        paste_x_offset = int(round(abs(current_x) * (video_width / crop_w_on_source)))
    else: actual_crop_w = min(crop_w_on_source, img_w - crop_x1)
    if current_y < 0:
        actual_crop_h, crop_y1 = img_h, 0
        paste_y_offset = int(round(abs(current_y) * (video_height / crop_h_on_source)))
    else: actual_crop_h = min(crop_h_on_source, img_h - crop_y1)

    if actual_crop_w <= 0 or actual_crop_h <= 0:
        return np.zeros((video_height, video_width, 3), dtype=np.uint8)
    cropped_image_part = image_bgr[crop_y1: crop_y1 + actual_crop_h, crop_x1: crop_x1 + actual_crop_w]
    if cropped_image_part.size == 0:
        return np.zeros((video_height, video_width, 3), dtype=np.uint8)

    target_w_for_resize, target_h_for_resize = video_width, video_height
    if current_x < 0 or (crop_x1 + actual_crop_w < current_x + crop_w_on_source and current_x >=0) :
        target_w_for_resize = int(round(actual_crop_w * (video_width / crop_w_on_source)))
    if current_y < 0 or (crop_y1 + actual_crop_h < current_y + crop_h_on_source and current_y >=0):
        target_h_for_resize = int(round(actual_crop_h * (video_height / crop_h_on_source)))
    target_w_for_resize, target_h_for_resize = max(1, target_w_for_resize), max(1, target_h_for_resize)

    resized_crop = cv2.resize(cropped_image_part, (target_w_for_resize, target_h_for_resize), interpolation=cv2.INTER_LANCZOS4)
    final_frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
    current_paste_x, current_paste_y = paste_x_offset, paste_y_offset

    if target_w_for_resize < video_width and paste_x_offset == 0 : current_paste_x = (video_width - target_w_for_resize) // 2
    if target_h_for_resize < video_height and paste_y_offset == 0 : current_paste_y = (video_height - target_h_for_resize) // 2
    current_paste_x, current_paste_y = max(0, current_paste_x), max(0, current_paste_y)

    paste_h, paste_w = resized_crop.shape[:2]
    y_slice_end, x_slice_end = min(video_height, current_paste_y + paste_h), min(video_width, current_paste_x + paste_w)
    img_slice_h, img_slice_w = min(paste_h, y_slice_end - current_paste_y), min(paste_w, x_slice_end - current_paste_x)

    if img_slice_h > 0 and img_slice_w > 0:
        final_frame[current_paste_y:y_slice_end, current_paste_x:x_slice_end] = resized_crop[0:img_slice_h, 0:img_slice_w]
    return final_frame


def apply_ken_burns_effect( # Renamed to be public
    image_bgr: np.ndarray,
    video_width: int,
    video_height: int,
    movement_type: str,
    full_animation_duration_frames: int,
    generate_count: int,
    generate_from_frame_offset: int = 0
) -> List[np.ndarray]:
    """
    Applies a Ken Burns effect (pan and zoom) to an image.
    """
    frames = []
    img_h, img_w = image_bgr.shape[:2]
    zoom_level = 0.9
    crop_w_on_source = int(round(video_width / (1/zoom_level)))
    crop_h_on_source = int(round(video_height / (1/zoom_level)))
    crop_w_on_source, crop_h_on_source = max(1, crop_w_on_source), max(1, crop_h_on_source)

    start_x, start_y, end_x, end_y = _calculate_ken_burns_pan_coordinates(
        img_w, img_h, crop_w_on_source, crop_h_on_source, movement_type
    )

    if generate_count <= 0: return frames
    if full_animation_duration_frames <=0:
        for _ in range(generate_count): frames.append(np.zeros((video_height, video_width, 3), dtype=np.uint8))
        return frames

    for k in range(generate_count):
        current_animation_frame_index = generate_from_frame_offset + k
        frame = _generate_single_ken_burns_frame(
            image_bgr, video_width, video_height, img_w, img_h,
            crop_w_on_source, crop_h_on_source, start_x, start_y, end_x, end_y,
            current_animation_frame_index, full_animation_duration_frames
        )
        frames.append(frame)

    if not frames:
        frames.append(np.zeros((video_height, video_width, 3), dtype=np.uint8))
    return frames
