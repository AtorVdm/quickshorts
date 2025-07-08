#!/usr/bin/env python3
"""
Main script for generating YouTube Shorts from a video background.

This script takes a source text file, generates a narration script using OpenAI,
uses a specified video as background, adds narration audio (Azure Cognitive Services),
captions (Captacity), and background music.
"""

import sys
import os
import re
from typing import List, Tuple, Dict, Any

# Import project modules
import short_narration
import short_video
from config import AppConfig, load_source_material, initialize_openai_client, NARRATION_VIDEO_PROMPT_FILENAME
import openai # For type hinting

def generate_script_for_video(
    oai_client: openai.OpenAI,
    source_material_text: str,
    narration_system_prompt: str, # Actual prompt string
    config: AppConfig
) -> Tuple[str, List[Dict[str, str]], List[str]]:
    """Generates script for video narration. Loads from file if it exists.
    Returns raw_response_text, parsed_script_data (for narration audio), and narration_sentences list.
    """
    response_file = os.path.join(config.base_output_dir, "response_video.txt")

    raw_response_text: str
    # Check if settings_file_path is provided. If so, it might indicate a desire to regenerate or use specific settings
    # that shouldn't rely on a generic cached response. So, only use cache if no settings_file_path.
    if os.path.exists(response_file) and not config.settings_file_path:
        print(f"Loading existing video script from {os.path.basename(response_file)}.")
        with open(response_file, 'r', encoding='utf-8') as f:
            raw_response_text = f.read()
    else:
        print("Generating video script with OpenAI...")
        try:
            if not narration_system_prompt:
                print("Error: Video narration system prompt is not loaded or empty.")
                sys.exit(1)

            response = oai_client.chat.completions.create(
                model=config.openai_script_model,
                messages=[
                    {"role": "system", "content": narration_system_prompt},
                    {"role": "user", "content": f"Create a YouTube short narration based on the following source material:\n\n{source_material_text}"}
                ]
            )
            raw_response_text = response.choices[0].message.content or ""
            # Basic sanitization
            raw_response_text = raw_response_text.replace("’", "'").replace("`", "'").replace("…", "...").replace("“", '"').replace("”", '"')
            with open(response_file, "w", encoding='utf-8') as f:
                f.write(raw_response_text)
            print(f"Video script saved to {response_file}")
        except openai.APIError as e:
            print(f"OpenAI API Error (video script generation): {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error (video script generation): {e}")
            sys.exit(1)

    # Use short_narration.parse_narration_text to parse the script
    # This function is expected to handle prompts that may or may not contain image descriptions.
    # For video prompts, it should correctly extract only narration.
    parsed_script_data, narration_sentences = short_narration.parse_narration_text(raw_response_text)

    if not narration_sentences:
        print("Warning: No narration sentences parsed by short_narration.parse_narration_text. This might result in a silent video or errors.")
    if not parsed_script_data:
        # This case should ideally be handled by parse_narration_text returning empty lists
        # but as a safeguard:
        print("Warning: Parsed script data is empty. Audio generation might fail or produce no audio.")
        # Ensure parsed_script_data is an empty list if parse_narration_text returns None or similar for data part
        parsed_script_data = []


    return raw_response_text, parsed_script_data, narration_sentences


def main_video() -> None:
    """Main script execution for video-based short generation."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <source_file_path> [settings_file_path] [video_background_path]")
        sys.exit(1)

    source_file_arg = sys.argv[1]
    settings_file_arg = None
    video_background_arg = os.path.join("resources", "video.webm") # Default

    # Process arguments more robustly
    # Example: video_based_short.py source.txt
    # Example: video_based_short.py source.txt settings.json
    # Example: video_based_short.py source.txt resources/my_video.webm
    # Example: video_based_short.py source.txt settings.json resources/my_video.webm

    idx = 2
    if len(sys.argv) > idx:
        # Is it a settings file? (heuristic: ends with .json and exists)
        if sys.argv[idx].endswith(".json") and os.path.exists(sys.argv[idx]):
            settings_file_arg = sys.argv[idx]
            idx += 1
        # Is it a video file? (heuristic: next arg exists)
        elif len(sys.argv) > idx : # if it's not json, assume it could be video path
            if os.path.exists(sys.argv[idx]): # check if it exists before assigning
                 video_background_arg = sys.argv[idx]
                 idx +=1 # increment only if this was treated as video
            elif sys.argv[idx].endswith(".mp4"): # if it ends with mp4 but doesn't exist
                print(f"Warning: Specified video file {sys.argv[idx]} not found. Will try default.")
                # video_background_arg remains default
                idx +=1 # consume this arg anyway
            # if it's not json and not an existing video, what is it? could be settings without .json
            # for now, if settings_file_arg is still None, assume it's settings
            elif settings_file_arg is None: # Check if settings_file_arg has not been assigned yet
                 settings_file_arg = sys.argv[idx] # Assign as settings file
                 idx +=1


    if len(sys.argv) > idx: # If there's another argument after processing settings/video
         #This means settings was first (or mistaken for video), now process video
        if os.path.exists(sys.argv[idx]):
            video_background_arg = sys.argv[idx]
        elif sys.argv[idx].endswith(".mp4"): # if it's an mp4 file but doesn't exist
             print(f"Warning: Specified video file {sys.argv[idx]} not found. Will try default.")
        # If it's not an existing file or an mp4, it could be a settings file if not already assigned.
        elif settings_file_arg is None: # Check if settings_file_arg is still available to be assigned
            settings_file_arg = sys.argv[idx]


    if not os.path.exists(video_background_arg):
        print(f"Warning: Video background file '{video_background_arg}' not found.")
        default_video_bg_path = os.path.join("resources", "video.webm")
        if video_background_arg != default_video_bg_path: # if specified was different from default
            print(f"Attempting to use default video background: '{default_video_bg_path}'")
            video_background_arg = default_video_bg_path

        if not os.path.exists(video_background_arg): # Check default again
            print(f"Error: Default video background '{video_background_arg}' also not found. Please ensure a video background is available at resources/video.webm or provide a valid path.")
            sys.exit(1)


    # Initialize Configuration
    config = AppConfig(source_file_path=source_file_arg, settings_file_path=settings_file_arg)
    print(f"Processing video-based short: {config.short_name}")
    print(f"Output directory: {config.base_output_dir}")
    print(f"Using video background: {video_background_arg}")
    if settings_file_arg:
        print(f"Using settings file: {settings_file_arg}")
    else:
        print("No settings file specified, using default settings.")


    # Load the specific narration prompt for video
    try:
        # NARRATION_VIDEO_PROMPT_FILENAME is imported from config
        with open(NARRATION_VIDEO_PROMPT_FILENAME, 'r', encoding='utf-8') as f:
            video_narration_prompt_text = f.read()
        config.narration_system_prompt = video_narration_prompt_text # Override the one loaded by AppConfig
        print(f"Loaded video narration prompt from {NARRATION_VIDEO_PROMPT_FILENAME}")
    except FileNotFoundError:
        print(f"Error: Video narration prompt file not found at {NARRATION_VIDEO_PROMPT_FILENAME}.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading video narration prompt file {NARRATION_VIDEO_PROMPT_FILENAME}: {e}")
        sys.exit(1)

    # Load source material
    source_material_content = load_source_material(config.source_file_path)

    # Initialize API Clients
    try:
        oai_client = initialize_openai_client(api_key=config.openai_api_key, base_url=config.openai_base_url)
    except ValueError as e:
        print(f"Error initializing OpenAI client: {e}")
        sys.exit(1)

    if not config.azure_api_key or not config.azure_region:
        print("Warning: Azure API key/region not fully configured. Narration audio generation might fail.")

    # Script Generation
    _, parsed_video_script_data, narrations_list = generate_script_for_video(
        oai_client, source_material_content, config.narration_system_prompt, config
    )

    if not narrations_list:
        print("No narration sentences generated/parsed. Video will likely be silent or very short, or generation might fail.")
        # Allow to continue, subsequent functions should handle empty lists gracefully.

    # Narration Generation
    print("Generating narration audio for video short...")
    short_narration.create_narration_audio_files(
        data=parsed_video_script_data,
        output_folder=config.narration_output_dir,
        azure_subscription_key=config.azure_api_key,
        azure_region=config.azure_region
    )

    # Video Creation from Background Video
    print("Generating video from background...")
    # Ensure short_video module is imported, which it is.
    # The function create_short_from_background_video will be added to short_video.py
    short_video.create_short_from_background_video(
        background_video_path=video_background_arg,
        narration_sentences=narrations_list, # Pass the list of strings
        narration_audio_dir=config.narration_output_dir, # Pass narration dir for duration calculation
        background_music_file=config.background_music_file_path,
        base_dir=config.base_output_dir,
        final_output_filename=os.path.basename(config.final_video_file_path),
        caption_settings=config.caption_settings,
        video_width=config.video_width,
        video_height=config.video_height,
        frame_rate=config.frame_rate
    )

    print(f"DONE! Your video (from background) should be at: {config.final_video_file_path}")

if __name__ == "__main__":
    main_video()
