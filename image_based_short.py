#!/usr/bin/env python3
"""
Main script for generating YouTube Shorts from images.

This script takes a source text file, generates a narration script and image prompts using OpenAI,
creates audio narration using Azure Cognitive Services, generates images using DALL-E,
and finally combines them into a video with captions and background music.
"""

import sys
import os
from typing import Dict, List, Tuple

# Import project modules
import short_narration
import short_images
import short_video
from config import AppConfig, load_source_material, initialize_openai_client # Updated import
import openai # For type hinting


def generate_script_and_image_prompts(
    oai_client: openai.OpenAI,
    source_material_text: str,
    config: AppConfig
) -> Tuple[str, List[Dict[str, str]], List[str]]:
    """Generates script and image prompts. Loads from files if they exist."""
    response_file = os.path.join(config.base_output_dir, "response.txt")

    raw_response_text: str
    if os.path.exists(response_file) :
        print(f"Loading existing script from {os.path.basename(response_file)}.")
        with open(response_file, 'r', encoding='utf-8') as f:
            raw_response_text = f.read()
    else:
        print("Generating script with OpenAI...")
        try:
            # Ensure narration_system_prompt is loaded
            if not config.narration_system_prompt:
                print("Error: Narration system prompt is not loaded.")
                sys.exit(1)

            response = oai_client.chat.completions.create(
                model=config.openai_script_model,
                messages=[
                    {"role": "system", "content": config.narration_system_prompt},
                    {"role": "user", "content": f"Create a YouTube short narration based on the following source material:\n\n{source_material_text}"}
                ]
            )
            raw_response_text = response.choices[0].message.content or ""
            # Basic sanitization
            raw_response_text = raw_response_text.replace("’", "'").replace("`", "'").replace("…", "...").replace("“", '"').replace("”", '"')
            with open(response_file, "w", encoding='utf-8') as f:
                f.write(raw_response_text)
            print(f"Script saved to {response_file}")
        except openai.APIError as e:
            print(f"OpenAI API Error (script generation): {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error (script generation): {e}")
            sys.exit(1)

    parsed_data, narration_sentences = short_narration.parse_narration_text(raw_response_text)

    return raw_response_text, parsed_data, narration_sentences


def main() -> None:
    """Main script execution for image-based short generation."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <source_file_path> [settings_file_path] [--with-video]")
        sys.exit(1)

    source_file_arg = sys.argv[1]
    settings_file_arg = sys.argv[2] if len(sys.argv) > 2 else None
    create_video = (sys.argv[3] == "--with-video") if len(sys.argv) > 3 else False

    # Initialize Configuration
    config = AppConfig(source_file_path=source_file_arg, settings_file_path=settings_file_arg)
    print(f"Processing short: {config.short_name}")
    print(f"Output directory: {config.base_output_dir}")

    # Load source material
    source_material_content = load_source_material(config.source_file_path)

    # Initialize API Clients
    try:
        oai_client = initialize_openai_client(api_key=config.openai_api_key, base_url=config.openai_base_url)
        # The line short_images.set_openai_client(oai_client) will be removed later
        # as the client will be passed directly.
    except ValueError as e:
        print(f"Error initializing OpenAI client: {e}")
        sys.exit(1)

    if not config.azure_api_key or not config.azure_region:
        print("Warning: Azure API key/region not fully configured. Narration might fail.")
    # Azure credentials will be passed directly to create_narration_audio_files

    # Script Generation
    _, parsed_script_data, narrations_list = generate_script_and_image_prompts(
        oai_client, source_material_content, config
    )
    if not narrations_list:
        print("No narration sentences generated/parsed. Exiting.")
        sys.exit(1)

    # Narration Generation
    print("Generating narration audio...")
    short_narration.create_narration_audio_files(
        data=parsed_script_data,
        output_folder=config.narration_output_dir,
        azure_subscription_key=config.azure_api_key,
        azure_region=config.azure_region
    )

    # Image Generation
    print("Generating images...")
    # This call will be updated when short_images.py is refactored
    short_images.create_from_data(
        data=parsed_script_data,
        output_dir=config.image_output_dir,
        client=oai_client, # Pass client directly
        image_model=config.openai_image_model
    )

    if not create_video:
        print("Skipping video generation.")
        return

    # Video Creation
    print("Generating video...")
    short_video.create_short_video(
        narration_sentences=narrations_list,
        background_music_file=config.background_music_file_path,
        base_dir=config.base_output_dir,
        final_output_filename=os.path.basename(config.final_video_file_path), # Get just filename
        caption_settings=config.caption_settings,
        video_width=config.video_width,
        video_height=config.video_height,
        frame_rate=config.frame_rate,
        fade_duration_ms=config.fade_duration_ms
    )

    print(f"DONE! Your video should be at: {config.final_video_file_path}")

if __name__ == "__main__":
    main()
