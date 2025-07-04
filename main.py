#!/usr/bin/env python3
"""
Main script for generating YouTube Shorts.

This script takes a source text file, generates a narration script and image prompts using OpenAI,
creates audio narration using Azure Cognitive Services, generates images using DALL-E,
and finally combines them into a video with captions.
"""

import json
import sys
import os
from typing import Dict, Any, List, Tuple, Optional

# Import project modules
import short_narration
import short_images
import short_video
from openai_client import initialize_openai_client # Corrected import
import openai # For type hinting

# System prompt for YouTube Short narration generation (defined once)
NARRATION_SYSTEM_PROMPT_TEXT = """You are a YouTube short narration generator. You generate 50 seconds to 70 seconds of narration. The shorts you create have a background that fades from image to image as the narration is going on.

You will need to generate descriptions of images for each of the sentences in the short. The image descriptions should use the key words from the content they describe. They will be passed to an AI image generator. Mention that the image should occupy the entire picture without cuts on any side. DO NOT IN ANY CIRCUMSTANCES use names of celebrities or people in the image descriptions. It is illegal to generate images of celebrities. Only describe persons without their names. Do not reference any real person or group in the image descriptions. Don't mention the female figure or other sexual content in the images because they are not allowed.

You are however allowed to use any content, including real names in the narration. Only image descriptions are restricted.

Note that the narration will be fed into a text-to-speech engine, so don't use special characters.

Respond with a pair of an image description in square brackets and a narration below it. Both of them should be on their own lines, as follows:

###

[Description of a background image]

Narrator: "One sentence of narration"

[Description of a background image]

Narrator: "One sentence of narration"

[Description of a background image]

Narrator: "One sentence of narration"

###

The short should be 6 sentences maximum.

You should add a description of a fitting backround image in between all of the narrations. It will later be used to generate an image with AI.
"""

class AppConfig:
    """Holds all configuration for the application."""
    def __init__(self, source_file_path: str, settings_file_path: Optional[str]):
        # File paths
        self.source_file_path: str = source_file_path
        self.settings_file_path: Optional[str] = settings_file_path

        # Derived paths (initialized in _setup_paths)
        self.short_name: str = ""
        self.base_output_dir: str = ""
        self.narration_output_dir: str = ""
        self.image_output_dir: str = ""
        self.final_video_file_path: str = ""

        # API Keys & Endpoints (from environment or config file)
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.openai_base_url: Optional[str] = os.getenv("OPENAI_API_URL")
        self.azure_api_key: Optional[str] = os.getenv("AZURE_API_KEY")
        self.azure_region: Optional[str] = os.getenv("AZURE_REGION")

        # Model Settings
        self.openai_script_model: str = "openai/gpt-4o"
        self.openai_image_model: str = "openai/dall-e-3"

        # Video Settings
        self.video_width: int = 1080
        self.video_height: int = 1920
        self.frame_rate: int = 30
        self.fade_duration_ms: int = 1000

        self.caption_settings: Dict[str, Any] = {}
        self.narration_system_prompt: str = NARRATION_SYSTEM_PROMPT_TEXT

        self._load_settings_from_file() # Load from JSON, potentially overriding above
        self._setup_paths() # Setup derived paths

    def _load_settings_from_file(self) -> None:
        if not self.settings_file_path:
            return
        try:
            with open(self.settings_file_path, 'r', encoding='utf-8') as f:
                json_settings = json.load(f)

            self.openai_api_key = json_settings.get("OPENAI_API_KEY", self.openai_api_key)
            self.openai_base_url = json_settings.get("OPENAI_API_URL", self.openai_base_url)
            self.azure_api_key = json_settings.get("AZURE_API_KEY", self.azure_api_key)
            self.azure_region = json_settings.get("AZURE_REGION", self.azure_region)

            self.openai_script_model = json_settings.get("OPENAI_SCRIPT_MODEL", self.openai_script_model)
            self.openai_image_model = json_settings.get("OPENAI_IMAGE_MODEL", self.openai_image_model)

            self.video_width = json_settings.get("VIDEO_WIDTH", self.video_width)
            self.video_height = json_settings.get("VIDEO_HEIGHT", self.video_height)
            self.frame_rate = json_settings.get("FRAME_RATE", self.frame_rate)
            self.fade_duration_ms = json_settings.get("FADE_DURATION_MS", self.fade_duration_ms)

            self.caption_settings = json_settings.get("CAPTION_SETTINGS", self.caption_settings) # Ensure default is kept if not in file
            self.narration_system_prompt = json_settings.get("NARRATION_SYSTEM_PROMPT", self.narration_system_prompt)
            print(f"Loaded and applied settings from {self.settings_file_path}")
        except FileNotFoundError:
            print(f"Info: Settings file not found at {self.settings_file_path}. Using default/env settings.")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {self.settings_file_path}. Using default/env settings.")
        except Exception as e:
            print(f"Error reading settings file {self.settings_file_path}: {e}. Using default/env settings.")

    def _setup_paths(self) -> None:
        self.short_name = os.path.splitext(os.path.basename(self.source_file_path))[0]
        self.base_output_dir = os.path.join("shorts", self.short_name)
        self.narration_output_dir = os.path.join(self.base_output_dir, "narrations")
        self.image_output_dir = os.path.join(self.base_output_dir, "images")
        self.final_video_file_path = os.path.join(self.base_output_dir, f"{self.short_name}.mp4")

        os.makedirs(self.base_output_dir, exist_ok=True)
        os.makedirs(self.narration_output_dir, exist_ok=True)
        os.makedirs(self.image_output_dir, exist_ok=True)


def load_source_material(file_path: str) -> str:
    """Loads source material from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Source file not found at {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading source file {file_path}: {e}")
        sys.exit(1)


def generate_script_and_image_prompts(
    oai_client: openai.OpenAI,
    source_material_text: str,
    config: AppConfig
) -> Tuple[str, List[Dict[str, str]], List[str]]:
    """
    Generates narration script and image prompts using OpenAI.
    Saves response and parsed data to files. If files exist, loads from them.
    """
    response_file = os.path.join(config.base_output_dir, "response.txt")
    data_file = os.path.join(config.base_output_dir, "data.json")

    raw_response_text: str
    if os.path.exists(response_file) and os.path.exists(data_file):
        print(f"Found existing {os.path.basename(response_file)} and {os.path.basename(data_file)}. Loading script.")
        with open(response_file, 'r', encoding='utf-8') as f:
            raw_response_text = f.read()
    else:
        print("Generating script with OpenAI...")
        try:
            response = oai_client.chat.completions.create(
                model=config.openai_script_model,
                messages=[
                    {"role": "system", "content": config.narration_system_prompt},
                    {"role": "user", "content": f"Create a YouTube short narration based on the following source material:\n\n{source_material_text}"}
                ]
            )
            raw_response_text = response.choices[0].message.content or ""
            raw_response_text = raw_response_text.replace("’", "'").replace("`", "'").replace("…", "...").replace("“", '"').replace("”", '"')
            with open(response_file, "w", encoding='utf-8') as f:
                f.write(raw_response_text)
            print(f"Script saved to {response_file}")
        except openai.APIError as e:
            print(f"OpenAI API Error during script generation: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error during script generation: {e}")
            sys.exit(1)

    parsed_data, narration_sentences = short_narration.parse_narration_text(raw_response_text)

    if not (os.path.exists(response_file) and os.path.exists(data_file)) or not os.path.exists(data_file):
        with open(data_file, "w", encoding='utf-8') as f:
            json.dump(parsed_data, f, ensure_ascii=False, indent=2)
        print(f"Parsed data saved to {data_file}")

    return raw_response_text, parsed_data, narration_sentences


def main() -> None:
    """Main execution function."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <source_file_path> [settings_file_path]")
        sys.exit(1)

    source_file_arg = sys.argv[1]
    settings_file_arg = sys.argv[2] if len(sys.argv) > 2 else None

    # --- Initialize Configuration ---
    config = AppConfig(source_file_path=source_file_arg, settings_file_path=settings_file_arg)
    print(f"Processing short: {config.short_name}")
    print(f"Output will be in: {config.base_output_dir}")

    # --- Load source material ---
    source_material_content = load_source_material(config.source_file_path)

    # --- Initialize API Clients ---
    try:
        oai_client = initialize_openai_client(api_key=config.openai_api_key, base_url=config.openai_base_url)
        short_images.set_openai_client(oai_client) # For modules that might use a global client
    except ValueError as e:
        print(f"Error initializing OpenAI client: {e}")
        sys.exit(1)

    if config.azure_api_key and config.azure_region:
        short_narration.initialize_azure_credentials(config.azure_api_key, config.azure_region)
    else:
        print("Warning: Azure API key or region not fully configured in AppConfig. Narration might fail.")

    # --- Script Generation ---
    _, parsed_script_data, narrations_list = generate_script_and_image_prompts(
        oai_client, source_material_content, config
    )
    if not narrations_list:
        print("No narration sentences were generated or parsed. Exiting.")
        sys.exit(1)

    # --- Narration Generation ---
    print("Generating narration audio...")
    short_narration.create_narration_audio_files(
        data=parsed_script_data,
        output_folder=config.narration_output_dir,
        azure_subscription_key=config.azure_api_key, # Pass explicitly
        azure_region=config.azure_region           # Pass explicitly
    )

    # --- Image Generation ---
    print("Generating images...")
    short_images.create_from_data(
        data=parsed_script_data,
        output_dir=config.image_output_dir,
        client=oai_client,
        image_model=config.openai_image_model # Ensure image model is passed
    )

    # --- Video Creation ---
    print("Generating video...")
    short_video.create_short_video(
        narration_sentences=narrations_list,
        base_dir=config.base_output_dir,
        final_output_filename=os.path.basename(config.final_video_file_path),
        caption_settings=config.caption_settings,
        video_width=config.video_width,
        video_height=config.video_height,
        frame_rate=config.frame_rate,
        fade_duration_ms=config.fade_duration_ms
    )

    print(f"DONE! Your video should be at: {config.final_video_file_path}")

if __name__ == "__main__":
    main()
