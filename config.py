#!/usr/bin/env python3
"""
Configuration script for generating YouTube Shorts.

This script defines the AppConfig class for managing application settings,
API keys, and paths. It also provides utility functions for loading
source material and initializing the OpenAI client.
"""

import json
import sys
import os
from typing import Dict, Any, List, Tuple, Optional

import openai # For type hinting

# Global variable to hold the initialized OpenAI client.
# This is set by initialize_openai_client and retrieved by get_openai_client.
_initialized_openai_client: Optional[openai.OpenAI] = None

NARRATION_PROMPT_FILENAME = "resources/narration_with_images_prompt.txt"

class AppConfig:
    """Application configuration."""
    def __init__(self, source_file_path: str, settings_file_path: Optional[str]):
        # File paths
        self.source_file_path: str = source_file_path
        self.settings_file_path: Optional[str] = settings_file_path

        # Derived paths
        self.short_name: str = ""
        self.base_output_dir: str = ""
        self.narration_output_dir: str = ""
        self.image_output_dir: str = ""
        self.final_video_file_path: str = ""
        self.background_music_file_path: str = ""

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
        self.narration_system_prompt: str = "" # Loaded in _load_narration_prompt

        self._load_settings_from_file()
        self._load_narration_prompt(NARRATION_PROMPT_FILENAME)
        self._setup_paths()

    def _load_narration_prompt(self, prompt_filename) -> None:
        """Loads the narration system prompt from a file."""
        try:
            with open(prompt_filename, 'r', encoding='utf-8') as f:
                self.narration_system_prompt = f.read()
            print(f"Loaded narration prompt from {prompt_filename}")
        except FileNotFoundError:
            print(f"Error: Narration prompt file not found at {prompt_filename}. Please create it or check path.")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading narration prompt file {prompt_filename}: {e}")
            sys.exit(1)


    def _load_settings_from_file(self) -> None:
        """Loads settings from a JSON file, overriding defaults."""
        if not self.settings_file_path or not os.path.exists(self.settings_file_path):
            print(f"Info: Settings file not provided or not found at {self.settings_file_path}. Using default/env settings.")
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

            self.caption_settings = json_settings.get("CAPTION_SETTINGS", self.caption_settings)
            print(f"Loaded and applied settings from {self.settings_file_path}")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {self.settings_file_path}. Using default/env settings.")
        except Exception as e:
            print(f"Error reading settings file {self.settings_file_path}: {e}. Using default/env settings.")

    def _setup_paths(self) -> None:
        """Sets up output directory paths based on the source file name."""
        self.short_name = os.path.splitext(os.path.basename(self.source_file_path))[0]
        self.base_output_dir = os.path.join("shorts", self.short_name)
        self.narration_output_dir = os.path.join(self.base_output_dir, "narrations")
        self.image_output_dir = os.path.join(self.base_output_dir, "images")
        self.final_video_file_path = os.path.join(self.base_output_dir, f"{self.short_name}.mp4")
        self.background_music_file_path = os.path.join("resources", "background_aow.mp3")

        os.makedirs(self.base_output_dir, exist_ok=True)
        os.makedirs(self.narration_output_dir, exist_ok=True)
        os.makedirs(self.image_output_dir, exist_ok=True)


def load_source_material(file_path: str) -> str:
    """Loads source material from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Source file not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading source file {file_path}: {e}")
        sys.exit(1)


def initialize_openai_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> openai.OpenAI:
    """Initializes and returns an OpenAI client, also setting the module-level client instance.

    Args:
        api_key: The OpenAI API key. Defaults to os.getenv("OPENAI_API_KEY").
        base_url: The base URL for the OpenAI API. Defaults to os.getenv("OPENAI_API_URL").

    Returns:
        An OpenAI client instance.

    Raises:
        ValueError: If the OpenAI API key is not found.
    """
    global _initialized_openai_client
    actual_api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
    actual_base_url = base_url if base_url is not None else os.getenv("OPENAI_API_URL") # Can be None

    if not actual_api_key:
        raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable or pass it explicitly during initialization.")

    # Create the client
    current_client = openai.OpenAI(
        api_key=actual_api_key,
        base_url=actual_base_url, # OpenAI constructor handles None for base_url
    )
    _initialized_openai_client = current_client
    return current_client

def get_openai_client() -> openai.OpenAI:
    """Returns the initialized OpenAI client.

    Returns:
        The OpenAI client instance.

    Raises:
        RuntimeError: If the client has not been initialized by calling initialize_openai_client first.
    """
    if _initialized_openai_client is None:
        # This situation should ideally be prevented by ensuring initialize_openai_client
        # is called before this function is ever needed by modules that don't receive the client directly.
        # However, with direct passing of the client, this function's usage might diminish.
        raise RuntimeError("OpenAI client has not been initialized. Call initialize_openai_client first.")
    return _initialized_openai_client
