import base64
import os
from typing import List, Dict, Optional # Removed Any
import openai

_openai_client: Optional[openai.OpenAI] = None

def set_openai_client(client: openai.OpenAI) -> None:
    """Sets the global OpenAI client for this module."""
    global _openai_client
    _openai_client = client

def _get_openai_client() -> openai.OpenAI:
    """Retrieves the configured OpenAI client."""
    if _openai_client is None:
        raise RuntimeError("OpenAI client for short_images not set.")
    return _openai_client

def create_from_data(
    data: List[Dict[str, str]],
    output_dir: str,
    client: Optional[openai.OpenAI] = None,
    image_model: str = "dall-e-3"  # Added image_model argument
) -> None:
    """
    Creates images from the given data using the provided OpenAI client and image model.

    Args:
        data: List of dicts, items with "type": "image" are processed.
        output_dir: Directory to save generated images.
        client: Initialized OpenAI client. Uses global if None.
        image_model: The DALL-E model to use for image generation.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    active_client = client if client else _get_openai_client()
    if not active_client:
        print("Error: OpenAI client not available for image generation.")
        return

    image_number = 0
    for element in data:
        if element.get("type") != "image":
            continue
        image_number += 1
        image_name = f"image_{image_number}.webp"
        image_full_path = os.path.join(output_dir, image_name)

        if os.path.exists(image_full_path):
            print(f"{image_name} already exists. Skipping.")
            continue

        description = element.get("description")
        if not description:
            print(f"Warning: Image element missing 'description'. Skipping image {image_number}.")
            continue

        full_prompt = description + ". Vertical image, fully filling the canvas."
        # Pass the image_model to generate_image
        generate_image(active_client, full_prompt, image_full_path, model=image_model)


def generate_image(
    client: openai.OpenAI,
    prompt: str,
    output_file: str,
    size: str = "1024x1792",
    model: str = "dall-e-3"
) -> None:
    """
    Generates a single image using the OpenAI API and saves it to a file.

    Args:
        client: An initialized OpenAI client.
        prompt: The prompt to generate the image from.
        output_file: Path to save the generated image.
        size: Image size (e.g., "1024x1792").
        model: The DALL-E model to use.
    """
    try:
        response = client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            quality="standard",
            response_format="b64_json",
            n=1,
        )
        image_b64 = response.data[0].b64_json
        if image_b64:
            with open(output_file, "wb") as f:
                f.write(base64.b64decode(image_b64))
            print(f"Image generated ({model}) and saved to {output_file}")
        else:
            print(f"Error: No image data received for prompt: {prompt}")
    except openai.APIError as e:
        print(f"OpenAI API Error generating image (model {model}, prompt '{prompt}'): {e}")
    except Exception as e:
        print(f"Unexpected error during image generation (model {model}, prompt '{prompt}'): {e}")
