#!/usr/bin/env python3
import os
from typing import Optional

import openai

# This global client is initialized by main.py and then used by other modules.
# It's typed as Optional because it's None until main.py initializes it.
_initialized_client: Optional[openai.OpenAI] = None

def initialize_openai_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> openai.OpenAI:
    """Initializes and returns an OpenAI client, also setting the module-level client instance.

    This function should ideally be called once at the beginning of the application.

    Args:
        api_key: The OpenAI API key. Defaults to os.getenv("OPENAI_API_KEY").
        base_url: The base URL for the OpenAI API. Defaults to os.getenv("OPENAI_API_URL").

    Returns:
        An OpenAI client instance.

    Raises:
        ValueError: If the OpenAI API key is not found.
    """
    global _initialized_client
    actual_api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
    actual_base_url = base_url if base_url is not None else os.getenv("OPENAI_API_URL") # Can be None

    if not actual_api_key:
        raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable or pass it explicitly during initialization.")

    # Create the client
    current_client = openai.OpenAI(
        api_key=actual_api_key,
        base_url=actual_base_url, # OpenAI constructor handles None for base_url
    )
    _initialized_client = current_client
    return current_client

def get_openai_client() -> openai.OpenAI:
    """Returns the initialized OpenAI client.

    This function relies on initialize_openai_client having been called previously.

    Returns:
        The OpenAI client instance.

    Raises:
        RuntimeError: If the client has not been initialized by calling initialize_openai_client first.
    """
    if _initialized_client is None:
        raise RuntimeError("OpenAI client has not been initialized. Call initialize_openai_client first.")
    return _initialized_client
