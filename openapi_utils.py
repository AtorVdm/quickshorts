import os

def get_openai_key() -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key

    print("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
    return None
