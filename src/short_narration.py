import os
from typing import List, Dict, Tuple, Optional

from azure.cognitiveservices import speech as speechsdk

# Global store for Azure credentials, can be set by main.py via initialize_azure_credentials
_azure_credentials: Optional[Dict[str, str]] = None

def initialize_azure_credentials(api_key: str, region: str) -> None:
    """
    Initializes Azure credentials for speech synthesis.
    This function is intended to be called by main.py.
    Args:
        api_key: Azure Speech Service API key.
        region: Azure Speech Service region.
    Raises:
        ValueError: If api_key or region is not provided.
    """
    global _azure_credentials
    if not api_key or not region:
        raise ValueError("Azure API key and region must be provided for initialization.")
    _azure_credentials = {"api_key": api_key, "region": region}

def _get_initialized_azure_credentials() -> Dict[str, str]:
    """
    Retrieves the initialized Azure credentials.
    Raises RuntimeError if credentials have not been set via initialize_azure_credentials.
    """
    if _azure_credentials is None:
        raise RuntimeError("Azure credentials not initialized. Call initialize_azure_credentials first from main.py.")
    return _azure_credentials

def parse_narration_text(narration_text: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Parses raw narration text into structured data for image and text elements.

    Args:
        narration_text: The raw string output from the narration generation model.

    Returns:
        A tuple containing:
        - data: List of dicts ({"type": "image/text", "description/content": "..."}).
        - narrations_list: List of just the narration sentences.
    """
    data: List[Dict[str, str]] = []
    narrations_list: List[str] = []
    lines = narration_text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line: continue
        if line.startswith('Narrator: '):
            text_content = line.replace('Narrator: ', '', 1).strip('"')
            data.append({"type": "text", "content": text_content})
            narrations_list.append(text_content)
        elif line.startswith('[') and line.endswith(']'):
            description = line.strip('[]')
            data.append({"type": "image", "description": description})
    return data, narrations_list

def create_narration_audio_files(
    data: List[Dict[str, str]],
    output_folder: str,
    azure_subscription_key: Optional[str] = None, # Explicitly passed from AppConfig
    azure_region: Optional[str] = None          # Explicitly passed from AppConfig
) -> None:
    """
    Generates narration audio files for each text element in the data.

    Args:
        data: Parsed data list from parse_narration_text.
        output_folder: Directory to save narration MP3 files.
        azure_subscription_key: Azure Speech Service API key.
        azure_region: Azure Speech Service region.
    """
    os.makedirs(output_folder, exist_ok=True)

    key_to_use = azure_subscription_key
    region_to_use = azure_region

    if not key_to_use or not region_to_use:
        try:
            creds = _get_initialized_azure_credentials()
            key_to_use = creds["api_key"]
            region_to_use = creds["region"]
        except RuntimeError as e:
            print(f"Error: Azure credentials not provided and not initialized globally. {e} Skipping audio generation.")
            return

    if not key_to_use or not region_to_use:
        print("Error: Azure API key or region missing. Skipping audio generation.")
        return

    narration_count = 0
    for element in data:
        if element.get("type") == "text":
            narration_count += 1
            content = element.get("content")
            if not content:
                print(f"Warning: Text element missing 'content'. Skipping narration {narration_count}.")
                continue

            output_file = os.path.join(output_folder, f"narration_{narration_count}.mp3")
            if os.path.exists(output_file):
                print(f"{os.path.basename(output_file)} already exists. Skipping audio generation.")
                continue

            _generate_azure_tts_audio(content, output_file, key_to_use, region_to_use)

def _generate_azure_tts_audio(
    text: str,
    output_filename: str,
    subscription_key: str,
    region: str
) -> None:
    """
    Generates a single audio file from text using Azure Text-to-Speech.
    """
    ssml_text = f"""
    <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" 
           xmlns:mstts="http://www.w3.org/2001/mstts" xml:lang="en-GB">
      <voice name="en-GB-OllieMultilingualNeural">
        <mstts:express-as>
          <prosody rate="+20%">
            {text}
          </prosody>
        </mstts:express-as>
      </voice>
    </speak>
    """

    speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region)
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Audio48Khz192KBitRateMonoMp3
    )
    audio_config = speechsdk.audio.AudioOutputConfig(filename=output_filename)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    try:
        result = synthesizer.speak_ssml_async(ssml_text).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print(f"Speech synthesized and saved to {output_filename}")
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print(f"Speech synthesis canceled: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error and cancellation_details.error_details:
                print(f"Error details: {cancellation_details.error_details}")
    except Exception as e:
        print(f"Error during Azure TTS for '{os.path.basename(output_filename)}': {e}")
        if os.path.exists(output_filename):
            try:
                os.remove(output_filename)
                print(f"Cleaned up partially written file: {output_filename}")
            except OSError as oe:
                print(f"Error cleaning up file {output_filename}: {oe}")