import os
from typing import List, Dict, Tuple, Optional

from azure.cognitiveservices import speech as speechsdk

def parse_narration_text(narration_text: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Parses raw narration text into image descriptions and narration sentences.

    Args:
        narration_text: Raw output from the narration generation model.

    Returns:
        Tuple: (list of dicts for images/text, list of narration sentences).
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
    azure_subscription_key: Optional[str] = None,
    azure_region: Optional[str] = None
) -> None:
    """
    Generates narration audio files using Azure TTS.

    Args:
        data: Parsed data from parse_narration_text.
        output_folder: Directory to save MP3 files.
        azure_subscription_key: Azure Speech API key.
        azure_region: Azure Speech region.
    """
    os.makedirs(output_folder, exist_ok=True)

    if not azure_subscription_key or not azure_region:
        print("Error: Azure API key or region not provided. Skipping audio generation.")
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
                print(f"Skipping existing audio: {os.path.basename(output_file)}")
                continue

            _generate_azure_tts_audio(content, output_file, azure_subscription_key, azure_region) # Use parameters directly

def _generate_azure_tts_audio(
    text: str,
    output_filename: str,
    subscription_key: str, # Parameter name kept for clarity within this function
    region: str            # Parameter name kept for clarity
) -> None:
    """Generates a single audio file from text using Azure TTS."""
    # Ensure parameters are not None before proceeding, though create_narration_audio_files should check
    if not subscription_key or not region:
        print(f"Error: Missing Azure credentials for generating {output_filename}.")
        return

    ssml_text = f"""
    <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"
           xmlns:mstts="http://www.w3.org/2001/mstts" xml:lang="en-GB">
      <voice name="en-GB-MiaNeural">
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