import os

from azure.cognitiveservices import speech as speechsdk

azure_subscription_key = os.getenv("AZURE_API_KEY")
azure_region = os.getenv("AZURE_REGION")

def parse(narration):
    data = []
    narrations = []
    lines = narration.split("\n")
    for line in lines:
        if line.startswith('Narrator: '):
            text = line.replace('Narrator: ', '')
            data.append({
                "type": "text",
                "content": text.strip('"'),
            })
            narrations.append(text.strip('"'))
        elif line.startswith('['):
            background = line.strip('[]')
            data.append({
                "type": "image",
                "description": background,
            })
    return data, narrations

def create(data, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    n = 0
    for element in data:
        if element["type"] != "text":
            continue
        n += 1
        output_file = os.path.join(output_folder, f"narration_{n}.mp3")

        if os.path.exists(output_file):
            print(f"narration_{n}.mp3 already exists. Skipping audio generation.")
            continue

        generate_azure_audio_ssml(element["content"], output_file)

def generate_azure_audio_ssml(text, output_filename):
    if os.path.exists(output_filename):
        print(f"{output_filename} already exists. Skipping audio generation.")
        return
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

    speech_config = speechsdk.SpeechConfig(subscription=azure_subscription_key, region=azure_region)
    # Set output format to 48 kHz 192KBit mono mp3
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Audio48Khz192KBitRateMonoMp3
    )

    audio_config = speechsdk.audio.AudioOutputConfig(filename=output_filename)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    result = synthesizer.speak_ssml_async(ssml_text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(f"Speech synthesized and saved to {output_filename}")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print(f"Speech synthesis canceled: {cancellation_details.reason}")
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(f"Error details: {cancellation_details.error_details}")