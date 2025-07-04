#!/usr/bin/env python3

import json
import sys
import os

import short_narration
import short_images
import short_video
from openai_client import client

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <source_file> [settings_file]")
    sys.exit(1)

with open(sys.argv[1]) as f:
    source_material = f.read()

caption_settings = {}
if len(sys.argv) > 2:
    with open(sys.argv[2]) as f:
        caption_settings = json.load(f)

short_name = str(sys.argv[1].split(".")[0])
output_file = short_name + ".mp4"

basedir = os.path.join("shorts", short_name)
if not os.path.exists(basedir):
    os.makedirs(basedir)

response_text = ""
if os.path.exists(os.path.join(basedir, "response.txt")) & os.path.exists(os.path.join(basedir, "data.json")):
    response_text = open(os.path.join(basedir, "response.txt")).read()
    print(f"data.json and response.txt already exists. Skipping text generation.")
else:
    print("Generating script...")

    response = client.chat.completions.create(
        model="openai/gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """You are a YouTube short narration generator. You generate 40 seconds to 60 seconds of narration. The shorts you create have a background that fades from image to image as the narration is going on.
    
    You will need to generate descriptions of images for each of the sentences in the short. They will be passed to an AI image generator. DO NOT IN ANY CIRCUMSTANCES use names of celebrities or people in the image descriptions. It is illegal to generate images of celebrities. Only describe persons without their names. Do not reference any real person or group in the image descriptions. Don't mention the female figure or other sexual content in the images because they are not allowed.
    
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
            },
            {
                "role": "user",
                "content": f"Create a YouTube short narration based on the following source material:\n\n{source_material}"
            }
        ]
    )

    response_text = response.choices[0].message.content
    response_text.replace("’", "'").replace("`", "'").replace("…", "...").replace("“", '"').replace("”", '"')

    with open(os.path.join(basedir, "response.txt"), "w") as f:
        f.write(response_text)

data, narrations = short_narration.parse(response_text)
with open(os.path.join(basedir, "data.json"), "w") as f:
    json.dump(data, f, ensure_ascii=False)

print(f"Generating narration...")
short_narration.create(data, os.path.join(basedir, "narrations"))

print("Generating images...")
short_images.create_from_data(data, os.path.join(basedir, "images"))

print("Generating video...")
short_video.create(narrations, basedir, output_file, caption_settings)

print(f"DONE! Here's your video: {os.path.join(basedir, output_file)}")
