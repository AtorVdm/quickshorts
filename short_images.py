import base64
import os

from openai_client import client

def create_from_data(data, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_number = 0
    for element in data:
        if element["type"] != "image":
            continue
        image_number += 1
        image_name = f"image_{image_number}.webp"
        image_full_path = os.path.join(output_dir, image_name)
        if os.path.exists(image_full_path):
            print(f"{image_name} already exists. Skipping image generation.")
            continue
        generate(element["description"] + ". Vertical image, fully filling the canvas.", image_full_path)


def generate(prompt, output_file, size="1024x1792"):
    response = client.images.generate(
        model="openai/dall-e-3",
        prompt=prompt,
        size=size,
        quality="standard",
        response_format="b64_json",
        n=1,
    )

    image_b64 = response.data[0].b64_json

    with open(output_file, "wb") as f:
        f.write(base64.b64decode(image_b64))
    print(f"Image generated and saved to {output_file}")
