# Quickshorts

Quickshorts is a tool for making AI generated short videos. Captions with word highlighting are generated with [Captacity](https://github.com/unconv/captacity), which uses [OpenAI Whisper](https://github.com/openai/whisper).

## Quick Start

First, add your API-keys to the environment:

```console
$ export OPENAI_API_URL=YOUR_OPENAI_API_URL
$ export OPENAI_API_KEY=YOUR_OPENAI_API_KEY
$ export AZURE_API_KEY=YOUR_AZURE_API_KEY
$ export AZURE_REGION=YOUR_AZURE_REGION
```

Then, put your source content in a file, for example `some_story.txt` and run the `main.py`:

```console
$ ./main.py some_story.txt
Generating script with OpenAI...
Generating narration audio...
Generating images...
Generating video...
Creating segments for captions...
Extracting audio...
Generating video elements...
Rendering video...
DONE! Here's your video: shorts/some_story/some_story.avi
```

## Caption styling

Optionally, you can specify a settings file to define settings for the caption styling:

```console
$ ./main.py some_story.txt settings.json
```

The settings file can look like this, for example:

```json
{
    "font": "Bangers-Regular.ttf",
    "font_size": 130,
    "font_color": "yellow",

    "stroke_width": 3,
    "stroke_color": "black",

    "highlight_current_word": true,
    "word_highlight_color": "red",

    "line_count": 2,

    "padding": 50,

    "shadow_strength": 1.0,
    "shadow_blur": 0.1
}
```
