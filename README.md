# ZukiPy

A Python package for interacting with the Zukijourney API, providing easy access to AI models for chat, image generation, speech processing, and more.

## Installation

Install using pip:

```sh
pip install git+https://github.com/EgehanKilicarslan/zukiPy
```

## Features
- Chat completion with AI models
- Image generation and upscaling
- Text-to-speech and speech-to-text conversion
- Language translation
- Text embeddings generation

## Quick Start
```python
from zukiPy.client import ZukiClient

# Initialize the client
client = ZukiClient(
    api_key="your-api-key",
    model="gpt-3.5-turbo"  # default model
)

# Chat completion
response = client.chat.send("What is the meaning of life?")
print(response)

# Generate images
image_url = client.image.generate(
    prompt="A beautiful sunset over mountains",
    width=512,
    height=512
)

# Text-to-speech
from zukiPy.enum import TtsModals
audio_path = client.audio.tts(
    text="Hello, world!",
    voice=TtsModals.ALLOY
)

# Translation
from zukiPy.enum import Language
translated_text = client.other.translate(
    text="Hello, world!",
    source_language=Language.ENGLISH,
    target_language=Language.SPANISH
)

## Configuration Options
The ZukiClient accepts several configuration parameters:
client = ZukiClient(
    api_key="your-api-key",            # Required
    model="gpt-3.5-turbo",             # Required
    role="user",                       # Default: "user"
    system_prompt="You are helpful.",  # Default system prompt
    temperature=0.7,                   # Default: 0.7
    generations=1,                     # Number of generations for images
    width=512,                        # Default image width
    height=512,                       # Default image height
    timeout=30,                       # API timeout in seconds
    download_images=False,            # Auto-download generated images
    download_audio=False,             # Auto-download audio files
)
```

## API Handlers

```python
###### Chat Handler ######
response = client.chat.send(
    message="Your message",
    model=None,              # Optional: override default model
    role=None,              # Optional: override default role
    system_prompt=None,     # Optional: override system prompt
    temperature=None        # Optional: override temperature
)

###### Image Handler ######

# Generate image
image = client.image.generate(
    prompt="Your prompt",
    generations=None,       # Optional: number of images
    width=None,            # Optional: image width
    height=None,           # Optional: image height
    negative_prompt=None    # Optional: what to avoid in generation
)

# Upscale image
client.image.upscale(
    file_path=Path("image.png")
)

###### Audio Handler ######

# Text to Speech
from zukiPy.enum import TtsModals, ElevenlabsModals, SpeechifyModals

audio = client.audio.tts(
    text="Your text",
    voice=TtsModals.ALLOY  # Choose from TtsModals, ElevenlabsModals, or SpeechifyModals
)

# Speech to Text
text = client.audio.stt(
    file_path=Path("audio.mp3")
)

##### Other Handler ######

# Translation
translated = client.other.translate(
    text="Hello",
    source_language=Language.ENGLISH,
    target_language=Language.SPANISH
)

# Generate embeddings
from zukiPy.enum import EmbeddingType

embeddings = client.other.embed(
    input="Your text",
    encoding_format=EmbeddingType.FLOAT
)
```


##### Requirements
- Python ≥ 3.10
- requests ≥ 2.25.0

##### License
- MIT License - see the [LICENSE](LICENSE) file for details.

#### Author
- Egehan Kılıçarslan (contact@egehankilicarslan.me)

#### Links
- [GitHub Repository](https://github.com/EgehanKilicarslan/zukiPy)
- [Issue Tracker](https://github.com/EgehanKilicarslan/zukiPy/issues)