import logging
import shutil
from pathlib import Path

import requests
from requests.exceptions import RequestException

from . import API_BASE
from .enum import ElevenlabsModals, EmbeddingType, Language, SpeechifyModals, TtsModals
from .helper import ZukiModels


class ChatHandler:
    def __init__(
        self,
        api_key: str,
        model: str,
        role: str,
        system_prompt: str,
        temperature: float,
        timeout: int,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._role = role
        self._system_prompt = system_prompt
        self._temperature = temperature
        self._timeout = timeout
        self._logger = logging.getLogger(__name__)

    def send(
        self,
        message: str,
        model: str | None = None,
        role: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        timeout: int | None = None,
    ) -> str | None:
        """Send a message to the chat model.

        Args:
            message: The message to send
            model: Override default model
            role: Override default role
            system_prompt: Override default system prompt
            temperature: Override default temperature
            timeout: Override default timeout

        Returns:
            The model's response text or None if request failed
        """
        try:
            model = model or self._model
            if ZukiModels.get_model_type(model) != "chat":
                raise ValueError(f"Model {model} is not a chat model")

            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt or self._system_prompt},
                    {"role": role or self._role, "content": message},
                ],
                "temperature": temperature or self._temperature,
            }

            response = requests.post(
                f"{API_BASE}{ZukiModels.get_model_endpoint(model)}",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
                json=payload,
                timeout=timeout or self._timeout,
            )
            response.raise_for_status()

            data = response.json()
            if not data.get("choices") or not data["choices"][0].get("message"):
                raise ValueError("Invalid response format from API")

            return data["choices"][0]["message"]["content"]

        except RequestException as e:
            self._logger.error(f"API request failed: {str(e)}")
            return None
        except (ValueError, KeyError) as e:
            self._logger.error(f"Error processing request/response: {str(e)}")
            return None
        except Exception as e:
            self._logger.error(f"Unexpected error: {str(e)}")
            return None


class ImageHandler:
    def __init__(
        self,
        api_key: str,
        model: str,
        width: int = 512,
        height: int = 512,
        generations: int = 1,
        negative_prompt: str | None = None,
        timeout: int = 30,
        download_images: bool = False,
        download_images_path: Path = Path("images"),
        backup_path: Path = Path("backup"),
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._width = width
        self._height = height
        self._generations = generations
        self._negative_prompt = negative_prompt
        self._download_images = download_images
        self._download_images_path = download_images_path
        self._timeout = timeout
        self._backup_path = backup_path
        self._logger = logging.getLogger(__name__)

    def generate(
        self,
        prompt: str,
        generations: int | None = None,
        model: str | None = None,
        negative_prompt: str | None = None,
        width: int | None = None,
        height: int | None = None,
        download_images: bool | None = None,
        download_images_path: Path | None = None,
        timeout: int | None = None,
    ) -> str | Path | None:
        """
        Generate an image using the API.

        Args:
            prompt: Prompt for the image generation
            generations: Number of images to generate
            model: Override default model
            negative_prompt: Override default negative prompt
            width: Override default image width
            height: Override default image height
            download_images: Override default download_images setting
            download_images_path: Override default download path
            timeout: Override default timeout

        Returns:
            URL of the generated image, Path to downloaded image, or None if generation failed
        """
        try:
            model = model or self._model
            if ZukiModels.get_model_type(model) != "image":
                raise ValueError(f"Model {model} is not an image model")

            # Construct payload more efficiently
            payload = {
                "prompt": prompt,
                "n": max(generations or self._generations, 1),
                "model": model,
                "width": width or self._width,
                "height": height or self._height,
            }

            # Add size string after width/height are set
            payload["size"] = f"{payload['width']}x{payload['height']}"

            # Only add negative prompt if it exists
            if negative_prompt or self._negative_prompt:
                payload["negative_prompt"] = negative_prompt or self._negative_prompt

            response = requests.post(
                f"{API_BASE}{ZukiModels.get_model_endpoint(model)}",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
                json=payload,
                timeout=timeout or self._timeout,
            )
            response.raise_for_status()

            data = response.json()
            if "detail" in data:
                raise ValueError(f"API error: {data['detail']}")

            image_url = data["data"][0]["url"]

            should_download = (
                download_images if download_images is not None else self._download_images
            )
            if should_download:
                try:
                    download_path = download_images_path or self._download_images_path
                    download_path.mkdir(parents=True, exist_ok=True)

                    image_path = download_path / Path(image_url).name
                    img_response = requests.get(image_url, timeout=timeout or self._timeout)
                    img_response.raise_for_status()

                    with open(image_path, "wb") as f:
                        f.write(img_response.content)
                    return image_path
                except Exception as e:
                    self._logger.error(f"Image download failed: {str(e)}")
                    return image_url
            return image_url

        except Exception as e:
            self._logger.error(f"Image generation failed: {str(e)}")
            return None

    def upscale(self, file_path: Path, backup_path: Path | None = None) -> bool:
        """
        Upscale an image using the API.

        Args:
            file_path: Path to the image file to upscale
            backup_path: Path to store a backup of the original image

        Returns:
            True if successful, False otherwise
        """
        try:
            if not file_path.is_file():
                raise ValueError(f"File not found: {file_path}")

            if file_path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                raise ValueError(f"Unsupported file format: {file_path.suffix}")

            backup_path = backup_path or self._backup_path
            backup_file = backup_path / file_path.name
            backup_path.mkdir(parents=True, exist_ok=True)

            shutil.copy2(file_path, backup_file)

            with open(file_path, "rb") as f:
                response = requests.post(
                    f"{API_BASE}/v1/images/upscale",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    files={"file": (file_path.name, f)},
                    data={"model": "upscale"},
                    timeout=self._timeout,
                )
                response.raise_for_status()

                if response.content:
                    with open(file_path, "wb") as out_file:
                        out_file.write(response.content)
                    return True
                raise ValueError("Empty response received from API")

        except Exception as e:
            self._logger.error(f"Image upscale failed: {str(e)}")
            if "backup_file" in locals() and backup_file.exists():
                backup_file.replace(file_path)
            return False


class AudioHandler:
    def __init__(
        self,
        api_key: str,
        model: str,
        download_audio: bool,
        download_audio_path: Path,
        timeout: int,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._download_audio = download_audio
        self._download_audio_path = download_audio_path
        self._timeout = timeout
        self._logger = logging.getLogger(__name__)

    def tts(
        self,
        text: str,
        voice: TtsModals | ElevenlabsModals | SpeechifyModals,
        model: str | None = None,
        download_audio: bool | None = None,
        download_audio_path: Path | None = None,
        timeout: int | None = None,
    ) -> str | Path | None:
        """Generate speech from text using the API.

        Args:
            text: Text to convert to speech
            voice: Voice to use (e.g., 'alloy')
            model: Override default TTS model
            download_audio: Override default download setting
            download_audio_path: Override default download path
            timeout: Override default timeout

        Returns:
            URL of the generated audio, Path to downloaded file, or None if generation failed
        """
        if not text.strip():
            self._logger.error("Input text cannot be empty")
            return None

        try:
            model = model or self._model
            ZukiModels.get_model(model)

            response = requests.post(
                f"{API_BASE}/v1/audio/speech",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
                json={"model": model, "voice": voice.value, "input": text},
                timeout=timeout or self._timeout,
            )
            response.raise_for_status()

            if not response.content:
                raise ValueError("Empty response received from API")

            # Determine if we should download based on override or default setting
            should_download = download_audio if download_audio is not None else self._download_audio

            if not should_download:
                return response.headers.get("Location")  # Return URL if not downloading

            # Handle audio download
            download_path = download_audio_path or self._download_audio_path
            download_path.mkdir(parents=True, exist_ok=True)

            file_name = f"{model}-{voice}-{hash(text)}.mp3"
            audio_path = download_path / file_name

            try:
                with open(audio_path, "wb") as f:
                    f.write(response.content)
                return audio_path
            except OSError as e:
                self._logger.error(f"Failed to save audio file: {str(e)}")
                return None

        except requests.exceptions.RequestException as e:
            self._logger.error(f"API request failed: {str(e)}")
        except ValueError as e:
            self._logger.error(f"Validation error: {str(e)}")
        except Exception as e:
            self._logger.error(f"Unexpected error in TTS generation: {str(e)}")

        return None

    def stt(
        self,
        file_path: Path,
        model: str | None = None,
        download_audio: bool | None = None,
        download_audio_path: Path | None = None,
        timeout: int | None = None,
    ) -> str | None:
        """Convert speech to text using the API.

        Args:
            file_path: Path to the audio file
            model: Override default STT model
            download_audio: Override default setting for saving transcript
            download_audio_path: Override default path for saving transcript
            timeout: Override default timeout

        Returns:
            Transcribed text or None if transcription failed
        """
        if not file_path.is_file():
            self._logger.error(f"Audio file not found: {file_path}")
            return None

        try:
            model = model or self._model
            ZukiModels.get_model(model)

            # Check file extension
            if file_path.suffix.lower() not in (
                ".mp3",
                ".mp4",
                ".mpeg",
                ".mpga",
                ".m4a",
                ".wav",
                ".webm",
            ):
                raise ValueError(f"Unsupported audio format: {file_path.suffix}")

            # Validate file size before sending
            if file_path.stat().st_size > 25 * 1024 * 1024:  # 25MB limit
                raise ValueError("File size exceeds 25MB limit")

            with open(file_path, "rb") as audio_file:
                try:
                    response = requests.post(
                        f"{API_BASE}/v1/audio/transcriptions",
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                        },
                        files={"file": (file_path.name, audio_file)},
                        data={"model": model},
                        timeout=timeout or self._timeout,
                    )
                    response.raise_for_status()

                    # Handle 500 error specifically
                    if response.status_code == 500:
                        self._logger.error("Server error occurred. Retrying...")
                        # Add retry logic here if needed
                        return None

                    data = response.json()
                    if not data or "text" not in data:
                        raise ValueError("Invalid response format from API")

                    transcribed_text = data["text"].strip()

                    # Handle text file download
                    should_download = (
                        download_audio if download_audio is not None else self._download_audio
                    )
                    if should_download:
                        download_path = download_audio_path or self._download_audio_path
                        download_path.mkdir(parents=True, exist_ok=True)

                        transcript_file = download_path / f"{file_path.stem}-transcript.txt"
                        try:
                            with open(transcript_file, "w", encoding="utf-8") as f:
                                f.write(transcribed_text)
                            self._logger.info(f"Transcript saved to: {transcript_file}")
                        except OSError as e:
                            self._logger.error(f"Failed to save transcript file: {str(e)}")

                    return transcribed_text

                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 500:
                        self._logger.error("Internal server error occurred")
                    else:
                        self._logger.error(f"HTTP error occurred: {str(e)}")
                    return None

        except requests.exceptions.RequestException as e:
            self._logger.error(f"API request failed: {str(e)}")
        except ValueError as e:
            self._logger.error(f"Validation error: {str(e)}")
        except Exception as e:
            self._logger.error(f"Unexpected error in STT transcription: {str(e)}")

        return None


class OtherHandler:
    def __init__(
        self,
        api_key: str,
        model: str,
        role: str,
        system_prompt: str,
        temperature: float,
        timeout: int,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._role = role
        self._system_prompt = system_prompt
        self._temperature = temperature
        self._timeout = timeout
        self._logger = logging.getLogger(__name__)

    def translate(
        self, text: str, source_language: Language, target_language: Language
    ) -> str | None:
        """Translate text between languages.

        Args:
            text: Text to translate
            source_language: Source language enum
            target_language: Target language enum

        Returns:
            Translated text or None if translation failed
        """
        try:
            if not text.strip():
                raise ValueError("Input text cannot be empty")

            if source_language == target_language:
                return text

            payload = {
                "text": text,
                "src": source_language.value,
                "target": target_language.value,
                "model": "google-translate",
            }

            response = requests.post(
                f"{API_BASE}/v1/text/translations",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()

            data = response.json()
            if not data:
                raise ValueError("Empty response received from API")

            # Try to get translation first, fall back to text field
            translation = data.get("translation")
            if translation:
                return translation.strip()

            # Fall back to text field if translation is not available
            text_result = data.get("text")
            if text_result:
                return text_result.strip()

            raise ValueError("No translation or text field in response")

        except requests.exceptions.RequestException as e:
            self._logger.error(f"API request failed: {str(e)}")
            return None
        except ValueError as e:
            self._logger.error(f"Validation error: {str(e)}")
            return None
        except Exception as e:
            self._logger.error(f"Unexpected error in translation: {str(e)}")
            return None

    def embed(
        self,
        input: str,
        encoding_format: EmbeddingType,
        model: str | None = None,
        timeout: int | None = None,
    ) -> list[float] | str | None:
        """Generate embeddings for the input text.

        Args:
            input: Text to generate embeddings for
            encoding_format: Type of encoding to use
            model: Override default embeddings model
            timeout: Override default timeout

        Returns:
            List of float values for float encoding,
            Base64 string for base64 encoding,
            or None if generation failed
        """
        try:
            model = model or self._model
            timeout = timeout or self._timeout

            if ZukiModels.get_model_type(model) != "embeddings":
                raise ValueError(f"Model {model} is not an embeddings model")

            if not input.strip():
                raise ValueError("Input text cannot be empty")

            payload = {
                "model": model,
                "input": [input],
                "encoding_format": encoding_format.value,
            }

            response = requests.post(
                f"{API_BASE}{ZukiModels.get_model_endpoint(model)}",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()

            data = response.json()

            # Handle the specific API response format
            if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                for item in data["data"]:
                    # Handle base64 encoded embedding
                    if isinstance(item, dict) and "embedding" in item:
                        return item["embedding"]
                    # Handle float array embedding
                    if isinstance(item, list):
                        return item
                    # Handle object with base64 or float array
                    if isinstance(item, dict) and any(k in item for k in ["base64", "float"]):
                        return item.get("base64") or item.get("float")

            raise ValueError("Invalid or missing embedding data in response")

        except requests.exceptions.RequestException as e:
            self._logger.error(f"API request failed: {str(e)}")
            return None
        except ValueError as e:
            self._logger.error(f"Validation error: {str(e)}")
            return None
        except Exception as e:
            self._logger.error(f"Unexpected error in embedding generation: {str(e)}")
            return None
