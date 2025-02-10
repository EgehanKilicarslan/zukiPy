import logging
from pathlib import Path

from .side import AudioHandler, ChatHandler, ImageHandler, OtherHandler


class ZukiClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        role: str = "user",
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        generations: int = 1,
        negative_prompt: str | None = None,
        width: int = 512,
        height: int = 512,
        timeout: int = 30,
        download_images: bool = False,
        download_audio: bool = False,
        download_images_path: Path = Path("images"),
        download_audio_path: Path = Path("audio"),
        backup_path: Path = Path("backup"),
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.role = role
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.generations = generations
        self.negative_prompt = negative_prompt
        self.width = width
        self.height = height
        self.timeout = timeout
        self.download_images = download_images
        self.download_audio = download_audio
        self.download_images_path = download_images_path
        self.download_audio_path = download_audio_path
        self.backup_path = backup_path
        self.logger = logging.getLogger(__name__)

        if not self.download_images_path.exists():
            self.download_images_path.mkdir(parents=True)
        if not self.download_audio_path.exists():
            self.download_audio_path.mkdir(parents=True)
        if not self.backup_path.exists():
            self.backup_path.mkdir(parents=True)

        self._chat_handler = None
        self._image_handler = None
        self._audio_handler = None
        self._other_handler = None

    @property
    def chat(self) -> ChatHandler:
        """Get the chat handler instance."""
        if self._chat_handler is None:
            self._chat_handler = ChatHandler(
                api_key=self.api_key,
                model=self.model,
                role=self.role,
                system_prompt=self.system_prompt,
                temperature=self.temperature,
                timeout=self.timeout,
            )
        return self._chat_handler

    @property
    def image(self) -> ImageHandler:
        """Get the image handler instance."""
        if self._image_handler is None:
            self._image_handler = ImageHandler(
                api_key=self.api_key,
                model=self.model,
                width=self.width,
                height=self.height,
                generations=self.generations,
                negative_prompt=self.negative_prompt,
                timeout=self.timeout,
                download_images=self.download_images,
                download_images_path=self.download_images_path,
                backup_path=self.backup_path,
            )
        return self._image_handler

    @property
    def audio(self) -> AudioHandler:
        """Get the audio handler instance."""
        if self._audio_handler is None:
            self._audio_handler = AudioHandler(
                api_key=self.api_key,
                model=self.model,
                timeout=self.timeout,
                download_audio=self.download_audio,
                download_audio_path=self.download_audio_path,
            )
        return self._audio_handler

    @property
    def other(self) -> OtherHandler:
        """Get the other handler instance."""
        if self._other_handler is None:
            self._other_handler = OtherHandler(
                api_key=self.api_key,
                model=self.model,
                role=self.role,
                system_prompt=self.system_prompt,
                temperature=self.temperature,
                timeout=self.timeout,
            )
        return self._other_handler
