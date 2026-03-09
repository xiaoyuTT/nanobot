"""Speech-to-Text (STT) service wrapper."""

from loguru import logger
import httpx
from pathlib import Path


class STTService:
    """Speech-to-Text 服务"""

    def __init__(self, config):
        self.config = config
        self.api_url = config.api_url
        self.api_key = config.api_key
        self.language = config.language

    async def transcribe(self, audio_path: Path | str) -> str:
        """
        将音频文件转换为文本

        参数：
        - audio_path: 音频文件路径

        返回：
        - 转录的文本
        """
        if not self.config.enabled:
            logger.warning("STT is not enabled")
            return ""

        try:
            audio_path = Path(audio_path)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            # 调用本地 STT API
            async with httpx.AsyncClient() as client:
                with open(audio_path, "rb") as f:
                    files = {
                        "audio": (audio_path.name, f, "audio/mpeg")
                    }
                    params = {
                        "language": self.language
                    }

                    response = await client.post(
                        self.api_url,
                        files=files,
                        params=params,
                        headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                    )

                    result = response.json()

                    # 假设 API 返回格式为 {"text": "转录的文本"}
                    text = result.get("text", "")
                    logger.info(f"STT transcribed: {text[:100]}...")
                    return text

        except Exception as e:
            logger.error(f"STT transcription failed: {e}")
            return ""