"""Text-to-Speech (TTS) service wrapper."""

from loguru import logger
import httpx
from pathlib import Path


class TTSService:
    """Text-to-Speech 服务"""

    def __init__(self, config):
        self.config = config
        self.api_url = config.api_url
        self.api_key = config.api_key
        self.voice = config.voice
        self.format = config.format
        self.speed = config.speed

    async def synthesize(self, text: str, output_path: Path | str = None) -> bytes | None:
        """
        将文本转换为语音

        参数：
        - text: 要转换的文本
        - output_path: 输出文件路径（可选）

        返回：
        - 音频数据或 None
        """
        if not self.config.enabled:
            logger.warning("TTS is not enabled")
            return None

        try:
            if not text.strip():
                return None

            # 调用本地 TTS API
            async with httpx.AsyncClient() as client:
                payload = {
                    "text": text,
                    "voice": self.voice,
                    "format": self.format,
                    "speed": self.speed
                }

                response = await client.post(
                    self.api_url,
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                )

                if response.status_code != 200:
                    logger.error(f"TTS API error: {response.status_code}")
                    return None

                audio_data = response.content

                # 如果指定了输出路径，保存文件
                if output_path:
                    output_path = Path(output_path)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(audio_data)
                    logger.info(f"TTS saved to: {output_path}")

                logger.info(f"TTS synthesized {len(audio_data)} bytes")
                return audio_data

        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return None