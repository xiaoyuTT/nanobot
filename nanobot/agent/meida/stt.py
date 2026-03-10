"""Speech-to-Text (STT) service wrapper."""

from loguru import logger
import torch
from qwen_asr import Qwen3ASRModel
from pathlib import Path


class STTService:
    """Speech-to-Text 服务"""

    def __init__(self, config):
        self.config = config
        self.language = config.language
        
        # 初始化本地ASR模型
        if config.enabled:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            logger.info(f"Initializing ASR model on device: {device}")
            self.model = Qwen3ASRModel.from_pretrained(
                "/opt/models/Qwen3-ASR-0.6B",
                dtype=dtype,
                device_map=device,
                max_inference_batch_size=4,
                max_new_tokens=256,
            )
        else:
            self.model = None

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

            # 使用本地ASR模型进行转录
            results = self.model.transcribe(
                audio=str(audio_path),
                language=self.language,
            )

            text = results[0].text
            logger.info(f"STT transcribed: {text[:100]}...")
            return text

        except Exception as e:
            logger.error(f"STT transcription failed: {e}")
            return ""