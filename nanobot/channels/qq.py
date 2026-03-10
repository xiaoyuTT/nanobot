"""QQ channel implementation using botpy SDK."""

import asyncio
import os
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import QQConfig
from nanobot.agent.meida.stt import STTService


try:
    import botpy
    from botpy.message import C2CMessage

    QQ_AVAILABLE = True
except ImportError:
    QQ_AVAILABLE = False
    botpy = None
    C2CMessage = None

if TYPE_CHECKING:
    from botpy.message import C2CMessage


def _make_bot_class(channel: "QQChannel") -> "type[botpy.Client]":
    """Create a botpy Client subclass bound to the given channel."""
    intents = botpy.Intents(public_messages=True, direct_message=True)

    class _Bot(botpy.Client):
        def __init__(self):
            # Disable botpy's file log — nanobot uses loguru; default "botpy.log" fails on read-only fs
            super().__init__(intents=intents, ext_handlers=False)

        async def on_ready(self):
            logger.info("QQ bot ready: {}", self.robot.name)

        async def on_c2c_message_create(self, message: "C2CMessage"):
            await channel._on_message(message)

        async def on_direct_message_create(self, message):
            await channel._on_message(message)

    return _Bot


class QQChannel(BaseChannel):
    """QQ channel using botpy SDK with WebSocket connection."""

    name = "qq"

    def __init__(self, config: QQConfig, bus: MessageBus):
        super().__init__(config, bus)
        self.config: QQConfig = config
        self._client: "botpy.Client | None" = None
        self._processed_ids: deque = deque(maxlen=1000)
        self._stt_service = None
        
    def set_stt_service(self, stt_service: STTService):
        """Set the STT service for voice message processing."""
        self._stt_service = stt_service

    async def start(self) -> None:
        """Start the QQ bot."""
        if not QQ_AVAILABLE:
            logger.error("QQ SDK not installed. Run: pip install qq-botpy")
            return

        if not self.config.app_id or not self.config.secret:
            logger.error("QQ app_id and secret not configured")
            return

        self._running = True
        BotClass = _make_bot_class(self)
        self._client = BotClass()

        logger.info("QQ bot started (C2C private message)")
        await self._run_bot()

    async def _run_bot(self) -> None:
        """Run the bot connection with auto-reconnect."""
        while self._running:
            try:
                await self._client.start(appid=self.config.app_id, secret=self.config.secret)
            except Exception as e:
                logger.warning("QQ bot error: {}", e)
            if self._running:
                logger.info("Reconnecting QQ bot in 5 seconds...")
                await asyncio.sleep(5)

    async def stop(self) -> None:
        """Stop the QQ bot."""
        self._running = False
        if self._client:
            try:
                await self._client.close()
            except Exception:
                pass
        logger.info("QQ bot stopped")

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through QQ."""
        if not self._client:
            logger.warning("QQ client not initialized")
            return
        try:
            msg_id = msg.metadata.get("message_id")
            await self._client.api.post_c2c_message(
                openid=msg.chat_id,
                msg_type=0,
                content=msg.content,
                msg_id=msg_id,
            )
        except Exception as e:
            logger.error("Error sending QQ message: {}", e)

    async def _download_voice(self, message: "C2CMessage") -> str | None:
        """Download voice message from QQ."""
        try:
            # 检查消息是否包含语音
            if not hasattr(message, 'attachments'):
                logger.info("Message has no attachments attribute")
                return None
            
            # 检查attachments是否有值
            if not message.attachments:
                logger.info("Message attachments is empty")
                return None
            
            # 找到语音附件
            voice_attachment = None
            try:
                for attachment in message.attachments: 
                    # 如果是对象，使用getattr
                    attachment_type = getattr(attachment, 'type', None) or getattr(attachment, 'content_type', None)
                    filename = getattr(attachment, 'filename', '')
                    # 检查content_type是否为语音类型，或文件名是否包含音频扩展名
                    if attachment_type in ['audio', 'voice'] or any(ext in filename.lower() for ext in ['.slk', '.silk', '.mp3', '.wav']):
                        voice_attachment = attachment
                        break
            except Exception as e:
                logger.error(f"Error iterating attachments: {e}")
                return None
            
            if not voice_attachment:
                logger.info("No voice attachment found")
                return None
            
            # 下载语音文件
            voice_url = getattr(voice_attachment, 'url', None)
            if not voice_url:
                logger.info("No voice URL found")
                return None
            
            # 去除URL中的反引号和空格
            voice_url = voice_url.strip().strip('`')
            
            # 创建保存目录
            voice_dir = Path(__file__).parent.parent.parent / "data" / "voice" / "qq"
            voice_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成文件名，强制使用.slk扩展名
            # 使用时间戳和随机数生成安全的文件名，避免特殊字符
            import time
            import random
            timestamp = int(time.time() * 1000)
            random_suffix = random.randint(1000, 9999)
            filename = f"voice_{timestamp}_{random_suffix}.slk"
            voice_path = voice_dir / filename
            
            # 下载文件
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(voice_url)
                response.raise_for_status()
                file_content = response.content
                logger.info(f"Downloaded file size: {len(file_content)} bytes")
                with open(voice_path, "wb") as f:
                    f.write(file_content)
            
            # 检查是否需要格式转换
            if voice_path.suffix.lower() in ['.slk']:
                # 转换为 wav 格式
                import subprocess
                wav_path = voice_path.with_suffix('.wav')
                try:
                    # 使用 silk-v3-decoder 的 converter.sh 脚本进行转换
                    # 使用相对路径指向项目内的 silk-v3-decoder 目录
                    converter_script = Path(__file__).parent.parent.parent / "silk-v3-decoder" / "converter.sh"
                    result = subprocess.run(
                        ['sh', str(converter_script), str(voice_path), 'wav'], 
                        check=False, 
                        capture_output=True, 
                        text=True
                    )
                    if result.returncode != 0:
                        logger.error(f"Silk converter error: {result.stderr}")
                        return None
                    voice_path = wav_path
                    logger.info(f"Converted silk/slk to WAV: {voice_path}")
                except Exception as e:
                    logger.error(f"Error converting silk/slk to WAV: {e}")
                    return None
            return str(voice_path)
            
        except Exception as e:
            logger.error(f"Error downloading voice message: {e}")
            return None

    async def _on_message(self, data: "C2CMessage") -> None:
        """Handle incoming message from QQ."""
        try:
            # Dedup by message ID
            if data.id in self._processed_ids:
                return
            self._processed_ids.append(data.id)

            author = data.author
            user_id = str(getattr(author, 'id', None) or getattr(author, 'user_openid', 'unknown'))
            content = (data.content or "").strip()
            logger.info(f"Received QQ message from {user_id}: {content}")
            # 下载并处理语音消息
            voice_path = await self._download_voice(data)
            media = []
            
            # 如果有语音文件且STT服务可用，进行语音转文本
            if voice_path and self._stt_service:
                try:
                    # 调用STT服务转文本
                    transcribed_text = await self._stt_service.transcribe(voice_path)
                    if transcribed_text:
                        content = f"[语音消息]: {transcribed_text}"
                        logger.info(f"Transcribed voice message: {transcribed_text[:100]}...")
                except Exception as e:
                    logger.error(f"Error transcribing voice message: {e}")
                
                # 将语音文件路径添加到media列表
                media = [voice_path]
            
            # 如果消息内容为空且没有语音，直接返回
            if not content and not media:
                return

            await self._handle_message(
                sender_id=user_id,
                chat_id=user_id,
                content=content,
                media=media,
                metadata={"message_id": data.id},
            )
        except Exception:
            logger.exception("Error handling QQ message")
