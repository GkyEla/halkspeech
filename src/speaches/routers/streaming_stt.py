"""
Streaming STT WebSocket Endpoint

VAD olmadan, 16kHz PCM16 mono audio ile dusuk TTFT streaming transkripsiyon.
HTTP endpoint'lerini etkilemez.

Usage:
    ws://server:port/v1/audio/transcriptions/stream?model=Systran/faster-whisper-large-v3&language=tr

Protocol:
    Client -> Server: Binary PCM16 audio chunks (16kHz mono, little-endian)
    Client -> Server: JSON {"type": "audio.done"} to signal end
    Server -> Client: JSON {"type": "transcript.partial", "text": "..."}
    Server -> Client: JSON {"type": "transcript.final", "text": "..."}
"""

import asyncio
import logging
import struct
import time
from io import BytesIO
from typing import Literal

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from numpy.typing import NDArray
from pydantic import BaseModel
import soundfile as sf

from speaches.dependencies import (
    ConfigDependency,
    WhisperModelManagerDependency,
    get_config,
    get_whisper_model_manager,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["streaming-stt"])

# Constants
SAMPLE_RATE = 16000  # 16kHz
BYTES_PER_SAMPLE = 2  # PCM16 = 2 bytes per sample
CHANNELS = 1  # Mono

# Streaming config - Ultra low latency defaults
MIN_CHUNK_DURATION_MS = 200  # Minimum audio before first transcription (was 500)
CHUNK_DURATION_MS = 300  # Process every N ms of audio (was 1000)
MIN_CHUNK_SAMPLES = int(SAMPLE_RATE * MIN_CHUNK_DURATION_MS / 1000)
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)


class StreamingConfig(BaseModel):
    """Configuration for streaming transcription."""
    sample_rate: int = 16000
    chunk_duration_ms: int = 300  # More frequent updates
    min_chunk_duration_ms: int = 200  # Faster first response
    enable_vad: bool = False
    vad_threshold: float = 0.5


class TranscriptMessage(BaseModel):
    """Message sent to client with transcription results."""
    type: Literal["transcript.partial", "transcript.final", "error", "ready"]
    text: str = ""
    latency_ms: float = 0.0
    audio_duration_ms: float = 0.0


def pcm16_bytes_to_float32(audio_bytes: bytes) -> NDArray[np.float32]:
    """Convert PCM16 little-endian bytes to float32 numpy array."""
    # PCM16: signed 16-bit integer, little-endian
    num_samples = len(audio_bytes) // BYTES_PER_SAMPLE
    samples = struct.unpack(f'<{num_samples}h', audio_bytes)
    # Normalize to [-1.0, 1.0]
    return np.array(samples, dtype=np.float32) / 32768.0


def float32_to_wav_bytes(audio: NDArray[np.float32], sample_rate: int = SAMPLE_RATE) -> BytesIO:
    """Convert float32 audio to WAV file bytes for transcription."""
    buffer = BytesIO()
    sf.write(
        buffer,
        audio,
        samplerate=sample_rate,
        subtype="PCM_16",
        endian="LITTLE",
        format="wav",
    )
    buffer.seek(0)
    return buffer


class StreamingTranscriber:
    """Handles streaming transcription with low latency."""

    def __init__(
        self,
        whisper_model_manager: WhisperModelManagerDependency,
        model: str,
        language: str | None = None,
        config: StreamingConfig | None = None,
    ):
        self.whisper_model_manager = whisper_model_manager
        self.model = model
        self.language = language
        self.config = config or StreamingConfig()

        # Audio buffer
        self.audio_buffer: NDArray[np.float32] = np.array([], dtype=np.float32)
        self.processed_samples = 0
        self.total_samples = 0

        # Transcription state
        self.previous_text = ""
        self.is_done = False

    @property
    def buffer_duration_ms(self) -> float:
        """Current buffer duration in milliseconds."""
        return len(self.audio_buffer) / self.config.sample_rate * 1000

    @property
    def unprocessed_samples(self) -> int:
        """Number of samples not yet processed."""
        return len(self.audio_buffer) - self.processed_samples

    def append_audio(self, audio_chunk: NDArray[np.float32]) -> None:
        """Append audio chunk to buffer."""
        self.audio_buffer = np.append(self.audio_buffer, audio_chunk)
        self.total_samples += len(audio_chunk)

    def should_transcribe(self) -> bool:
        """Check if we have enough audio for transcription."""
        if self.is_done:
            return self.unprocessed_samples > 0

        # Calculate sample thresholds from config
        min_samples = int(self.config.sample_rate * self.config.min_chunk_duration_ms / 1000)
        chunk_samples = int(self.config.sample_rate * self.config.chunk_duration_ms / 1000)

        # First transcription: wait for minimum duration
        if self.processed_samples == 0:
            return self.unprocessed_samples >= min_samples

        # Subsequent: process every chunk_samples
        return self.unprocessed_samples >= chunk_samples

    async def transcribe(self) -> TranscriptMessage | None:
        """Transcribe current buffer and return result."""
        if len(self.audio_buffer) == 0:
            return None

        start_time = time.perf_counter()

        # Convert to WAV for transcription
        wav_buffer = float32_to_wav_bytes(self.audio_buffer, self.config.sample_rate)

        try:
            with self.whisper_model_manager.load_model(self.model) as whisper:
                segments, info = whisper.transcribe(
                    wav_buffer,
                    language=self.language,
                    task="transcribe",
                    vad_filter=self.config.enable_vad,
                    without_timestamps=True,  # Faster - no timestamp calculation
                    beam_size=1,  # Greedy decoding - fastest
                    best_of=1,  # No sampling
                    temperature=0.0,  # Deterministic - faster
                    condition_on_previous_text=False,  # Don't condition - faster for streaming
                    compression_ratio_threshold=2.4,  # Default
                    no_speech_threshold=0.6,  # Default
                    initial_prompt=None,  # No prompt overhead
                )

                # Collect all segment texts
                text = "".join(segment.text for segment in segments).strip()

            latency_ms = (time.perf_counter() - start_time) * 1000
            audio_duration_ms = len(self.audio_buffer) / self.config.sample_rate * 1000

            # Update state
            self.processed_samples = len(self.audio_buffer)

            # Determine message type
            msg_type: Literal["transcript.partial", "transcript.final"] = (
                "transcript.final" if self.is_done else "transcript.partial"
            )

            # Only send if text changed or final
            if text != self.previous_text or self.is_done:
                self.previous_text = text
                return TranscriptMessage(
                    type=msg_type,
                    text=text,
                    latency_ms=latency_ms,
                    audio_duration_ms=audio_duration_ms,
                )

            return None

        except Exception as e:
            logger.exception(f"Transcription error: {e}")
            return TranscriptMessage(
                type="error",
                text=str(e),
            )

    def mark_done(self) -> None:
        """Mark audio stream as complete."""
        self.is_done = True


@router.websocket("/v1/audio/transcriptions/stream")
async def stream_transcription(
    ws: WebSocket,
    model: str,
    language: str | None = None,
    sample_rate: int = 16000,
    min_chunk_ms: int = 200,  # Minimum audio before first transcript
    chunk_ms: int = 300,  # How often to produce transcripts
    enable_vad: bool = False,
) -> None:
    """
    WebSocket endpoint for streaming audio transcription.

    Optimized for ULTRA LOW TTFT (Time to First Transcript) with 16kHz PCM16 mono input.

    Query Parameters:
        model: Whisper model to use (e.g., "Systran/faster-whisper-large-v3")
        language: Language code (e.g., "tr", "en") or None for auto-detect
        sample_rate: Input audio sample rate (default: 16000)
        min_chunk_ms: Minimum audio before first transcript (default: 200ms, min: 100ms)
        chunk_ms: How often to produce transcripts after first (default: 300ms)
        enable_vad: Enable voice activity detection (default: False for lowest latency)

    Protocol:
        1. Client sends binary PCM16 audio chunks (16kHz mono, little-endian)
        2. Server sends JSON transcript messages as audio is processed
        3. Client sends {"type": "audio.done"} when finished
        4. Server sends final transcript and closes connection

    For lowest TTFT, use: min_chunk_ms=100&chunk_ms=200
    """
    # Get dependencies
    config = get_config()
    whisper_model_manager = get_whisper_model_manager()

    await ws.accept()
    logger.info(f"Streaming STT connection accepted: model={model}, language={language}, sample_rate={sample_rate}")

    # Send ready message
    await ws.send_json(TranscriptMessage(type="ready", text="Connection established").model_dump())

    # Enforce minimum values for safety
    min_chunk_ms = max(100, min_chunk_ms)  # At least 100ms
    chunk_ms = max(100, chunk_ms)  # At least 100ms

    # Create transcriber
    streaming_config = StreamingConfig(
        sample_rate=sample_rate,
        min_chunk_duration_ms=min_chunk_ms,
        chunk_duration_ms=chunk_ms,
        enable_vad=enable_vad,
    )
    transcriber = StreamingTranscriber(
        whisper_model_manager=whisper_model_manager,
        model=model,
        language=language,
        config=streaming_config,
    )

    try:
        while True:
            try:
                # Receive message with timeout
                message = await asyncio.wait_for(ws.receive(), timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("WebSocket timeout, closing connection")
                break

            if message["type"] == "websocket.disconnect":
                break

            # Handle binary audio data
            if "bytes" in message and message["bytes"]:
                audio_bytes = message["bytes"]
                audio_chunk = pcm16_bytes_to_float32(audio_bytes)

                # Handle sample rate mismatch
                if sample_rate != SAMPLE_RATE:
                    # Resample to 16kHz
                    ratio = SAMPLE_RATE / sample_rate
                    target_length = int(len(audio_chunk) * ratio)
                    audio_chunk = np.interp(
                        np.linspace(0, len(audio_chunk), target_length),
                        np.arange(len(audio_chunk)),
                        audio_chunk
                    ).astype(np.float32)

                transcriber.append_audio(audio_chunk)

                # Check if we should transcribe
                if transcriber.should_transcribe():
                    result = await transcriber.transcribe()
                    if result:
                        await ws.send_json(result.model_dump())

            # Handle text/JSON messages
            elif "text" in message and message["text"]:
                import json
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "audio.done":
                        # Final transcription
                        transcriber.mark_done()
                        if transcriber.should_transcribe():
                            result = await transcriber.transcribe()
                            if result:
                                await ws.send_json(result.model_dump())
                        break
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON message: {message['text']}")

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
        try:
            await ws.send_json(TranscriptMessage(type="error", text=str(e)).model_dump())
        except:
            pass
    finally:
        try:
            await ws.close()
        except:
            pass
        logger.info(f"Streaming STT session ended. Total audio: {transcriber.buffer_duration_ms:.0f}ms")
