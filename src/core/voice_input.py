"""
VoiceEngine: sounddevice によるマイク録音と faster-whisper による音声認識を管理する。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE = "float32"
MIN_RECORDING_SECONDS = 0.5


@dataclass
class AudioDevice:
    """オーディオデバイスの情報。"""

    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    is_default_input: bool = False
    is_default_output: bool = False


class VoiceEngine:
    """マイク録音 + faster-whisper 推論を管理する。"""

    def __init__(self, config: dict | None = None):
        self._config = config or {}
        voice_cfg = self._config.get("voice", {})

        self._model_size: str = voice_cfg.get("whisper_model", "small")
        self._device: str = voice_cfg.get("device", "cuda")
        self._compute_type: str = voice_cfg.get("compute_type", "float16")
        self._language: str = voice_cfg.get("language", "ja")
        self._input_device: Optional[int] = voice_cfg.get("input_device")
        self._output_device: Optional[int] = voice_cfg.get("output_device")

        self._whisper_model = None
        self._stream = None
        self._audio_buffer: list[np.ndarray] = []
        self._recording = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # プロパティ
    # ------------------------------------------------------------------

    @property
    def is_model_loaded(self) -> bool:
        return self._whisper_model is not None

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def input_device(self) -> Optional[int]:
        return self._input_device

    @input_device.setter
    def input_device(self, device_id: Optional[int]):
        self._input_device = device_id

    @property
    def output_device(self) -> Optional[int]:
        return self._output_device

    @output_device.setter
    def output_device(self, device_id: Optional[int]):
        self._output_device = device_id

    @property
    def model_size(self) -> str:
        return self._model_size

    @property
    def whisper_device(self) -> str:
        return self._device

    @property
    def compute_type(self) -> str:
        return self._compute_type

    @property
    def language(self) -> str:
        return self._language

    # ------------------------------------------------------------------
    # デバイス列挙
    # ------------------------------------------------------------------

    @staticmethod
    def list_audio_devices() -> list[AudioDevice]:
        """利用可能なオーディオデバイスを列挙する。"""
        try:
            import sounddevice as sd
        except ImportError:
            logger.warning("sounddevice is not installed")
            return []

        devices: list[AudioDevice] = []
        try:
            default_input, default_output = sd.default.device
            for idx, info in enumerate(sd.query_devices()):
                devices.append(
                    AudioDevice(
                        index=idx,
                        name=info["name"],
                        max_input_channels=info["max_input_channels"],
                        max_output_channels=info["max_output_channels"],
                        is_default_input=(idx == default_input),
                        is_default_output=(idx == default_output),
                    )
                )
        except Exception as e:
            logger.error("Failed to enumerate audio devices: %s", e)
        return devices

    @staticmethod
    def list_input_devices() -> list[AudioDevice]:
        return [d for d in VoiceEngine.list_audio_devices() if d.max_input_channels > 0]

    @staticmethod
    def list_output_devices() -> list[AudioDevice]:
        return [d for d in VoiceEngine.list_audio_devices() if d.max_output_channels > 0]

    # ------------------------------------------------------------------
    # Whisper モデル管理
    # ------------------------------------------------------------------

    def load_model(
        self,
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
    ) -> None:
        """Whisper モデルをロードする。"""
        if model_size:
            self._model_size = model_size
        if device:
            self._device = device
        if compute_type:
            self._compute_type = compute_type

        self.unload_model()

        from faster_whisper import WhisperModel

        logger.info(
            "Loading Whisper model: %s (device=%s, compute_type=%s)",
            self._model_size,
            self._device,
            self._compute_type,
        )
        self._whisper_model = WhisperModel(
            self._model_size,
            device=self._device,
            compute_type=self._compute_type,
        )
        logger.info("Whisper model loaded successfully.")

    def unload_model(self) -> None:
        """Whisper モデルをアンロードする。"""
        if self._whisper_model is not None:
            logger.info("Unloading Whisper model")
            del self._whisper_model
            self._whisper_model = None

    def update_settings(
        self,
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        language: str | None = None,
    ) -> None:
        """設定を更新する。モデル関連の設定が変わった場合は次回使用時に再ロードされる。"""
        needs_reload = False
        if model_size and model_size != self._model_size:
            self._model_size = model_size
            needs_reload = True
        if device and device != self._device:
            self._device = device
            needs_reload = True
        if compute_type and compute_type != self._compute_type:
            self._compute_type = compute_type
            needs_reload = True
        if language is not None:
            self._language = language

        if needs_reload and self._whisper_model is not None:
            self.unload_model()

    # ------------------------------------------------------------------
    # 録音
    # ------------------------------------------------------------------

    def start_recording(self) -> None:
        """マイク録音を開始する。"""
        import sounddevice as sd

        with self._lock:
            if self._recording:
                return
            self._audio_buffer.clear()
            self._recording = True

        def _callback(indata, frames, time_info, status):
            if status:
                logger.warning("sounddevice status: %s", status)
            with self._lock:
                if self._recording:
                    self._audio_buffer.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            device=self._input_device,
            callback=_callback,
        )
        self._stream.start()
        logger.info("Recording started (device=%s)", self._input_device)

    def stop_recording(self) -> np.ndarray | None:
        """録音を停止し、音声データを返す。短すぎる場合は None を返す。"""
        # ストリーム参照の取得・解除をロック内で行い、
        # 別スレッドの _callback との競合を防ぐ
        with self._lock:
            self._recording = False
            stream = self._stream
            self._stream = None

        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as e:
                logger.warning("Error stopping stream: %s", e)

        with self._lock:
            if not self._audio_buffer:
                return None
            audio = np.concatenate(self._audio_buffer, axis=0).flatten()
            self._audio_buffer.clear()

        duration = len(audio) / SAMPLE_RATE
        if duration < MIN_RECORDING_SECONDS:
            logger.info("Recording too short (%.2fs), discarding", duration)
            return None

        logger.info("Recording stopped: %.2fs, %d samples", duration, len(audio))
        return audio

    # ------------------------------------------------------------------
    # 文字起こし
    # ------------------------------------------------------------------

    def transcribe(self, audio: np.ndarray) -> str:
        """音声データを文字起こしする。モデル未ロードなら RuntimeError を送出する。"""
        if self._whisper_model is None:
            raise RuntimeError("Whisperモデルが未ロードです。音声パネルからモデルをロードしてください。")

        segments, info = self._whisper_model.transcribe(
            audio,
            language=self._language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        result = " ".join(text_parts).strip()
        logger.info("Transcription: %s (lang=%s, prob=%.2f)", result, info.language, info.language_probability)
        return result

    # ------------------------------------------------------------------
    # config 書き出し用
    # ------------------------------------------------------------------

    def get_config(self) -> dict:
        """現在の設定を辞書として返す。"""
        return {
            "input_device": self._input_device,
            "output_device": self._output_device,
            "whisper_model": self._model_size,
            "device": self._device,
            "compute_type": self._compute_type,
            "language": self._language,
            "auto_send": self._config.get("voice", {}).get("auto_send", False),
        }
