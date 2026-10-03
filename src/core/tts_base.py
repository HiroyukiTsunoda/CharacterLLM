"""
TTSEngineBase: TTS エンジンの抽象基底クラス。

Qwen3-TTS 等のバックエンドが共通で実装するインターフェースを定義する。
"""

from __future__ import annotations

import io
import logging
import re
import threading
import wave
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.core.character import Character

logger = logging.getLogger(__name__)


class TTSEngineBase(ABC):
    """TTS エンジンの抽象基底クラス。"""

    def __init__(self, config: dict | None = None):
        self._config = config or {}
        tts_cfg = self._config.get("tts", {})

        self._enabled: bool = tts_cfg.get("enabled", False)
        self._use_gpu: bool = tts_cfg.get("use_gpu", True)
        self._auto_play: bool = tts_cfg.get("auto_play", True)

        self._model_cache: dict[str, object] = {}
        self._lock = threading.Lock()
        self._playback_stop = threading.Event()

    # ------------------------------------------------------------------
    # 共通プロパティ
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    @property
    def use_gpu(self) -> bool:
        return self._use_gpu

    @use_gpu.setter
    def use_gpu(self, value: bool):
        if value != self._use_gpu:
            self._use_gpu = value
            self.clear_cache()

    @property
    def auto_play(self) -> bool:
        return self._auto_play

    @auto_play.setter
    def auto_play(self, value: bool):
        self._auto_play = value

    @property
    def device(self) -> str:
        if self._use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
            logger.warning("CUDA is not available; falling back to CPU.")
        return "cpu"

    @property
    def cuda_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # キャラクター別モデルキャッシュ（共通実装）
    # ------------------------------------------------------------------

    def is_character_loaded(self, character_id: str) -> bool:
        with self._lock:
            return character_id in self._model_cache

    def loaded_character_ids(self) -> list[str]:
        with self._lock:
            return list(self._model_cache.keys())

    def clear_cache(self) -> None:
        with self._lock:
            self._model_cache.clear()
        logger.info("%s: model cache cleared.", type(self).__name__)

    def unload_character(self, character_id: str) -> None:
        with self._lock:
            if character_id in self._model_cache:
                del self._model_cache[character_id]
                logger.info("%s: model unloaded: %s",
                            type(self).__name__, character_id)

    # ------------------------------------------------------------------
    # 抽象メソッド（各エンジンが実装する）
    # ------------------------------------------------------------------

    @abstractmethod
    def synthesize(
        self, text: str, character: Character, *, force: bool = False,
    ) -> tuple[int, np.ndarray]:
        """テキストを音声に変換する。(sample_rate, audio_int16) を返す。"""
        ...

    @abstractmethod
    def load_model_for_character(self, character: Character) -> None:
        """キャラクターの TTS モデルを明示的にメモリへロードする。"""
        ...

    @abstractmethod
    def unload_all(self) -> None:
        """全リソースを解放する。"""
        ...

    @abstractmethod
    def get_config(self) -> dict:
        """現在の設定を辞書として返す。"""
        ...

    # ------------------------------------------------------------------
    # 音声再生（共通実装）
    # ------------------------------------------------------------------

    def play_audio(
        self,
        sr: int,
        audio: np.ndarray,
        output_device: int | None = None,
    ) -> None:
        """sounddevice で音声を再生する（ブロッキング）。"""
        import sounddevice as sd

        self._playback_stop.clear()

        if audio.dtype != np.int16:
            if audio.dtype in (np.float32, np.float64):
                audio = (audio * 32767).astype(np.int16)

        logger.info(
            "Playing audio: %d samples, sr=%d, device=%s",
            len(audio), sr, output_device,
        )

        sd.play(audio, samplerate=sr, device=output_device)
        sd.wait()

    def stop_playback(self) -> None:
        """再生中の音声を停止する。"""
        try:
            import sounddevice as sd
            sd.stop()
        except Exception as e:
            logger.warning("Failed to stop playback: %s", e)
        self._playback_stop.set()

    # ------------------------------------------------------------------
    # ユーティリティ（共通実装）
    # ------------------------------------------------------------------

    def synthesize_to_wav(self, text: str, character: Character) -> bytes:
        """テキストを WAV バイト列に変換する。"""
        sr, audio = self.synthesize(text, character)
        return self._audio_to_wav(sr, audio)

    @staticmethod
    def _audio_to_wav(sr: int, audio: np.ndarray) -> bytes:
        """numpy 音声データを WAV バイト列に変換する。"""
        if audio.dtype in (np.float32, np.float64):
            audio = (audio * 32767).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(audio.tobytes())
        buf.seek(0)
        return buf.read()

    @staticmethod
    def _clean_text_for_tts(text: str) -> str:
        """LLM 応答テキストから TTS に不適切な要素を除去する。"""
        t = text.strip()
        t = re.sub(r"```[\s\S]*?```", "", t)
        t = re.sub(r"`[^`]+`", "", t)
        t = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", t)
        t = re.sub(r"#{1,6}\s*", "", t)
        t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
        t = re.sub(r"[*_~|>]+", "", t)
        t = re.sub(r"[^\u3000-\u9FFF\u30A0-\u30FF\u3040-\u309F"
                    r"\uFF00-\uFFEF\u4E00-\u9FAF"
                    r"a-zA-Za-zA-Z0-90-9"
                    r"、。！？!?.,;:…ー〜（）()「」『』【】\s\n]", "", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    @staticmethod
    def _split_text(text: str, max_len: int = 100) -> list[str]:
        """テキストを句読点で区切り、max_len 以下のセグメントに分割する。"""
        sentences = re.split(r"(?<=[。！？!?\n])", text)
        segments: list[str] = []
        buf = ""
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if len(buf) + len(s) > max_len and buf:
                segments.append(buf)
                buf = s
            else:
                buf += s
        if buf:
            segments.append(buf)
        return segments if segments else [text[:max_len]]
