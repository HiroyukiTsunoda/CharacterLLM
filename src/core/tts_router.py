"""
TTSRouter: Qwen3-TTS エンジンへの委譲ルーター。

外部から見れば TTSEngineBase と同じインターフェースで使える。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from src.core.tts_base import TTSEngineBase

if TYPE_CHECKING:
    from src.core.character import Character

logger = logging.getLogger(__name__)


class TTSRouter(TTSEngineBase):
    """Qwen3-TTS エンジンへの委譲ルーター。"""

    def __init__(self, config: dict | None = None):
        super().__init__(config)

        self._qwen3_engine = None
        self._qwen3_init_attempted = False

    @property
    def qwen3_engine(self):
        """Qwen3TTSEngine を遅延初期化で取得する。"""
        if self._qwen3_engine is None and not self._qwen3_init_attempted:
            self._qwen3_init_attempted = True
            try:
                from src.core.qwen3_tts_engine import Qwen3TTSEngine
                self._qwen3_engine = Qwen3TTSEngine(config=self._config)
                if not self._qwen3_engine.available:
                    logger.info(
                        "Qwen3-TTS is not available (qwen-tts not installed)."
                    )
            except Exception as e:
                logger.warning("Failed to initialize Qwen3TTSEngine: %s", e)
        return self._qwen3_engine

    @property
    def qwen3_available(self) -> bool:
        """Qwen3-TTS が利用可能かどうか。"""
        engine = self.qwen3_engine
        return engine is not None and engine.available

    # ------------------------------------------------------------------
    # プロパティのオーバーライド
    # ------------------------------------------------------------------

    @TTSEngineBase.enabled.setter
    def enabled(self, value: bool):
        self._enabled = value
        if self._qwen3_engine is not None:
            self._qwen3_engine.enabled = value

    @TTSEngineBase.use_gpu.setter
    def use_gpu(self, value: bool):
        if value != self._use_gpu:
            self._use_gpu = value
            if self._qwen3_engine is not None:
                self._qwen3_engine.use_gpu = value

    @TTSEngineBase.auto_play.setter
    def auto_play(self, value: bool):
        self._auto_play = value
        if self._qwen3_engine is not None:
            self._qwen3_engine.auto_play = value

    # ------------------------------------------------------------------
    # TTSEngineBase 実装
    # ------------------------------------------------------------------

    def _get_engine(self) -> TTSEngineBase:
        """Qwen3-TTS エンジンを取得する。利用不可なら例外。"""
        engine = self.qwen3_engine
        if engine is None or not engine.available:
            raise RuntimeError(
                "Qwen3-TTS が利用できません。'pip install qwen-tts' を実行してください。"
            )
        return engine

    def synthesize(
        self, text: str, character: Character, *, force: bool = False,
    ) -> tuple[int, np.ndarray]:
        return self._get_engine().synthesize(text, character, force=force)

    def try_register_cached_model(self, character: Character) -> bool:
        if self._qwen3_engine is not None:
            return self._qwen3_engine.try_register_cached_model(character)
        return False

    def load_model_for_character(self, character: Character) -> None:
        self._get_engine().load_model_for_character(character)

    def is_character_loaded(self, character_id: str) -> bool:
        if self._qwen3_engine is not None:
            return self._qwen3_engine.is_character_loaded(character_id)
        return False

    def loaded_character_ids(self) -> list[str]:
        if self._qwen3_engine is not None:
            return self._qwen3_engine.loaded_character_ids()
        return []

    def unload_character(self, character_id: str) -> None:
        if self._qwen3_engine is not None:
            self._qwen3_engine.unload_character(character_id)

    def unload_all(self) -> None:
        if self._qwen3_engine is not None:
            self._qwen3_engine.unload_all()
        logger.info("TTSRouter: all engines unloaded.")

    def clear_cache(self) -> None:
        if self._qwen3_engine is not None:
            self._qwen3_engine.clear_cache()

    def get_config(self) -> dict:
        """設定を返す。"""
        config: dict = {
            "enabled": self._enabled,
            "use_gpu": self._use_gpu,
            "auto_play": self._auto_play,
        }
        if self._qwen3_engine is not None:
            qwen3_cfg = self._qwen3_engine.get_config()
            config["qwen3"] = qwen3_cfg.get("qwen3", {})
        else:
            qwen3_defaults = self._config.get("tts", {}).get("qwen3", {})
            config["qwen3"] = qwen3_defaults or {
                "default_model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                "dtype": "bfloat16",
                "flash_attention": True,
            }
        return config
