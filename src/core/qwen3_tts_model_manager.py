"""
Qwen3TTSModelManager: Qwen3-TTS モデルの一覧・ダウンロード状況管理。

Qwen3-TTS は qwen-tts パッケージ + HuggingFace Hub 経由でモデルを
自動ダウンロードする。本マネージャーはモデルバリアントの定義と UI 向けの情報提供を行う。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Qwen3TTSModelInfo:
    """Qwen3-TTS モデルバリアントの情報。"""
    id: str
    name: str
    repo_id: str
    mode: str
    size: str
    description: str


KNOWN_MODELS: list[Qwen3TTSModelInfo] = [
    # 0.6B シリーズ
    Qwen3TTSModelInfo(
        id="qwen3-tts-0.6b-base",
        name="Qwen3-TTS 0.6B Base (音声クローン)",
        repo_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        mode="base",
        size="0.6B",
        description="0.6B パラメータ、低レイテンシ、3秒音声クローン対応",
    ),
    Qwen3TTSModelInfo(
        id="qwen3-tts-0.6b-custom",
        name="Qwen3-TTS 0.6B CustomVoice",
        repo_id="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        mode="custom_voice",
        size="0.6B",
        description="0.6B パラメータ、事前定義スピーカー + 感情制御",
    ),
    # 1.7B シリーズ
    Qwen3TTSModelInfo(
        id="qwen3-tts-1.7b-base",
        name="Qwen3-TTS 1.7B Base (音声クローン)",
        repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        mode="base",
        size="1.7B",
        description="1.7B パラメータ、高品質、3秒音声クローン対応",
    ),
    Qwen3TTSModelInfo(
        id="qwen3-tts-1.7b-custom",
        name="Qwen3-TTS 1.7B CustomVoice",
        repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        mode="custom_voice",
        size="1.7B",
        description="1.7B パラメータ、高品質、事前定義スピーカー + 感情制御",
    ),
    Qwen3TTSModelInfo(
        id="qwen3-tts-1.7b-design",
        name="Qwen3-TTS 1.7B VoiceDesign",
        repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        mode="voice_design",
        size="1.7B",
        description="1.7B パラメータ、高品質、テキスト記述でボイス生成",
    ),
]


class Qwen3TTSModelManager:
    """Qwen3-TTS モデルバリアントの管理。"""

    def __init__(self):
        self._models = {m.id: m for m in KNOWN_MODELS}

    def list_models(self) -> list[Qwen3TTSModelInfo]:
        """利用可能な全モデルバリアントを返す。"""
        return list(KNOWN_MODELS)

    def get_model(self, model_id: str) -> Optional[Qwen3TTSModelInfo]:
        """ID でモデル情報を取得する。"""
        return self._models.get(model_id)

    def get_model_by_repo(self, repo_id: str) -> Optional[Qwen3TTSModelInfo]:
        """HuggingFace repo ID でモデル情報を取得する。"""
        for m in KNOWN_MODELS:
            if m.repo_id == repo_id:
                return m
        return None

    def is_cached(self, repo_id: str) -> bool:
        """モデルが HuggingFace キャッシュに存在するか確認する。"""
        try:
            from huggingface_hub import try_to_load_from_cache
            result = try_to_load_from_cache(repo_id, "config.json")
            return result is not None
        except Exception:
            return False

    def list_models_with_status(self) -> list[dict]:
        """モデル一覧にキャッシュ状態を付与して返す。"""
        result = []
        for m in KNOWN_MODELS:
            info = {
                "id": m.id,
                "name": m.name,
                "repo_id": m.repo_id,
                "mode": m.mode,
                "size": m.size,
                "description": m.description,
                "cached": self.is_cached(m.repo_id),
            }
            result.append(info)
        return result
