"""
OpenAIProvider: OpenAI API (ChatGPT) を使った推論プロバイダー。

ModelLoader と同じインターフェースを提供し、ChatEngine から
透過的に利用できる。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


def _friendly_error(exc: Exception) -> str:
    """OpenAI の例外を日本語のわかりやすいメッセージに変換する。"""
    msg = str(exc)
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)

    if code == 429 or "insufficient_quota" in msg:
        return (
            "APIクォータ（利用枠）が不足しています。\n"
            "OpenAI の Billing ページでクレジットを追加してください。\n"
            "https://platform.openai.com/account/billing"
        )
    if code == 401 or "invalid_api_key" in msg or "Incorrect API key" in msg:
        return "APIキーが無効です。正しいキーを .env に設定してください。"
    if code == 404 or "model_not_found" in msg:
        return "指定されたモデルが見つかりません。別のモデルを選択してください。"
    if "rate_limit" in msg.lower():
        return "レートリミットに達しました。しばらく待ってから再試行してください。"

    return msg


@dataclass
class OpenAIModelEntry:
    """選択可能な OpenAI モデル定義。"""
    model_id: str
    display_name: str
    supports_thinking: bool = False


AVAILABLE_MODELS: list[OpenAIModelEntry] = [
    OpenAIModelEntry("gpt-4o", "GPT-4o (最新・高性能)"),
    OpenAIModelEntry("gpt-4o-mini", "GPT-4o mini (高速・低コスト)"),
    OpenAIModelEntry("gpt-4-turbo", "GPT-4 Turbo"),
    OpenAIModelEntry("gpt-3.5-turbo", "GPT-3.5 Turbo (最安)"),
    OpenAIModelEntry("o3-mini", "o3-mini (推論モデル)", supports_thinking=True),
]


class OpenAIProvider:
    """OpenAI API を使った推論プロバイダー。

    ModelLoader と同じプロパティ・メソッドを公開するため、
    ChatEngine は provider を差し替えるだけで利用できる。
    """

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client = None
        self._current_model: str | None = None
        self._supports_thinking: bool = False

    # ------------------------------------------------------------------
    # ModelLoader 互換プロパティ
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._client is not None and self._current_model is not None

    @property
    def supports_thinking(self) -> bool:
        return self._supports_thinking

    @property
    def uses_think_tags(self) -> bool:
        return False

    @property
    def template_inserts_think(self) -> bool:
        return False

    @property
    def current_model(self) -> str | None:
        return self._current_model

    @property
    def api_key_set(self) -> bool:
        return bool(self._api_key)

    # ------------------------------------------------------------------
    # 接続管理
    # ------------------------------------------------------------------

    def set_api_key(self, api_key: str) -> None:
        self._api_key = api_key
        self._client = None

    def connect(self, model_id: str = "gpt-4o") -> None:
        """OpenAI クライアントを初期化し、モデルを選択する。"""
        if not self._api_key:
            raise ValueError(
                "OpenAI API キーが設定されていません。\n"
                ".env ファイルに OPENAI_API_KEY を設定してください。"
            )

        from openai import OpenAI
        self._client = OpenAI(api_key=self._api_key, max_retries=1)
        self._current_model = model_id

        entry = self._find_entry(model_id)
        self._supports_thinking = entry.supports_thinking if entry else False

        logger.info("OpenAI provider connected: model=%s", model_id)

    def disconnect(self) -> None:
        """プロバイダーを切断する。"""
        self._client = None
        self._current_model = None
        self._supports_thinking = False
        logger.info("OpenAI provider disconnected.")

    def test_connection(self) -> str:
        """API 接続をテストし、結果メッセージを返す。"""
        if not self._api_key:
            return "APIキーが未設定です。"
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self._api_key, max_retries=0)
            response = client.chat.completions.create(
                model=self._current_model or "gpt-4o-mini",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )
            return f"接続成功 (モデル: {response.model})"
        except Exception as e:
            return f"接続失敗: {_friendly_error(e)}"

    # ------------------------------------------------------------------
    # 推論
    # ------------------------------------------------------------------

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
        repeat_penalty: float = 1.1,
        stream: bool = True,
    ) -> Iterator[str] | str:
        """ChatEngine 互換の推論メソッド。"""
        if self._client is None or self._current_model is None:
            raise RuntimeError(
                "OpenAI プロバイダーが接続されていません。先に connect() を呼んでください。"
            )

        # repeat_penalty → frequency_penalty に概算マッピング
        # llama.cpp の repeat_penalty 1.0 = ペナルティなし、OpenAI の frequency_penalty 0.0 = なし
        freq_penalty = max(0.0, min(2.0, (repeat_penalty - 1.0) * 2.0))

        kwargs = dict(
            model=self._current_model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            frequency_penalty=freq_penalty,
            stream=stream,
        )

        # o-series モデルは temperature/top_p をサポートしない
        if self._current_model.startswith("o"):
            kwargs.pop("temperature", None)
            kwargs.pop("top_p", None)
            kwargs.pop("frequency_penalty", None)

        try:
            if stream:
                return self._generate_stream(kwargs)
            else:
                response = self._client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(_friendly_error(e)) from e

    def _generate_stream(self, kwargs: dict) -> Iterator[str]:
        """ストリーミング推論。トークンごとに yield する。"""
        try:
            stream = self._client.chat.completions.create(**kwargs)
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise RuntimeError(_friendly_error(e)) from e

    # ------------------------------------------------------------------
    # ヘルパー
    # ------------------------------------------------------------------

    @staticmethod
    def get_available_models() -> list[OpenAIModelEntry]:
        return AVAILABLE_MODELS.copy()

    @staticmethod
    def _find_entry(model_id: str) -> OpenAIModelEntry | None:
        for entry in AVAILABLE_MODELS:
            if entry.model_id == model_id:
                return entry
        return None
