"""
InferenceProvider: 推論プロバイダーの共通インターフェース定義。

ModelLoader (ローカルGGUF) と OpenAIProvider (OpenAI API) はこの Protocol を
構造的に満たす。ChatEngine はこの型を通してプロバイダーを透過的に扱う。
"""

from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable


@runtime_checkable
class InferenceProvider(Protocol):
    """推論プロバイダーが満たすべきインターフェース。"""

    @property
    def is_loaded(self) -> bool:
        """モデルがロード済み（または接続済み）か。"""
        ...

    @property
    def supports_thinking(self) -> bool:
        """思考ブロック（<think> 等）を出力するモデルか。"""
        ...

    @property
    def uses_think_tags(self) -> bool:
        """システムプロンプトで <think> タグの使用を指示すべきか。"""
        ...

    @property
    def template_inserts_think(self) -> bool:
        """チャットテンプレートが <think> を自動挿入するか。"""
        ...

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
        repeat_penalty: float = 1.1,
        stream: bool = True,
    ) -> Iterator[str] | str:
        """メッセージリストから応答を生成する。

        stream=True ならトークンの Iterator、False なら完成テキストを返す。
        """
        ...
