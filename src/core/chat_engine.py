"""
ChatEngine: キャラクターとモデルを繋ぎ、チャット履歴を管理して推論を実行する。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterator, Optional

from src.core.character import Character, CharacterManager
from src.core.model_loader import ModelLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# チャットメッセージ
# ---------------------------------------------------------------------------

@dataclass
class ChatMessage:
    """1つのチャットメッセージ。"""
    role: str          # "system" | "user" | "assistant"
    content: str
    character_id: str = ""
    think_content: str = ""   # <think>...</think> の中身（表示用に保持）

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


# ---------------------------------------------------------------------------
# ChatEngine
# ---------------------------------------------------------------------------

class ChatEngine:
    """チャットの中枢。キャラクター・モデル・履歴を統合する。"""

    def __init__(
        self,
        model_loader: ModelLoader,
        character_manager: CharacterManager,
    ):
        self.model_loader = model_loader
        self.character_manager = character_manager

        self._current_character: Optional[Character] = None
        self._history: list[ChatMessage] = []
        self._max_history: int = 50  # 最大履歴数（system除く）
        self._last_think_content: str = ""  # 直近の推論で取得した思考内容

    # ------------------------------------------------------------------
    # プロパティ
    # ------------------------------------------------------------------

    @property
    def current_character(self) -> Optional[Character]:
        return self._current_character

    @property
    def history(self) -> list[ChatMessage]:
        return list(self._history)

    @property
    def is_ready(self) -> bool:
        """モデルとキャラクター両方がセットされているか。"""
        return self.model_loader.is_loaded and self._current_character is not None

    # ------------------------------------------------------------------
    # キャラクター操作
    # ------------------------------------------------------------------

    def set_character(self, character_id: str) -> Character:
        """アクティブなキャラクターを変更する。"""
        character = self.character_manager.get_character(character_id)
        if character is None:
            raise ValueError(f"キャラクターが見つかりません: {character_id}")
        self._current_character = character
        logger.info("Character set: %s (%s)", character.name, character.id)
        return character

    def set_character_direct(self, character: Character) -> None:
        """Characterオブジェクトを直接セットする。"""
        self._current_character = character

    # ------------------------------------------------------------------
    # 履歴管理
    # ------------------------------------------------------------------

    def clear_history(self) -> None:
        """チャット履歴をクリア。"""
        self._history.clear()
        logger.info("Chat history cleared.")

    def add_message(self, role: str, content: str, think_content: str = "") -> ChatMessage:
        """履歴にメッセージを追加。"""
        msg = ChatMessage(
            role=role,
            content=content,
            character_id=self._current_character.id if self._current_character else "",
            think_content=think_content,
        )
        self._history.append(msg)

        # 履歴が長くなりすぎたら古いものを削除（systemは除く）
        user_assistant_msgs = [m for m in self._history if m.role != "system"]
        if len(user_assistant_msgs) > self._max_history:
            # 最も古いuser/assistantメッセージを削除
            for i, m in enumerate(self._history):
                if m.role != "system":
                    self._history.pop(i)
                    break

        return msg

    # <think> ブロック使用の指示（システムプロンプトに付加）
    _THINK_INSTRUCTION = (
        "\n\n応答する前に、必ず<think>タグ内で思考してください。"
        "思考は日本語で行ってください。"
        "形式: <think>思考内容</think>応答内容"
    )

    def _build_messages(self, user_input: str) -> list[dict]:
        """LLMに送信するメッセージリストを組み立てる。"""
        messages: list[dict] = []

        # 1. システムプロンプト（キャラクター + <think>タグ対応モデルに思考指示）
        if self._current_character:
            sys_msg = self._current_character.get_system_message()
            if self.model_loader.uses_think_tags:
                sys_msg["content"] += self._THINK_INSTRUCTION
            messages.append(sys_msg)

        # 2. 過去の会話履歴
        for msg in self._history:
            messages.append(msg.to_dict())

        # 3. 今回のユーザー入力
        messages.append({"role": "user", "content": user_input})

        return messages

    # ------------------------------------------------------------------
    # 動的 max_tokens 判定
    # ------------------------------------------------------------------

    # 詳細な回答を求めるキーワード（優先度高）
    _DETAIL_KEYWORDS = ("教えて", "詳しく", "理由", "なぜ", "どうして")
    # 質問・相談・説明要求を示すキーワード
    _QUESTION_KEYWORDS = (
        "?", "？", "どう", "何", "なに", "いつ", "どこ", "誰", "だれ",
        "どれ", "どの", "ですか", "ますか", "かな", "だろう",
        "相談", "説明", "教え", "知りたい", "わからない", "分からない",
    )

    # <think> ブロックに割り当てる追加トークン予算
    _THINK_TOKEN_BUDGET = 512

    def _estimate_max_tokens(self, user_input: str) -> int:
        """ユーザー入力の長さ・内容に応じて max_tokens を動的に決定する。

        ルール（優先度順）:
          1. 「教えて」「詳しく」「理由」等を含む → 768
          2. 質問 / 相談 / 説明要求            → 512
          3. 通常会話（〜80文字）              → 256
          4. 短文（〜20文字）                  → 128

        返値は応答本文用のトークン数。<think> 予算は呼び出し側で加算する。
        """
        text = user_input.strip()

        # 優先度1: 詳細要求キーワード
        if any(kw in text for kw in self._DETAIL_KEYWORDS):
            return 768

        # 優先度2: 質問・相談系
        if any(kw in text for kw in self._QUESTION_KEYWORDS):
            return 512

        # 優先度3/4: 文字数ベース
        if len(text) <= 20:
            return 128

        return 256

    # ------------------------------------------------------------------
    # 思考ブロック除去（Qwen3 <think> + gpt-oss analysis channel）
    # ------------------------------------------------------------------

    # Qwen3: <think>...</think>
    _THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)
    _THINK_EXTRACT_RE = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)

    # gpt-oss: <|channel|>analysis<|message|>...<|end|> ... <|channel|>final<|message|>
    _GPT_OSS_ANALYSIS_RE = re.compile(
        r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>",
        flags=re.DOTALL,
    )
    _GPT_OSS_FINAL_RE = re.compile(
        r".*<\|channel\|>final<\|message\|>",
        flags=re.DOTALL,
    )

    @staticmethod
    def strip_think(text: str) -> str:
        """応答テキストから思考ブロックを除去する。

        Qwen3 の <think>...</think> と gpt-oss の analysis チャネルの
        両方に対応。
        """
        result = ChatEngine._THINK_RE.sub("", text)
        result = ChatEngine._GPT_OSS_FINAL_RE.sub("", result)
        return result.lstrip("\n ")

    @staticmethod
    def extract_think(text: str) -> str:
        """応答テキストから思考内容を抽出する。"""
        match = ChatEngine._THINK_EXTRACT_RE.search(text)
        if match:
            return match.group(1).strip()
        match = ChatEngine._GPT_OSS_ANALYSIS_RE.search(text)
        if match:
            return match.group(1).strip()
        return ""

    def get_last_think_content(self) -> str:
        """直近の推論で取得した思考内容を返す。"""
        return self._last_think_content

    # ストリーム検出に必要な最小文字数（"<|channel|>" = 11 文字）
    _DETECT_WINDOW = 11
    # <think> タグの部分一致を保持するためのマージン
    _THINK_TAG_LEN = len("<think>")

    def _filter_think_stream(
        self, token_stream: Iterator[str]
    ) -> Iterator[str]:
        """ストリーミングトークンから思考ブロックを除去して yield する。

        Qwen3 の <think>...</think> と gpt-oss の analysis チャネルに対応。
        <think> は応答の前後どちらに出現しても検出する。
        思考内容は self._last_think_content に保存される。
        """
        buffer = ""
        pending = ""
        phase = "detect"  # "detect" | "in_think" | "in_analysis" | "passthrough"
        self._last_think_content = ""

        for token in token_stream:
            if phase == "passthrough":
                pending += token
                think_idx = pending.find("<think>")
                if think_idx != -1:
                    if think_idx > 0:
                        yield pending[:think_idx]
                    buffer = pending[think_idx:]
                    pending = ""
                    phase = "in_think"
                else:
                    safe = len(pending) - self._THINK_TAG_LEN + 1
                    if safe > 0:
                        yield pending[:safe]
                        pending = pending[safe:]
                continue

            buffer += token

            if phase == "detect":
                stripped = buffer.lstrip()
                if stripped.startswith("<think>"):
                    phase = "in_think"
                elif stripped.startswith("<|channel|>"):
                    phase = "in_analysis"
                elif len(stripped) >= self._DETECT_WINDOW:
                    phase = "passthrough"
                    pending = buffer
                    buffer = ""
                    think_idx = pending.find("<think>")
                    if think_idx != -1:
                        if think_idx > 0:
                            yield pending[:think_idx]
                        buffer = pending[think_idx:]
                        pending = ""
                        phase = "in_think"
                    else:
                        safe = len(pending) - self._THINK_TAG_LEN + 1
                        if safe > 0:
                            yield pending[:safe]
                            pending = pending[safe:]
                continue

            if phase == "in_think":
                end_idx = buffer.find("</think>")
                if end_idx != -1:
                    phase = "passthrough"
                    think_start = buffer.find("<think>") + len("<think>")
                    self._last_think_content = buffer[think_start:end_idx].strip()
                    remaining = buffer[end_idx + len("</think>"):]
                    remaining = remaining.lstrip("\n ")
                    buffer = ""
                    pending = ""
                    if remaining:
                        yield remaining

            elif phase == "in_analysis":
                final_marker = "<|channel|>final<|message|>"
                marker_idx = buffer.find(final_marker)
                if marker_idx != -1:
                    phase = "passthrough"
                    analysis_start = "<|channel|>analysis<|message|>"
                    a_idx = buffer.find(analysis_start)
                    if a_idx != -1:
                        a_content_start = a_idx + len(analysis_start)
                        a_end = buffer.find("<|end|>", a_content_start)
                        if a_end != -1 and a_end < marker_idx:
                            self._last_think_content = buffer[a_content_start:a_end].strip()
                    remaining = buffer[marker_idx + len(final_marker):]
                    remaining = remaining.lstrip("\n ")
                    buffer = ""
                    pending = ""
                    if remaining:
                        yield remaining

        if pending:
            yield pending
        if buffer and phase not in ("in_think", "in_analysis"):
            yield buffer

    # ------------------------------------------------------------------
    # チャット実行
    # ------------------------------------------------------------------

    def chat(self, user_input: str) -> str:
        """ユーザー入力を受け取り、応答を返す（非ストリーミング）。"""
        if not self.is_ready:
            raise RuntimeError("モデルまたはキャラクターが設定されていません。")

        messages = self._build_messages(user_input)
        params = self._current_character.generation_params
        response_tokens = self._estimate_max_tokens(user_input)
        think_budget = self._THINK_TOKEN_BUDGET if self.model_loader.supports_thinking else 0
        max_tokens = response_tokens + think_budget

        logger.debug(
            "max_tokens: %d (response=%d + think=%d, input length=%d)",
            max_tokens, response_tokens, think_budget, len(user_input),
        )

        raw_response = self.model_loader.generate(
            messages=messages,
            temperature=params.temperature,
            top_p=params.top_p,
            max_tokens=max_tokens,
            repeat_penalty=params.repeat_penalty,
            stream=False,
        )

        self._last_think_content = self.extract_think(raw_response)
        response = self.strip_think(raw_response)

        # 履歴に追加（<think> 除去済みテキスト + 思考内容）
        self.add_message("user", user_input)
        self.add_message("assistant", response, think_content=self._last_think_content)

        return response

    def chat_stream(self, user_input: str) -> Iterator[str]:
        """ストリーミングチャット。トークンごとにyieldする。

        <think>...</think> ブロックは自動的にフィルタリングされる。
        呼び出し側で全トークンを結合した後、finalize_stream()を呼ぶこと。
        """
        if not self.is_ready:
            raise RuntimeError("モデルまたはキャラクターが設定されていません。")

        messages = self._build_messages(user_input)
        params = self._current_character.generation_params
        response_tokens = self._estimate_max_tokens(user_input)
        think_budget = self._THINK_TOKEN_BUDGET if self.model_loader.supports_thinking else 0
        max_tokens = response_tokens + think_budget

        logger.debug(
            "max_tokens: %d (response=%d + think=%d, input length=%d)",
            max_tokens, response_tokens, think_budget, len(user_input),
        )

        # ユーザーメッセージを先に履歴追加
        self._last_think_content = ""
        self.add_message("user", user_input)

        raw_stream = self.model_loader.generate(
            messages=messages,
            temperature=params.temperature,
            top_p=params.top_p,
            max_tokens=max_tokens,
            repeat_penalty=params.repeat_penalty,
            stream=True,
        )

        return self._filter_think_stream(raw_stream)

    def finalize_stream(self, full_response: str) -> None:
        """ストリーミング完了後にアシスタント応答を履歴に追加する。"""
        clean = self.strip_think(full_response)
        think = self._last_think_content or self.extract_think(full_response)
        self._last_think_content = think
        self.add_message("assistant", clean, think_content=think)

    # ------------------------------------------------------------------
    # シリアライズ（DB保存用）
    # ------------------------------------------------------------------

    def get_history_dicts(self) -> list[dict]:
        """履歴を辞書リストで返す（DB保存用）。"""
        return [
            {
                "role": m.role,
                "content": m.content,
                "character_id": m.character_id,
                "think_content": m.think_content,
            }
            for m in self._history
        ]

    def load_history(self, messages: list[dict]) -> None:
        """辞書リストから履歴を復元する。"""
        self._history.clear()
        for m in messages:
            self._history.append(
                ChatMessage(
                    role=m["role"],
                    content=m["content"],
                    character_id=m.get("character_id", ""),
                    think_content=m.get("think_content", ""),
                )
            )
