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
raw_logger = logging.getLogger("llm_raw")


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
        "\n\n応答する前に、必ず<think>タグの中で考えをまとめてから応答してください。"
        "<think>タグの外には応答だけを書いてください。"
    )

    # 動的 max_tokens に応じた応答長ガイダンス（システムプロンプト末尾に付加）
    _LENGTH_GUIDANCE = {
        128: "\n\n【応答の長さ】1〜2文の短い返答にしてください。挨拶には挨拶で返す程度で十分です。",
        256: "\n\n【応答の長さ】簡潔に応答してください。3〜5文程度を目安にしてください。",
        512: "\n\n【応答の長さ】質問や相談には丁寧に答えつつ、冗長にならないよう心がけてください。",
        768: "",
    }

    def _build_messages(self, user_input: str, response_tokens: int = 256) -> list[dict]:
        """LLMに送信するメッセージリストを組み立てる。"""
        messages: list[dict] = []
        template_think = self.model_loader.template_inserts_think

        # 1. システムプロンプト（キャラクター + 応答長ガイダンス + <think>タグ指示）
        if self._current_character:
            sys_msg = self._current_character.get_system_message()
            guidance = self._LENGTH_GUIDANCE.get(response_tokens, "")
            if guidance:
                sys_msg["content"] += guidance
            if self.model_loader.uses_think_tags and not template_think:
                sys_msg["content"] += self._THINK_INSTRUCTION
            messages.append(sys_msg)

        # 2. 過去の会話履歴
        for msg in self._history:
            d = msg.to_dict()
            if template_think and d["role"] == "assistant":
                # テンプレートがアシスタント応答に <think> を挿入するため、
                # 履歴の応答には </think> を補い整合性を保つ。
                # 思考内容があれば含める（モデルが自然な応答を継続できるよう）。
                think = msg.think_content
                if think:
                    d["content"] = think + "\n</think>\n" + d["content"]
                else:
                    d["content"] = "</think>\n" + d["content"]
            messages.append(d)

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
    # Nemotron等は思考が非常に長くなるため十分な余裕を持たせる
    _THINK_TOKEN_BUDGET = 2048

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

    # Qwen3 テンプレート由来: </think>思考内容</think> （開始タグなし）
    _BARE_CLOSE_THINK_RE = re.compile(
        r"^\s*</think>\s*(.*?)</think>\s*", flags=re.DOTALL,
    )

    # gpt-oss: <|channel|>analysis<|message|>...<|end|> ... <|channel|>final<|message|>
    _GPT_OSS_ANALYSIS_RE = re.compile(
        r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>",
        flags=re.DOTALL,
    )
    _GPT_OSS_FINAL_RE = re.compile(
        r".*<\|channel\|>final<\|message\|>",
        flags=re.DOTALL,
    )

    # 思考内容/応答内容 セクションマーカー（任意の前置文字に対応）
    # モデルが \dag, †, Hath, ✖️ 等の前置を付ける場合や前置なしの場合を全て検出
    _SECTION_RE = re.compile(
        r"[^\n]*思考内容\s*.*?[^\n]*応答内容[ \t]*\n?",
        flags=re.DOTALL,
    )
    _SECTION_THINK_EXTRACT_RE = re.compile(
        r"[^\n]*思考内容\s*(.*?)\s*[^\n]*応答内容",
        flags=re.DOTALL,
    )

    @staticmethod
    def strip_think(text: str) -> str:
        """応答テキストから思考ブロックを除去する。

        対応フォーマット:
          - <think>...</think>
          - </think>...</think> (テンプレート由来・開始タグなし)
          - テンプレート自動挿入 <think> (Nemotron/Qwen3: 生テキスト...</think>応答)
          - gpt-oss analysis channel
          - [任意前置]思考内容 ... [任意前置]応答内容
        """
        # 1. ペアになった <think>...</think> を除去
        result = ChatEngine._THINK_RE.sub("", text)

        # 2. ペア除去後に残った </think> はテンプレート自動挿入の <think> を
        #    閉じるもの (Nemotron/Qwen3)。最後の </think> 以降が応答本文。
        last_close = result.rfind("</think>")
        if last_close != -1:
            result = result[last_close + len("</think>"):]
            return result.lstrip("\n ")

        # 3. gpt-oss analysis channel
        result = ChatEngine._GPT_OSS_FINAL_RE.sub("", result)

        # 3.5. gpt-oss end-of-turn: <|end|> 以降を除去（繰り返し生成の防止）
        end_idx = result.find("<|end|>")
        if end_idx != -1:
            result = result[:end_idx]

        # 4. 思考内容/応答内容 セクションマーカー
        result = ChatEngine._SECTION_RE.sub("", result)

        return result.lstrip("\n ")

    @staticmethod
    def extract_think(text: str) -> str:
        """応答テキストから思考内容を抽出する。"""
        parts: list[str] = []

        # 1. <think>...</think> ペアの中身
        for m in ChatEngine._THINK_EXTRACT_RE.finditer(text):
            content = m.group(1).strip()
            if content:
                parts.append(content)

        # 2. ペア除去後に残った standalone </think> の前のテキスト
        #    (テンプレート自動挿入パターン: Nemotron/Qwen3)
        no_pairs = ChatEngine._THINK_RE.sub("", text)
        last_close = no_pairs.rfind("</think>")
        if last_close != -1:
            raw_think = no_pairs[:last_close].strip()
            raw_think = raw_think.replace("</think>", "").strip()
            if raw_think:
                parts.append(raw_think)

        if parts:
            return "\n".join(parts)

        # 3. gpt-oss analysis channel
        match = ChatEngine._GPT_OSS_ANALYSIS_RE.search(text)
        if match:
            return match.group(1).strip()

        # 4. 思考内容/応答内容 セクションマーカー
        match = ChatEngine._SECTION_THINK_EXTRACT_RE.search(text)
        if match:
            return match.group(1).strip()

        return ""

    def get_last_think_content(self) -> str:
        """直近の推論で取得した思考内容を返す。"""
        return self._last_think_content

    # ストリーム検出に必要な最小文字数（"<|channel|>" = 11 文字）
    _DETECT_WINDOW = 11
    # 思考ストリームマーカー（UI 側で検出して思考セクションに振り分ける）
    THINK_STREAM_START = "\x00\x01THINK\x02\x00"
    THINK_STREAM_END = "\x00\x01/THINK\x02\x00"

    # タグ部分一致を保持するマージン（</think> = 8 文字が最長）
    _STREAM_MARGIN = len("</think>")  # 8

    # 思考内容/応答内容 キーワード（前置文字不問で検出する）
    _THINK_KW = "思考内容"
    _RESPONSE_KW = "応答内容"

    @staticmethod
    def _find_section_kw(text: str, keyword: str) -> tuple[int, int]:
        """keyword を含む行の開始位置と keyword 末尾位置を返す。

        Returns (line_start, keyword_end) or (-1, -1) if not found.
        """
        kw_idx = text.find(keyword)
        if kw_idx == -1:
            return -1, -1
        line_start = text.rfind("\n", 0, kw_idx)
        line_start = line_start + 1 if line_start != -1 else 0
        return line_start, kw_idx + len(keyword)

    def _check_passthrough(self, text: str):
        """passthrough 中にテキスト内の思考開始マーカーを検査する。

        Returns (phase, buffer, pending, to_yield).
        """
        # gpt-oss end-of-turn: <|end|> 以降は不要な繰り返しなので打ち切る
        end_idx = text.find("<|end|>")
        if end_idx != -1:
            return "done", "", "", text[:end_idx] if end_idx > 0 else ""

        think_idx = text.find("<think>")
        sec_start, _ = self._find_section_kw(text, self._THINK_KW)

        idx, kind = -1, None
        if think_idx != -1:
            idx, kind = think_idx, "in_think"
        if sec_start != -1 and (idx == -1 or sec_start < idx):
            idx, kind = sec_start, "in_section_think"

        if kind is not None:
            return kind, text[idx:], "", text[:idx] if idx > 0 else ""

        to_yield = ""
        safe = len(text) - self._STREAM_MARGIN + 1
        if safe > 0:
            to_yield = text[:safe]
            text = text[safe:]
        return "passthrough", "", text, to_yield

    def _filter_think_stream(
        self, token_stream: Iterator[str]
    ) -> Iterator[str]:
        """ストリーミングトークンを処理し思考/応答を分離して yield する。

        思考ブロックは THINK_STREAM_START / THINK_STREAM_END マーカーで
        囲んで yield される。UI 側でマーカーを検出し思考セクションへ振り分ける。

        対応フォーマット:
          - Qwen3: <think>...</think>
          - テンプレート自動挿入: 生テキスト...<think>...</think>...</think>応答
          - gpt-oss: <|channel|>analysis ... <|channel|>final
          - [任意前置]思考内容 ... [任意前置]応答内容
        """
        buffer = ""
        pending = ""
        phase = "detect"
        self._last_think_content = ""
        prefix_stripped = False
        _raw_output_tokens: list[str] = []
        template_inserts_think = self.model_loader.template_inserts_think

        _CLOSE_THINK = "</think>"
        _CLOSE_THINK_LEN = len(_CLOSE_THINK)

        for token in token_stream:
            _raw_output_tokens.append(token)
            # --- passthrough: 応答トークンをそのまま流す ---
            if phase == "passthrough":
                pending += token
                new_phase, new_buf, new_pend, to_yield = (
                    self._check_passthrough(pending)
                )
                if new_phase == "done":
                    if to_yield:
                        yield to_yield
                    pending = ""
                    buffer = ""
                    break
                elif new_phase != "passthrough":
                    if to_yield:
                        yield to_yield
                    buffer = new_buf
                    pending = ""
                    phase = new_phase
                    prefix_stripped = False
                else:
                    if to_yield:
                        yield to_yield
                    pending = new_pend
                continue

            buffer += token

            # --- detect: ストリーム冒頭のフォーマット判定 ---
            if phase == "detect":
                stripped = buffer.lstrip()
                if stripped.startswith("<think>"):
                    phase = "in_think"
                    prefix_stripped = False
                elif stripped.startswith("</think>"):
                    # Qwen3: チャットテンプレートが <think> を自動挿入済み。
                    # ストリームは </think> で始まり、その後に思考内容、
                    # 再度 </think> で閉じられてから応答が続く。
                    close_end = buffer.find("</think>") + len("</think>")
                    buffer = buffer[close_end:]
                    phase = "in_think"
                    prefix_stripped = True
                    yield self.THINK_STREAM_START
                elif stripped.startswith("<|channel|>"):
                    phase = "in_analysis"
                elif self._THINK_KW in stripped:
                    phase = "in_section_think"
                    prefix_stripped = False
                elif len(stripped) >= self._DETECT_WINDOW:
                    if template_inserts_think:
                        # テンプレートが <think> を自動挿入しているが、
                        # モデル出力がタグなしで始まった (Nemotron パターン)。
                        # 生テキストは暗黙的な思考ブロック内。
                        phase = "in_implicit_think"
                        yield self.THINK_STREAM_START
                    else:
                        new_phase, new_buf, new_pend, to_yield = (
                            self._check_passthrough(buffer)
                        )
                        if to_yield:
                            yield to_yield
                        buffer = new_buf
                        pending = new_pend
                        phase = new_phase
                        if new_phase != "passthrough":
                            prefix_stripped = False
                continue

            # --- in_think: <think>...</think> ---
            if phase == "in_think":
                if not prefix_stripped:
                    idx = buffer.find("<think>")
                    if idx != -1:
                        buffer = buffer[idx + len("<think>"):]
                        prefix_stripped = True
                        yield self.THINK_STREAM_START

                if prefix_stripped:
                    end_idx = buffer.find(_CLOSE_THINK)
                    if end_idx != -1:
                        tail = buffer[:end_idx]
                        if tail:
                            self._last_think_content += tail
                            yield tail
                        yield self.THINK_STREAM_END

                        remaining = buffer[end_idx + _CLOSE_THINK_LEN:]
                        remaining = remaining.lstrip("\n ")
                        buffer = ""
                        pending = ""
                        phase = "passthrough"
                        prefix_stripped = False

                        if remaining:
                            sec_start, _ = self._find_section_kw(
                                remaining, self._THINK_KW,
                            )
                            if sec_start != -1:
                                if sec_start > 0:
                                    yield remaining[:sec_start]
                                buffer = remaining[sec_start:]
                                phase = "in_section_think"
                                prefix_stripped = False
                            else:
                                remaining = remaining.replace(
                                    _CLOSE_THINK, "",
                                )
                                if remaining:
                                    yield remaining
                    else:
                        safe = len(buffer) - _CLOSE_THINK_LEN + 1
                        if safe > 0:
                            chunk = buffer[:safe]
                            self._last_think_content += chunk
                            yield chunk
                            buffer = buffer[safe:]

            # --- in_section_think: [前置]思考内容...[前置]応答内容 ---
            elif phase == "in_section_think":
                if not prefix_stripped:
                    _, kw_end = self._find_section_kw(
                        buffer, self._THINK_KW,
                    )
                    if kw_end != -1:
                        buffer = buffer[kw_end:]
                        prefix_stripped = True
                        if self._last_think_content:
                            self._last_think_content += "\n"
                        yield self.THINK_STREAM_START

                if prefix_stripped:
                    resp_start, resp_end = self._find_section_kw(
                        buffer, self._RESPONSE_KW,
                    )
                    if resp_start != -1:
                        tail = buffer[:resp_start]
                        if tail:
                            self._last_think_content += tail
                            yield tail
                        yield self.THINK_STREAM_END

                        remaining = buffer[resp_end:]
                        remaining = remaining.lstrip("\n ")
                        remaining = remaining.replace(_CLOSE_THINK, "")
                        buffer = ""
                        pending = ""
                        phase = "passthrough"
                        prefix_stripped = False
                        if remaining:
                            yield remaining
                    else:
                        kw_len = len(self._RESPONSE_KW)
                        safe = len(buffer) - kw_len + 1
                        if safe > 0:
                            chunk = buffer[:safe]
                            self._last_think_content += chunk
                            yield chunk
                            buffer = buffer[safe:]

            # --- in_analysis: gpt-oss analysis channel ---
            elif phase == "in_analysis":
                final_marker = "<|channel|>final<|message|>"
                marker_idx = buffer.find(final_marker)
                if marker_idx != -1:
                    analysis_start = "<|channel|>analysis<|message|>"
                    a_idx = buffer.find(analysis_start)
                    analysis_text = ""
                    if a_idx != -1:
                        a_start = a_idx + len(analysis_start)
                        a_end = buffer.find("<|end|>", a_start)
                        if a_end != -1 and a_end < marker_idx:
                            analysis_text = buffer[a_start:a_end].strip()

                    if analysis_text:
                        self._last_think_content = analysis_text
                        yield self.THINK_STREAM_START
                        yield analysis_text
                        yield self.THINK_STREAM_END

                    remaining = buffer[marker_idx + len(final_marker):]
                    remaining = remaining.lstrip("\n ")
                    # gpt-oss end-of-turn: <|end|> 以降を除去
                    _eidx = remaining.find("<|end|>")
                    if _eidx != -1:
                        remaining = remaining[:_eidx]
                    buffer = ""
                    pending = ""
                    phase = "passthrough"
                    prefix_stripped = False
                    if remaining:
                        yield remaining

            # --- in_implicit_think: テンプレート自動挿入 <think> ---
            # <think> はモデル出力中に literal に出現しうるため depth
            # tracking は行わず、最初の </think> で思考終了とする。
            # 残り文字列に後続の </think> がある場合は strip_think が
            # finalize 時に正しく処理する。
            elif phase == "in_implicit_think":
                close_idx = buffer.find(_CLOSE_THINK)
                if close_idx != -1:
                    tail = buffer[:close_idx]
                    if tail:
                        self._last_think_content += tail
                        yield tail
                    yield self.THINK_STREAM_END

                    remaining = buffer[close_idx + _CLOSE_THINK_LEN:]
                    remaining = remaining.lstrip("\n ")
                    buffer = ""
                    pending = ""
                    phase = "passthrough"
                    prefix_stripped = False

                    if remaining:
                        yield remaining
                else:
                    safe = len(buffer) - _CLOSE_THINK_LEN + 1
                    if safe > 0:
                        chunk = buffer[:safe]
                        self._last_think_content += chunk
                        yield chunk
                        buffer = buffer[safe:]

        # --- ストリーム終端処理 ---
        # gpt-oss end-of-turn marker が残っていたら除去
        _end_marker = "<|end|>"
        _ei = pending.find(_end_marker)
        if _ei != -1:
            pending = pending[:_ei]
        _ei = buffer.find(_end_marker)
        if _ei != -1:
            buffer = buffer[:_ei]

        if pending:
            yield pending
        if buffer:
            in_think_active = (
                (phase in ("in_think", "in_section_think") and prefix_stripped)
                or phase == "in_implicit_think"
            )
            if in_think_active:
                if buffer.strip():
                    self._last_think_content += buffer
                    yield buffer
                yield self.THINK_STREAM_END
            elif phase not in ("in_think", "in_analysis", "in_section_think"):
                yield buffer

        raw_text = "".join(_raw_output_tokens)
        raw_logger.info("=== LLM Raw Output (stream) ===\n%s\n=== END ===", raw_text)

    # ------------------------------------------------------------------
    # チャット実行
    # ------------------------------------------------------------------

    def chat(self, user_input: str) -> str:
        """ユーザー入力を受け取り、応答を返す（非ストリーミング）。"""
        if not self.is_ready:
            raise RuntimeError("モデルまたはキャラクターが設定されていません。")

        params = self._current_character.generation_params
        response_tokens = self._estimate_max_tokens(user_input)
        messages = self._build_messages(user_input, response_tokens)
        think_budget = self._THINK_TOKEN_BUDGET if self.model_loader.supports_thinking else 0
        max_tokens = response_tokens + think_budget

        logger.debug(
            "max_tokens: %d (response=%d + think=%d, input length=%d)",
            max_tokens, response_tokens, think_budget, len(user_input),
        )

        raw_logger.info("=== User Input ===\n%s\n=== END ===", user_input)

        raw_response = self.model_loader.generate(
            messages=messages,
            temperature=params.temperature,
            top_p=params.top_p,
            max_tokens=max_tokens,
            repeat_penalty=params.repeat_penalty,
            stream=False,
        )

        raw_logger.info("=== LLM Raw Output (non-stream) ===\n%s\n=== END ===", raw_response)

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

        params = self._current_character.generation_params
        response_tokens = self._estimate_max_tokens(user_input)
        messages = self._build_messages(user_input, response_tokens)
        think_budget = self._THINK_TOKEN_BUDGET if self.model_loader.supports_thinking else 0
        max_tokens = response_tokens + think_budget

        logger.debug(
            "max_tokens: %d (response=%d + think=%d, input length=%d)",
            max_tokens, response_tokens, think_budget, len(user_input),
        )

        # ユーザーメッセージを先に履歴追加
        self._last_think_content = ""
        self.add_message("user", user_input)

        raw_logger.info("=== User Input ===\n%s\n=== END ===", user_input)

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
