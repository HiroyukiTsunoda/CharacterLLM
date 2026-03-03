"""
ChatWidget: 吹き出しスタイルのチャット表示、ストリーミング対応、メッセージ入力欄。
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import Qt, Signal, QThread, QTimer
from PySide6.QtGui import QFont, QKeyEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.core.chat_engine import ChatEngine
from src.core.tts_base import TTSEngineBase
from src.ui.styles import CHAT_BUBBLE_ASSISTANT, CHAT_BUBBLE_USER, COLORS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 推論ワーカースレッド
# ---------------------------------------------------------------------------

class InferenceWorker(QThread):
    """別スレッドでストリーミング推論を実行する。

    ストリームから THINK_STREAM_START / END マーカーを検出し、
    思考トークンと応答トークンを別シグナルで通知する。
    """

    token_received = Signal(str)
    think_token_received = Signal(str)
    finished_signal = Signal(str)
    error_signal = Signal(str)

    def __init__(self, chat_engine: ChatEngine, user_input: str, parent=None):
        super().__init__(parent)
        self.chat_engine = chat_engine
        self.user_input = user_input
        self._full_response = ""
        self._think_content = ""

    def run(self):
        try:
            stream = self.chat_engine.chat_stream(self.user_input)
            in_think = False
            for token in stream:
                if token == ChatEngine.THINK_STREAM_START:
                    in_think = True
                    continue
                if token == ChatEngine.THINK_STREAM_END:
                    in_think = False
                    continue
                if in_think:
                    self._think_content += token
                    self.think_token_received.emit(token)
                else:
                    self._full_response += token
                    self.token_received.emit(token)
            self.chat_engine.finalize_stream(self._full_response)

            # LLM (llama.cpp) の CUDA 操作を完全に完了させてから
            # finished_signal を発信する。後続の TTS が GPU を安全に使えるようにする。
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass

            self.finished_signal.emit(self._full_response)
        except Exception as e:
            logger.error("Inference error: %s", e)
            self.error_signal.emit(str(e))


# ---------------------------------------------------------------------------
# TTS ワーカースレッド
# ---------------------------------------------------------------------------

class TTSWorker(QThread):
    """別スレッドで音声合成と再生を実行する。"""

    finished_signal = Signal()
    error_signal = Signal(str)

    def __init__(
        self,
        tts_engine: TTSEngineBase,
        text: str,
        character,
        output_device: int | None,
        parent=None,
    ):
        super().__init__(parent)
        self._tts_engine = tts_engine
        self._text = text
        self._character = character
        self._output_device = output_device

    def run(self):
        try:
            sr, audio = self._tts_engine.synthesize(self._text, self._character)
            self._tts_engine.play_audio(sr, audio, self._output_device)
            self.finished_signal.emit()
        except Exception as e:
            logger.error("TTS playback error: %s", e)
            self.error_signal.emit(str(e))


# ---------------------------------------------------------------------------
# チャットバブル
# ---------------------------------------------------------------------------

class ChatBubble(QFrame):
    """1つのメッセージを表す吹き出しウィジェット。"""

    def __init__(self, text: str, is_user: bool, character_name: str = "", parent=None):
        super().__init__(parent)
        self.is_user = is_user
        self._think_expanded = False
        self._setup_ui(text, character_name)

    def _setup_ui(self, text: str, character_name: str):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(2)

        # 名前ラベル
        if not self.is_user and character_name:
            name_label = QLabel(character_name)
            name_label.setStyleSheet(
                f"color: {COLORS['accent']}; font-size: 12px; font-weight: bold; background: transparent;"
            )
            name_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
            layout.addWidget(name_label)
        elif self.is_user:
            name_label = QLabel("あなた")
            name_label.setStyleSheet(
                f"color: {COLORS['text_secondary']}; font-size: 12px; background: transparent;"
            )
            name_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            layout.addWidget(name_label)

        # バブル本体
        bubble_container = QHBoxLayout()
        bubble_container.setContentsMargins(0, 0, 0, 0)

        # バブルフレーム（背景色・角丸を担当）
        self.bubble_frame = QFrame()
        self.bubble_frame.setObjectName("bubbleContent")
        self.bubble_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        bubble_layout = QVBoxLayout(self.bubble_frame)
        bubble_layout.setContentsMargins(14, 10, 14, 10)
        bubble_layout.setSpacing(6)

        # 思考セクション（アシスタントのみ）
        if not self.is_user:
            self.think_header = QLabel("▶ 思考")
            self.think_header.setCursor(Qt.CursorShape.PointingHandCursor)
            self.think_header.setVisible(False)
            self.think_header.setStyleSheet(f"""
                color: {COLORS['text_muted']};
                font-size: 12px;
                background: transparent;
                padding: 0px;
            """)
            self.think_header.mousePressEvent = lambda _e: self._toggle_think()
            bubble_layout.addWidget(self.think_header)

            self.think_label = QLabel("")
            self.think_label.setWordWrap(True)
            self.think_label.setVisible(False)
            self.think_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self.think_label.setStyleSheet(f"""
                color: {COLORS['text_secondary']};
                font-size: 13px;
                background-color: rgba(26, 27, 46, 150);
                border-radius: 6px;
                padding: 8px;
            """)
            bubble_layout.addWidget(self.think_label)

        # 応答テキスト
        self.text_label = QLabel(text)
        self.text_label.setWordWrap(True)
        self.text_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.text_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.text_label.setStyleSheet(f"""
            color: {'white' if self.is_user else COLORS['text_primary']};
            font-size: 14px;
            background: transparent;
            padding: 0px;
        """)
        bubble_layout.addWidget(self.text_label)

        # バブルフレームのスタイル
        if self.is_user:
            self.bubble_frame.setStyleSheet(f"""
                QFrame#bubbleContent {{
                    background-color: {COLORS['user_bubble']};
                    border-radius: 12px;
                }}
            """)
            bubble_container.addStretch()
            bubble_container.addWidget(self.bubble_frame)
        else:
            self.bubble_frame.setStyleSheet(f"""
                QFrame#bubbleContent {{
                    background-color: {COLORS['assistant_bubble']};
                    border-radius: 12px;
                }}
            """)
            bubble_container.addWidget(self.bubble_frame)
            bubble_container.addStretch()

        layout.addLayout(bubble_container)
        self.setStyleSheet("QFrame { background: transparent; border: none; }")

    # ------------------------------------------------------------------
    # 思考セクションの展開/折りたたみ
    # ------------------------------------------------------------------

    def _toggle_think(self):
        """思考セクションの表示/非表示を切り替える。"""
        if self.is_user or not hasattr(self, "think_label"):
            return
        self._think_expanded = not self._think_expanded
        self.think_label.setVisible(self._think_expanded)
        self.think_header.setText("▼ 思考" if self._think_expanded else "▶ 思考")

    def set_think_content(self, text: str):
        """思考内容をセットし、トグルヘッダーを表示する。"""
        if self.is_user or not hasattr(self, "think_header") or not text:
            return
        self.think_header.setVisible(True)
        self.think_label.setText(text)

    def append_think_text(self, text: str):
        """ストリーミング中に思考テキストを追加する。

        初回呼び出し時にヘッダーを表示する（折りたたみ状態）。
        """
        if self.is_user or not hasattr(self, "think_label"):
            return
        if not self.think_header.isVisible():
            self.think_header.setVisible(True)
            self._think_expanded = False
            self.think_label.setVisible(False)
            self.think_header.setText("▶ 思考中...")
        current = self.think_label.text()
        self.think_label.setText(current + text)

    def finish_think_stream(self):
        """思考ストリーミング完了。ヘッダーテキストを通常表示に戻す。"""
        if self.is_user or not hasattr(self, "think_header"):
            return
        if self._think_expanded:
            self.think_header.setText("▼ 思考")
        else:
            self.think_header.setText("▶ 思考")

    # ------------------------------------------------------------------
    # テキスト操作
    # ------------------------------------------------------------------

    def append_text(self, text: str):
        """ストリーミング中にテキストを追加する。"""
        current = self.text_label.text()
        self.text_label.setText(current + text)

    def set_text(self, text: str):
        """テキストを置換する。"""
        self.text_label.setText(text)


# ---------------------------------------------------------------------------
# メッセージ入力欄
# ---------------------------------------------------------------------------

class MessageInput(QPlainTextEdit):
    """Shift+Enterで改行、Enterで送信するテキスト入力欄。"""

    submit_signal = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPlaceholderText("メッセージを入力... (Enter で送信、Shift+Enter で改行)")
        self.setMaximumHeight(120)
        self.setMinimumHeight(44)
        self._composing = False

    def inputMethodEvent(self, event):
        self._composing = bool(event.preeditString())
        super().inputMethodEvent(event)

    def inputMethodQuery(self, query):
        if query == Qt.InputMethodQuery.ImCursorRectangle:
            rect = self.cursorRect()
            return rect.translated(0, self.fontMetrics().height() + 2)
        return super().inputMethodQuery(query)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self._composing:
                super().keyPressEvent(event)
                return
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                super().keyPressEvent(event)
            else:
                text = self.toPlainText().strip()
                if text:
                    self.submit_signal.emit(text)
                    self.clear()
        else:
            super().keyPressEvent(event)


# ---------------------------------------------------------------------------
# ChatWidget 本体
# ---------------------------------------------------------------------------

class ChatWidget(QWidget):
    """チャット画面全体（メッセージ一覧 + 入力欄）。"""

    BUBBLE_WIDTH_RATIO = 0.75

    def __init__(
        self,
        chat_engine: ChatEngine,
        voice_engine=None,
        tts_engine: TTSEngineBase | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.chat_engine = chat_engine
        self._voice_engine = voice_engine
        self._tts_engine = tts_engine
        self._auto_send = True
        self._worker: Optional[InferenceWorker] = None
        self._tts_worker: Optional[TTSWorker] = None
        self._current_bubble: Optional[ChatBubble] = None
        self._bubbles: list[ChatBubble] = []
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ---- メッセージ表示エリア ----
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet(f"QScrollArea {{ border: none; background-color: {COLORS['bg_dark']}; }}")

        self.messages_container = QWidget()
        self.messages_layout = QVBoxLayout(self.messages_container)
        self.messages_layout.setContentsMargins(16, 16, 16, 16)
        self.messages_layout.setSpacing(8)
        self.messages_layout.addStretch()  # メッセージを下に寄せる

        self.scroll_area.setWidget(self.messages_container)
        main_layout.addWidget(self.scroll_area, stretch=1)

        # ---- 音声自動送信トグル ----
        self.voice_bar = QWidget()
        voice_bar_layout = QHBoxLayout(self.voice_bar)
        voice_bar_layout.setContentsMargins(16, 4, 16, 0)
        voice_bar_layout.setSpacing(6)
        voice_bar_layout.addStretch()

        self.auto_send_check = QCheckBox("音声入力を自動送信")
        self.auto_send_check.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 12px;"
        )
        self.auto_send_check.setChecked(self._auto_send)
        self.auto_send_check.toggled.connect(self._on_auto_send_toggled)
        voice_bar_layout.addWidget(self.auto_send_check)

        self.voice_bar.setVisible(self._voice_engine is not None)
        main_layout.addWidget(self.voice_bar)

        # ---- 入力エリア ----
        input_frame = QFrame()
        input_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_medium']};
                border-top: 1px solid {COLORS['border']};
            }}
        """)
        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(16, 8, 16, 8)
        input_layout.setSpacing(8)

        self.message_input = MessageInput()
        self.message_input.submit_signal.connect(self.send_message)
        input_layout.addWidget(self.message_input, stretch=1)

        # マイクボタン（VoiceEngine がある場合のみ）
        self.voice_button = None
        if self._voice_engine is not None:
            from src.ui.voice_button import VoiceButton

            self.voice_button = VoiceButton(self._voice_engine)
            self.voice_button.transcription_done.connect(self._on_transcription_done)
            self.voice_button.error_signal.connect(
                lambda err: self._show_status(f"音声認識エラー: {err}", error=True)
            )
            self.voice_button.state_changed.connect(self._on_voice_state_changed)
            input_layout.addWidget(self.voice_button)

        self.send_button = QPushButton("送信")
        self.send_button.setMinimumHeight(44)
        self.send_button.setMinimumWidth(80)
        self.send_button.clicked.connect(self._on_send_clicked)
        input_layout.addWidget(self.send_button)

        main_layout.addWidget(input_frame)

        # ---- ステータス表示 ----
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px; padding: 2px 16px;")
        self.status_label.setVisible(False)
        main_layout.addWidget(self.status_label)

    # ------------------------------------------------------------------
    # メッセージ送信
    # ------------------------------------------------------------------

    def _on_send_clicked(self):
        text = self.message_input.toPlainText().strip()
        if text:
            self.message_input.clear()
            self.send_message(text)

    def send_message(self, text: str):
        """ユーザーメッセージを送信し、ストリーミング推論を開始する。"""
        if not self.chat_engine.is_ready:
            self._show_status("モデルとキャラクターを選択してください。", error=True)
            return

        if self._worker is not None and self._worker.isRunning():
            return  # 推論中は重複実行しない

        # ユーザーバブル追加
        self._add_bubble(text, is_user=True)

        # アシスタントバブル（空）を先に追加
        char_name = ""
        if self.chat_engine.current_character:
            char_name = self.chat_engine.current_character.name
        self._current_bubble = self._add_bubble("", is_user=False, character_name=char_name)

        # UI状態を「生成中」に
        self._set_generating(True)

        # ワーカースレッド開始
        self._worker = InferenceWorker(self.chat_engine, text, self)
        self._worker.token_received.connect(self._on_token)
        self._worker.think_token_received.connect(self._on_think_token)
        self._worker.finished_signal.connect(self._on_generation_done)
        self._worker.error_signal.connect(self._on_generation_error)
        self._worker.start()

    def _on_token(self, token: str):
        """応答トークン受信時: バブルにテキスト追加。"""
        if self._current_bubble:
            self._current_bubble.append_text(token)
        self._scroll_to_bottom()

    def _on_think_token(self, token: str):
        """思考トークン受信時: バブルの思考セクションにテキスト追加。"""
        if self._current_bubble:
            self._current_bubble.append_think_text(token)
        self._scroll_to_bottom()

    def _on_generation_done(self, full_response: str):
        """生成完了。最終テキストを確定し思考セクションのヘッダーを更新する。"""
        self._set_generating(False)
        clean = ""
        if self._current_bubble:
            clean = ChatEngine.strip_think(full_response)
            self._current_bubble.set_text(clean)

            if self._worker and self._worker._think_content:
                self._current_bubble.set_think_content(
                    self._worker._think_content,
                )
            self._current_bubble.finish_think_stream()
        self._current_bubble = None
        self._scroll_to_bottom()

        self._try_tts_playback(clean)

    def _on_generation_error(self, error: str):
        """生成エラー。"""
        self._set_generating(False)
        if self._current_bubble:
            self._current_bubble.set_text(f"エラー: {error}")
        self._current_bubble = None
        self._show_status(f"推論エラー: {error}", error=True)

    # ------------------------------------------------------------------
    # 音声入力
    # ------------------------------------------------------------------

    def _on_transcription_done(self, text: str):
        """音声認識完了。auto_send に応じて送信 or 挿入。"""
        if self._auto_send:
            self.send_message(text)
        else:
            self.message_input.setPlainText(text)
            self.message_input.setFocus()

    def _on_auto_send_toggled(self, checked: bool):
        self._auto_send = checked

    def _on_voice_state_changed(self, state: str):
        """録音/認識状態に応じてステータスを表示。"""
        if state == "recording":
            self._show_status("録音中...")
        elif state == "transcribing":
            self._show_status("音声認識中...")
        else:
            self.status_label.setVisible(False)

    def set_auto_send(self, enabled: bool):
        """外部から auto_send を設定する。"""
        self._auto_send = enabled
        self.auto_send_check.setChecked(enabled)

    # ------------------------------------------------------------------
    # TTS 自動再生
    # ------------------------------------------------------------------

    def _try_tts_playback(self, text: str):
        """条件を満たせば応答テキストを TTS で再生する。"""
        if not text or not self._tts_engine:
            logger.debug("TTS auto-play skipped: no text or no engine")
            return
        if not self._tts_engine.enabled:
            logger.info("TTS auto-play skipped: global TTS disabled")
            return
        if not self._tts_engine.auto_play:
            logger.info("TTS auto-play skipped: auto_play is off")
            return

        character = self.chat_engine.current_character
        if character is None:
            logger.info("TTS auto-play skipped: no character selected")
            return
        if not character.tts_params.enabled:
            logger.info("TTS auto-play skipped: character '%s' TTS disabled", character.name)
            return

        if self._tts_worker is not None and self._tts_worker.isRunning():
            self._tts_engine.stop_playback()

        output_device = None
        if self._voice_engine:
            output_device = self._voice_engine.output_device

        self._show_status("音声を再生中...")
        self._tts_worker = TTSWorker(
            self._tts_engine, text, character, output_device, self,
        )
        self._tts_worker.finished_signal.connect(self._on_tts_finished)
        self._tts_worker.error_signal.connect(self._on_tts_error)
        self._tts_worker.start()

    def _on_tts_finished(self):
        self.status_label.setVisible(False)

    def _on_tts_error(self, error: str):
        self._show_status(f"TTS エラー: {error}", error=True)

    def stop_tts(self):
        """外部から TTS 再生を停止する。"""
        if self._tts_engine:
            self._tts_engine.stop_playback()

    # ------------------------------------------------------------------
    # ヘルパー
    # ------------------------------------------------------------------

    def _add_bubble(self, text: str, is_user: bool, character_name: str = "") -> ChatBubble:
        """チャットバブルを追加する。"""
        bubble = ChatBubble(text, is_user, character_name, self.messages_container)
        bubble.bubble_frame.setMaximumWidth(self._bubble_max_width())
        self._bubbles.append(bubble)
        count = self.messages_layout.count()
        self.messages_layout.insertWidget(count - 1, bubble)
        self._scroll_to_bottom()
        return bubble

    def _bubble_max_width(self) -> int:
        viewport_w = self.scroll_area.viewport().width()
        return max(400, int(viewport_w * self.BUBBLE_WIDTH_RATIO))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        max_w = self._bubble_max_width()
        for b in self._bubbles:
            b.bubble_frame.setMaximumWidth(max_w)

    def _scroll_to_bottom(self):
        """スクロールを一番下に。"""
        QTimer.singleShot(10, lambda: self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        ))

    def _set_generating(self, generating: bool):
        """生成中のUI状態切替。"""
        self.send_button.setEnabled(not generating)
        self.message_input.setEnabled(not generating)
        if generating:
            self.send_button.setText("生成中...")
        else:
            self.send_button.setText("送信")
            self.message_input.setFocus()

    def _show_status(self, text: str, error: bool = False):
        """ステータスラベルを表示。"""
        color = COLORS['error'] if error else COLORS['text_muted']
        self.status_label.setStyleSheet(f"color: {color}; font-size: 12px; padding: 2px 16px;")
        self.status_label.setText(text)
        self.status_label.setVisible(True)
        QTimer.singleShot(5000, lambda: self.status_label.setVisible(False))

    def clear_chat(self):
        """チャット画面をクリア。"""
        while self.messages_layout.count() > 1:
            item = self.messages_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self._bubbles.clear()
        self.chat_engine.clear_history()

    def load_history_to_ui(self):
        """chat_engineの履歴をUIに反映する。"""
        char_name = ""
        if self.chat_engine.current_character:
            char_name = self.chat_engine.current_character.name
        for msg in self.chat_engine.history:
            if msg.role == "user":
                self._add_bubble(msg.content, is_user=True)
            elif msg.role == "assistant":
                bubble = self._add_bubble(msg.content, is_user=False, character_name=char_name)
                if msg.think_content:
                    bubble.set_think_content(msg.think_content)
