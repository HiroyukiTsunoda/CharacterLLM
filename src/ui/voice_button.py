"""
VoiceButton: ホールド操作で録音し、WhisperWorker で文字起こしを実行するボタン。
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import QPushButton

from src.core.voice_input import VoiceEngine
from src.ui.styles import COLORS

logger = logging.getLogger(__name__)


class WhisperWorker(QThread):
    """別スレッドで音声認識を実行する。"""

    transcription_done = Signal(str)
    error_signal = Signal(str)

    def __init__(self, voice_engine: VoiceEngine, audio: np.ndarray, parent=None):
        super().__init__(parent)
        self.voice_engine = voice_engine
        self.audio = audio

    def run(self):
        try:
            text = self.voice_engine.transcribe(self.audio)
            if text:
                self.transcription_done.emit(text)
            else:
                self.error_signal.emit("音声を認識できませんでした")
        except Exception as e:
            logger.error("Whisper transcription error: %s", e)
            self.error_signal.emit(str(e))


class VoiceButton(QPushButton):
    """ホールド操作で録音→文字起こしを行うマイクボタン。

    - mousePressEvent: 録音開始
    - mouseReleaseEvent: 録音停止 → WhisperWorker で認識
    """

    transcription_done = Signal(str)
    error_signal = Signal(str)
    state_changed = Signal(str)  # "idle" / "recording" / "transcribing"

    _STYLE_IDLE = f"""
        QPushButton {{
            background-color: {COLORS['bg_light']};
            border: 1px solid {COLORS['border']};
            border-radius: 6px;
            color: {COLORS['text_primary']};
            font-size: 18px;
            padding: 4px;
        }}
        QPushButton:hover {{
            background-color: {COLORS['bg_lighter']};
            border-color: {COLORS['accent']};
        }}
    """

    _STYLE_RECORDING = f"""
        QPushButton {{
            background-color: {COLORS['error']};
            border: 1px solid {COLORS['error']};
            border-radius: 6px;
            color: white;
            font-size: 18px;
            padding: 4px;
        }}
    """

    _STYLE_TRANSCRIBING = f"""
        QPushButton {{
            background-color: {COLORS['warning']};
            border: 1px solid {COLORS['warning']};
            border-radius: 6px;
            color: white;
            font-size: 18px;
            padding: 4px;
        }}
        QPushButton:disabled {{
            background-color: {COLORS['warning']};
            color: white;
        }}
    """

    def __init__(self, voice_engine: VoiceEngine, parent=None):
        super().__init__(parent)
        self._voice_engine = voice_engine
        self._worker: Optional[WhisperWorker] = None
        self._state = "idle"

        self.setText("\U0001f3a4")
        self.setToolTip("長押しで音声入力")
        self.setMinimumHeight(44)
        self.setMinimumWidth(44)
        self.setMaximumWidth(44)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._apply_style()

    # ------------------------------------------------------------------
    # 状態管理
    # ------------------------------------------------------------------

    def _set_state(self, state: str):
        self._state = state
        self._apply_style()
        self.state_changed.emit(state)

    def _apply_style(self):
        if self._state == "recording":
            self.setStyleSheet(self._STYLE_RECORDING)
            self.setText("\u23f9")
            self.setToolTip("録音中... 離すと認識開始")
        elif self._state == "transcribing":
            self.setStyleSheet(self._STYLE_TRANSCRIBING)
            self.setText("\u23f3")
            self.setToolTip("認識中...")
            self.setEnabled(False)
        else:
            self.setStyleSheet(self._STYLE_IDLE)
            self.setText("\U0001f3a4")
            self.setToolTip("長押しで音声入力")
            self.setEnabled(True)

    # ------------------------------------------------------------------
    # マウスイベント（ホールド操作）
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._state == "idle":
            self._start_recording()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._state == "recording":
            self._stop_recording()
        super().mouseReleaseEvent(event)

    # ------------------------------------------------------------------
    # 録音 → 認識
    # ------------------------------------------------------------------

    def _start_recording(self):
        if not self._voice_engine.is_model_loaded:
            self.error_signal.emit("Whisperモデルが未ロードです。音声パネルからモデルをロードしてください。")
            return
        try:
            self._voice_engine.start_recording()
            self._set_state("recording")
        except Exception as e:
            logger.error("Failed to start recording: %s", e)
            self.error_signal.emit(f"録音開始エラー: {e}")

    def _stop_recording(self):
        try:
            audio = self._voice_engine.stop_recording()
        except Exception as e:
            logger.error("Failed to stop recording: %s", e)
            self.error_signal.emit(f"録音停止エラー: {e}")
            self._set_state("idle")
            return

        if audio is None:
            self._set_state("idle")
            return

        self._set_state("transcribing")
        self._worker = WhisperWorker(self._voice_engine, audio, self)
        self._worker.transcription_done.connect(self._on_transcription_done)
        self._worker.error_signal.connect(self._on_transcription_error)
        self._worker.start()

    def _on_transcription_done(self, text: str):
        self._set_state("idle")
        self.transcription_done.emit(text)

    def _on_transcription_error(self, error: str):
        self._set_state("idle")
        self.error_signal.emit(error)
