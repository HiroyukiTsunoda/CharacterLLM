"""
VoicePanel: サイドバー「音声」タブ。入力/出力デバイス選択、Whisper 設定、
TTS 設定（Qwen3-TTS）を提供する。
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import Signal, QThread
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.core.voice_input import VoiceEngine
from src.core.tts_base import TTSEngineBase
from src.ui import install_wheel_guard
from src.ui.styles import COLORS

logger = logging.getLogger(__name__)

WHISPER_MODELS = [
    ("tiny", "tiny (~75MB, 低精度・高速)"),
    ("base", "base (~140MB)"),
    ("small", "small (~460MB)"),
    ("medium", "medium (~1.5GB)"),
    ("large-v3", "large-v3 (~3GB, 高精度)"),
    ("large-v3-turbo", "large-v3-turbo (~1.6GB, 高精度・高速, 推奨)"),
]

WHISPER_DEVICES = [
    ("cpu", "CPU"),
    ("cuda", "CUDA (GPU)"),
]

TTS_DEVICES = [
    ("cpu", "CPU"),
    ("cuda", "CUDA (GPU)"),
]


# ------------------------------------------------------------------
# バックグラウンドワーカー
# ------------------------------------------------------------------

class WhisperLoadWorker(QThread):
    finished_signal = Signal()
    error_signal = Signal(str)

    def __init__(self, voice_engine: VoiceEngine, parent=None):
        super().__init__(parent)
        self._voice_engine = voice_engine

    def run(self):
        try:
            self._voice_engine.load_model()
            self.finished_signal.emit()
        except Exception as e:
            logger.error("Whisper model load failed: %s", e)
            self.error_signal.emit(str(e))


class TTSLoadWorker(QThread):
    """キャラクターの TTS モデルをメモリにロードする。"""
    finished_signal = Signal()
    error_signal = Signal(str)

    def __init__(self, tts_engine: TTSEngineBase, character, parent=None):
        super().__init__(parent)
        self._tts_engine = tts_engine
        self._character = character

    def run(self):
        try:
            self._tts_engine.load_model_for_character(self._character)
            self.finished_signal.emit()
        except Exception as e:
            logger.error("TTS model load failed: %s", e)
            self.error_signal.emit(str(e))


class TTSTestWorker(QThread):
    finished_signal = Signal()
    error_signal = Signal(str)

    def __init__(self, tts_engine, text, character, output_device, parent=None):
        super().__init__(parent)
        self._tts_engine = tts_engine
        self._text = text
        self._character = character
        self._output_device = output_device

    def run(self):
        try:
            sr, audio = self._tts_engine.synthesize(self._text, self._character, force=True)
            self._tts_engine.play_audio(sr, audio, self._output_device)
            self.finished_signal.emit()
        except Exception as e:
            logger.error("TTS test failed: %s", e)
            self.error_signal.emit(str(e))


# ------------------------------------------------------------------
# VoicePanel
# ------------------------------------------------------------------

class VoicePanel(QWidget):
    """音声設定パネル（サイドバータブ）。"""

    input_device_changed = Signal(object)
    output_device_changed = Signal(object)
    whisper_settings_changed = Signal(dict)
    tts_config_changed = Signal()
    config_changed = Signal()
    vram_changed = Signal()

    def __init__(
        self,
        voice_engine: VoiceEngine,
        tts_engine: TTSEngineBase | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._voice_engine = voice_engine
        self._tts_engine = tts_engine
        self._load_worker: WhisperLoadWorker | None = None
        self._tts_load_worker: TTSLoadWorker | None = None
        self._tts_test_worker: TTSTestWorker | None = None
        self._current_character = None
        self._setup_ui()
        self._refresh_devices()
        self._load_from_engine()
        self._update_whisper_status()
        if self._tts_engine:
            self._load_tts_from_engine()
            self._update_tts_status()

    # ------------------------------------------------------------------
    # 外部 API
    # ------------------------------------------------------------------

    def set_current_character(self, character) -> None:
        self._current_character = character
        self._update_tts_character_display()

    # ------------------------------------------------------------------
    # UI 構築
    # ------------------------------------------------------------------

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        self._build_input_device_ui(layout)
        self._build_output_device_ui(layout)
        self._build_whisper_ui(layout)
        self._build_tts_global_ui(layout)
        self._build_tts_char_ui(layout)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addStretch()

        install_wheel_guard(
            self,
            self.input_combo, self.output_combo,
            self.model_combo, self.device_combo,
            self.tts_device_combo,
            self.qwen3_model_combo,
        )

        scroll.setWidget(container)
        outer.addWidget(scroll)

    # ---- 入力デバイス ----

    def _build_input_device_ui(self, layout):
        grp = QGroupBox("入力デバイス（マイク）")
        gl = QVBoxLayout()
        gl.setSpacing(6)
        self.input_combo = QComboBox()
        self.input_combo.currentIndexChanged.connect(self._on_input_device_changed)
        gl.addWidget(self.input_combo)
        bl = QHBoxLayout()
        btn = QPushButton("デバイス更新")
        btn.setProperty("secondary", True)
        btn.clicked.connect(self._refresh_devices)
        bl.addWidget(btn)
        bl.addStretch()
        gl.addLayout(bl)
        grp.setLayout(gl)
        layout.addWidget(grp)

    # ---- 出力デバイス ----

    def _build_output_device_ui(self, layout):
        grp = QGroupBox("出力デバイス（スピーカー）")
        gl = QVBoxLayout()
        gl.setSpacing(6)
        self.output_combo = QComboBox()
        self.output_combo.currentIndexChanged.connect(self._on_output_device_changed)
        gl.addWidget(self.output_combo)
        bl = QHBoxLayout()
        btn = QPushButton("デバイス更新")
        btn.setProperty("secondary", True)
        btn.clicked.connect(self._refresh_devices)
        bl.addWidget(btn)
        bl.addStretch()
        gl.addLayout(bl)
        grp.setLayout(gl)
        layout.addWidget(grp)

    # ---- Whisper ----

    def _build_whisper_ui(self, layout):
        grp = QGroupBox("音声認識 (Whisper)")
        fl = QFormLayout()
        fl.setSpacing(8)

        self.model_combo = QComboBox()
        for v, lb in WHISPER_MODELS:
            self.model_combo.addItem(lb, v)
        self.model_combo.currentIndexChanged.connect(self._on_whisper_settings_changed)
        fl.addRow("モデル:", self.model_combo)

        self.device_combo = QComboBox()
        for v, lb in WHISPER_DEVICES:
            self.device_combo.addItem(lb, v)
        self.device_combo.currentIndexChanged.connect(self._on_whisper_settings_changed)
        fl.addRow("デバイス:", self.device_combo)

        self.language_edit = QLineEdit()
        self.language_edit.setPlaceholderText("ja")
        self.language_edit.setMaximumWidth(80)
        self.language_edit.editingFinished.connect(self._on_whisper_settings_changed)
        fl.addRow("言語:", self.language_edit)

        bl = QHBoxLayout()
        self.whisper_load_btn = QPushButton("モデルロード")
        self.whisper_load_btn.clicked.connect(self._on_whisper_load)
        bl.addWidget(self.whisper_load_btn)
        self.whisper_unload_btn = QPushButton("アンロード")
        self.whisper_unload_btn.setProperty("secondary", True)
        self.whisper_unload_btn.clicked.connect(self._on_whisper_unload)
        self.whisper_unload_btn.setEnabled(False)
        bl.addWidget(self.whisper_unload_btn)
        fl.addRow(bl)

        self.whisper_progress = QProgressBar()
        self.whisper_progress.setRange(0, 0)
        self.whisper_progress.setVisible(False)
        fl.addRow(self.whisper_progress)

        self.whisper_status_label = QLabel("未ロード")
        self.whisper_status_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        self.whisper_status_label.setWordWrap(True)
        fl.addRow("状態:", self.whisper_status_label)

        grp.setLayout(fl)
        layout.addWidget(grp)

    # ---- TTS グローバル ----

    def _build_tts_global_ui(self, layout):
        grp = QGroupBox("音声合成 (Qwen3-TTS)")
        fl = QFormLayout()
        fl.setSpacing(8)

        self.tts_enabled_check = QCheckBox("TTS を有効にする")
        self.tts_enabled_check.toggled.connect(self._on_tts_enabled_changed)
        fl.addRow(self.tts_enabled_check)

        self.tts_auto_play_check = QCheckBox("応答を自動再生")
        self.tts_auto_play_check.toggled.connect(self._on_tts_auto_play_changed)
        fl.addRow(self.tts_auto_play_check)

        self.tts_device_combo = QComboBox()
        for v, lb in TTS_DEVICES:
            self.tts_device_combo.addItem(lb, v)
        self.tts_device_combo.currentIndexChanged.connect(self._on_tts_device_changed)
        fl.addRow("デバイス:", self.tts_device_combo)

        bl = QHBoxLayout()
        self.tts_unload_btn = QPushButton("全モデル解放")
        self.tts_unload_btn.setProperty("secondary", True)
        self.tts_unload_btn.clicked.connect(self._on_tts_unload_all)
        self.tts_unload_btn.setEnabled(False)
        bl.addWidget(self.tts_unload_btn)
        fl.addRow(bl)

        self.tts_status_label = QLabel("無効")
        self.tts_status_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        self.tts_status_label.setWordWrap(True)
        fl.addRow("状態:", self.tts_status_label)

        grp.setLayout(fl)
        layout.addWidget(grp)

    # ---- キャラクター別 TTS ----

    def _build_tts_char_ui(self, layout):
        grp = QGroupBox("キャラクター別 TTS 設定")
        fl = QFormLayout()
        fl.setSpacing(6)

        self.tts_char_label = QLabel("キャラクター未選択")
        self.tts_char_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        fl.addRow(self.tts_char_label)

        self.tts_char_enabled_check = QCheckBox("このキャラクターの TTS を有効にする")
        self.tts_char_enabled_check.toggled.connect(self._on_char_tts_toggled)
        fl.addRow(self.tts_char_enabled_check)

        # Qwen3-TTS パラメータ
        self.qwen3_model_combo = QComboBox()
        self.qwen3_model_combo.setToolTip("Qwen3-TTS モデルバリアントを選択")
        self.qwen3_model_combo.currentIndexChanged.connect(self._on_qwen3_model_selected)
        fl.addRow("モデル:", self.qwen3_model_combo)

        self.qwen3_mode_label = QLabel("—")
        self.qwen3_mode_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 11px;"
        )
        fl.addRow("モード:", self.qwen3_mode_label)

        self.qwen3_voice_edit = QLineEdit()
        self.qwen3_voice_edit.setPlaceholderText("vivian")
        self.qwen3_voice_edit.setToolTip("CustomVoice 用スピーカー名")
        self.qwen3_voice_edit.editingFinished.connect(self._on_qwen3_params_changed)
        fl.addRow("ボイス名:", self.qwen3_voice_edit)

        self.qwen3_instructions_edit = QTextEdit()
        self.qwen3_instructions_edit.setPlaceholderText("例: 優しい声でゆっくり話して")
        self.qwen3_instructions_edit.setToolTip("感情・スタイル指示テキスト")
        self.qwen3_instructions_edit.setMaximumHeight(60)
        self.qwen3_instructions_edit.textChanged.connect(self._on_qwen3_params_changed)
        fl.addRow("指示:", self.qwen3_instructions_edit)

        # 音声クローン用
        self.qwen3_ref_audio_label = QLabel("参照音声:")
        ref_bl = QHBoxLayout()
        self.qwen3_ref_audio_edit = QLineEdit()
        self.qwen3_ref_audio_edit.setPlaceholderText("参照音声ファイルパス (.wav)")
        self.qwen3_ref_audio_edit.editingFinished.connect(self._on_qwen3_params_changed)
        ref_bl.addWidget(self.qwen3_ref_audio_edit)
        self.qwen3_ref_audio_btn = QPushButton("...")
        self.qwen3_ref_audio_btn.setMaximumWidth(30)
        self.qwen3_ref_audio_btn.clicked.connect(self._on_qwen3_browse_ref_audio)
        ref_bl.addWidget(self.qwen3_ref_audio_btn)
        fl.addRow(self.qwen3_ref_audio_label, ref_bl)

        self.qwen3_ref_text_edit = QLineEdit()
        self.qwen3_ref_text_edit.setPlaceholderText("参照音声のテキスト")
        self.qwen3_ref_text_edit.editingFinished.connect(self._on_qwen3_params_changed)
        fl.addRow("参照テキスト:", self.qwen3_ref_text_edit)

        # ロード / アンロード / ステータス / テスト
        load_bl = QHBoxLayout()
        self.tts_char_load_btn = QPushButton("メモリにロード")
        self.tts_char_load_btn.setToolTip("このキャラクターの TTS モデルを GPU/CPU メモリにロード")
        self.tts_char_load_btn.clicked.connect(self._on_char_tts_load)
        load_bl.addWidget(self.tts_char_load_btn)
        self.tts_char_unload_btn = QPushButton("アンロード")
        self.tts_char_unload_btn.setProperty("secondary", True)
        self.tts_char_unload_btn.setToolTip("このキャラクターの TTS モデルをメモリから解放")
        self.tts_char_unload_btn.clicked.connect(self._on_char_tts_unload)
        self.tts_char_unload_btn.setEnabled(False)
        load_bl.addWidget(self.tts_char_unload_btn)
        fl.addRow(load_bl)

        self.tts_char_load_progress = QProgressBar()
        self.tts_char_load_progress.setRange(0, 0)
        self.tts_char_load_progress.setVisible(False)
        fl.addRow(self.tts_char_load_progress)

        self.tts_char_status_label = QLabel("")
        self.tts_char_status_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        self.tts_char_status_label.setWordWrap(True)
        fl.addRow("状態:", self.tts_char_status_label)

        tbl = QHBoxLayout()
        self.tts_test_btn = QPushButton("テスト発話")
        self.tts_test_btn.clicked.connect(self._on_tts_test)
        tbl.addWidget(self.tts_test_btn)
        self.tts_stop_btn = QPushButton("停止")
        self.tts_stop_btn.setProperty("secondary", True)
        self.tts_stop_btn.clicked.connect(self._on_tts_stop)
        self.tts_stop_btn.setEnabled(False)
        tbl.addWidget(self.tts_stop_btn)
        fl.addRow(tbl)

        grp.setLayout(fl)
        layout.addWidget(grp)

        self._populate_qwen3_model_combo()
        self._update_qwen3_mode_visibility()

    # ==================================================================
    # Qwen3-TTS モデルコンボ
    # ==================================================================

    _MODE_DISPLAY: dict[str, str] = {
        "custom_voice": "CustomVoice (事前定義スピーカー)",
        "voice_design": "VoiceDesign (テキスト記述)",
        "base": "Base (音声クローン)",
    }

    @staticmethod
    def _get_mode_for_repo(repo_id: str) -> str:
        """repo_id からモデルがサポートするモードを返す。"""
        try:
            from src.core.qwen3_tts_model_manager import Qwen3TTSModelManager
            info = Qwen3TTSModelManager().get_model_by_repo(repo_id)
            if info:
                return info.mode
        except Exception:
            pass
        return "custom_voice"

    def _populate_qwen3_model_combo(self):
        """Qwen3-TTS モデル選択ドロップダウンを初期化する。"""
        self.qwen3_model_combo.blockSignals(True)
        self.qwen3_model_combo.clear()
        try:
            from src.core.qwen3_tts_model_manager import Qwen3TTSModelManager
            mgr = Qwen3TTSModelManager()
            for m in mgr.list_models():
                self.qwen3_model_combo.addItem(m.name, m.repo_id)
        except Exception:
            self.qwen3_model_combo.addItem(
                "Qwen3-TTS 0.6B Base", "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            )
        self.qwen3_model_combo.blockSignals(False)

    def _update_qwen3_mode_visibility(self):
        """選択中モデルのモードに応じてウィジェットの表示を切り替える。"""
        repo_id = self.qwen3_model_combo.currentData() or ""
        mode = self._get_mode_for_repo(repo_id)

        self.qwen3_mode_label.setText(self._MODE_DISPLAY.get(mode, mode))

        is_custom = (mode == "custom_voice")
        is_base = (mode == "base")

        self.qwen3_voice_edit.setVisible(is_custom)
        self.qwen3_ref_audio_edit.setVisible(is_base)
        self.qwen3_ref_audio_btn.setVisible(is_base)
        self.qwen3_ref_audio_label.setVisible(is_base)
        self.qwen3_ref_text_edit.setVisible(is_base)

    def _on_qwen3_model_selected(self, _idx):
        """Qwen3-TTS モデルが選択された時の処理。"""
        char = self._current_character
        if char is None:
            return
        repo_id = self.qwen3_model_combo.currentData() or ""
        char.tts_params.qwen3_model = repo_id
        char.tts_params.qwen3_mode = self._get_mode_for_repo(repo_id)

        self._update_qwen3_mode_visibility()

        if self._tts_engine:
            self._tts_engine.unload_character(char.id)
        self._update_char_load_status()
        self.tts_config_changed.emit()
        self.config_changed.emit()

    def _on_qwen3_params_changed(self):
        """Qwen3-TTS パラメータが変更された時の処理。"""
        char = self._current_character
        if char is None:
            return
        tts = char.tts_params
        tts.qwen3_voice = self.qwen3_voice_edit.text().strip()
        tts.qwen3_instructions = self.qwen3_instructions_edit.toPlainText().strip()
        tts.qwen3_ref_audio = self.qwen3_ref_audio_edit.text().strip()
        tts.qwen3_ref_text = self.qwen3_ref_text_edit.text().strip()
        self.tts_config_changed.emit()
        self.config_changed.emit()

    def _on_qwen3_browse_ref_audio(self):
        """参照音声ファイルを選択するダイアログを表示する。"""
        path, _ = QFileDialog.getOpenFileName(
            self, "参照音声を選択", "",
            "音声ファイル (*.wav *.mp3 *.flac *.ogg);;すべてのファイル (*)",
        )
        if path:
            self.qwen3_ref_audio_edit.setText(path)
            self._on_qwen3_params_changed()

    # ==================================================================
    # デバイスリスト更新
    # ==================================================================

    def _refresh_devices(self):
        saved_input = self._voice_engine.input_device
        saved_output = self._voice_engine.output_device

        self.input_combo.blockSignals(True)
        self.output_combo.blockSignals(True)
        self.input_combo.clear()
        self.output_combo.clear()
        self.input_combo.addItem("デフォルト", None)
        self.output_combo.addItem("デフォルト", None)

        input_devices = VoiceEngine.list_input_devices()
        output_devices = VoiceEngine.list_output_devices()

        input_idx = 0
        for i, dev in enumerate(input_devices):
            sfx = " (default)" if dev.is_default_input else ""
            self.input_combo.addItem(f"{dev.name}{sfx}", dev.index)
            if dev.index == saved_input:
                input_idx = i + 1

        output_idx = 0
        for i, dev in enumerate(output_devices):
            sfx = " (default)" if dev.is_default_output else ""
            self.output_combo.addItem(f"{dev.name}{sfx}", dev.index)
            if dev.index == saved_output:
                output_idx = i + 1

        self.input_combo.setCurrentIndex(input_idx)
        self.output_combo.setCurrentIndex(output_idx)
        self.input_combo.blockSignals(False)
        self.output_combo.blockSignals(False)

        cnt = len(input_devices)
        self.status_label.setText(f"入力デバイス: {cnt}個検出" if cnt > 0 else "入力デバイスが見つかりません")

    # ==================================================================
    # エンジンから読み込み
    # ==================================================================

    def _load_from_engine(self):
        self.model_combo.blockSignals(True)
        self.device_combo.blockSignals(True)

        ms = self._voice_engine.model_size
        for i in range(self.model_combo.count()):
            if self.model_combo.itemData(i) == ms:
                self.model_combo.setCurrentIndex(i)
                break
        dv = self._voice_engine.whisper_device
        for i in range(self.device_combo.count()):
            if self.device_combo.itemData(i) == dv:
                self.device_combo.setCurrentIndex(i)
                break
        self.language_edit.setText(self._voice_engine.language)

        self.model_combo.blockSignals(False)
        self.device_combo.blockSignals(False)

    def _load_tts_from_engine(self):
        if not self._tts_engine:
            return
        self.tts_enabled_check.blockSignals(True)
        self.tts_auto_play_check.blockSignals(True)
        self.tts_device_combo.blockSignals(True)

        self.tts_enabled_check.setChecked(self._tts_engine.enabled)
        self.tts_auto_play_check.setChecked(self._tts_engine.auto_play)
        dv = "cuda" if self._tts_engine.use_gpu else "cpu"
        for i in range(self.tts_device_combo.count()):
            if self.tts_device_combo.itemData(i) == dv:
                self.tts_device_combo.setCurrentIndex(i)
                break

        self.tts_enabled_check.blockSignals(False)
        self.tts_auto_play_check.blockSignals(False)
        self.tts_device_combo.blockSignals(False)

    # ==================================================================
    # 入出力デバイスハンドラ
    # ==================================================================

    def _on_input_device_changed(self, _idx):
        did = self.input_combo.currentData()
        self._voice_engine.input_device = did
        self.input_device_changed.emit(did)
        self.config_changed.emit()

    def _on_output_device_changed(self, _idx):
        did = self.output_combo.currentData()
        self._voice_engine.output_device = did
        self.output_device_changed.emit(did)
        self.config_changed.emit()

    # ==================================================================
    # Whisper
    # ==================================================================

    def _on_whisper_settings_changed(self):
        ms = self.model_combo.currentData()
        dv = self.device_combo.currentData()
        lang = self.language_edit.text().strip() or "ja"
        ct = "float16" if dv == "cuda" else "int8"
        self._voice_engine.update_settings(model_size=ms, device=dv, compute_type=ct, language=lang)
        self.whisper_settings_changed.emit({"whisper_model": ms, "device": dv, "compute_type": ct, "language": lang})
        self.config_changed.emit()
        self._update_whisper_status()

    def _on_whisper_load(self):
        if self._load_worker and self._load_worker.isRunning():
            return
        self.whisper_load_btn.setEnabled(False)
        self.whisper_load_btn.setText("ロード中...")
        self.whisper_unload_btn.setEnabled(False)
        self.whisper_progress.setVisible(True)
        self.whisper_status_label.setText("ダウンロード/ロード中...")
        self.whisper_status_label.setStyleSheet(f"color: {COLORS['warning']}; font-size: 11px;")
        self._load_worker = WhisperLoadWorker(self._voice_engine, self)
        self._load_worker.finished_signal.connect(self._on_whisper_load_finished)
        self._load_worker.error_signal.connect(self._on_whisper_load_error)
        self._load_worker.start()

    def _on_whisper_load_finished(self):
        self.whisper_load_btn.setText("モデルロード")
        self.whisper_load_btn.setEnabled(True)
        self.whisper_unload_btn.setEnabled(True)
        self.whisper_progress.setVisible(False)
        self._update_whisper_status()
        self.vram_changed.emit()

    def _on_whisper_load_error(self, err):
        self.whisper_load_btn.setText("モデルロード")
        self.whisper_load_btn.setEnabled(True)
        self.whisper_progress.setVisible(False)
        self.whisper_status_label.setText(f"エラー: {err}")
        self.whisper_status_label.setStyleSheet(f"color: {COLORS['error']}; font-size: 11px;")

    def _on_whisper_unload(self):
        self._voice_engine.unload_model()
        self.whisper_unload_btn.setEnabled(False)
        self._update_whisper_status()
        self.vram_changed.emit()

    def _update_whisper_status(self):
        if self._voice_engine.is_model_loaded:
            m = self._voice_engine.model_size
            d = self._voice_engine.whisper_device
            self.whisper_status_label.setText(f"ロード済み: {m} ({d})")
            self.whisper_status_label.setStyleSheet(f"color: {COLORS['success']}; font-size: 11px;")
            self.whisper_unload_btn.setEnabled(True)
        else:
            self.whisper_status_label.setText("未ロード")
            self.whisper_status_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
            self.whisper_unload_btn.setEnabled(False)

    # ==================================================================
    # TTS グローバル設定
    # ==================================================================

    def _on_tts_enabled_changed(self, checked):
        if self._tts_engine:
            self._tts_engine.enabled = checked
        self._update_tts_status()
        self.tts_config_changed.emit()
        self.config_changed.emit()

    def _on_tts_auto_play_changed(self, checked):
        if self._tts_engine:
            self._tts_engine.auto_play = checked
        self.tts_config_changed.emit()
        self.config_changed.emit()

    def _on_tts_device_changed(self, _idx):
        dv = self.tts_device_combo.currentData()
        if self._tts_engine:
            self._tts_engine.use_gpu = (dv == "cuda")
        self._update_tts_status()
        self.tts_config_changed.emit()
        self.config_changed.emit()

    def _on_tts_unload_all(self):
        if self._tts_engine:
            self._tts_engine.unload_all()
        self.tts_unload_btn.setEnabled(False)
        self._update_tts_status()
        self._update_char_load_status()
        self.vram_changed.emit()

    def _update_tts_status(self):
        if not self._tts_engine:
            self.tts_status_label.setText("TTSEngine 未初期化")
            self.tts_status_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
            return

        gpu_warn = ""
        if self._tts_engine.use_gpu and not self._tts_engine.cuda_available:
            gpu_warn = " [CUDA不可→CPU使用]"

        if not self._tts_engine.enabled:
            self.tts_status_label.setText(f"無効{gpu_warn}")
            self.tts_status_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
            self.tts_unload_btn.setEnabled(False)
            return

        qwen3_ok = getattr(self._tts_engine, "qwen3_available", False)
        if not qwen3_ok:
            self.tts_status_label.setText(
                f"有効 (Qwen3-TTS 利用不可: pip install qwen-tts が必要){gpu_warn}"
            )
            self.tts_status_label.setStyleSheet(f"color: {COLORS['warning']}; font-size: 11px;")
            self.tts_unload_btn.setEnabled(False)
            return

        loaded = self._tts_engine.loaded_character_ids()
        if loaded:
            self.tts_status_label.setText(
                f"Qwen3-TTS 有効 ({self._tts_engine.device}) / ロード済みキャラ: {len(loaded)}個{gpu_warn}"
            )
            self.tts_status_label.setStyleSheet(f"color: {COLORS['success']}; font-size: 11px;")
            self.tts_unload_btn.setEnabled(True)
        else:
            self.tts_status_label.setText(f"Qwen3-TTS 有効 (モデル未ロード){gpu_warn}")
            self.tts_status_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
            self.tts_unload_btn.setEnabled(False)

    # ==================================================================
    # キャラクター別 TTS
    # ==================================================================

    def _update_tts_character_display(self):
        char = self._current_character
        if char is None:
            self.tts_char_label.setText("キャラクター未選択")
            self._set_char_tts_widgets_enabled(False)
            return

        self.tts_char_label.setText(f"キャラクター: {char.name}")
        self._set_char_tts_widgets_enabled(True)

        tts = char.tts_params
        self._block_char_signals(True)

        self.tts_char_enabled_check.setChecked(tts.enabled)

        if tts.qwen3_model:
            for i in range(self.qwen3_model_combo.count()):
                if self.qwen3_model_combo.itemData(i) == tts.qwen3_model:
                    self.qwen3_model_combo.setCurrentIndex(i)
                    break
        self.qwen3_voice_edit.setText(tts.qwen3_voice)
        self.qwen3_instructions_edit.setPlainText(tts.qwen3_instructions)
        self.qwen3_ref_audio_edit.setText(tts.qwen3_ref_audio)
        self.qwen3_ref_text_edit.setText(tts.qwen3_ref_text)

        self._update_qwen3_mode_visibility()
        self._block_char_signals(False)
        self._update_char_load_status()

    def _block_char_signals(self, block: bool):
        for w in (
            self.tts_char_enabled_check,
            self.qwen3_model_combo,
            self.qwen3_voice_edit, self.qwen3_instructions_edit,
            self.qwen3_ref_audio_edit, self.qwen3_ref_text_edit,
        ):
            w.blockSignals(block)

    def _set_char_tts_widgets_enabled(self, enabled):
        for w in (
            self.tts_char_enabled_check,
            self.tts_char_load_btn, self.tts_char_unload_btn,
            self.tts_test_btn,
            self.qwen3_model_combo,
            self.qwen3_voice_edit, self.qwen3_instructions_edit,
            self.qwen3_ref_audio_edit, self.qwen3_ref_audio_btn,
            self.qwen3_ref_text_edit,
        ):
            w.setEnabled(enabled)

    def _on_char_tts_toggled(self, checked):
        if self._current_character is None:
            return
        self._current_character.tts_params.enabled = checked
        self.tts_config_changed.emit()
        self.config_changed.emit()

    # ==================================================================
    # キャラクター TTS ロード / アンロード
    # ==================================================================

    def _on_char_tts_load(self):
        char = self._current_character
        if not char or not self._tts_engine:
            return

        qwen3_ok = getattr(self._tts_engine, "qwen3_available", False)
        if not qwen3_ok:
            self.tts_char_status_label.setText(
                "Qwen3-TTS 利用不可 (pip install qwen-tts が必要)"
            )
            self.tts_char_status_label.setStyleSheet(f"color: {COLORS['error']}; font-size: 11px;")
            return

        register_fn = getattr(self._tts_engine, "try_register_cached_model", None)
        if register_fn and register_fn(char):
            self._update_char_load_status()
            self._update_tts_status()
            return

        if self._tts_load_worker and self._tts_load_worker.isRunning():
            return

        self.tts_char_load_btn.setEnabled(False)
        self.tts_char_load_btn.setText("ロード中...")
        self.tts_char_load_progress.setVisible(True)
        self.tts_char_status_label.setText("Qwen3-TTS ロード中...")
        self.tts_char_status_label.setStyleSheet(f"color: {COLORS['warning']}; font-size: 11px;")

        self._tts_load_worker = TTSLoadWorker(self._tts_engine, char, self)
        self._tts_load_worker.finished_signal.connect(self._on_char_tts_load_finished)
        self._tts_load_worker.error_signal.connect(self._on_char_tts_load_error)
        self._tts_load_worker.start()

    def _on_char_tts_load_finished(self):
        self.tts_char_load_progress.setVisible(False)
        self._update_char_load_status()
        self._update_tts_status()
        self.vram_changed.emit()

    def _on_char_tts_load_error(self, err):
        self.tts_char_load_progress.setVisible(False)
        self._update_char_load_status()
        self.tts_char_status_label.setText(f"ロードエラー: {err}")
        self.tts_char_status_label.setStyleSheet(f"color: {COLORS['error']}; font-size: 11px;")

    def _on_char_tts_unload(self):
        char = self._current_character
        if not char or not self._tts_engine:
            return
        self._tts_engine.unload_character(char.id)
        self._update_char_load_status()
        self._update_tts_status()
        self.vram_changed.emit()

    def _update_char_load_status(self):
        char = self._current_character
        if not char or not self._tts_engine:
            self.tts_char_status_label.setText("")
            self.tts_char_unload_btn.setEnabled(False)
            self.tts_char_load_btn.setText("メモリにロード")
            self.tts_char_load_btn.setEnabled(False)
            return

        qwen3_ok = getattr(self._tts_engine, "qwen3_available", False)
        if not qwen3_ok:
            self.tts_char_load_btn.setText("メモリにロード")
            self.tts_char_load_btn.setEnabled(False)
            self.tts_char_status_label.setText(
                "Qwen3-TTS 利用不可 (pip install qwen-tts が必要)"
            )
            self.tts_char_status_label.setStyleSheet(f"color: {COLORS['error']}; font-size: 11px;")
            self.tts_char_unload_btn.setEnabled(False)
            return

        register_fn = getattr(self._tts_engine, "try_register_cached_model", None)
        if register_fn:
            register_fn(char)

        loaded = self._tts_engine.is_character_loaded(char.id)

        self.tts_char_load_btn.setText("メモリにロード")
        self.tts_char_load_btn.setEnabled(True)
        if loaded:
            self.tts_char_status_label.setText(f"Qwen3-TTS ロード済み ({self._tts_engine.device})")
            self.tts_char_status_label.setStyleSheet(f"color: {COLORS['success']}; font-size: 11px;")
            self.tts_char_unload_btn.setEnabled(True)
        else:
            self.tts_char_status_label.setText("Qwen3-TTS 未ロード")
            self.tts_char_status_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
            self.tts_char_unload_btn.setEnabled(False)

    # ==================================================================
    # テスト発話
    # ==================================================================

    def _on_tts_test(self):
        if not self._tts_engine or not self._current_character:
            return
        if self._tts_test_worker and self._tts_test_worker.isRunning():
            return
        output_device = self.output_combo.currentData()
        self.tts_test_btn.setEnabled(False)
        self.tts_test_btn.setText("再生中...")
        self.tts_stop_btn.setEnabled(True)
        self._tts_test_worker = TTSTestWorker(
            self._tts_engine, "こんにちは、テスト音声です。",
            self._current_character, output_device, self,
        )
        self._tts_test_worker.finished_signal.connect(self._on_tts_test_finished)
        self._tts_test_worker.error_signal.connect(self._on_tts_test_error)
        self._tts_test_worker.start()

    def _on_tts_test_finished(self):
        self.tts_test_btn.setText("テスト発話")
        self.tts_test_btn.setEnabled(True)
        self.tts_stop_btn.setEnabled(False)

        char = self._current_character
        if char is not None and not char.tts_params.enabled:
            char.tts_params.enabled = True
            self.tts_char_enabled_check.blockSignals(True)
            self.tts_char_enabled_check.setChecked(True)
            self.tts_char_enabled_check.blockSignals(False)
            self.tts_config_changed.emit()
            self.config_changed.emit()
            logger.info(
                "Auto-enabled TTS for character '%s' after successful test.",
                char.name,
            )

        self._update_tts_status()
        self._update_char_load_status()

    def _on_tts_test_error(self, err):
        self.tts_test_btn.setText("テスト発話")
        self.tts_test_btn.setEnabled(True)
        self.tts_stop_btn.setEnabled(False)
        self.tts_char_status_label.setText(f"テストエラー: {err}")
        self.tts_char_status_label.setStyleSheet(f"color: {COLORS['error']}; font-size: 11px;")

    def _on_tts_stop(self):
        if self._tts_engine:
            self._tts_engine.stop_playback()
        self.tts_stop_btn.setEnabled(False)
