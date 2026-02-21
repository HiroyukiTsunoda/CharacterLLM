"""
VoicePanel: サイドバー「音声」タブ。入力/出力デバイス選択、Whisper 設定、
Style-Bert-VITS2 TTS 設定（モデルダウンロード・ロード/アンロード含む）を提供する。
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import Signal, QThread
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.core.voice_input import VoiceEngine
from src.core.voice_output import TTSEngine
from src.core.tts_model_manager import TTSModelManager
from src.ui import install_wheel_guard
from src.ui.styles import COLORS

logger = logging.getLogger(__name__)

WHISPER_MODELS = [
    ("tiny", "tiny (~75MB, 低精度・高速)"),
    ("base", "base (~140MB)"),
    ("small", "small (~460MB, 推奨)"),
    ("medium", "medium (~1.5GB)"),
    ("large-v3", "large-v3 (~3GB, 高精度)"),
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


class BERTLoadWorker(QThread):
    finished_signal = Signal()
    error_signal = Signal(str)

    def __init__(self, tts_engine: TTSEngine, parent=None):
        super().__init__(parent)
        self._tts_engine = tts_engine

    def run(self):
        try:
            self._tts_engine.load_bert()
            self.finished_signal.emit()
        except Exception as e:
            logger.error("BERT model load failed: %s", e)
            self.error_signal.emit(str(e))


class TTSModelDownloadWorker(QThread):
    """HuggingFace からの TTS モデルダウンロードを別スレッドで実行する。"""
    finished_signal = Signal(str)
    error_signal = Signal(str)

    def __init__(self, model_manager: TTSModelManager, model_id: str, parent=None):
        super().__init__(parent)
        self._mgr = model_manager
        self._model_id = model_id

    def run(self):
        try:
            self._mgr.download_model(self._model_id)
            self.finished_signal.emit(self._model_id)
        except Exception as e:
            logger.error("TTS model download failed: %s", e)
            self.error_signal.emit(str(e))


class TTSLoadWorker(QThread):
    """キャラクターの TTS モデルをメモリにロードする。"""
    finished_signal = Signal()
    error_signal = Signal(str)

    def __init__(self, tts_engine: TTSEngine, character, parent=None):
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
        tts_engine: TTSEngine | None = None,
        model_manager: TTSModelManager | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._voice_engine = voice_engine
        self._tts_engine = tts_engine
        self._model_manager = model_manager
        self._load_worker: WhisperLoadWorker | None = None
        self._bert_worker: BERTLoadWorker | None = None
        self._dl_worker: TTSModelDownloadWorker | None = None
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
        self._refresh_model_combo()

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

        # ---- 入力デバイス ----
        self._build_input_device_ui(layout)
        # ---- 出力デバイス ----
        self._build_output_device_ui(layout)
        # ---- Whisper ----
        self._build_whisper_ui(layout)
        # ---- TTS グローバル ----
        self._build_tts_global_ui(layout)
        # ---- キャラクター別 TTS ----
        self._build_tts_char_ui(layout)

        # ---- ステータス ----
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
            self.tts_model_select_combo,
            self.tts_speaker_id_spin, self.tts_length_spin,
            self.tts_sdp_spin, self.tts_noise_spin,
            self.tts_noisew_spin, self.tts_style_weight_spin,
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
        grp = QGroupBox("音声合成 (Style-Bert-VITS2)")
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
        self.bert_load_btn = QPushButton("BERTロード")
        self.bert_load_btn.setToolTip("SBV2 用 BERT モデルをロード（初回はダウンロード）")
        self.bert_load_btn.clicked.connect(self._on_bert_load)
        bl.addWidget(self.bert_load_btn)
        self.bert_unload_btn = QPushButton("全モデル解放")
        self.bert_unload_btn.setProperty("secondary", True)
        self.bert_unload_btn.clicked.connect(self._on_tts_unload_all)
        self.bert_unload_btn.setEnabled(False)
        bl.addWidget(self.bert_unload_btn)
        fl.addRow(bl)

        self.bert_progress = QProgressBar()
        self.bert_progress.setRange(0, 0)
        self.bert_progress.setVisible(False)
        fl.addRow(self.bert_progress)

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

        # モデル選択ドロップダウン
        self.tts_model_select_combo = QComboBox()
        self.tts_model_select_combo.setToolTip("ダウンロード済みモデルから選択")
        self.tts_model_select_combo.currentIndexChanged.connect(self._on_model_selected)
        fl.addRow("TTSモデル:", self.tts_model_select_combo)

        # ロード / アンロード
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

        # スタイル
        self.tts_style_edit = QLineEdit("Neutral")
        self.tts_style_edit.setToolTip("スタイル名 (例: Neutral, Happy, Sad)")
        self.tts_style_edit.editingFinished.connect(self._on_char_tts_params_changed)
        fl.addRow("スタイル:", self.tts_style_edit)

        # speaker_id
        self.tts_speaker_id_spin = QSpinBox()
        self.tts_speaker_id_spin.setRange(0, 99)
        self.tts_speaker_id_spin.valueChanged.connect(self._on_char_tts_params_changed)
        fl.addRow("話者ID:", self.tts_speaker_id_spin)

        # 話速
        self.tts_length_spin = QDoubleSpinBox()
        self.tts_length_spin.setRange(0.1, 5.0)
        self.tts_length_spin.setSingleStep(0.05)
        self.tts_length_spin.setValue(1.0)
        self.tts_length_spin.setToolTip("話速 (1.0=標準、大きいほどゆっくり)")
        self.tts_length_spin.valueChanged.connect(self._on_char_tts_params_changed)
        fl.addRow("話速:", self.tts_length_spin)

        # sdp_ratio
        self.tts_sdp_spin = QDoubleSpinBox()
        self.tts_sdp_spin.setRange(0.0, 1.0)
        self.tts_sdp_spin.setSingleStep(0.05)
        self.tts_sdp_spin.setValue(0.2)
        self.tts_sdp_spin.valueChanged.connect(self._on_char_tts_params_changed)
        fl.addRow("SDP Ratio:", self.tts_sdp_spin)

        # noise
        self.tts_noise_spin = QDoubleSpinBox()
        self.tts_noise_spin.setRange(0.0, 2.0)
        self.tts_noise_spin.setSingleStep(0.05)
        self.tts_noise_spin.setValue(0.6)
        self.tts_noise_spin.valueChanged.connect(self._on_char_tts_params_changed)
        fl.addRow("Noise:", self.tts_noise_spin)

        # noise_w
        self.tts_noisew_spin = QDoubleSpinBox()
        self.tts_noisew_spin.setRange(0.0, 2.0)
        self.tts_noisew_spin.setSingleStep(0.05)
        self.tts_noisew_spin.setValue(0.8)
        self.tts_noisew_spin.valueChanged.connect(self._on_char_tts_params_changed)
        fl.addRow("Noise W:", self.tts_noisew_spin)

        # style_weight
        self.tts_style_weight_spin = QDoubleSpinBox()
        self.tts_style_weight_spin.setRange(0.0, 50.0)
        self.tts_style_weight_spin.setSingleStep(0.5)
        self.tts_style_weight_spin.setValue(5.0)
        self.tts_style_weight_spin.valueChanged.connect(self._on_char_tts_params_changed)
        fl.addRow("Style Weight:", self.tts_style_weight_spin)

        # テスト発話
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

    # ==================================================================
    # モデル選択コンボの更新
    # ==================================================================

    def _refresh_model_combo(self):
        """キャラクター TTS モデル選択ドロップダウンを更新する（全モデル表示）。"""
        self.tts_model_select_combo.blockSignals(True)
        self.tts_model_select_combo.clear()
        self.tts_model_select_combo.addItem("(未設定)", "")
        if self._model_manager:
            for info in self._model_manager.list_downloadable():
                suffix = " DL済" if info["downloaded"] else ""
                self.tts_model_select_combo.addItem(f'{info["name"]}{suffix}', info["id"])
        self.tts_model_select_combo.blockSignals(False)

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

    # ==================================================================
    # BERT ロード / アンロード
    # ==================================================================

    def _on_bert_load(self):
        if not self._tts_engine:
            return
        if self._bert_worker and self._bert_worker.isRunning():
            return
        self.bert_load_btn.setEnabled(False)
        self.bert_load_btn.setText("ロード中...")
        self.bert_progress.setVisible(True)
        self.tts_status_label.setText("BERT ダウンロード/ロード中...")
        self.tts_status_label.setStyleSheet(f"color: {COLORS['warning']}; font-size: 11px;")
        self._bert_worker = BERTLoadWorker(self._tts_engine, self)
        self._bert_worker.finished_signal.connect(self._on_bert_load_finished)
        self._bert_worker.error_signal.connect(self._on_bert_load_error)
        self._bert_worker.start()

    def _on_bert_load_finished(self):
        self.bert_load_btn.setText("BERTロード")
        self.bert_load_btn.setEnabled(True)
        self.bert_unload_btn.setEnabled(True)
        self.bert_progress.setVisible(False)
        self._update_tts_status()
        self.vram_changed.emit()

    def _on_bert_load_error(self, err):
        self.bert_load_btn.setText("BERTロード")
        self.bert_load_btn.setEnabled(True)
        self.bert_progress.setVisible(False)
        self.tts_status_label.setText(f"BERT エラー: {err}")
        self.tts_status_label.setStyleSheet(f"color: {COLORS['error']}; font-size: 11px;")

    def _on_tts_unload_all(self):
        if self._tts_engine:
            self._tts_engine.unload_all()
        self.bert_unload_btn.setEnabled(False)
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
            self.bert_unload_btn.setEnabled(False)
            return
        if self._tts_engine.is_bert_loaded:
            loaded = self._tts_engine.loaded_character_ids()
            self.tts_status_label.setText(
                f"BERT ロード済み ({self._tts_engine.device}) / TTS モデル: {len(loaded)}個{gpu_warn}"
            )
            self.tts_status_label.setStyleSheet(f"color: {COLORS['success']}; font-size: 11px;")
            self.bert_unload_btn.setEnabled(True)
        else:
            self.tts_status_label.setText(f"有効 (BERT 未ロード){gpu_warn}")
            self.tts_status_label.setStyleSheet(f"color: {COLORS['warning']}; font-size: 11px;")
            self.bert_unload_btn.setEnabled(False)

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
        self.tts_style_edit.setText(tts.style)
        self.tts_speaker_id_spin.setValue(tts.speaker_id)
        self.tts_length_spin.setValue(tts.length)
        self.tts_sdp_spin.setValue(tts.sdp_ratio)
        self.tts_noise_spin.setValue(tts.noise)
        self.tts_noisew_spin.setValue(tts.noise_w)
        self.tts_style_weight_spin.setValue(tts.style_weight)

        # モデル選択コンボを同期
        self._sync_model_combo_to_char(tts)

        self._block_char_signals(False)
        self._update_char_load_status()

    def _sync_model_combo_to_char(self, tts):
        """キャラクターの model_path からモデル選択コンボのインデックスを合わせる。"""
        self.tts_model_select_combo.blockSignals(True)
        matched = False
        if tts.model_path and self._model_manager:
            for i in range(1, self.tts_model_select_combo.count()):
                mid = self.tts_model_select_combo.itemData(i)
                lm = self._model_manager.get_local_model(mid)
                if lm and str(lm.model_path) == tts.model_path:
                    self.tts_model_select_combo.setCurrentIndex(i)
                    matched = True
                    break
        if not matched:
            self.tts_model_select_combo.setCurrentIndex(0)
        self.tts_model_select_combo.blockSignals(False)

    def _block_char_signals(self, block: bool):
        for w in (
            self.tts_char_enabled_check, self.tts_style_edit,
            self.tts_speaker_id_spin, self.tts_length_spin,
            self.tts_sdp_spin, self.tts_noise_spin,
            self.tts_noisew_spin, self.tts_style_weight_spin,
            self.tts_model_select_combo,
        ):
            w.blockSignals(block)

    def _set_char_tts_widgets_enabled(self, enabled):
        for w in (
            self.tts_char_enabled_check, self.tts_model_select_combo,
            self.tts_char_load_btn, self.tts_char_unload_btn,
            self.tts_style_edit, self.tts_speaker_id_spin,
            self.tts_length_spin, self.tts_sdp_spin,
            self.tts_noise_spin, self.tts_noisew_spin,
            self.tts_style_weight_spin, self.tts_test_btn,
        ):
            w.setEnabled(enabled)

    def _on_model_selected(self, _idx):
        """ドロップダウンからモデルを選択した時、キャラクターのパスを自動設定する。"""
        char = self._current_character
        if char is None or not self._model_manager:
            return
        mid = self.tts_model_select_combo.currentData()
        if not mid:
            char.tts_params.model_path = ""
            char.tts_params.config_path = ""
            char.tts_params.style_vec_path = ""
        else:
            lm = self._model_manager.get_local_model(mid)
            if lm:
                char.tts_params.model_path = str(lm.model_path)
                char.tts_params.config_path = str(lm.config_path)
                char.tts_params.style_vec_path = str(lm.style_vec_path)
            else:
                char.tts_params.model_path = ""
                char.tts_params.config_path = ""
                char.tts_params.style_vec_path = ""

        if self._tts_engine:
            self._tts_engine.unload_character(char.id)
        self._update_char_load_status()
        self.tts_config_changed.emit()
        self.config_changed.emit()

    def _on_char_tts_toggled(self, checked):
        if self._current_character is None:
            return
        self._current_character.tts_params.enabled = checked
        self.tts_config_changed.emit()
        self.config_changed.emit()

    def _on_char_tts_params_changed(self):
        char = self._current_character
        if char is None:
            return
        tts = char.tts_params
        tts.style = self.tts_style_edit.text().strip() or "Neutral"
        tts.speaker_id = self.tts_speaker_id_spin.value()
        tts.length = self.tts_length_spin.value()
        tts.sdp_ratio = self.tts_sdp_spin.value()
        tts.noise = self.tts_noise_spin.value()
        tts.noise_w = self.tts_noisew_spin.value()
        tts.style_weight = self.tts_style_weight_spin.value()
        self.tts_config_changed.emit()
        self.config_changed.emit()

    # ==================================================================
    # キャラクター TTS モデルダウンロード（キャラクターセクション内）
    # ==================================================================

    def _start_char_download(self, model_id: str):
        """選択されたモデルをキャラクター TTS セクションからダウンロードする。"""
        if not self._model_manager:
            return
        if self._dl_worker and self._dl_worker.isRunning():
            return

        self.tts_char_load_btn.setEnabled(False)
        self.tts_char_load_btn.setText("ダウンロード中...")
        self.tts_char_load_progress.setVisible(True)
        self.tts_char_status_label.setText(f"ダウンロード中...")
        self.tts_char_status_label.setStyleSheet(f"color: {COLORS['warning']}; font-size: 11px;")

        self._dl_worker = TTSModelDownloadWorker(self._model_manager, model_id, self)
        self._dl_worker.finished_signal.connect(self._on_char_download_finished)
        self._dl_worker.error_signal.connect(self._on_char_download_error)
        self._dl_worker.start()

    def _on_char_download_finished(self, model_id: str):
        self.tts_char_load_progress.setVisible(False)
        self.tts_char_status_label.setText(f"ダウンロード完了")
        self.tts_char_status_label.setStyleSheet(f"color: {COLORS['success']}; font-size: 11px;")

        self._refresh_model_combo()

        char = self._current_character
        if char and self._model_manager:
            lm = self._model_manager.get_local_model(model_id)
            if lm:
                char.tts_params.model_path = str(lm.model_path)
                char.tts_params.config_path = str(lm.config_path)
                char.tts_params.style_vec_path = str(lm.style_vec_path)
                self.tts_config_changed.emit()
                self.config_changed.emit()
            self._sync_model_combo_to_char(char.tts_params)

        self._update_char_load_status()

    def _on_char_download_error(self, err: str):
        self.tts_char_load_progress.setVisible(False)
        self._update_char_load_status()
        self.tts_char_status_label.setText(f"ダウンロードエラー: {err}")
        self.tts_char_status_label.setStyleSheet(f"color: {COLORS['error']}; font-size: 11px;")

    # ==================================================================
    # キャラクター TTS ロード / アンロード
    # ==================================================================

    def _on_char_tts_load(self):
        char = self._current_character
        if not char or not self._tts_engine:
            return

        mid = self.tts_model_select_combo.currentData()
        if not mid:
            self.tts_char_status_label.setText("TTS モデルを選択してください")
            self.tts_char_status_label.setStyleSheet(f"color: {COLORS['error']}; font-size: 11px;")
            return

        lm = self._model_manager.get_local_model(mid) if self._model_manager else None
        if not lm:
            self._start_char_download(mid)
            return

        if not char.tts_params.model_path:
            self.tts_char_status_label.setText("TTS モデルを選択してください")
            self.tts_char_status_label.setStyleSheet(f"color: {COLORS['error']}; font-size: 11px;")
            return
        if self._tts_load_worker and self._tts_load_worker.isRunning():
            return

        self.tts_char_load_btn.setEnabled(False)
        self.tts_char_load_btn.setText("ロード中...")
        self.tts_char_load_progress.setVisible(True)
        self.tts_char_status_label.setText("ロード中...")
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

        mid = self.tts_model_select_combo.currentData()
        is_downloaded = bool(
            self._model_manager and mid and self._model_manager.get_local_model(mid)
        )

        if mid and not is_downloaded:
            self.tts_char_load_btn.setText("ダウンロード")
            self.tts_char_load_btn.setEnabled(True)
        else:
            self.tts_char_load_btn.setText("メモリにロード")
            self.tts_char_load_btn.setEnabled(bool(mid))

        loaded = self._tts_engine.is_character_loaded(char.id)
        if loaded:
            self.tts_char_status_label.setText(f"ロード済み ({self._tts_engine.device})")
            self.tts_char_status_label.setStyleSheet(f"color: {COLORS['success']}; font-size: 11px;")
            self.tts_char_unload_btn.setEnabled(True)
        else:
            if mid and not is_downloaded:
                self.tts_char_status_label.setText("モデル未ダウンロード")
            elif mid:
                self.tts_char_status_label.setText("未ロード")
            else:
                self.tts_char_status_label.setText("モデル未設定")
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
