"""
SettingsDialog: GPU設定、コンテキスト長、デフォルトパラメータ等の設定画面。
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from src.ui.styles import COLORS

logger = logging.getLogger(__name__)


class SettingsDialog(QDialog):
    """アプリケーション設定ダイアログ。"""

    settings_saved = Signal(dict)

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("設定")
        self.setMinimumWidth(450)
        self._setup_ui()
        self._load_config()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # ---- GPU設定 ----
        gpu_group = QGroupBox("GPU設定")
        gpu_layout = QFormLayout()
        gpu_layout.setSpacing(8)

        self.gpu_layers_spin = QSpinBox()
        self.gpu_layers_spin.setRange(-1, 200)
        self.gpu_layers_spin.setSpecialValueText("全レイヤー (-1)")
        self.gpu_layers_spin.setToolTip("-1 = 全レイヤーをGPUにオフロード")
        gpu_layout.addRow("GPUレイヤー数:", self.gpu_layers_spin)

        self.main_gpu_spin = QSpinBox()
        self.main_gpu_spin.setRange(0, 7)
        self.main_gpu_spin.setToolTip("使用するGPUのインデックス (通常は0)")
        gpu_layout.addRow("メインGPU:", self.main_gpu_spin)

        gpu_group.setLayout(gpu_layout)
        layout.addWidget(gpu_group)

        # ---- 推論設定 ----
        inf_group = QGroupBox("推論設定")
        inf_layout = QFormLayout()
        inf_layout.setSpacing(8)

        self.ctx_length_spin = QSpinBox()
        self.ctx_length_spin.setRange(512, 32768)
        self.ctx_length_spin.setSingleStep(512)
        self.ctx_length_spin.setToolTip("コンテキストウィンドウのサイズ (トークン数)")
        inf_layout.addRow("コンテキスト長:", self.ctx_length_spin)

        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 2.0)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setToolTip("デフォルトのTemperature (キャラクター設定で上書き可)")
        inf_layout.addRow("Temperature:", self.temperature_spin)

        self.top_p_spin = QDoubleSpinBox()
        self.top_p_spin.setRange(0.0, 1.0)
        self.top_p_spin.setSingleStep(0.05)
        inf_layout.addRow("Top P:", self.top_p_spin)

        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(64, 4096)
        self.max_tokens_spin.setSingleStep(64)
        inf_layout.addRow("最大トークン数:", self.max_tokens_spin)

        self.repeat_penalty_spin = QDoubleSpinBox()
        self.repeat_penalty_spin.setRange(1.0, 2.0)
        self.repeat_penalty_spin.setSingleStep(0.05)
        inf_layout.addRow("繰り返しペナルティ:", self.repeat_penalty_spin)

        inf_group.setLayout(inf_layout)
        layout.addWidget(inf_group)

        # ---- UI設定 ----
        ui_group = QGroupBox("UI設定")
        ui_layout = QFormLayout()
        ui_layout.setSpacing(8)

        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(10, 24)
        self.font_size_spin.setSingleStep(1)
        ui_layout.addRow("フォントサイズ:", self.font_size_spin)

        self.theme_combo = QComboBox()
        self.theme_combo.addItem("ダーク", "dark")
        ui_layout.addRow("テーマ:", self.theme_combo)

        ui_group.setLayout(ui_layout)
        layout.addWidget(ui_group)

        # ---- ボタン ----
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("キャンセル")
        cancel_btn.setProperty("secondary", True)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        save_btn = QPushButton("保存")
        save_btn.clicked.connect(self._on_save)
        btn_layout.addWidget(save_btn)

        layout.addLayout(btn_layout)

    def _load_config(self):
        """現在の設定をフォームに読み込む。"""
        gpu = self.config.get("gpu", {})
        self.gpu_layers_spin.setValue(gpu.get("n_gpu_layers", -1))
        self.main_gpu_spin.setValue(gpu.get("main_gpu", 0))

        inf = self.config.get("inference", {})
        self.ctx_length_spin.setValue(inf.get("context_length", 4096))
        self.temperature_spin.setValue(inf.get("temperature", 0.7))
        self.top_p_spin.setValue(inf.get("top_p", 0.9))
        self.max_tokens_spin.setValue(inf.get("max_tokens", 512))
        self.repeat_penalty_spin.setValue(inf.get("repeat_penalty", 1.1))

        ui = self.config.get("ui", {})
        self.font_size_spin.setValue(ui.get("font_size", 14))

    def _on_save(self):
        """設定を保存。"""
        new_config = {
            "gpu": {
                "n_gpu_layers": self.gpu_layers_spin.value(),
                "main_gpu": self.main_gpu_spin.value(),
            },
            "inference": {
                "context_length": self.ctx_length_spin.value(),
                "temperature": self.temperature_spin.value(),
                "top_p": self.top_p_spin.value(),
                "max_tokens": self.max_tokens_spin.value(),
                "repeat_penalty": self.repeat_penalty_spin.value(),
            },
            "ui": {
                "theme": self.theme_combo.currentData(),
                "font_size": self.font_size_spin.value(),
            },
        }
        self.settings_saved.emit(new_config)
        self.accept()
