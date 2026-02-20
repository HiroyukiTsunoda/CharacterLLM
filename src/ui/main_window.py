"""
MainWindow: アプリのメインウィンドウ。サイドバー + チャットエリアのレイアウト。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.core.character import CharacterManager
from src.core.chat_engine import ChatEngine
from src.core.model_loader import ModelLoader
from src.ui.character_panel import CharacterPanel
from src.ui.chat_widget import ChatWidget
from src.ui.model_panel import ModelPanel
from src.ui.styles import COLORS, DARK_THEME, SIDEBAR_STYLE

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """アプリケーションのメインウィンドウ。"""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # コアシステム初期化
        self.model_loader = ModelLoader(
            models_dir=config.get("models_dir", "models"),
            config=config,
        )
        self.character_manager = CharacterManager(
            characters_dir=config.get("characters_dir", "characters"),
        )
        self.chat_engine = ChatEngine(
            model_loader=self.model_loader,
            character_manager=self.character_manager,
        )

        self._settings_dialog = None

        self._setup_ui()
        self._setup_menu()
        self._connect_signals()
        self._update_status()

    # ------------------------------------------------------------------
    # UI構築
    # ------------------------------------------------------------------

    def _setup_ui(self):
        self.setWindowTitle("CharacterLLM")
        ui_cfg = self.config.get("ui", {})
        self.resize(
            ui_cfg.get("window_width", 1920),
            ui_cfg.get("window_height", 1080),
        )
        self.setMinimumSize(800, 600)

        # ダークテーマ適用
        self.setStyleSheet(DARK_THEME)

        # 中央ウィジェット
        central = QWidget()
        self.setCentralWidget(central)

        # メインレイアウト: スプリッター (サイドバー | チャット)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # ---- サイドバー ----
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setStyleSheet(SIDEBAR_STYLE)
        sidebar.setMinimumWidth(260)

        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        # アプリタイトル
        title_widget = QWidget()
        title_layout = QVBoxLayout(title_widget)
        title_layout.setContentsMargins(16, 16, 16, 8)

        title_label = QLabel("CharacterLLM")
        title_label.setStyleSheet(f"""
            color: {COLORS['accent']};
            font-size: 20px;
            font-weight: bold;
        """)
        title_layout.addWidget(title_label)

        subtitle = QLabel("キャラクターAIチャット")
        subtitle.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        title_layout.addWidget(subtitle)

        sidebar_layout.addWidget(title_widget)

        # サイドバータブ
        self.sidebar_tabs = QTabWidget()
        self.sidebar_tabs.setDocumentMode(True)

        # キャラクターパネル
        self.character_panel = CharacterPanel(self.character_manager)
        self.sidebar_tabs.addTab(self.character_panel, "キャラクター")

        # モデルパネル
        self.model_panel = ModelPanel(self.model_loader)
        self.sidebar_tabs.addTab(self.model_panel, "モデル")

        sidebar_layout.addWidget(self.sidebar_tabs, stretch=1)

        self.splitter.addWidget(sidebar)

        # ---- チャットエリア ----
        chat_area = QWidget()
        chat_layout = QVBoxLayout(chat_area)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        chat_layout.setSpacing(0)

        # チャットヘッダー
        self.chat_header = QWidget()
        header_layout = QHBoxLayout(self.chat_header)
        header_layout.setContentsMargins(16, 8, 16, 8)

        self.char_name_label = QLabel("キャラクター未選択")
        self.char_name_label.setStyleSheet(f"""
            color: {COLORS['text_primary']};
            font-size: 16px;
            font-weight: bold;
        """)
        header_layout.addWidget(self.char_name_label)

        header_layout.addStretch()

        self.model_status_label = QLabel("モデル未ロード")
        self.model_status_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px;")
        header_layout.addWidget(self.model_status_label)

        clear_btn = QPushButton("チャットクリア")
        clear_btn.setProperty("secondary", True)
        clear_btn.clicked.connect(self._on_clear_chat)
        header_layout.addWidget(clear_btn)

        self.chat_header.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['bg_medium']};
                border-bottom: 1px solid {COLORS['border']};
            }}
        """)
        chat_layout.addWidget(self.chat_header)

        # チャットウィジェット
        self.chat_widget = ChatWidget(self.chat_engine)
        chat_layout.addWidget(self.chat_widget, stretch=1)

        self.splitter.addWidget(chat_area)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setCollapsible(0, False)
        self.splitter.setSizes([408, 792])

        main_layout.addWidget(self.splitter)

        # ---- ステータスバー ----
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("準備完了")

    # ------------------------------------------------------------------
    # メニューバー
    # ------------------------------------------------------------------

    def _setup_menu(self):
        menubar = self.menuBar()

        # ファイルメニュー
        file_menu = menubar.addMenu("ファイル")

        clear_action = QAction("チャットクリア", self)
        clear_action.triggered.connect(self._on_clear_chat)
        file_menu.addAction(clear_action)

        file_menu.addSeparator()

        settings_action = QAction("設定", self)
        settings_action.triggered.connect(self._on_settings)
        file_menu.addAction(settings_action)

        file_menu.addSeparator()

        quit_action = QAction("終了", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # ヘルプメニュー
        help_menu = menubar.addMenu("ヘルプ")

        about_action = QAction("このアプリについて", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    # ------------------------------------------------------------------
    # シグナル接続
    # ------------------------------------------------------------------

    def _connect_signals(self):
        # キャラクター選択時
        self.character_panel.character_selected.connect(self._on_character_selected)

        # モデルロード/アンロード
        self.model_panel.model_loaded.connect(self._on_model_loaded)
        self.model_panel.model_unloaded.connect(self._on_model_unloaded)

    # ------------------------------------------------------------------
    # イベントハンドラ
    # ------------------------------------------------------------------

    def _on_character_selected(self, character_id: str):
        """キャラクターが選択された時。"""
        try:
            char = self.chat_engine.set_character(character_id)
            self.char_name_label.setText(char.name)
            self.status_bar.showMessage(f"キャラクター変更: {char.name}")
            self._update_status()
        except ValueError as e:
            logger.error("Character selection error: %s", e)

    def _on_model_loaded(self, model_path: str):
        """モデルがロードされた時。"""
        name = Path(model_path).stem
        self.model_status_label.setText(f"モデル: {name}")
        self.model_status_label.setStyleSheet(f"color: {COLORS['success']}; font-size: 12px;")
        self.status_bar.showMessage(f"モデルロード完了: {name}")
        self._update_status()

    def _on_model_unloaded(self):
        """モデルがアンロードされた時。"""
        self.model_status_label.setText("モデル未ロード")
        self.model_status_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px;")
        self.status_bar.showMessage("モデルをアンロードしました")
        self._update_status()

    def _on_clear_chat(self):
        """チャットクリア。"""
        self.chat_widget.clear_chat()
        self.status_bar.showMessage("チャットをクリアしました")

    def _on_settings(self):
        """設定ダイアログを開く。"""
        from src.ui.settings_dialog import SettingsDialog
        if self._settings_dialog is None:
            self._settings_dialog = SettingsDialog(self.config, parent=self)
            self._settings_dialog.settings_saved.connect(self._on_settings_saved)
        self._settings_dialog.exec()

    def _on_settings_saved(self, new_config: dict):
        """設定が保存された時。"""
        self.config.update(new_config)
        # config.jsonに書き戻す
        try:
            config_path = Path("config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            self.status_bar.showMessage("設定を保存しました")
        except Exception as e:
            logger.error("Failed to save config: %s", e)

    def _on_about(self):
        """アプリ情報ダイアログ。"""
        QMessageBox.about(
            self,
            "CharacterLLM について",
            "CharacterLLM v1.0\n\n"
            "キャラクターAIとチャットできるデスクトップアプリケーション。\n\n"
            "GPU推論 (CUDA) 対応\n"
            "対応モデル: Qwen3, Llama, Mistral 等 (GGUF形式)",
        )

    # ------------------------------------------------------------------
    # 状態管理
    # ------------------------------------------------------------------

    def _update_status(self):
        """ステータスバーの情報を更新。"""
        parts = []
        if self.chat_engine.current_character:
            parts.append(f"キャラクター: {self.chat_engine.current_character.name}")
        if self.model_loader.is_loaded:
            name = Path(self.model_loader.current_model_path).stem
            parts.append(f"モデル: {name}")
        if parts:
            self.status_bar.showMessage(" | ".join(parts))

    # ------------------------------------------------------------------
    # ウィンドウクローズ
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        """アプリ終了時にモデルをアンロード。"""
        if self.model_loader.is_loaded:
            self.model_loader.unload_model()
        event.accept()
