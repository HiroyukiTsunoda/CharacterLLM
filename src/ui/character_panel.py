"""
CharacterPanel: キャラクター選択リスト、新規作成・編集フォーム。
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.core.character import Character, CharacterManager, GenerationParams, Personality
from src.ui.styles import COLORS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# キャラクター編集ダイアログ
# ---------------------------------------------------------------------------

class CharacterEditDialog(QDialog):
    """キャラクターの新規作成・編集ダイアログ。"""

    def __init__(self, character: Character | None = None, parent=None):
        super().__init__(parent)
        self.character = character
        self.setWindowTitle("キャラクター編集" if character else "キャラクター新規作成")
        self.setMinimumWidth(500)
        self.setMinimumHeight(600)
        self._setup_ui()

        if character:
            self._load_character(character)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # ---- 基本情報 ----
        basic_group = QGroupBox("基本情報")
        basic_layout = QFormLayout()
        basic_layout.setSpacing(8)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("キャラクター名を入力")
        basic_layout.addRow("名前:", self.name_edit)

        self.avatar_edit = QLineEdit()
        self.avatar_edit.setPlaceholderText("アバター画像ファイル名 (オプション)")
        basic_layout.addRow("アバター:", self.avatar_edit)

        basic_group.setLayout(basic_layout)
        layout.addWidget(basic_group)

        # ---- 性格設定 ----
        personality_group = QGroupBox("性格設定")
        personality_layout = QFormLayout()
        personality_layout.setSpacing(8)

        self.system_prompt_edit = QPlainTextEdit()
        self.system_prompt_edit.setPlaceholderText("キャラクターの基本設定を記述\n例: あなたは優しい執事のセバスチャンです。丁寧で品のある言葉遣いをします。")
        self.system_prompt_edit.setMinimumHeight(100)
        personality_layout.addRow("システムプロンプト:", self.system_prompt_edit)

        self.tone_edit = QLineEdit()
        self.tone_edit.setPlaceholderText("例: 丁寧、優しい、ツンデレ")
        personality_layout.addRow("口調:", self.tone_edit)

        self.speech_style_edit = QLineEdit()
        self.speech_style_edit.setPlaceholderText("例: 「〜でございます」「〜なのだ」")
        personality_layout.addRow("話し方:", self.speech_style_edit)

        self.background_edit = QPlainTextEdit()
        self.background_edit.setPlaceholderText("キャラクターの背景・設定")
        self.background_edit.setMaximumHeight(80)
        personality_layout.addRow("背景:", self.background_edit)

        personality_group.setLayout(personality_layout)
        layout.addWidget(personality_group)

        # ---- 生成パラメータ ----
        params_group = QGroupBox("生成パラメータ")
        params_layout = QFormLayout()
        params_layout.setSpacing(8)

        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 2.0)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setValue(0.7)
        self.temperature_spin.setToolTip("高い値ほど創造的な回答になります")
        params_layout.addRow("Temperature:", self.temperature_spin)

        self.top_p_spin = QDoubleSpinBox()
        self.top_p_spin.setRange(0.0, 1.0)
        self.top_p_spin.setSingleStep(0.05)
        self.top_p_spin.setValue(0.9)
        params_layout.addRow("Top P:", self.top_p_spin)

        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(64, 4096)
        self.max_tokens_spin.setSingleStep(64)
        self.max_tokens_spin.setValue(512)
        params_layout.addRow("最大トークン数:", self.max_tokens_spin)

        self.repeat_penalty_spin = QDoubleSpinBox()
        self.repeat_penalty_spin.setRange(1.0, 2.0)
        self.repeat_penalty_spin.setSingleStep(0.05)
        self.repeat_penalty_spin.setValue(1.1)
        params_layout.addRow("繰り返しペナルティ:", self.repeat_penalty_spin)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # ---- ボタン ----
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("キャンセル")
        cancel_btn.setProperty("secondary", True)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        save_btn = QPushButton("保存")
        save_btn.clicked.connect(self.accept)
        btn_layout.addWidget(save_btn)

        layout.addLayout(btn_layout)

    def _load_character(self, char: Character):
        """既存キャラクターのデータをフォームに読み込む。"""
        self.name_edit.setText(char.name)
        self.avatar_edit.setText(char.avatar)
        self.system_prompt_edit.setPlainText(char.personality.system_prompt)
        self.tone_edit.setText(char.personality.tone)
        self.speech_style_edit.setText(char.personality.speech_style)
        self.background_edit.setPlainText(char.personality.background)
        self.temperature_spin.setValue(char.generation_params.temperature)
        self.top_p_spin.setValue(char.generation_params.top_p)
        self.max_tokens_spin.setValue(char.generation_params.max_tokens)
        self.repeat_penalty_spin.setValue(char.generation_params.repeat_penalty)

    def get_character_data(self) -> dict:
        """フォームの入力値を辞書で返す。"""
        return {
            "name": self.name_edit.text().strip(),
            "avatar": self.avatar_edit.text().strip(),
            "personality": {
                "system_prompt": self.system_prompt_edit.toPlainText().strip(),
                "tone": self.tone_edit.text().strip(),
                "speech_style": self.speech_style_edit.text().strip(),
                "background": self.background_edit.toPlainText().strip(),
            },
            "generation_params": {
                "temperature": self.temperature_spin.value(),
                "top_p": self.top_p_spin.value(),
                "max_tokens": self.max_tokens_spin.value(),
                "repeat_penalty": self.repeat_penalty_spin.value(),
            },
        }


# ---------------------------------------------------------------------------
# CharacterPanel
# ---------------------------------------------------------------------------

class CharacterPanel(QWidget):
    """サイドバーに配置するキャラクター選択・管理パネル。"""

    character_selected = Signal(str)  # character_id

    def __init__(self, character_manager: CharacterManager, parent=None):
        super().__init__(parent)
        self.character_manager = character_manager
        self._setup_ui()
        self.refresh_list()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ヘッダー
        header = QLabel("キャラクター")
        header.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 15px; font-weight: bold;")
        layout.addWidget(header)

        # キャラクターリスト
        self.char_list = QListWidget()
        self.char_list.currentItemChanged.connect(self._on_selection_changed)
        layout.addWidget(self.char_list, stretch=1)

        # ボタン群
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)

        self.add_btn = QPushButton("新規")
        self.add_btn.clicked.connect(self._on_add)
        btn_layout.addWidget(self.add_btn)

        self.edit_btn = QPushButton("編集")
        self.edit_btn.setProperty("secondary", True)
        self.edit_btn.clicked.connect(self._on_edit)
        btn_layout.addWidget(self.edit_btn)

        self.delete_btn = QPushButton("削除")
        self.delete_btn.setProperty("secondary", True)
        self.delete_btn.clicked.connect(self._on_delete)
        btn_layout.addWidget(self.delete_btn)

        layout.addLayout(btn_layout)

    # ------------------------------------------------------------------
    # リスト操作
    # ------------------------------------------------------------------

    def refresh_list(self):
        """キャラクターリストを再読込。"""
        self.char_list.clear()
        for char in self.character_manager.list_characters():
            item = QListWidgetItem(char.name)
            item.setData(Qt.ItemDataRole.UserRole, char.id)
            self.char_list.addItem(item)

    def select_character(self, character_id: str):
        """プログラムからキャラクターを選択する。"""
        for i in range(self.char_list.count()):
            item = self.char_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == character_id:
                self.char_list.setCurrentItem(item)
                break

    def _on_selection_changed(self, current: QListWidgetItem, previous: QListWidgetItem):
        if current:
            char_id = current.data(Qt.ItemDataRole.UserRole)
            self.character_selected.emit(char_id)

    # ------------------------------------------------------------------
    # CRUD操作
    # ------------------------------------------------------------------

    def _on_add(self):
        """新規キャラクター作成。"""
        dialog = CharacterEditDialog(parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            data = dialog.get_character_data()
            if not data["name"]:
                QMessageBox.warning(self, "エラー", "名前を入力してください。")
                return
            char = self.character_manager.create_character(
                name=data["name"],
                system_prompt=data["personality"]["system_prompt"],
                tone=data["personality"]["tone"],
                speech_style=data["personality"]["speech_style"],
                background=data["personality"]["background"],
                temperature=data["generation_params"]["temperature"],
                top_p=data["generation_params"]["top_p"],
                max_tokens=data["generation_params"]["max_tokens"],
                repeat_penalty=data["generation_params"]["repeat_penalty"],
                avatar=data["avatar"],
            )
            self.refresh_list()
            self.select_character(char.id)

    def _on_edit(self):
        """選択中のキャラクターを編集。"""
        current = self.char_list.currentItem()
        if not current:
            return
        char_id = current.data(Qt.ItemDataRole.UserRole)
        char = self.character_manager.get_character(char_id)
        if not char:
            return

        dialog = CharacterEditDialog(character=char, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            data = dialog.get_character_data()
            char.name = data["name"]
            char.avatar = data["avatar"]
            char.personality = Personality.from_dict(data["personality"])
            char.generation_params = GenerationParams.from_dict(data["generation_params"])
            self.character_manager.save_character(char)
            self.refresh_list()
            self.select_character(char.id)

    def _on_delete(self):
        """選択中のキャラクターを削除。"""
        current = self.char_list.currentItem()
        if not current:
            return
        char_id = current.data(Qt.ItemDataRole.UserRole)
        char = self.character_manager.get_character(char_id)
        if not char:
            return

        reply = QMessageBox.question(
            self,
            "削除確認",
            f"キャラクター「{char.name}」を削除しますか？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.character_manager.delete_character(char_id)
            self.refresh_list()
