"""
QSSスタイルシート: ダークテーマを中心としたアプリ全体のスタイル定義。
"""

# ---------------------------------------------------------------------------
# カラーパレット
# ---------------------------------------------------------------------------

COLORS = {
    "bg_dark": "#1a1b2e",
    "bg_medium": "#232540",
    "bg_light": "#2d2f52",
    "bg_lighter": "#383a64",
    "accent": "#7c6ff7",
    "accent_hover": "#9a8fff",
    "accent_pressed": "#6358d4",
    "text_primary": "#e8e8f0",
    "text_secondary": "#a0a0b8",
    "text_muted": "#6b6b88",
    "border": "#3a3c60",
    "success": "#4caf50",
    "warning": "#ff9800",
    "error": "#f44336",
    "user_bubble": "#7c6ff7",
    "assistant_bubble": "#2d2f52",
    "scrollbar_bg": "#1a1b2e",
    "scrollbar_handle": "#3a3c60",
}


# ---------------------------------------------------------------------------
# メインスタイルシート
# ---------------------------------------------------------------------------

DARK_THEME = f"""
/* ===== ベース ===== */
QMainWindow, QWidget {{
    background-color: {COLORS['bg_dark']};
    color: {COLORS['text_primary']};
    font-family: "Segoe UI", "Yu Gothic UI", "Meiryo", sans-serif;
    font-size: 14px;
}}

/* ===== スクロールバー ===== */
QScrollBar:vertical {{
    background: {COLORS['scrollbar_bg']};
    width: 8px;
    margin: 0;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {COLORS['scrollbar_handle']};
    min-height: 30px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical:hover {{
    background: {COLORS['accent']};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: none;
}}

/* ===== ボタン ===== */
QPushButton {{
    background-color: {COLORS['accent']};
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: bold;
    font-size: 13px;
}}
QPushButton:hover {{
    background-color: {COLORS['accent_hover']};
}}
QPushButton:pressed {{
    background-color: {COLORS['accent_pressed']};
}}
QPushButton:disabled {{
    background-color: {COLORS['bg_lighter']};
    color: {COLORS['text_muted']};
}}

/* セカンダリボタン */
QPushButton[secondary="true"] {{
    background-color: {COLORS['bg_light']};
    border: 1px solid {COLORS['border']};
}}
QPushButton[secondary="true"]:hover {{
    background-color: {COLORS['bg_lighter']};
}}

/* ===== テキスト入力 ===== */
QLineEdit, QTextEdit, QPlainTextEdit {{
    background-color: {COLORS['bg_medium']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 8px 12px;
    selection-background-color: {COLORS['accent']};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
    border-color: {COLORS['accent']};
}}

/* ===== コンボボックス ===== */
QComboBox {{
    background-color: {COLORS['bg_medium']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 6px 12px;
    min-width: 100px;
}}
QComboBox:hover {{
    border-color: {COLORS['accent']};
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}
QComboBox QAbstractItemView {{
    background-color: {COLORS['bg_medium']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border']};
    selection-background-color: {COLORS['accent']};
}}

/* ===== ラベル ===== */
QLabel {{
    color: {COLORS['text_primary']};
}}
QLabel[secondary="true"] {{
    color: {COLORS['text_secondary']};
    font-size: 12px;
}}

/* ===== スライダー ===== */
QSlider::groove:horizontal {{
    border: none;
    height: 4px;
    background: {COLORS['bg_lighter']};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {COLORS['accent']};
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}}
QSlider::handle:horizontal:hover {{
    background: {COLORS['accent_hover']};
}}

/* ===== スピンボックス ===== */
QSpinBox, QDoubleSpinBox {{
    background-color: {COLORS['bg_medium']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 4px 8px;
}}

/* ===== タブ ===== */
QTabWidget::pane {{
    border: 1px solid {COLORS['border']};
    background-color: {COLORS['bg_dark']};
    border-radius: 6px;
}}
QTabBar::tab {{
    background-color: {COLORS['bg_dark']};
    color: {COLORS['text_secondary']};
    padding: 8px 20px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
    outline: none;
}}
QTabBar::tab:selected {{
    background-color: {COLORS['bg_light']};
    color: {COLORS['text_primary']};
    border-bottom: 2px solid {COLORS['accent']};
}}
QTabBar::tab:hover {{
    background-color: {COLORS['bg_light']};
}}

/* ===== リストウィジェット ===== */
QListWidget {{
    background-color: {COLORS['bg_dark']};
    border: none;
    outline: none;
}}
QListWidget::item {{
    background-color: {COLORS['bg_medium']};
    color: {COLORS['text_primary']};
    border-radius: 6px;
    padding: 10px 12px;
    margin: 2px 4px;
}}
QListWidget::item:selected {{
    background-color: {COLORS['accent']};
    color: white;
}}

/* ===== グループボックス ===== */
QGroupBox {{
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    padding: 0 8px;
}}

/* ===== プログレスバー ===== */
QProgressBar {{
    background-color: {COLORS['bg_medium']};
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background-color: {COLORS['accent']};
    border-radius: 4px;
}}

/* ===== メニューバー ===== */
QMenuBar {{
    background-color: {COLORS['bg_dark']};
    color: {COLORS['text_primary']};
}}
QMenuBar::item:selected {{
    background-color: {COLORS['bg_light']};
}}
QMenu {{
    background-color: {COLORS['bg_medium']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border']};
}}
QMenu::item:selected {{
    background-color: {COLORS['accent']};
}}

/* ===== ステータスバー ===== */
QStatusBar {{
    background-color: {COLORS['bg_dark']};
    color: {COLORS['text_secondary']};
    font-size: 12px;
}}

/* ===== スプリッター ===== */
QSplitter::handle {{
    background-color: {COLORS['border']};
    width: 1px;
}}
"""

# ---------------------------------------------------------------------------
# サイドバー専用スタイル
# ---------------------------------------------------------------------------

SIDEBAR_STYLE = f"""
QWidget#sidebar {{
    background-color: {COLORS['bg_medium']};
    border-right: 1px solid {COLORS['border']};
}}
"""

# ---------------------------------------------------------------------------
# チャットバブル用スタイル
# ---------------------------------------------------------------------------

CHAT_BUBBLE_USER = f"""
    background-color: {COLORS['user_bubble']};
    color: white;
    border-radius: 12px;
    padding: 10px 14px;
    font-size: 14px;
"""

CHAT_BUBBLE_ASSISTANT = f"""
    background-color: {COLORS['assistant_bubble']};
    color: {COLORS['text_primary']};
    border-radius: 12px;
    padding: 10px 14px;
    font-size: 14px;
"""
