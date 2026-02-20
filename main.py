"""
CharacterLLM - キャラクターAIチャット Windowsアプリケーション

エントリポイント。config.jsonを読み込み、PySide6アプリケーションを起動する。
"""

import json
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# ロギング設定
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO"):
    """ログ出力を設定する。"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "app.log", encoding="utf-8"),
        ],
    )


# ---------------------------------------------------------------------------
# 設定読込
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.json") -> dict:
    """config.json を読み込む。なければデフォルト値を返す。"""
    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # デフォルト設定
    default_config = {
        "models_dir": "models",
        "characters_dir": "characters",
        "database_path": "data/chat_history.db",
        "default_model": None,
        "default_character": "default_assistant",
        "gpu": {
            "n_gpu_layers": -1,
            "main_gpu": 0,
        },
        "inference": {
            "context_length": 4096,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 512,
            "repeat_penalty": 1.1,
        },
        "ui": {
            "theme": "dark",
            "window_width": 1920,
            "window_height": 1080,
            "font_size": 14,
        },
    }

    # デフォルト設定を保存
    with open(path, "w", encoding="utf-8") as f:
        json.dump(default_config, f, ensure_ascii=False, indent=2)

    return default_config


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main():
    """アプリケーションを起動する。"""
    # 作業ディレクトリをスクリプトのあるディレクトリに変更
    os.chdir(Path(__file__).parent)

    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("CharacterLLM starting...")

    config = load_config()
    logger.info("Config loaded.")

    # PySide6 アプリケーション
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFont

    app = QApplication(sys.argv)

    # アプリ全体のフォント設定
    font_size = config.get("ui", {}).get("font_size", 14)
    font = QFont("Segoe UI", font_size)
    font.setFamilies(["Segoe UI", "Yu Gothic UI", "Meiryo", "sans-serif"])
    app.setFont(font)

    # アプリケーション情報
    app.setApplicationName("CharacterLLM")
    app.setOrganizationName("CharacterLLM")
    app.setApplicationVersion("1.0.0")

    # メインウィンドウ
    from src.ui.main_window import MainWindow
    window = MainWindow(config)
    window.show()

    # デフォルトキャラクターを選択
    default_char = config.get("default_character")
    if default_char:
        try:
            window.character_panel.select_character(default_char)
        except Exception:
            logger.warning("Default character not found: %s", default_char)

    logger.info("Application started.")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
