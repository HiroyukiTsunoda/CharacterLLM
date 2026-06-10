"""
CharacterLLM - キャラクターAIチャット Windowsアプリケーション

エントリポイント。config.jsonを読み込み、PySide6アプリケーションを起動する。
"""

import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

# llama-cpp-python (ggml) と PyTorch が同一 GPU 上で非同期 CUDA カーネルを
# 実行すると cuBLAS ハンドルが競合する。全カーネルを同期実行にして回避する。
# CUDA ライブラリがロードされる前に設定する必要がある。
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

# ---------------------------------------------------------------------------
# ロギング設定
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _purge_old_lines(log_path: Path, max_age: timedelta):
    """ログファイルから *max_age* より古い行を削除する。"""
    if not log_path.exists():
        return
    cutoff = datetime.now() - max_age
    ts_re = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s")
    kept: list[str] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = ts_re.match(line)
            if m:
                ts = datetime.strptime(m.group(1), LOG_DATE_FORMAT)
                if ts >= cutoff:
                    kept.append(line)
            elif kept:
                kept.append(line)
    with open(log_path, "w", encoding="utf-8") as f:
        f.writelines(kept)


def setup_logging(level: str = "INFO"):
    """ログ出力を設定する。"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    _purge_old_lines(log_dir / "app.log", timedelta(hours=24))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "app.log", encoding="utf-8"),
        ],
    )

    # LLM 生出力専用ロガー（フィルタ前の生テキストをそのまま記録）
    raw_logger = logging.getLogger("llm_raw")
    raw_logger.setLevel(logging.DEBUG)
    raw_logger.propagate = False
    raw_handler = logging.FileHandler(log_dir / "llm_raw.log", encoding="utf-8")
    raw_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
    raw_logger.addHandler(raw_handler)


# ---------------------------------------------------------------------------
# 設定読込
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.json") -> dict:
    """config.json を読み込む。なければデフォルト値を返す。"""
    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return _default_config()


def _default_config() -> dict:
    """デフォルト設定を返す。"""
    return {
        "models_dir": "models",
        "characters_dir": "characters",
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
        "voice": {
            "input_device": None,
            "output_device": None,
            "whisper_model": "small",
            "device": "cuda",
            "compute_type": "float16",
            "language": "ja",
            "auto_send": False,
        },
        "tts": {
            "enabled": False,
            "use_gpu": True,
            "auto_play": True,
            "qwen3": {
                "default_model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                "dtype": "bfloat16",
                "flash_attention": True,
            },
        },
        "openai": {
            "model": "gpt-5.4-mini",
            "reasoning_effort": "low",
            "enabled": False,
        },
    }


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main():
    """アプリケーションを起動する。"""
    # 作業ディレクトリをスクリプトのあるディレクトリに変更
    os.chdir(Path(__file__).parent)

    load_dotenv()
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("CharacterLLM starting...")

    config = load_config()
    logger.info("Config loaded.")

    # 初回起動時はデフォルト設定をファイルに書き出す
    if not Path("config.json").exists():
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        logger.info("Default config.json created.")

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
