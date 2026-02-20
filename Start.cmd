@echo off
chcp 65001 >nul 2>&1
title CharacterLLM

cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo [CharacterLLM] 仮想環境が見つかりません。セットアップを開始します...
    echo.
    python -m venv venv
    if errorlevel 1 (
        echo [エラー] Python が見つかりません。Python 3.11以上をインストールしてください。
        pause
        exit /b 1
    )
    call venv\Scripts\activate.bat
    echo [CharacterLLM] llama-cpp-python CUDA版をインストール中...
    pip install https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.4-cu124/llama_cpp_python-0.3.4-cp312-cp312-win_amd64.whl
    echo [CharacterLLM] 依存パッケージをインストール中...
    pip install -r requirements.txt
    echo.
    echo [CharacterLLM] セットアップ完了！
    echo.
) else (
    call venv\Scripts\activate.bat
)

echo [CharacterLLM] アプリを起動します...
python main.py

if errorlevel 1 (
    echo.
    echo [エラー] アプリが異常終了しました。
    pause
)
