@echo off
chcp 65001 >nul 2>&1
title llama-cpp-python CUDA Build

cd /d "%~dp0"

:: ---- Step 1: CUDA VS Integration を修復 ----
echo [Build] CUDA Visual Studio Integration を確認中...

set "CUDA_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "VS_CUSTOM=C:\Program Files\Microsoft Visual Studio\2022\Professional\MSBuild\Microsoft\VC\v170\BuildCustomizations"

if not exist "%VS_CUSTOM%" (
    set "VS_CUSTOM=C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Microsoft\VC\v170\BuildCustomizations"
)
if not exist "%VS_CUSTOM%" (
    set "VS_CUSTOM=C:\Program Files\Microsoft Visual Studio\2022\Enterprise\MSBuild\Microsoft\VC\v170\BuildCustomizations"
)

:: CUDA toolset ファイルがVSにコピーされているか確認
if not exist "%VS_CUSTOM%\CUDA 12.8.targets" (
    echo [Build] CUDA MSBuild ファイルを Visual Studio にコピーします...
    echo [Build] 管理者権限が必要な場合があります。
    copy "%CUDA_DIR%\extras\visual_studio_integration\MSBuildExtensions\*" "%VS_CUSTOM%\"
    if errorlevel 1 (
        echo.
        echo [エラー] コピーに失敗しました。このスクリプトを右クリック→「管理者として実行」してください。
        pause
        exit /b 1
    )
    echo [Build] CUDA MSBuild ファイルをコピーしました。
) else (
    echo [Build] CUDA VS Integration: OK
)

echo.

:: ---- Step 2: 開発者環境 + ビルド ----
echo [Build] Visual Studio 開発者環境をセットアップ中...

set "VSCMD="
if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
    set "VSCMD=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
)
if not defined VSCMD if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    set "VSCMD=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
)
if not defined VSCMD if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" (
    set "VSCMD=C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
)
if not defined VSCMD (
    echo [エラー] Visual Studio 2022 が見つかりません。
    pause
    exit /b 1
)
call "%VSCMD%"
echo.

:: ---- Step 3: ビルド ----
call venv\Scripts\activate.bat

:: VS Generator を使用 (Ninja ではなく)
set CMAKE_GENERATOR=
set CMAKE_ARGS=-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89
set CUDACXX=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe

echo [Build] llama-cpp-python をCUDA付きでビルド中...
echo [Build] Visual Studio Generator + CUDA 12.8
echo [Build] 数分かかります。お待ちください...
echo.

pip install llama-cpp-python --force-reinstall --no-cache-dir

if errorlevel 1 (
    echo.
    echo [エラー] ビルドに失敗しました。
    pause
    exit /b 1
)

echo.
echo [Build] 完了！CUDA対応の llama-cpp-python がインストールされました。
echo [Build] Start.cmd でアプリを起動してください。
echo.
pause
