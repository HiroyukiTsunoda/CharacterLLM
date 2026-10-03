@echo off
chcp 65001 >nul 2>&1
title llama-cpp-python CUDA Build

cd /d "%~dp0"

set "VENV_PYTHON=%~dp0venv\Scripts\python.exe"
set "VENV_PIP=%~dp0venv\Scripts\pip.exe"
set "BUILD_DIR=%~dp0_build_llama_cpp_python"

if not exist "%VENV_PYTHON%" (
    echo [Error] venv not found.
    pause
    exit /b 1
)

:: ---- Step 1: CUDA VS Integration ----
echo [Build] Checking CUDA VS Integration...

set "CUDA_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "VS_CUSTOM=C:\Program Files\Microsoft Visual Studio\2022\Professional\MSBuild\Microsoft\VC\v170\BuildCustomizations"

if not exist "%VS_CUSTOM%" (
    set "VS_CUSTOM=C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Microsoft\VC\v170\BuildCustomizations"
)
if not exist "%VS_CUSTOM%" (
    set "VS_CUSTOM=C:\Program Files\Microsoft Visual Studio\2022\Enterprise\MSBuild\Microsoft\VC\v170\BuildCustomizations"
)

if not exist "%VS_CUSTOM%\CUDA 12.8.targets" (
    echo [Build] Copying CUDA MSBuild files...
    copy "%CUDA_DIR%\extras\visual_studio_integration\MSBuildExtensions\*" "%VS_CUSTOM%\"
    if errorlevel 1 (
        echo [Error] Copy failed. Run as administrator.
        pause
        exit /b 1
    )
) else (
    echo [Build] CUDA VS Integration: OK
)
echo.

:: ---- Step 2: VS developer environment ----
echo [Build] Setting up Visual Studio environment...

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
    echo [Error] Visual Studio 2022 not found.
    pause
    exit /b 1
)
call "%VSCMD%"
echo.

:: ---- Step 3: Clone source ----
if exist "%BUILD_DIR%" (
    echo [Build] Removing old build directory...
    rd /s /q "%BUILD_DIR%"
)

echo [Build] Cloning JamePeng/llama-cpp-python v0.3.30 (Qwen3.5 supported)...
git clone --recurse-submodules https://github.com/JamePeng/llama-cpp-python.git "%BUILD_DIR%"
if errorlevel 1 (
    echo [Error] git clone failed.
    pause
    exit /b 1
)
echo [Build] Clone complete.
echo.

:: ---- Step 4: CUDA build ----
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set CMAKE_ARGS=-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89
set CUDACXX=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe
set FORCE_CMAKE=1

echo [Build] Building with CUDA support...
echo [Build] This takes 10-15 minutes. Please wait...
echo.

"%VENV_PIP%" install "%BUILD_DIR%" --force-reinstall --no-cache-dir
if errorlevel 1 (
    echo.
    echo [Error] Build failed.
    pause
    exit /b 1
)
echo.

:: ---- Step 5: Verify ----
echo [Build] Verifying installation...
"%VENV_PYTHON%" -c "import llama_cpp; print('Version:', llama_cpp.__version__)"
if errorlevel 1 (
    echo [Error] Import failed.
    pause
    exit /b 1
)

"%VENV_PYTHON%" -c "import os,llama_cpp;d=os.path.join(os.path.dirname(llama_cpp.__file__),'lib');[print(' ',f,round(os.path.getsize(os.path.join(d,f))/1048576,1),'MB') for f in sorted(os.listdir(d)) if f.endswith('.dll')]"
echo.

:: ---- Step 6: Cleanup ----
echo [Build] Cleaning up...
rd /s /q "%BUILD_DIR%" 2>nul

echo.
echo ============================================
echo [Build] Done!
echo [Build] CUDA llama-cpp-python with Qwen3.5 support installed.
echo [Build] Run Start.cmd to launch the app.
echo ============================================
echo.
pause
