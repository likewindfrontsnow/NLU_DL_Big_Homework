@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Intelligent Note Agent - Launcher

set "VENV_NAME=llm_video_to_note"

set PYTHONIOENCODING=utf-8
set PYTHONLEGACYWINDOWSSTDIO=utf-8

echo ========================================================
echo        Initializing Intelligent Note Agent...
echo ========================================================
echo.

if exist "%VENV_NAME%\Scripts\python.exe" (
    echo [Check] Found existing environment: %VENV_NAME%
    set "TARGET_PYTHON=%VENV_NAME%\Scripts\python.exe"
    goto :InstallDeps
)

echo [Setup] First time setup / Configuring environment...

py -3 --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [Install] Creating virtual env using Python Launcher...
    py -3 -m venv "%VENV_NAME%"
    goto :CheckVenv
)

python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [Install] Creating virtual env using System Python...
    python -m venv "%VENV_NAME%"
    goto :CheckVenv
)

echo.
echo [ERROR] Python not found! Please install Python 3.10+ and add to PATH.
pause
exit /b

:CheckVenv
if not exist "%VENV_NAME%\Scripts\python.exe" (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b
)
set "TARGET_PYTHON=%VENV_NAME%\Scripts\python.exe"
echo [Success] Environment created!

:InstallDeps

if not exist "requirements.txt" goto :CheckBin

echo.
echo ========================================================
echo [2/3] Installing dependencies...
echo       (Please wait, downloading AI models may take time)
echo ========================================================
echo.

"%TARGET_PYTHON%" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

:CheckBin
if exist "bin" (
    set "PATH=%CD%\bin;%PATH%"
    echo.
    echo [Config] Local FFmpeg loaded.
)

:RunApp
echo.
echo [3/3] Starting App...
echo --------------------------------------------------------

"%TARGET_PYTHON%" -m streamlit run app.py

if %errorlevel% neq 0 (
    echo.
    echo [Error] App exited unexpectedly.
    pause
)